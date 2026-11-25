"""Run one real prompt through the from-scratch GPT-2 and write what the site shows.

For every block in the flow chart this records the source code the page displays
(read from `gpt2.py`, so the page cannot drift from the code that ran) and the real
tensors that went in and came out, trimmed to a readable slice.

    python export_site_data.py          # writes src/data/models/gpt2.json in the site
"""

import inspect
import json
import textwrap
from pathlib import Path

import torch

import gpt2
from gpt2 import (
    GPT2_ENCODING, apply_causal_mask, attention_scores, attention_weights, gelu, generate,
    merge_heads, split_heads, tokenize,
)
from load_hf_weights import build_gpt2_from_hf, load_hf_gpt2

PROMPT = "Alan Turing was a"
NUM_GENERATED_TOKENS = 10
NUM_TOP_TOKENS = 8
COLUMNS_SHOWN = 6
DECIMALS = 3
TRACED_LAYER = 0
LOGIT_TOLERANCE = 1e-3
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "src" / "data" / "models" / "gpt2.json"


# ---------------------------------------------------------------- source snippets

def source_of(*objects) -> str:
    return "\n\n".join(textwrap.dedent(inspect.getsource(obj)).rstrip() for obj in objects)


def lines_containing(obj, *needles: str) -> str:
    lines = inspect.getsource(obj).splitlines()
    picked = [line for line in lines if any(needle in line for needle in needles)]
    if not picked:
        raise ValueError(f"no line of {obj.__qualname__} contains {needles}")
    return textwrap.dedent("\n".join(picked))


def tokenizer_snippet() -> str:
    encoding_line = lines_containing(gpt2, "GPT2_ENCODING = tiktoken")
    return encoding_line + "\n\n\n" + source_of(gpt2.tokenize, gpt2.detokenize)


def build_snippets() -> dict[str, str]:
    attention_forward = gpt2.CausalSelfAttention.forward
    return {
        "gpt2": source_of(gpt2.GPT2),
        "tokenizer": tokenizer_snippet(),
        "embeddings": source_of(gpt2.Embeddings),
        "transformer-block": source_of(gpt2.TransformerBlock),
        "layer-norm": source_of(gpt2.LayerNorm),
        "attention": source_of(gpt2.CausalSelfAttention),
        "qkv-projection": lines_containing(attention_forward, "qkv_projection(x)"),
        "split-heads": source_of(split_heads),
        "attention-scores": source_of(attention_scores),
        "causal-mask": source_of(apply_causal_mask),
        "softmax": source_of(attention_weights),
        "weighted-sum": lines_containing(attention_forward, "weights @ values"),
        "merge-heads": source_of(merge_heads) + "\n\n\n" + lines_containing(attention_forward, "merge_heads(per"),
        "residual": lines_containing(gpt2.TransformerBlock.forward, "hidden = hidden +"),
        "mlp": source_of(gpt2.MLP),
        "up-projection": lines_containing(gpt2.MLP, "up_projection = ", "hidden_dim = config"),
        "gelu": source_of(gelu),
        "down-projection": lines_containing(gpt2.MLP, "down_projection = "),
        "final-norm": lines_containing(gpt2.GPT2, "final_norm"),
        "lm-head": source_of(gpt2.LMHead),
        "generate": source_of(generate),
    }


# ---------------------------------------------------------------- tensor previews

def token_texts(token_ids: torch.Tensor) -> list[str]:
    return [GPT2_ENCODING.decode([token_id]) for token_id in token_ids.tolist()]


def rounded(values: torch.Tensor) -> list:
    return [[round(value, DECIMALS) if abs(value) != float("inf") else None for value in row]
            for row in values.tolist()]


def matrix_preview(name: str, tensor: torch.Tensor, row_labels: list[str], note: str = "",
                   with_row_stats: bool = False) -> dict:
    preview = {
        "kind": "matrix",
        "name": name,
        "shape": list(tensor.shape),
        "note": note,
        "row_labels": row_labels,
        "values": rounded(tensor[:, :COLUMNS_SHOWN]),
        "hidden_columns": max(tensor.shape[1] - COLUMNS_SHOWN, 0),
    }
    if with_row_stats:
        preview["row_means"] = [round(v, DECIMALS) for v in tensor.mean(dim=-1).tolist()]
        preview["row_stds"] = [round(v, DECIMALS) for v in tensor.std(dim=-1, unbiased=False).tolist()]
    return preview


def head_preview(name: str, tensor: torch.Tensor, row_labels: list[str], head: int = 0) -> dict:
    note = f"head {head} of {tensor.shape[0]}"
    preview = matrix_preview(name, tensor[head], row_labels, note)
    preview["shape"] = list(tensor.shape)
    return preview


def square_preview(name: str, tensor: torch.Tensor, labels: list[str], head: int = 0) -> dict:
    return {
        "kind": "square",
        "name": name,
        "shape": list(tensor.shape),
        "note": f"head {head} of {tensor.shape[0]}",
        "row_labels": labels,
        "column_labels": labels,
        "values": rounded(tensor[head]),
    }


def token_preview(name: str, token_ids: torch.Tensor) -> dict:
    return {
        "kind": "tokens",
        "name": name,
        "shape": list(token_ids.shape),
        "tokens": [{"id": token_id, "text": text}
                   for token_id, text in zip(token_ids.tolist(), token_texts(token_ids))],
    }


def text_preview(name: str, text: str) -> dict:
    return {"kind": "text", "name": name, "text": text}


def top_tokens_preview(name: str, logits: torch.Tensor, labels: list[str]) -> dict:
    probabilities = torch.softmax(logits[-1], dim=-1)
    top = probabilities.topk(NUM_TOP_TOKENS)
    return {
        "kind": "top_tokens",
        "name": name,
        "shape": list(logits.shape),
        "note": f"after the last token, {labels[-1]!r}",
        "candidates": [{"text": GPT2_ENCODING.decode([token_id]), "probability": round(p, 4)}
                       for p, token_id in zip(top.values.tolist(), top.indices.tolist())],
    }


def predictions_preview(name: str, logits: torch.Tensor, labels: list[str]) -> dict:
    probabilities = torch.softmax(logits, dim=-1)
    best_probability, best_token = probabilities.max(dim=-1)
    return {
        "kind": "predictions",
        "name": name,
        "shape": list(logits.shape),
        "rows": [{"after": label, "predicted": GPT2_ENCODING.decode([token_id]), "probability": round(p, 4)}
                 for label, token_id, p in zip(labels, best_token.tolist(), best_probability.tolist())],
    }


# ---------------------------------------------------------------- recording a forward pass

def record_module_outputs(model: gpt2.GPT2, token_ids: torch.Tensor) -> tuple[dict, torch.Tensor]:
    traced_block = model.blocks[TRACED_LAYER]
    watched = {
        "embeddings": model.embeddings,
        "block_in": traced_block,
        "attention_norm": traced_block.attention_norm,
        "attention": traced_block.attention,
        "mlp_norm": traced_block.mlp_norm,
        "mlp": traced_block.mlp,
        "up_projection": traced_block.mlp.up_projection,
        "last_block": model.blocks[-1],
        "final_norm": model.final_norm,
    }
    captured = {}
    handles = [module.register_forward_hook(capture_into(captured, name)) for name, module in watched.items()]
    with torch.no_grad():
        logits = model(token_ids)
    for handle in handles:
        handle.remove()
    return captured, logits


def capture_into(captured: dict, name: str):
    def hook(_module, inputs, output):
        captured[name] = {"input": inputs[0].detach(), "output": output.detach()}
    return hook


@torch.no_grad()
def trace_attention(attention: gpt2.CausalSelfAttention, x: torch.Tensor, expected: torch.Tensor) -> dict:
    queries, keys, values = attention.qkv_projection(x).chunk(3, dim=-1)
    head_queries = split_heads(queries, attention.num_heads)
    head_keys = split_heads(keys, attention.num_heads)
    head_values = split_heads(values, attention.num_heads)
    scores = attention_scores(head_queries, head_keys)
    masked = apply_causal_mask(scores)
    weights = attention_weights(head_queries, head_keys)
    per_head_output = weights @ head_values
    merged = merge_heads(per_head_output)
    output = attention.output_projection(merged)
    if not torch.allclose(output, expected, atol=1e-5):
        raise AssertionError("step-by-step attention trace disagrees with CausalSelfAttention.forward")
    return {
        "queries": queries, "keys": keys, "values": values,
        "head_queries": head_queries, "head_keys": head_keys, "head_values": head_values,
        "scores": scores, "masked": masked, "weights": weights,
        "per_head_output": per_head_output, "merged": merged, "output": output,
    }


@torch.no_grad()
def generation_steps(model: gpt2.GPT2, token_ids: torch.Tensor) -> list[dict]:
    steps = []
    for _ in range(NUM_GENERATED_TOKENS):
        probabilities = torch.softmax(model(token_ids)[-1], dim=-1)
        next_token = probabilities.argmax()
        steps.append({"text": GPT2_ENCODING.decode([next_token.item()]),
                      "probability": round(probabilities[next_token].item(), 4)})
        token_ids = torch.cat([token_ids, next_token.unsqueeze(0)])
    if [s["text"] for s in steps] != token_texts(generate(model, token_ids[:-NUM_GENERATED_TOKENS], NUM_GENERATED_TOKENS)[-NUM_GENERATED_TOKENS:]):
        raise AssertionError("recorded generation steps disagree with gpt2.generate")
    return steps


# ---------------------------------------------------------------- parameter counts

def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def parameter_counts(model: gpt2.GPT2) -> dict[str, int]:
    block = model.blocks[0]
    return {
        "gpt2": parameter_count(model),
        "embeddings": parameter_count(model.embeddings),
        "transformer-block": parameter_count(block),
        "layer-norm": parameter_count(block.attention_norm),
        "attention": parameter_count(block.attention),
        "qkv-projection": parameter_count(block.attention.qkv_projection),
        "merge-heads": parameter_count(block.attention.output_projection),
        "mlp": parameter_count(block.mlp),
        "up-projection": parameter_count(block.mlp.up_projection),
        "down-projection": parameter_count(block.mlp.down_projection),
        "final-norm": parameter_count(model.final_norm),
        "lm-head": 0,
    }


# ---------------------------------------------------------------- assembling the page data

def build_examples(model: gpt2.GPT2, token_ids: torch.Tensor) -> dict:
    labels = token_texts(token_ids)
    captured, logits = record_module_outputs(model, token_ids)
    traced_block = model.blocks[TRACED_LAYER]
    attention = trace_attention(traced_block.attention, captured["attention_norm"]["output"],
                                captured["attention"]["output"])
    block_in = captured["block_in"]["input"]
    after_attention = block_in + attention["output"]
    up = captured["up_projection"]["output"]
    with torch.no_grad():
        token_part = model.embeddings.token_embedding(token_ids)
        position_part = model.embeddings.position_embedding(torch.arange(len(token_ids)))
    layer = f"layer {TRACED_LAYER}"

    def matrix(name, tensor, note="", **kwargs):
        return matrix_preview(name, tensor, labels, note, **kwargs)

    return {
        "gpt2": {"inputs": [text_preview("prompt", PROMPT)],
                 "outputs": [top_tokens_preview("next-token probabilities", logits, labels)]},
        "tokenizer": {"inputs": [text_preview("text", PROMPT)],
                      "outputs": [token_preview("token ids", token_ids)]},
        "embeddings": {"inputs": [token_preview("token ids", token_ids)],
                       "outputs": [matrix("token embedding", token_part),
                                   matrix("position embedding", position_part),
                                   matrix("sum = hidden", captured["embeddings"]["output"])]},
        "transformer-block": {"inputs": [matrix("hidden", block_in, layer)],
                              "outputs": [matrix("hidden", captured["block_in"]["output"], layer)]},
        "layer-norm": {"inputs": [matrix("hidden", block_in, layer, with_row_stats=True)],
                       "outputs": [matrix("normalized", captured["attention_norm"]["output"], layer,
                                          with_row_stats=True)]},
        "attention": {"inputs": [matrix("x", captured["attention"]["input"], layer)],
                      "outputs": [matrix("attention output", attention["output"], layer)]},
        "qkv-projection": {"inputs": [matrix("x", captured["attention"]["input"], layer)],
                           "outputs": [matrix("queries", attention["queries"]),
                                       matrix("keys", attention["keys"]),
                                       matrix("values", attention["values"])]},
        "split-heads": {"inputs": [matrix("queries", attention["queries"])],
                        "outputs": [head_preview("queries, split", attention["head_queries"], labels)]},
        "attention-scores": {"inputs": [head_preview("queries", attention["head_queries"], labels),
                                        head_preview("keys", attention["head_keys"], labels)],
                             "outputs": [square_preview("scores", attention["scores"], labels)]},
        "causal-mask": {"inputs": [square_preview("scores", attention["scores"], labels)],
                        "outputs": [square_preview("masked scores", attention["masked"], labels)]},
        "softmax": {"inputs": [square_preview("masked scores", attention["masked"], labels)],
                    "outputs": [square_preview("attention weights", attention["weights"], labels)]},
        "weighted-sum": {"inputs": [square_preview("weights", attention["weights"], labels),
                                    head_preview("values", attention["head_values"], labels)],
                         "outputs": [head_preview("per-head output", attention["per_head_output"], labels)]},
        "merge-heads": {"inputs": [head_preview("per-head output", attention["per_head_output"], labels)],
                        "outputs": [matrix("merged", attention["merged"]),
                                    matrix("after output projection", attention["output"])]},
        "residual": {"inputs": [matrix("hidden (skip path)", block_in, layer),
                                matrix("attention output", attention["output"], layer)],
                     "outputs": [matrix("hidden + attention", after_attention, layer)]},
        "mlp": {"inputs": [matrix("x", captured["mlp"]["input"], layer)],
                "outputs": [matrix("mlp output", captured["mlp"]["output"], layer)]},
        "up-projection": {"inputs": [matrix("x", captured["mlp"]["input"], layer)],
                          "outputs": [matrix("expanded", up, layer)]},
        "gelu": {"inputs": [matrix("expanded", up, layer)],
                 "outputs": [matrix("activated", gelu(up), layer)]},
        "down-projection": {"inputs": [matrix("activated", gelu(up), layer)],
                            "outputs": [matrix("mlp output", captured["mlp"]["output"], layer)]},
        "final-norm": {"inputs": [matrix("hidden", captured["final_norm"]["input"], "after layer 11",
                                         with_row_stats=True)],
                       "outputs": [matrix("normalized", captured["final_norm"]["output"],
                                          with_row_stats=True)]},
        "lm-head": {"inputs": [matrix("normalized", captured["final_norm"]["output"])],
                    "outputs": [predictions_preview("best guess at every position", logits, labels),
                                top_tokens_preview("next-token probabilities", logits, labels)]},
        "generate": {"inputs": [text_preview("prompt", PROMPT)],
                     "outputs": [{"kind": "steps", "name": "greedy steps",
                                  "steps": generation_steps(model, token_ids)}]},
    }, attention["weights"], logits


def verify_against_hugging_face(hf_model, model: gpt2.GPT2, token_ids: torch.Tensor, logits: torch.Tensor) -> float:
    with torch.no_grad():
        hf_logits = hf_model(token_ids.unsqueeze(0)).logits[0]
    difference = (logits - hf_logits).abs().max().item()
    if difference > LOGIT_TOLERANCE:
        raise AssertionError(f"logits differ from Hugging Face by {difference}")
    return difference


def main() -> None:
    torch.manual_seed(0)
    hf_model = load_hf_gpt2()
    model = build_gpt2_from_hf(hf_model)
    token_ids = tokenize(PROMPT)
    examples, weights, logits = build_examples(model, token_ids)
    difference = verify_against_hugging_face(hf_model, model, token_ids, logits)
    snippets = build_snippets()
    counts = parameter_counts(model)
    continuation = "".join(step["text"] for step in examples["generate"]["outputs"][0]["steps"])
    data = {
        "prompt": PROMPT,
        "continuation": continuation,
        "tokens": token_texts(token_ids),
        "traced_layer": TRACED_LAYER,
        "max_logit_difference_vs_hf": difference,
        "attention_weights_all_heads": rounded(weights.reshape(-1, weights.shape[-1])),
        "num_heads": weights.shape[0],
        "blocks": {block_id: {"snippet": snippets[block_id], "parameters": counts.get(block_id), **io}
                   for block_id, io in examples.items()},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {OUTPUT_PATH} | {PROMPT!r} ->{continuation!r} | max |Δlogit| vs HF = {difference:.2e}")


if __name__ == "__main__":
    main()
