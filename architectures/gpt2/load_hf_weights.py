"""Copy OpenAI's released GPT-2 weights (via Hugging Face) into the from-scratch model.

Hugging Face stores GPT-2's linear layers as `Conv1D`, whose weight is [in, out];
`nn.Linear` stores [out, in]. Those four matrices are transposed on the way in.
"""

import torch
from transformers import GPT2LMHeadModel

from gpt2 import GPT2, GPT2Config

HF_MODEL_NAME = "gpt2"
CONV1D_SUFFIXES = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def load_hf_gpt2() -> GPT2LMHeadModel:
    return GPT2LMHeadModel.from_pretrained(HF_MODEL_NAME).eval()


def build_gpt2_from_hf(hf_model: GPT2LMHeadModel) -> GPT2:
    model = GPT2(GPT2Config()).eval()
    hf_state = hf_model.state_dict()
    with torch.no_grad():
        for our_name, hf_name in parameter_name_pairs(model.config):
            our_parameter = model.get_parameter(our_name)
            hf_tensor = hf_state[hf_name]
            if hf_name.endswith(CONV1D_SUFFIXES):
                hf_tensor = hf_tensor.T
            if our_parameter.shape != hf_tensor.shape:
                raise ValueError(f"{our_name} {tuple(our_parameter.shape)} != {hf_name} {tuple(hf_tensor.shape)}")
            our_parameter.copy_(hf_tensor)
    return model


def parameter_name_pairs(config: GPT2Config) -> list[tuple[str, str]]:
    pairs = [
        ("embeddings.token_embedding.weight", "transformer.wte.weight"),
        ("embeddings.position_embedding.weight", "transformer.wpe.weight"),
        ("final_norm.scale", "transformer.ln_f.weight"),
        ("final_norm.shift", "transformer.ln_f.bias"),
    ]
    for layer in range(config.num_layers):
        pairs += block_parameter_name_pairs(f"blocks.{layer}", f"transformer.h.{layer}")
    return pairs


def block_parameter_name_pairs(ours: str, theirs: str) -> list[tuple[str, str]]:
    return [
        (f"{ours}.attention_norm.scale", f"{theirs}.ln_1.weight"),
        (f"{ours}.attention_norm.shift", f"{theirs}.ln_1.bias"),
        (f"{ours}.attention.qkv_projection.weight", f"{theirs}.attn.c_attn.weight"),
        (f"{ours}.attention.qkv_projection.bias", f"{theirs}.attn.c_attn.bias"),
        (f"{ours}.attention.output_projection.weight", f"{theirs}.attn.c_proj.weight"),
        (f"{ours}.attention.output_projection.bias", f"{theirs}.attn.c_proj.bias"),
        (f"{ours}.mlp_norm.scale", f"{theirs}.ln_2.weight"),
        (f"{ours}.mlp_norm.shift", f"{theirs}.ln_2.bias"),
        (f"{ours}.mlp.up_projection.weight", f"{theirs}.mlp.c_fc.weight"),
        (f"{ours}.mlp.up_projection.bias", f"{theirs}.mlp.c_fc.bias"),
        (f"{ours}.mlp.down_projection.weight", f"{theirs}.mlp.c_proj.weight"),
        (f"{ours}.mlp.down_projection.bias", f"{theirs}.mlp.c_proj.bias"),
    ]
