import pytest
import torch

from gpt2 import generate, tokenize
from load_hf_weights import build_gpt2_from_hf, load_hf_gpt2

LOGIT_TOLERANCE = 1e-3
PROMPT = "The quick brown fox jumps over the lazy dog because"


@pytest.fixture(scope="module")
def models():
    hf_model = load_hf_gpt2()
    return hf_model, build_gpt2_from_hf(hf_model)


@torch.no_grad()
def test_logits_match_hugging_face(models):
    hf_model, our_model = models
    token_ids = tokenize(PROMPT)
    hf_logits = hf_model(token_ids.unsqueeze(0)).logits[0]
    our_logits = our_model(token_ids)
    assert (our_logits - hf_logits).abs().max().item() < LOGIT_TOLERANCE


def test_greedy_generation_matches_hugging_face(models):
    hf_model, our_model = models
    token_ids = tokenize(PROMPT)
    num_new_tokens = 8
    hf_tokens = hf_model.generate(
        token_ids.unsqueeze(0), max_new_tokens=num_new_tokens, do_sample=False, pad_token_id=50256
    )[0]
    assert torch.equal(generate(our_model, token_ids, num_new_tokens), hf_tokens)
