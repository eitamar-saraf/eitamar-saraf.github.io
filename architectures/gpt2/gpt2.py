"""GPT-2 (124M), written from scratch to be read top-down.

Every class and function here is one block in the site's flow chart, and the page
shows this exact source. The code favours being obvious over being fast:

- one sequence at a time, so tensors are [tokens, features] with no batch dimension
- inference only, so there is no dropout and no training loop
- no KV cache and no fused kernels
"""

import math
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int = 50257
    context_length: int = 1024
    embedding_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_expansion: int = 4
    layer_norm_eps: float = 1e-5


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = LayerNorm(config.embedding_dim, config.layer_norm_eps)
        self.lm_head = LMHead(self.embeddings.token_embedding)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embeddings(token_ids)      # [T] -> [T, 768]
        for block in self.blocks:
            hidden = block(hidden)               # [T, 768] -> [T, 768], 12 times
        hidden = self.final_norm(hidden)         # [T, 768]
        return self.lm_head(hidden)              # [T, 768] -> [T, 50257]


GPT2_ENCODING = tiktoken.get_encoding("gpt2")


def tokenize(text: str) -> torch.Tensor:
    return torch.tensor(GPT2_ENCODING.encode(text))


def detokenize(token_ids: torch.Tensor) -> str:
    return GPT2_ENCODING.decode(token_ids.tolist())


class Embeddings(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(len(token_ids))
        return self.token_embedding(token_ids) + self.position_embedding(positions)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.attention_norm = LayerNorm(config.embedding_dim, config.layer_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = LayerNorm(config.embedding_dim, config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * normalized + self.shift


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.qkv_projection = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries, keys, values = self.qkv_projection(x).chunk(3, dim=-1)    # 3 x [T, 768]
        queries = split_heads(queries, self.num_heads)                      # [12, T, 64]
        keys = split_heads(keys, self.num_heads)
        values = split_heads(values, self.num_heads)
        weights = attention_weights(queries, keys)                          # [12, T, T]
        per_head_output = weights @ values                                  # [12, T, 64]
        return self.output_projection(merge_heads(per_head_output))         # [T, 768]


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    num_tokens, dim = x.shape
    return x.view(num_tokens, num_heads, dim // num_heads).transpose(0, 1)


def attention_weights(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    scores = attention_scores(queries, keys)
    return torch.softmax(apply_causal_mask(scores), dim=-1)


def attention_scores(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    head_dim = queries.shape[-1]
    return queries @ keys.transpose(-2, -1) / math.sqrt(head_dim)


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    num_tokens = scores.shape[-1]
    future = torch.triu(torch.ones(num_tokens, num_tokens, dtype=torch.bool), diagonal=1)
    return scores.masked_fill(future, float("-inf"))


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    num_heads, num_tokens, head_dim = x.shape
    return x.transpose(0, 1).reshape(num_tokens, num_heads * head_dim)


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        hidden_dim = config.mlp_expansion * config.embedding_dim
        self.up_projection = nn.Linear(config.embedding_dim, hidden_dim)
        self.down_projection = nn.Linear(hidden_dim, config.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_projection(gelu(self.up_projection(x)))


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class LMHead(nn.Module):
    def __init__(self, token_embedding: nn.Embedding):
        super().__init__()
        self.token_embedding = token_embedding

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.token_embedding.weight.T


@torch.no_grad()
def generate(model: GPT2, token_ids: torch.Tensor, num_new_tokens: int) -> torch.Tensor:
    for _ in range(num_new_tokens):
        logits = model(token_ids[-model.config.context_length:])
        next_token = logits[-1].argmax()
        token_ids = torch.cat([token_ids, next_token.unsqueeze(0)])
    return token_ids
