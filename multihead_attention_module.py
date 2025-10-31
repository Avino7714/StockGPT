import numpy as np
import pandas as pd
import torch
from torch import device, nn


class MultiheadAttentionModule(nn.Module):
    def __init__(
        self, embedding_dim, context_dim, num_heads, dropout_rate=0.05, qkv_bias=False
    ):
        super().__init__()
        self.w_key = nn.Linear(embedding_dim, context_dim, bias=qkv_bias)
        self.w_value = nn.Linear(embedding_dim, context_dim, bias=qkv_bias)
        self.w_query = nn.Linear(embedding_dim, context_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = self.context_dim // self.num_heads
        self.out_proj = nn.Linear(context_dim, context_dim)

    def forward(self, embedding_tensor):
        b, num_tokens, _ = embedding_tensor.shape  # we already know dim of embedding
        query = self.w_query(embedding_tensor)
        key = self.w_key(embedding_tensor)
        value = self.w_value(embedding_tensor)

        key = key.view(b, num_tokens, self.num_heads, self.head_dim)
        value = value.view(b, num_tokens, self.num_heads, self.head_dim)
        query = query.view(b, num_tokens, self.num_heads, self.head_dim)

        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query = query.transpose(1, 2)

        # masked self-attention
        attn_score = query @ key.transpose(2, 3)

        # buffer, manually move to gpu...
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=embedding_tensor.device),
            diagonal=1,
        )
        attn_score.masked_fill(mask.bool(), -torch.inf)

        attn_weight = torch.softmax(attn_score / key.shape[-1] ** 0.5, dim=-1)
        attn_weight = self.dropout(attn_weight)

        context_vec = (attn_weight @ value).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.context_dim)
        context_vec = self.out_proj(context_vec)

        return context_vec


if __name__ == "__main__":
    torch.manual_seed(123)

    vocab_size = 400  # does not matter actually
    emb_dim = 300  # say
    embedding = nn.Embedding(vocab_size, emb_dim)
    print(embedding.weight.shape)

    total_emb = embedding(torch.arange(256)).contiguous().view(4, 64, 300)

    mha = MultiheadAttentionModule(
        embedding_dim=emb_dim,
        context_dim=200,
        num_heads=8,
        dropout_rate=0.05,
        qkv_bias=False,
    )

    print(x := mha(total_emb), x.shape)
