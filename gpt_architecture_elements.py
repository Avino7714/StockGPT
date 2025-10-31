import torch
from torch import nn

# ====================================


class LayerNorm(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(input_dim))
        self.shift = nn.Parameter(torch.zeros(input_dim))

    def forward(self, input_tensor):
        mean = input_tensor.mean(dim=-1, keepdim=True)
        var = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        layernorm_output = (input_tensor - mean) / torch.sqrt(var + self.eps)
        return self.scale * layernorm_output + self.shift


# ====================================


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        return (
            0.5
            * input_tensor
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2 / torch.pi))
                    * (input_tensor + 0.044715 * torch.pow(input_tensor, 3))
                )
            )
        )


# ====================================


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"], bias=cfg["qkv_bias"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"]),
        )

    def forward(self, input_tensor):
        return self.feed_forward(input_tensor)


# ====================================

if __name__ == "__main__":
    GPT_CONFIG = {
        "vocab_size": 401,
        "context_length": 256,
        "emb_dim": 300,
        "n_heads": 4,
        "n_layers": 4,
        "dropout_rate": 0.05,
        "qkv_bias": False,
    }

    # check LayerNorm functionality
    ln = LayerNorm(2)
    print(
        ln(torch.rand(2, 2)).mean(dim=-1, keepdim=True),
        ln(torch.rand(2, 2).var(dim=-1, keepdim=True)),
    )

    # check working of feed-forward and gelu
    ff = FeedForward(GPT_CONFIG)
    print(
        x := ff(
            torch.rand(
                # GPT_CONFIG["emb_dim"], GPT_CONFIG["emb_dim"]
                4,
                64,
                300,  # from attention example
            )
        ),
        x.shape,
    )
