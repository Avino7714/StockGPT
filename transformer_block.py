import torch
from torch import nn
from gpt_architecture_elements import FeedForward, LayerNorm
from multihead_attention_module import MultiheadAttentionModule


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiheadAttentionModule(
            embedding_dim=cfg["emb_dim"],
            context_dim=cfg["emb_dim"],
            # context_length = cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout_rate=cfg["dropout_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.layer_norm1 = LayerNorm(cfg["emb_dim"])
        self.layer_norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["dropout_rate"])

    def forward(self, input_tensor):

        shortcut = input_tensor
        x = self.layer_norm1(input_tensor)  # pre layer norm
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # dimensions must agree

        shortcut = x
        x = self.layer_norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


# ==========================================

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

    trf = TransformerBlock(GPT_CONFIG)
    print(x := trf(torch.rand(2, 256, 300)), x.shape)
