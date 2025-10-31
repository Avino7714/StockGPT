import torch
from torch import nn
from gpt_architecture_elements import LayerNorm
from transformer_block import TransformerBlock
from typing import Dict


class GPTModel(nn.Module):

    "Define a GPT2-like model object"

    def __init__(self, cfg: Dict):
        super().__init__()

        self.context_length = cfg["context_length"]
        self.emb_dim = cfg["emb_dim"]
        self.vocab_size = cfg["vocab_size"]

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["dropout_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, input_tensor):
        tok_embeds = self.tok_emb(input_tensor)
        pos_embeds = self.pos_emb(
            torch.arange(input_tensor.shape[1], device=input_tensor.device)
        )  # batch, n_tokens, embedding
        total_embeds = tok_embeds + pos_embeds

        x = self.drop_emb(total_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits

    def save_weights(self, save_name="gpt_model_weights.pth", data_parallel=False):

        "saves the model weights as pth file for future use"

        torch.save(self.state_dict(), save_name)
        print("Saved model parameters as {}".format(save_name))  # This is a nice line

    def load_weights_into_gpt(self, pth_file="gpt_model_weights.pth", device=None):
        """
        loads model weights from a previously saved pth file.
        device is either cpu or gpu. When training on gpu, and bringing into cpu,
        then use device = "cpu" and "cuda" for vice versa
        """

        if device is not None:
            self.load_state_dict(
                torch.load(pth_file, map_location=torch.device(device))
            )
        else:
            self.load_state_dict(torch.load(pth_file))
        print("Loaded weights into model successfully")


# =================================================


if __name__ == "__main__":
    GPT_CONFIG = {
        "vocab_size": 401,
        "context_length": 256,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 4,
        "dropout_rate": 0.05,
        "qkv_bias": False,
    }

    gpt = GPTModel(GPT_CONFIG)

    print("Model parameters : ", sum(p.numel() for p in gpt.parameters()))

    print("Try Saving")
    gpt.save_weights("trial.pth")

    print("Load weights")
    gpt.load_weights_into_gpt("trial.pth")
