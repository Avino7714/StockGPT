import torch
from torch import nn
from gpt_architecture_elements import LayerNorm
from transformer_block import TransformerBlock
from typing import Dict
from gpt_architecture_elements import LayerNorm

class OHLCV_GPTModel(nn.Module):

    "Define a GPT2-like model object for high dimensional input data, but one dim target. Does not inherit the original GPTmodel"

    def __init__(self, cfg: Dict, n_cols = 4):
        super().__init__()

        self.cfg = cfg
        self.n_cols = n_cols
        self.context_length = cfg["context_length"]
        self.emb_dim = cfg["emb_dim"]
        self.vocab_size = cfg["vocab_size"]

        self.tok_emb = nn.Embedding(self.cfg["vocab_size"], self.cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.combiner = nn.Sequential(
            nn.Linear(self.cfg["emb_dim"] * n_cols, self.cfg["emb_dim"]*(n_cols//2 + 1)),
            nn.Sigmoid(),
            nn.Linear(self.cfg["emb_dim"]*(n_cols//2 + 1), self.cfg["emb_dim"]),
            nn.Sigmoid(),
            nn.Linear(self.cfg["emb_dim"], self.cfg["emb_dim"])
        )
        self.ln = LayerNorm(self.cfg["emb_dim"])
        
        self.drop_emb = nn.Dropout(cfg["dropout_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, input_tensor):
        assert input_tensor.dim() == 3, "input tensor is not 3D data, gpt-model"

        pos_embeds = self.pos_emb(
            torch.arange(input_tensor.shape[1], device=input_tensor.device)
        )  # batch, n_tokens, embedding

        # prepare embeds
        tok_embedded = []
        for i in range(self.n_cols):
            tok_embedded.append(self.tok_emb(input_tensor[:,:,i]) + pos_embeds)
        tok_embedded= torch.cat(tok_embedded, dim = -1)
        total_embeds = self.combiner(tok_embedded)
        x = self.ln(total_embeds)
        
        x = self.drop_emb(x)
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
        "vocab_size": 50001,
        "context_length": 256,
        "emb_dim": 256,
        "n_heads": 4,
        "n_layers": 4,
        "dropout_rate": 0.05,
        "qkv_bias": False,
    }

    gpt = OHLCV_GPTModel(GPT_CONFIG, 5)
    print("Model parameters : ", sum(p.numel() for p in gpt.parameters()))

    input_tensor = torch.randint(low=0, high=50000, size=[3, 256, 5])
    # input_tensor = torch.zeros((25, 256, 5)).to(torch.int32)
    out = gpt(input_tensor)
    print(out.shape)  # WORKS DONE DONE DONE

    print("Try Saving")
    gpt.save_weights("trial.pth")
    print("Load weights")
    gpt.load_weights_into_gpt("trial.pth")
