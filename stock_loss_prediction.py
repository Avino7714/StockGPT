from typing import List, Tuple, Dict, Optional
import torch
import numpy as np
import pandas as pd
from torch.nn.functional import cross_entropy
from gpt_model import GPTModel


torch.manual_seed(123)

# ======================================


# TRAIN, TRAIN_dataload = create_data_loader_for_stock(
#     ...
# )

# VAL, VAL_dataload = create_data_loader_for_stock(

# )

# ======================================


def generate_next_stock_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int = 20,  # 20 day returns,
    context_size: int = 256,  # is always equal to context_length ?
    result_with_input: bool = True,  # returns the prediction combined with input
) -> torch.Tensor:
    """Rudimentary function to predict next annual return for stock.
    Returns the next few days prediction combined with the input.

    """

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat([idx_cond, idx_next], dim=1)

    return idx if result_with_input else idx[:, -max_new_tokens:]


# ======================================


def calc_loss_batch(
    input_batch: torch.Tensor, target_batch: torch.Tensor, model: GPTModel
):
    "Define the cross-entropy loss between the logits from inputs and target batch"

    input_batch = input_batch.long()
    target_batch = target_batch.long()

    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss


# ======================================


def calc_loss_loader(
    data_loader: torch.utils.data.DataLoader,
    model: GPTModel,
    device="cpu",
    num_batches: Optional[int] = None,
):
    "calculate loss function for a given set of batches together"

    total_loss = 0.0
    model = model.to(device)

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        # send data to device
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        if i < num_batches:
            loss = calc_loss_batch(
                input_batch=input_batch, target_batch=target_batch, model=model
            )
            total_loss += loss.item()

        else:
            break

    return total_loss / num_batches


# ======================================


def generate_next_stock(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int = 20,
    context_size: int = 256,
    temperature: float = 0.0,
    topk: Optional[int] = None,
    eos_id=None,
    result_with_input: bool = True,
) -> torch.Tensor:
    "generates next stock with top k sampling and temperature scaling"

    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :].squeeze(dim=0)
        # print(logits.shape)
        if topk is not None:
            top_logits, _ = torch.topk(logits, topk)
            if top_logits.dim() == 1:
                top_logits = top_logits.unsqueeze(dim=0)
            min_val = top_logits[:, -1]
            min_val = torch.reshape(min_val, (min_val.shape[0], 1))  # for comparison
            logits = torch.where(
                logits < min_val[None, :], torch.tensor(float("-inf")), logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1).squeeze(dim=0)
            # print(probs.shape)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat([idx, idx_next], dim=1)

    return idx if result_with_input else idx[:, -max_new_tokens:]


# ======================================


if __name__ == "__main__":
    from gpt_config import GPT_CONFIG

    GPT_CONFIG["vocab_size"] = 401
    model = GPTModel(cfg=GPT_CONFIG)
    encoded_tensor = torch.randint(
        low=0, high=401, size=(1, GPT_CONFIG["context_length"])
    )

    model.eval()
    out = generate_next_stock(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=GPT_CONFIG["predict_new_tokens"],
        context_size=GPT_CONFIG["context_length"],
        result_with_input=False,
    )

    print("Stock token prediction")
    print(out)
