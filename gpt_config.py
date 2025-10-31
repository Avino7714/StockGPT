import os
from stock_tokenizer import StockVocab

GPT_CONFIG = {
    "stock_path": os.path.join("stock_data", "stock_"),
    "min_bin": -10_000,  # classic
    "max_bin": 10_000,  # classic
    "min_vocab": 0,  # this is for ohlcv
    "max_vocab": 20_000,  # this is for ohlcv
    "bin_size": 50,
    "batch_size": 50,
    "stride": 4,
    "multiply_by": 1e4,  # this must be the same as max_bin --- DataLoader Ends, gptparams begins
    "context_length": 256,
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 4,
    "dropout_rate": 0.05,
    "qkv_bias": False,
    "predict_new_tokens": 1,
    "train_ratio": 0.9,  # this much of training data will be used for training, rest for validation.
    "num_epochs": 1,
    "adamw_learning_rate": 4e-4,
    "adamw_weight_decay": 0.1,
    "save_checkpoint_step": 100,  # save model training every ___ steps
    "save_path": os.path.join("."),
    "temperature": 1,
    "topk": 7,
}


def set_gpt_config_param(name, value):
    GPT_CONFIG[name] = value
