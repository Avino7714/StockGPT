import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Stock GPT""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Step 1 : Import all essentials, make sure they are right""")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd 
    import matplotlib.pyplot as plt
    import seaborn as sns
    return np, pd, plt, sns


@app.cell
def _():
    import torch
    from torch import nn
    torch.manual_seed(123)
    return nn, torch


@app.cell
def _():
    import os 
    import sys
    return os, sys


@app.cell
def _():
    from gpt_config import GPT_CONFIG, set_gpt_config_param
    # stock_path = os.path.join(GPT_CONFIG["stock_path"], "b")
    return GPT_CONFIG, set_gpt_config_param


@app.cell
def _():
    from stock_tokenizer import (
    StockTokenizer, StockVocab,
    create_data_loader_for_stock, create_data_loader_for_tokenizer,
    train_val_split, tokenizer_dataloader_dataframe)
    from data_cleaner import DataCleanerAssembler

    # Monthly Stock Tokenizer 
    from monthly_avg_assembler import MonthDataCleanerAssembler, MonthStockTokenizer
    return (
        DataCleanerAssembler,
        MonthDataCleanerAssembler,
        MonthStockTokenizer,
        StockTokenizer,
        StockVocab,
        create_data_loader_for_stock,
        create_data_loader_for_tokenizer,
        tokenizer_dataloader_dataframe,
        train_val_split,
    )


@app.cell
def _(set_gpt_config_param):
    set_gpt_config_param("save_checkpoint_step", 200)
    set_gpt_config_param("rolling_window", 20)
    return


@app.cell
def _(GPT_CONFIG, StockVocab, set_gpt_config_param):
    def set_gpt_vocab_size():
        "necessary precursor to run GPT"
        _s = StockVocab(GPT_CONFIG["min_bin"],
                        GPT_CONFIG["max_bin"],
                       GPT_CONFIG["bin_size"])

        set_gpt_config_param("vocab_size", _s.vocab_size)

    set_gpt_vocab_size()
    return (set_gpt_vocab_size,)


@app.cell
def _(GPT_CONFIG):
    GPT_CONFIG
    return


@app.cell
def _(GPT_CONFIG, MonthDataCleanerAssembler, os):

    mca = MonthDataCleanerAssembler(
        cfg = GPT_CONFIG,
        stock_path=os.path.join(GPT_CONFIG["stock_path"], "a"), # change this 
        is_batch=True,
        is_return=False
    )

    mcb = MonthDataCleanerAssembler(
        cfg = GPT_CONFIG,
        stock_path=os.path.join(GPT_CONFIG["stock_path"], "b"), # change this 
        is_batch=True,
        is_return=False
    )
    return mca, mcb


@app.cell
def _(mo):
    mo.md("""### Create Dataloader for training""")
    return


@app.cell
def _():
    from torch.utils.data import ConcatDataset
    return (ConcatDataset,)


@app.cell
def _(GPT_CONFIG, MonthDataCleanerAssembler, os):
    def func_concat_dataset(folder_name):
        total_concat_dataset = []
        for f in os.listdir(folder_name):
            total_concat_dataset.append(
                MonthDataCleanerAssembler(
                    cfg = GPT_CONFIG,
                    stock_path=os.path.join(GPT_CONFIG["stock_path"], folder_name), # change this 
                    is_batch=True,
                    is_return=False
                )
            )
        return total_concat_dataset
    return (func_concat_dataset,)


@app.cell
def _(
    ConcatDataset,
    GPT_CONFIG,
    create_data_loader_for_tokenizer,
    mca,
    mcb,
):
    # _all_datasets_in_folder = func_concat_dataset(os.path.join(GPT_CONFIG["stock_path"], "a") )
    train_loader = create_data_loader_for_tokenizer(
        ConcatDataset([mca.ConcatDataset(), mcb.ConcatDataset()]),
        cfg = GPT_CONFIG,
        batch_size=GPT_CONFIG["batch_size"],
        shuffle = True,
        num_workers=0 # change this 
    )
    return (train_loader,)


@app.cell
def _(train_loader):
    _i = iter(train_loader)
    print(next(_i))
    return


@app.cell
def _(mo):
    mo.md(
        """
        dca = DataCleanerAssembler(
            GPT_CONFIG,
            stock_path,
            is_return=False,
            is_batch=True
        )
        print(dca)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        train_loader = create_data_loader_for_tokenizer(
            dca.ConcatDataset(),
            GPT_CONFIG,
            batch_size=GPT_CONFIG["batch_size"]
        )
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Custom Validation Loader for evaluation""")
    return


@app.cell
def _(
    GPT_CONFIG,
    MonthDataCleanerAssembler,
    create_data_loader_for_tokenizer,
    os,
):
    _interested_stock = os.path.join(GPT_CONFIG["stock_path"],"h") # change this 

    _interested_stock = MonthDataCleanerAssembler(
        GPT_CONFIG, stock_path=_interested_stock,
        is_batch = True,
        is_return = False
    )

    custom_val_loader = create_data_loader_for_tokenizer(
        _interested_stock.ConcatDataset(), 
        GPT_CONFIG,
        batch_size=GPT_CONFIG["batch_size"],
        num_workers=0, # change this 
        shuffle=True 
    )
    return (custom_val_loader,)


@app.cell
def _(mo):
    mo.md(r"""### GPT Model, Parallelization and device""")
    return


@app.cell
def _():
    from gpt_model import GPTModel 
    from data_parallel import data_parallel, dist_parallel
    return GPTModel, data_parallel, dist_parallel


@app.cell
def _(GPTModel, GPT_CONFIG, data_parallel):
    close_prices_gpt_model = GPTModel(
        cfg=GPT_CONFIG
    )
    close_prices_gpt_model, device = data_parallel(close_prices_gpt_model)

    print(sum(p.numel() for p in close_prices_gpt_model.parameters()))
    return close_prices_gpt_model, device


@app.cell
def _(mo):
    mo.md(r"""### Model Training""")
    return


@app.cell
def _():
    from stock_train_mechanism import StockTrainer 
    from stock_loss_prediction import calc_loss_batch, calc_loss_loader, generate_next_stock, generate_next_stock_simple
    return (
        StockTrainer,
        calc_loss_batch,
        calc_loss_loader,
        generate_next_stock,
        generate_next_stock_simple,
    )


@app.cell
def _(GPT_CONFIG, StockTrainer, close_prices_gpt_model, device):
    trainer = StockTrainer(
        model = close_prices_gpt_model,
        cfg = GPT_CONFIG,
        load_from_checkpoint=None,
        device = device
    )

    print(trainer)
    return (trainer,)


@app.cell
def _(custom_val_loader, train_loader, trainer):
    trainer.engage_training(train_loader=train_loader,
                           num_epochs=1,
                           val_loader=custom_val_loader,
                           eval_freq=1000,
                            eval_iter=100
                           )
    return


@app.cell
def _(mo):
    mo.md(r"""End of Model Training""")
    return


@app.cell
def _(mo):
    mo.md("""### Model evaluation""")
    return


@app.cell
def _(mo):
    mo.md(
        """
        close_prices_gpt_model.load_weights_into_gpt(
            os.path.join("19000_steps", "gpt_training_checkpoint.pth"),
            device = "cpu"
        )
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""###> Make a temp testing dataset""")
    return


@app.cell
def _(mo):
    mo.md(
        """
        _aapl= DataCleanerAssembler(
            GPT_CONFIG,
            stock_path=os.path.join("stock_data","stock_","a","aapl.us.txt"),
            is_batch=False,
            is_return=False
        ).ConcatDataset()

        aapl_val = create_data_loader_for_tokenizer(
            _aapl,
            cfg = GPT_CONFIG,
            batch_size=GPT_CONFIG["batch_size"],
        )

        input_batch, target_batch = next(iter(aapl_val))
        print(input_batch.shape)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        aapl_results = generate_next_stock(
            model=close_prices_gpt_model,
            idx = input_batch[[3,7,22,47,34,14],:], # random signals of aapl stock
            result_with_input=True,
            temperature=1.0,
            topk=5
        )
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        plt.figure()
        for sig in aapl_results:
            plt.plot(sig, "o-")
        plt.show()
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
