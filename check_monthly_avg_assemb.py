import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    return np, pd


@app.cell
def _():
    from gpt_config import GPT_CONFIG, set_gpt_config_param
    set_gpt_config_param("rolling_window", 20)
    return GPT_CONFIG, set_gpt_config_param


@app.cell
def _():
    from monthly_avg_assembler import MonthDataCleanerAssembler, MonthStockTokenizer
    return MonthDataCleanerAssembler, MonthStockTokenizer


@app.cell
def _(pd):
    data = pd.read_csv(
        "stock_data/stock_/a/aapl.us.txt",
        usecols=["Date", "Close"],
        index_col="Date",
        parse_dates=["Date"]
    ).squeeze() 
    data
    return (data,)


@app.cell
def _(GPT_CONFIG, MonthDataCleanerAssembler):
    mca = MonthDataCleanerAssembler(
        cfg=GPT_CONFIG,
        stock_path="stock_data/stock_/a/aapl.us.txt",
        is_batch=False,
        is_return=False
    )
    return (mca,)


@app.cell
def _():
    from stock_tokenizer import create_data_loader_for_tokenizer
    return (create_data_loader_for_tokenizer,)


@app.cell
def _(GPT_CONFIG, create_data_loader_for_tokenizer, mca):
    d = create_data_loader_for_tokenizer(mca.ConcatDataset(), cfg = GPT_CONFIG, batch_size=100)
    return (d,)


@app.cell
def _(d):
    i = iter(d)
    to_compare = next(i)[1][0,:]
    return i, to_compare


if __name__ == "__main__":
    app.run()
