import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import yfinance as yf 
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import sys 
    import datetime as dt 
    sys.path.append("../stock_gpt/")
    return dt, mo, np, os, pd, plt, sns, sys, yf


@app.cell
def _():
    from utils import read_ohlcv
    from gpt_config import GPT_CONFIG, set_gpt_config_param
    from gpt_model import GPTModel
    from predict import Predict, compare_gpt_vs_hist_data
    return (
        GPTModel,
        GPT_CONFIG,
        Predict,
        compare_gpt_vs_hist_data,
        read_ohlcv,
        set_gpt_config_param,
    )


@app.cell
def _(set_gpt_config_param):
    set_gpt_config_param("vocab_size", 401)
    return


@app.cell
def _(mo):
    mo.md(r"""## Create models and predictor pipelines""")
    return


@app.cell
def _(GPTModel, GPT_CONFIG, Predict):
    def create_predictor(weights_path: str) -> Predict:
        gpt = GPTModel(GPT_CONFIG)
        gpt.load_weights_into_gpt(pth_file=weights_path, device = "cpu")
        gpt_predictor = Predict(model=gpt, cfg = GPT_CONFIG, device = "cpu")
        return gpt_predictor
    return (create_predictor,)


@app.cell
def _(create_predictor):
    forward_gptpred = create_predictor("../stock_gpt/0_93M_Complete_forward_avg20_70k_steps/gpt_training_checkpoint.pth")
    backward_gptpred = create_predictor("../stock_gpt/0_93M_Complete_Training_20_day_back_avg_70K_steps/gpt_training_checkpoint.pth")
    daily_gpt_pred = create_predictor("../stock_gpt/day_0_93M_19000_steps/gpt_training_checkpoint.pth") # this is perhaps the worst model
    return backward_gptpred, daily_gpt_pred, forward_gptpred


@app.cell
def _(mo):
    mo.md(r"""## Dataset""")
    return


@app.cell
def _(read_ohlcv):
    # CHANGE HERE 
    stock_data = read_ohlcv("yf_data/AMBUJACEM.NS.csv")
    stock_data = stock_data.iloc[-100:] # take the first 300 ticks 
    stock_data
    return (stock_data,)


@app.cell
def _(compare_gpt_vs_hist_data, forward_gptpred, plt, stock_data):
    forward_results = compare_gpt_vs_hist_data(gpt_predictor=forward_gptpred, hist_data=stock_data, plot=True)
    plt.show()
    return (forward_results,)


@app.cell
def _(backward_gptpred, compare_gpt_vs_hist_data, daily_gpt_pred, stock_data):
    backward_results = compare_gpt_vs_hist_data(gpt_predictor=backward_gptpred, hist_data=stock_data, plot=False)
    daily_results = compare_gpt_vs_hist_data(gpt_predictor=daily_gpt_pred, hist_data=stock_data, plot = False)
    return backward_results, daily_results


@app.cell
def _(mo):
    mo.md(r"""## Forward Predictor""")
    return


@app.cell
def _(pd, stock_data):
    _window = 20
    _indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=_window)
    close_forward = stock_data["Close"].pct_change().shift(_window).dropna().rolling(window = _indexer, min_periods=1).mean()
    return (close_forward,)


@app.cell
def _(close_forward, forward_results, plt):
    close_forward.plot(kind = "line", label = "close_forward")
    #(stock_data["Close"].pct_change()/5).plot(kind = "line", label = "stock")
    forward_results["gpt_preds"].plot(kind = "line", label = "gpt")
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Daily returns predictor

        - For large returns, the daily predictor can be predictor can be correct
        - Otherwise, daily returns is not a powerful model, needs to be trained better and further.
        """
    )
    return


@app.cell
def _(daily_results, plt, stock_data):
    plt.figure(figsize=[10,5])
    stock_data["Close"].pct_change().rolling(2).mean().plot(kind = "line", label = "close")
    #forward_results["gpt_preds"].plot(kind = "line", label = "forward")
    #backward_results["gpt_preds"].plot(kind = "line", label = "backward")
    daily_results["gpt_preds"].plot(kind = "line", label = "daily")
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Backward Predictor

        -  How does the backward prediction behave wrt real returns ? Pretty Excellent!!
        -  What happens when forward and backward gpt crossover? Simple moving average crossover strategy. Hope this works.
        """
    )
    return


@app.cell
def _(backward_results, stock_data):
    # how does the backward prediction behave wrt real returns ? Pretty Excellent!! 
    stock_data["Close"].pct_change().rolling(20).mean().plot(kind = "line", label = "c20")
    backward_results["gpt_preds"].plot(kind = "line", label = "backward")
    return


@app.cell
def _(backward_results, forward_results, plt, stock_data):
    # what happens when forward and backward gpt crossover? 

    plt.figure(figsize=[10,5])
    stock_data["Close"].pct_change().rolling(5).mean().plot(kind = "line", label = "c5")
    stock_data["Close"].pct_change().rolling(20).mean().plot(kind = "line", label = "c20")
    (forward_results["gpt_preds"]).plot(kind = "line", label = "forward") 
    backward_results["gpt_preds"].plot(kind = "line", label = "backward")
    plt.legend()
    return


@app.cell
def _(close_forward, plt, stock_data):
    #what happens when forward avg and backward avg intersect?
    # They are the same curve 
    close_forward.plot(kind = "line", label = "forward")
    stock_data["Close"].pct_change().rolling(5).mean().plot(kind = "line", label = "backward")
    plt.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
