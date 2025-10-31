import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""# Prediction and testing""")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import torch
    import datetime as dt
    import matplotlib.pyplot as plt 
    import seaborn as sns
    #today = dt.date.today()
    #today
    return dt, np, pd, plt, sns, torch


@app.cell
def _():
    from predict import Predict
    from gpt_config import GPT_CONFIG, set_gpt_config_param
    from gpt_model import GPTModel
    from stock_tokenizer import StockVocab
    return GPTModel, GPT_CONFIG, Predict, StockVocab, set_gpt_config_param


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
def _(dt, today):
    import yfinance as yf 
    india_data = yf.download("RELIANCE.NS TCS.NS INFY.NS",
                           start=today - dt.timedelta(380),
                           end = today,
                           interval='1d',
                           group_by="ticker")
    return india_data, yf


@app.cell
def get_close_returns():
    # important function 
    def get_close_returns(ydf):
        return (
            ydf.T.
            xs("Close", level = "Price").T.
            dropna().
            interpolate().
            pct_change().
            dropna()
        )
    return (get_close_returns,)


@app.cell
def _(get_close_returns, india_data):
    ind_returns = get_close_returns(india_data)
    ind_returns.plot(kind = "line")

        #ind_returns.to_csv("../Trading/ind_returns_trial.csv")
    return (ind_returns,)


@app.cell
def _(ind_returns):
    ind_returns.rolling(20).mean().iloc[-20:].plot(kind = "line")
    return


@app.cell
def _(mo):
    mo.md(r"""## Prediction configuration""")
    return


@app.cell
def _(GPTModel, GPT_CONFIG):
    # gpt model 
    close_gpt = GPTModel(GPT_CONFIG)

    # load 
    close_gpt.load_weights_into_gpt(
        pth_file="0_93M_Complete_forward_avg20_70k_steps/gpt_training_checkpoint.pth",
        device="cpu"
    )
    return (close_gpt,)


@app.cell
def _(set_gpt_config_param):
    set_gpt_config_param("temperature", 1)
    return


@app.cell
def _(ind_returns):
    ind_returns.dropna()
    return


@app.cell
def _(GPT_CONFIG, Predict, close_gpt, ind_returns):
    next_month_predictor = Predict(
        model=close_gpt,
        cfg = GPT_CONFIG,
        device="cpu"
    )
    next_month_predictor.inload(ind_returns.dropna(), is_return=True)
    return (next_month_predictor,)


@app.cell
def _(ind_returns, next_month_predictor):
    next_month_predictor.next(
        how_many=20,
        input_data=ind_returns.dropna(),
        is_return=True
    ).plot(kind = "line")
    return


@app.cell
def _(next_month_predictor, pd):
    _x = next_month_predictor.__probs__.tolist()
    _x = pd.DataFrame(list(zip(*_x)), columns = list('abc'), index=next_month_predictor.tokenizer.vocab/100)
    return


@app.cell
def _(next_month_predictor, np):
    next_month_predictor.long_short_guess_from_probs().astype(np.float64)*100
    return


@app.cell
def _(next_month_predictor):
    next_month_predictor.MGF(3)
    return


@app.cell
def _(mo):
    mo.md(r"""## Get massive data from NSE""")
    return


@app.cell
def _(np):
    G = np.random.default_rng(123)
    return (G,)


app._unparsable_cell(
    r"""
    def get_massive_data_bse(n_stocks = 100):

    
    """,
    name="_"
)


@app.cell
def _(pd):
    bse_stocks = pd.read_csv("../Trading/EQUITY_L.csv", 
                index_col="SYMBOL",
               usecols = ["SYMBOL", "NAME OF COMPANY", " SERIES", " DATE OF LISTING"])
               #names=["symbol","name", "listed_data", "paid_up_value", "face_value"])
    bse_stocks
    return (bse_stocks,)


@app.cell
def _(bse_stocks, get_close_returns, yf):
    _data = yf.download(tickers= list(map(lambda x : x + ".NS", bse_stocks.index[25:75].to_list()) ),
                       period="380d",interval="1d")
    bse_data = get_close_returns(_data)
    return (bse_data,)


@app.cell
def _(bse_data):
    bse_data
    return


@app.cell
def _(bse_data, next_month_predictor):
    next_month_predictor.inload(bse_data, is_return=True)
    return


@app.cell
def _(next_month_predictor):
    next_month_predictor.long_short_guess_from_probs()
    return


@app.cell
def _(next_month_predictor):
    next_month_predictor.next(10)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
