import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    return mo, np, pd, plt, sns


@app.cell
def _(mo):
    mo.md(
        r"""
        # Import Data

        Upload any file *downloaded from yahoo finance* 
        """
    )
    return


@app.cell
def _(pd):
    def read_ohlcv(filename):
        "read a given ohlcv data file downloaded from yfinance"
        return pd.read_csv(
            filename,
            skiprows=[0, 1, 2],  # remove the multiindex rows that cause trouble
            names=["Date", "Close", "High", "Low", "Open", "Volume"],
            index_col="Date",
            parse_dates=["Date"],
        )
    return (read_ohlcv,)


@app.cell
def _(mo):
    f= mo.ui.file_browser(multiple = False, filetypes = [".csv"])
    f
    return (f,)


@app.cell
def _(f, read_ohlcv):
    print(f"Uploaded {f.name()}")
    stock_data = read_ohlcv(f.path())
    stock_data
    return (stock_data,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Choose GPT Model 
        Input the path to the **pth** file, the weights will be loaded into model automatically
        """
    )
    return


@app.cell
def _():
    from gpt_model import GPTModel
    from gpt_config import GPT_CONFIG, set_gpt_config_param
    set_gpt_config_param("vocab_size", 401) # hard coded - annoying 
    set_gpt_config_param("emb_dim", 128)
    from predict import Predict
    return GPTModel, GPT_CONFIG, Predict, set_gpt_config_param


@app.cell
def _(GPT_CONFIG):
    GPT_CONFIG
    return


@app.cell
def _(mo):
    model_loc = mo.ui.file_browser(multiple = False, filetypes=[".pth"])
    model_loc
    return (model_loc,)


@app.cell
def _(GPTModel, GPT_CONFIG, Predict, model_loc):
    def create_predictor(weights_path: str) -> Predict:
        gpt = GPTModel(GPT_CONFIG)
        gpt.load_weights_into_gpt(pth_file=weights_path, device = "cpu")
        gpt_predictor = Predict(model=gpt, cfg = GPT_CONFIG, device = "cpu")
        return gpt_predictor
    gpt_predictor = create_predictor(model_loc.path())
    return create_predictor, gpt_predictor


@app.cell
def _(mo):
    data_input_slider = mo.ui.slider(
        start = 5,
        stop = 255,
        value = 254,
        label = "hyper parameter : how much stock returns to give in for each prediction ? >> "
    )
    data_input_slider
    return (data_input_slider,)


@app.cell
def _(Predict, data_input_slider, gpt_predictor, pd, stock_data):
    def gen_gpt_signals(
        predictor: Predict, hist_data, is_return=False, data_input_each_stage = 25
    ):

        close_data = hist_data["Close"]
        if not is_return:
            close_data = (
                close_data.pct_change().interpolate().dropna()
            ).rename("close_returns")

        nrows = close_data.shape[0]
        next_day = [0,0]
        day_after = [0,0]

        for i in range(3, nrows + 1):
            data_to_inload = close_data.iloc[:i].iloc[-data_input_each_stage:] # this is a hyperparameter 25 input days will give a robust 6 day average
            predictor.inload(data_to_inload, is_return=True)
            #next_day_avg = predictor.get_predict_probs().idxmax().squeeze()
            next_day_avg = predictor.next(how_many = 1).squeeze()
            next_day.append(next_day_avg)

        next_day = pd.Series(next_day, index = close_data.index, name = "tomorrow")
        hist_data = pd.concat([hist_data, close_data, next_day], axis = 1)

        hist_data["gpt_signal"] = (hist_data["tomorrow"] > 0).astype(int) # next_day
    
        return hist_data

    vectorized_stock_data = gen_gpt_signals(predictor=gpt_predictor,
                                            hist_data=stock_data,
                                            is_return = False,
                                           data_input_each_stage=data_input_slider.value)

    vectorized_stock_data[["Close", "close_returns", "tomorrow", "gpt_signal"]]
    return gen_gpt_signals, vectorized_stock_data


if __name__ == "__main__":
    app.run()
