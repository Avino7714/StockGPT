import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    return mo, np, pd, plt


@app.cell
def _():
    import sys 
    import os
    sys.path.append("../..") # for the gpt model 
    from gpt_config import GPT_CONFIG 
    from gpt_model import GPTModel
    from stock_tokenizer import StockTokenizer, StockVocab
    from stock_loss_prediction import generate_next_stock, generate_next_stock_simple
    from predict import Predict
    from typing import List, Dict, Tuple,Set, Optional, NamedTuple, Any
    return GPTModel, GPT_CONFIG, List, NamedTuple, Optional, Predict, Set


@app.cell
def _():
    import yfinance as yf
    return (yf,)


@app.cell
def _(mo):
    mo.md(r"""## Load Model""")
    return


@app.cell
def _(GPTModel, GPT_CONFIG):
    GPT_CONFIG["vocab_size"] = 401 # HARDCODE ALERT 
    close_prices_gpt = GPTModel(GPT_CONFIG)
    close_prices_gpt.load_weights_into_gpt("gpt_training_checkpoint.pth", device="cpu")
    return (close_prices_gpt,)


@app.cell
def _(mo):
    mo.md(r"""# Backtesting for a large number of stocks""")
    return


@app.cell
def _(np, pd):
    # get 100 random stocks from the list 
    _g = np.random.default_rng(14)
    idx = pd.IndexSlice
    close_data = pd.read_csv("all_tickers_5_years.csv",
                             header =[0,1,2], parse_dates=[0])
    dates = close_data.iloc[:,0] # get all dates
    close_data = close_data.loc[:, idx[:,"Close",:]].set_index(dates)
    _testing = 20
    close_data = close_data.iloc[:, :100]
    columns_to_drop = close_data.loc[:, close_data.isna().sum() > 0].columns
    close_data = close_data.drop(columns = columns_to_drop)
    close_data.columns = map(lambda x : x[0], close_data.columns)
    close_data.head()

    # _stocks_list = pd.read_csv("EQUITY_L.csv",
    #                          usecols=["SYMBOL"]).squeeze()
    # _stocks_list = _stocks_list.iloc[_g.integers(0, _stocks_list.shape[0], size=25)]
    # _stocks_list = _stocks_list.apply(lambda x : x + '.NS')
    # _stocks_list

    # last_year_data = yf.download(
    #     tickers=_stocks_list.tolist(),
    #     period="5y",
    #     interval="1d",
    #     group_by="column"
    # )
    # last_year_data = last_year_data[("Close")]
    # last_year_data.to_csv("last_5y_data_yf.csv")
    return (close_data,)


@app.cell
def _(GPTModel, GPT_CONFIG, List, Optional, Predict, Set, pd, yf):
    class BasicPortfolio:
        """The basic portfolio selector for StockGPT. Stock portfolio based strategy. Does not work for a single stock portfolio, yet"""

        def __init__(self, universe : List[str], model : GPTModel):
            self.portfolio : Set[str] = set()
            self.universe = universe 
            self.predictor = Predict(model, GPT_CONFIG)

            # make API for this if necessary
            self.long_stocks = None 
            self.short_stocks = None 

        # ---------------------------------------

        def daily_close_data_from_yfinance(self, period = "1y", interval = "1d"):
            "Obtain cleaned data from yfinance"
            last_year_data = yf.download(
                tickers= self.universe,
                period=period,
                interval=interval,
                group_by="column")[("Close")].interpolate().dropna()
            return last_year_data

        # ---------------------------------

        #does not work for a single stock index. Will work for universe > 1
        def get_predictions_for_universe(self, close_data : Optional[pd.DataFrame] = None, long_frac: float = 0.2, short_frac: float = 0.2):
            "fraction of universe to go long on, fraction of universe to go short on"

            if close_data is not None:
                self.predictor.inload(close_data, is_return = False)
            else:
                self.predictor.inload(self.daily_close_data_from_yfinance(), is_return = False)
            predictions = self.predictor.NEXT().squeeze().sort_values(ascending = False).rename("predicted_returns")
            assert long_frac + short_frac < 1.0, "Overlap error : The long stocks and short stock deciles must add to 1"
            self.long_stocks = predictions.iloc[:int(long_frac*predictions.shape[0])]
            self.short_stocks = predictions.iloc[-int(short_frac*predictions.shape[0]):]
            for x in self.long_stocks.index:
                self.portfolio.add(x)
            for x in self.short_stocks.index:
                 if x in self.portfolio:
                     self.portfolio.remove(x)

        # ----------------------------------

        def __repr__(self):
            return f"Present Stocks to consider for portfolio : {self.portfolio}\n"

        # ---------------------------------
    return (BasicPortfolio,)


@app.cell
def _(List, Optional, Set, pd, yf):
    class SimpletopGainTopLossPortfolio:
    
        def __init__(self, universe : List[str]):
            self.portfolio : Set[str] = set()
            self.universe = universe 

            # make API for this if necessary
            self.long_stocks = None 
            self.short_stocks = None 

            # ---------------------------------------

        def daily_close_data_from_yfinance(self, period = "1y", interval = "1d"):
            "Obtain cleaned data from yfinance"
            last_year_data = yf.download(
                tickers= self.universe,
                period=period,
                interval=interval,
                group_by="column")[("Close")].interpolate().dropna()
            return last_year_data

        # ---------------------------------

        #does not work for a single stock index. Will work for universe > 1
        def get_predictions_for_universe(self,
                                         close_data : Optional[pd.DataFrame] = None,
                                         long_frac: float = 0.2, short_frac: float = 0.2):
        
            "fraction of universe to go long on, fraction of universe to go short on"
            performance = (close_data.
                iloc[-10:].
                pct_change().
                mean().
                sort_values(ascending=False))
            self.long_stocks = performance[:int(long_frac * performance.shape[0])]
            self.short_stocks = performance[-int(short_frac * performance.shape[0]):]
            for x in self.long_stocks.index:
                self.portfolio.add(x)
            for x in self.short_stocks.index:
                 if x in self.portfolio:
                     self.portfolio.remove(x)

        
    return (SimpletopGainTopLossPortfolio,)


@app.cell(hide_code=True)
def _(BasicPortfolio, long_frac, pd, prediction, short_frac):
    class NPredPortfolio(BasicPortfolio):

        # ----------------------------------

        def n_predictions(self, n : int = 5):
            "Repeat the predictions n times and takes the most consistent stocks"
            freq = pd.Series(index=self.universe)
            while n>0:
                predictions = self.predictor.NEXT().squeeze().sort_values(ascending = False)
                long_stocks = predictions.iloc[:int(long_frac*predictions.shape[0])].index
                short_stocks = predictions.iloc[-int(short_frac*prediction.shape[0]):].index
                freq[long_stocks] += 1 
                freq[short_stocks] -= 1
                n-=1
            freq = freq.sort_values(ascending = False)
            return freq 

        # ----------------------------------

        def get_predictions_for_universe(self, n : int, long_frac = 0.1, short_frac = 0.1):
            "Obtain the stocks that are most consistently long and those that are most consistently short"
            freq = self.n_predictions(n)
            consistently_long = freq[:int(long_frac * freq.shape[0])]
            consistently_short = freq[-int(short_frac * freq.shape[0]):]
            self.portfolio.add(consistently_long.index)
            self.portfolio.remove(x for x in consistently_short.index if x in self.portfolio)
    return


@app.cell(hide_code=True)
def _(BasicPortfolio, pd):
    class SkewChoosePortfolio(BasicPortfolio):
        """Building on the Basic Portfolio class, go long if the skewness and return are both positive"""

        def MGF(self, moment = 1, normalize = True):
            "Obtain Moments for each stock from the probabilities of their tokens"

            vocab = pd.Series(self.predictor.probs.index)
            def make_moment_df(df, p):
                momdf = pd.DataFrame()
                for col in df.columns:
                    _val = vocab ** p * df[col]
                    momdf = pd.concat(momdf, _val, axis=1)
                momdf = momdf.sum()
                return momdf 
            mp = make_moment_df(self.predictor.get_predict_probs(), moment)

            if normalize:
                m2 = make_moment_df(self.predictor.get_predict_probs(), 2)    
                mp = mp/m2**(moment/2)
            return mp

        def get_predictions_for_universe(self, long_frac: float = 0.1, short_frac: float = 0.1):
            super().get_predictions_for_universe(long_frac, short_frac)
    return


@app.cell
def _(mo):
    mo.md(r"""# my own backtesting class""")
    return


@app.cell
def _(NamedTuple, np):
    class TickerState(NamedTuple):
        quantity : int = 0 # instantaneous - on the day, <0 if selling 
        action : str = "" # buy/sell
        pnl : float = 0.0 # on the day, not per unit, but inclusive of quantity
        # any other super important metric

        def __str__(self):
            return f""" {self.action} {np.abs(self.quantity)} """
    return (TickerState,)


@app.cell
def _(pd):
    class Calendar:

        def __init__(self, start_date, end_date):
            super().__init__()
            self.start_date = pd.to_datetime(start_date) 
            self.end_date = pd.to_datetime(end_date) 
            self.cal = pd.DatetimeIndex(pd.date_range(start=start_date, end=end_date)) # start off with this 
            if self.cal.empty:
                raise ValueError("Start date must be earlier than end date")

        def sync_cal_with_data(self, data_dates : pd.Series | pd.DatetimeIndex | pd.Index):
            "gets all the dates associated with a dataframe and syncs calendar with those dates. Intersecting time b/w them"
            take_dates_from_data = data_dates[(data_dates >= self.start_date) & (data_dates<=self.end_date)]
            self.cal = pd.DatetimeIndex(take_dates_from_data)
            if self.cal.empty:
                print("Calendar is empty : date-range not present in date. Using dates from data")
                self.cal = data_dates 

        def parse_calendar(self):
            "neatly returns iterable [non-list, non indexable] zip that yields yesterday, today, tomorrow"
            return zip(self.cal[:-2], self.cal[1:-1], self.cal[2:])

        def is_date_present_in_calendar(self, date : str | pd.DatetimeIndex):
            "Tells whether a given date is in the calendar on not"
            return pd.DatetimeIndex(date) in self.cal 

        def __repr__(self):
            return f"""
            Calendar {self.start_date} -> {self.end_date}
            No of days present : {len(self.cal)}
            """
    return (Calendar,)


@app.cell
def _(Calendar, Optional, TickerState, np, pd, yf):
    class PortfolioBacktester:

        def __init__(self,
                    portfolio,
                    hist_data : Optional[pd.DataFrame] = None, 
                    start_date : Optional[pd.Timestamp | str] = None, # datetime object 
                    end_date : Optional[pd.Timestamp | str] = None, # datetime ,
                    init_capital : float = 1_000_000,
                    init_stock : float = 1_00_000,
                    margin : float = 0.1,
                    commission : float = 0.0):

            assert any(__x is not None for __x in [hist_data, start_date, end_date] ), "Either Data or Dates must be provided"
            if hist_data is not None:
                self.hist_data = hist_data 
                if start_date is None or end_date is None: # any of them are None
                    self.calendar = Calendar(hist_data.index[0], hist_data.index[-1])
                else:
                    self.calendar = Calendar(start_date=start_date, end_date=end_date) 
            else:
                self.hist_data = None
                self.calendar = Calendar(start_date=start_date, end_date=end_date) 

            self.portfolio = portfolio
            self.universe = portfolio.universe # good        
            self.margin = margin
            self.commision = commission # not used yet
            self.init_capital = init_capital # initial capital for strategy
            self.init_stock = init_stock # amount of stock held, initial condition

        # ----------------------------------------------------

        def fetch_data_from_yfinance(self) -> pd.DataFrame:
            "Obtain cleaned data from yfinance"
            start_date = self.calendar.cal[0]
            end_date = self.calendar.cal[-1]

            hist_data = yf.download(
                tickers= self.universe.tolist(),
                start = start_date - pd.Timedelta(300), #an extra year
                end = end_date,
                group_by="column")[("Close")].interpolate()
            self.universe = hist_data.columns
            return hist_data

        # ----------------------------------------------------------------

        def update_ledger(self, action : str, quantity : int, ticker : str):
            "whenever there is a transaction, update the ledger"

            pnl = (self.tomorrows_actual_prices[ticker] - self.todays_prices[ticker]) * quantity # if buy then +ve, else qty -ve
            self.ledger.at[self.today, ticker] = TickerState(quantity = quantity, action=action, pnl = pnl) 

        # ----------------------------------------------------------------

        def update_equity(self):
            """ Update Equity stock value with the present day change, along with held portfolios and transactions """

            quantities = self.ledger.loc[:self.today].map(lambda x: x.quantity).sum()
            total = quantities * self.todays_prices
            stock_value = total.sum() # stocks bought
            self.equity.at[self.today,"Stock Value"] = stock_value + self.init_stock # initial condition

            # update cash - if no transaction has taken place 
            if np.isnan(self.equity.at[self.today, "Cash Value"]):
                self.equity.at[self.today,"Cash Value"] = self.equity.at[self.yesterday,'Cash Value']

        # ------------------------------------------------------------

        def buy(self, ticker, n : Optional[int] = 1, amt : Optional[float] = None):
            "buy desired number of stock either by no. of shares, or by nearest value <= amount"

            asset_price = self.todays_prices[ticker]        
            if amt is not None:
                n = int(amt/asset_price)
            elif n is None and amt is None:
                n = int(self.equity.loc[self.yesterday, "Cash Value"]/asset_price) # buy the whole thing if n is not there 

            if n == 0:
                self.update_ledger("failed to buy : not enough cash", 0, ticker)
                self.portfolio.portfolio.remove(ticker) # necessary to remove said ticker from portfolio
                return 

            # if all ok then:
            self.update_ledger("buy", n, ticker)
            if np.isnan(self.equity.at[self.today, "Cash Value"]):
                self.equity.at[self.today, "Cash Value"] = self.equity.at[self.yesterday, "Cash Value"] - n * asset_price
            else:
                self.equity.at[self.today, "Cash Value"] = self.equity.at[self.today, "Cash Value"] - n * asset_price

        # -------------------------------------------------------------

        def sell(self, ticker, n : Optional[int] = 1, amt : Optional[float] = None):
            "sell desired number of stocks either by no. of shares, or by nearest amount"

            asset_price = self.todays_prices[ticker]  
            quantity_of_stock_held = self.ledger.loc[:self.yesterday, ticker].map(lambda x: x.quantity).sum()
            if amt is not None:
                n = int(amt/asset_price)
            elif n is None and amt is None:
                n = quantity_of_stock_held # buy the whole thing if n is not there 

            if n > quantity_of_stock_held:
                self.update_ledger("failed to sell : not enough stocks to sell", 0, ticker)
                self.portfolio.portfolio.add(ticker)
                return 

            # if all ok then: 
            self.update_ledger("sell", -n, ticker)
            if np.isnan(self.equity.at[self.today, "Cash Value"]):
                self.equity.loc[self.today, "Cash Value"] = self.equity.at[self.yesterday, "Cash Value"] + n * asset_price
            else:
                self.equity.at[self.today, "Cash Value"] = self.equity.at[self.today, "Cash Value"] + n * asset_price

        # ----------------------------------------------------------

        def stop_strategy_triggers(self):
            """Defines conditions when the algorithm fails to give profits and time to quit"""

            # if cash present encroaches into margin set, then stop trading
            if self.today is None: # init stage 
                return False

            if self.equity.loc[self.today, "Cash Value"] <= self.margin * self.equity["Cash Value"].iloc[0]:
                return True 

            # if loss is greater than the margin,  break out 
            # There must always be cash on the margin 
            margin_capital = (1 - self.margin) * self.init_capital
            if self.equity.loc[self.today, "Cash Value"] < margin_capital:
                return True # don't go forward

        # ----------------------------------------------------------
        def init_backtest(self):
            "Initialize everython required for the backtest"

            if self.hist_data is None:
                self.hist_data = self.fetch_data_from_yfinance()
            self.calendar.sync_cal_with_data(self.hist_data.index) # sync the calendar between the obtained times from yfinance

            # define ledger
            self.ledger = pd.DataFrame(data = 0,
                                       index = self.calendar.cal,
                                       columns=self.universe).map(lambda x : TickerState(quantity=0, action="",pnl=0)) # good
            # define equity
            self.equity = pd.DataFrame(index = self.calendar.cal,
                                       columns = ["Stock Value", "Cash Value"],
                                       dtype = (np.float64, np.float64) ) 

            self.equity.loc[self.calendar.cal[0], "Cash Value"] = (1 - self.margin) * self.init_capital 
            self.equity.loc[self.calendar.cal[0], "Stock Value"] = self.init_stock # No stocks initially

            self.today = self.calendar.cal[0]
            self.todays_prices = self.hist_data.loc[self.today] 
            self.tomorrows_actual_prices = self.hist_data.loc[self.calendar.cal[1]]

        # ----------------------------------------------------------    

        def update_data_before_run(self, yesterday, today, tomorrow):
            "update the present day porfolio price"

            self.today = today
            self.tomorrow = tomorrow 
            self.yesterday = yesterday
            self.todays_prices = self.hist_data.loc[self.today]
            self.tomorrows_actual_prices = self.hist_data.loc[self.tomorrow]

        # ----------------------------------------------------------    

        def run_backtest(self):
            "Initialize and run all backtest here"

            # init ------------------------------------
            self.init_backtest()

            #-----------------------------------------
            # event driven - vectorize this; also use mpi here. Its possible 
            # run excluding the IC and final condition. 
            # 
            for i, (yesterday, today, tomorrow) in enumerate(self.calendar.parse_calendar()):
                self.update_data_before_run(yesterday, today, tomorrow)

                # 
                # EXIT CONDITIONS
                # 
                if self.stop_strategy_triggers():
                    print("Stop Condition triggered")
                    return  

                #
                # ENTRY CONDITIONS
                #

                #
                # STRATEGY
                # 
                if i%20 == 0: # do this on every20th day 
                    previous_portfolio = self.portfolio.portfolio.copy() # last time's portfolio
                    self.portfolio.get_predictions_for_universe(self.hist_data.loc[:self.today]) # update portfolio

                    # capital allocation 
                    weights = self.todays_prices/self.todays_prices.sum() # check whether sum of weights is one
                    cap_allocate = self.equity.loc[self.yesterday, "Cash Value"] * weights

                    predicted_shorting = self.portfolio.short_stocks.index.tolist()
                    for s in predicted_shorting:
                        if s in previous_portfolio:
                            self.sell(s, amt = cap_allocate[s])
                    predicted_longing = self.portfolio.long_stocks.index.tolist()
                    for l in predicted_longing:
                        if l not in previous_portfolio:
                            self.buy(l, amt = cap_allocate[l])

                # update : last step 
                self.update_equity()

            # ----------------------------
            # post - remove the the last row
            self.equity["Net Worth"] = self.equity.iloc[:-1].sum(axis = 1)
            self.total_pnl = self.equity["Net Worth"].diff().iloc[:-1]
            self.returns = self.equity["Net Worth"].pct_change().iloc[:-1]

    return (PortfolioBacktester,)


@app.cell
def _(
    BasicPortfolio,
    PortfolioBacktester,
    SimpletopGainTopLossPortfolio,
    close_data,
    close_prices_gpt,
):
    # make portfolio class
    # nifty100_portfolio = BasicPortfolio(universe=stock_universe, model=close_prices_gpt)
    gpt_portfolio_strategy = BasicPortfolio(universe=close_data.columns, model=close_prices_gpt) 
    topgain_toploss_strategy = SimpletopGainTopLossPortfolio(universe=close_data.columns)

    # make backtest class 
    bt_gpt = PortfolioBacktester(portfolio=gpt_portfolio_strategy,
                             hist_data=close_data,
                             # start_date = pd.to_datetime("2025-01-01"),
                             # end_date = pd.to_datetime("2025-01-31")
                            )
    # make backtest class 
    bt_simple_shuffle = PortfolioBacktester(portfolio=topgain_toploss_strategy,
                             hist_data=close_data,
                             # start_date = pd.to_datetime("2025-01-01"),
                             # end_date = pd.to_datetime("2025-01-31")
                            )

    return bt_gpt, bt_simple_shuffle


@app.cell
def _(bt_gpt, bt_simple_shuffle):
    # run backtest class for10 iterations and see 
    bt_gpt.run_backtest()
    bt_simple_shuffle.run_backtest()
    return


@app.cell
def _(bt_gpt, bt_simple_shuffle, plt):
    plt.plot(bt_gpt.equity["Net Worth"], label = "gpt")
    plt.plot(bt_simple_shuffle.equity["Net Worth"], label = "simple shuffle")
    plt.legend()
    return


@app.cell
def _(bt_gpt):
    (bt_gpt.equity["Net Worth"].iloc[-2] - bt_gpt.equity["Net Worth"].iloc[1])/bt_gpt.equity["Net Worth"].iloc[1]/5
    return


@app.cell
def _():
    # # this is how to debug 
    # transacts= bt.ledger.loc["2020-09-04"].apply(lambda x : x.quantity != 0)
    # print(bt.ledger.loc["2020-09-04", transacts == True])
    return


@app.cell
def _(mo):
    mo.md(r"""Market Index data""")
    return


@app.cell
def _(pd):
    index_ret_5y = pd.read_csv("NIFTY 500_Historical_PR_01052020to21052025.csv",
                              usecols = ["Date", "Close"])

    months = {'Jan' : 1, 'Feb' : 2, "Mar" : 3, "Apr": 4, "May": 5,
              "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10,
              "Nov":11, "Dec":12}

    def convert_to_dates(d):
        d = d.split()
        mm = str(months[d[1]])
        dd = d[0]
        yy = d[2]
        return pd.to_datetime("-".join([yy,mm,dd]))

    index_ret_5y["Date"] = index_ret_5y['Date'].apply(convert_to_dates)
    index_ret_5y = index_ret_5y.set_index(["Date"]).sort_index().pct_change()
    index_ret_5y
    return


@app.cell
def _():
    # plt.figure()
    # plt.plot((1 + index_ret_5y).cumprod(), label = "Index Returns")
    # plt.plot( ((1 + (bt.equity["Net Worth"] - 9_00_000).pct_change()).cumprod()), label = "StockGPT Returns")
    # plt.legend()
    return


@app.cell
def _():
    # plt.figure()
    # plt.plot((1 + index_ret_5y).cumprod() * 1_00_000, label = "Index Cap") 
    # plt.plot((bt.equity["Net Worth"] - 9_00_000), label = "StockGPT Cap")
    # plt.legend()
    return


@app.cell
def _():
    # pd.DataFrame({"index":(index_ret_5y).squeeze(),
    #               "gpt":((bt.equity["Net Worth"] - 9_00_000).pct_change())}).corr()
    return


if __name__ == "__main__":
    app.run()
