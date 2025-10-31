import pandas as pd
import numpy as np
import torch
from gpt_model import GPTModel
from stock_tokenizer import (
    StockVocab,
)
from stock_loss_prediction import (
    generate_next_stock,
    generate_next_stock_simple,
)
from typing import List, Dict, NamedTuple, Optional, Any
import datetime as dt

# ===================================================


class TensorDF(NamedTuple):

    "carry metadata when converting to torch.tensor"

    data: torch.Tensor
    columns: List[str]
    ind: List[Any]


# ===================================================


class Predict:
    def __init__(
        self,
        model: GPTModel,
        cfg: Dict,
        scale=10000,
        device: str = "cpu",
    ):
        """
        I use the given GPTModel to predict the next set of returns.
        I expect a dataframe with companies(or anything) as columns and dates as index(mandatory)
        """

        self.model = model
        self.model = self.model.to(device)
        self.cfg = cfg
        self.tokenizer = StockVocab(
            bin_min=cfg["min_bin"], bin_max=cfg["max_bin"], bin_size=cfg["bin_size"]
        )
        self.scale = scale
        self.device = device

    # --------------------------------------------------------
    def __encode__(self, data: pd.Series):
        encoded_data = self.tokenizer.encode((data * self.scale).astype(np.int32))
        return encoded_data

    # --------------------------------------------------------
    def __decode__(self, tokens: List):
        decoded_data = self.tokenizer.decode(tokens) / self.scale
        return decoded_data

    # --------------------------------------------------------
    def __clean__(self, data):
        return data.interpolate().dropna(
            # axis=0
        )  # drop the rows where na is there, even after interpolate

    # --------------------------------------------------------

    def inload(self, input_data: pd.DataFrame | pd.Series, is_return=False):
        "in-load dataframe into object and encode it"

        # make sure the data is of the correct context length : trim
        input_data = input_data.iloc[-self.cfg["context_length"] :]

        if len(input_data.shape) == 1:
            input_data = input_data.to_frame()

        # save the columns and the indices
        data_columns = input_data.columns.to_list()
        data_index = input_data.index.to_list()

        if not is_return:
            _data = input_data.pct_change()
        else:
            _data = input_data

        # encoding
        # print("encoding...")
        _data = self.__clean__(_data)
        _data = _data.apply(self.__encode__)
        _DATA = torch.tensor(_data.to_numpy().T)

        # make it object's own
        self.tensor_data = TensorDF(_DATA, data_columns, data_index)
        # print("DONE")

    # ----------------------------------------------------------------

    def next(
        self, how_many: int, input_data: Optional[pd.DataFrame] = None, is_return=False
    ) -> pd.DataFrame:
        "If you give me a Dataframe of stock returns, I will give you next `how_many` days prediction"

        if input_data is not None:
            self.inload(input_data, is_return=is_return)

        # prediction
        # print("Model Predicting...")
        _output: torch.Tensor = self.__generate__(how_many)
        if _output.dim() == 1:
            _output.unsqueeze(dim=0)

        # decoding
        _data_with_output: map[List[float]] = map(self.__decode__, _output.tolist())
        _final_df = pd.DataFrame(
            list(zip(*_data_with_output)),  # transpose List of Lists
            columns=self.tensor_data.columns,
        )

        return _final_df

    # --------------------------------------------------------

    def __generate__(self, how_many: int, simple=False):
        """
        calls the standard generate function multinomial form from stock_loss_prediction.
        Can/Should be replaced by function overloading, inheritance.
        Can be used for daily stock prediction
        """

        # if self.tensor_data.data:
        # raise ValueError("Data must be loaded into predictor before predictions!")

        _DATA = self.tensor_data.data
        if _DATA.dim() == 1:
            _DATA = _DATA.unsqueeze(dim=0)
        _DATA = _DATA.to(self.device)

        if not simple:
            return generate_next_stock(
                model=self.model,
                idx=_DATA,
                max_new_tokens=how_many,
                context_size=self.cfg["context_length"],
                temperature=self.cfg["temperature"],
                topk=self.cfg["topk"],
                result_with_input=False,
            )

        else:
            return generate_next_stock_simple(
                self.model,
                idx=_DATA,
                max_new_tokens=how_many,
                context_size=self.cfg["context_length"],
                result_with_input=False,
            )

    # -------------------------------------------------------

    @property
    def __probs__(self, use_temp=True):
        "Get the probability logits from the multinomial distribution and then intuit long_short from its pdf"
        self.model = self.model.to(self.device)
        self.model.eval()

        _DATA = self.tensor_data.data
        _DATA = _DATA.to(self.device)
        _DATA = _DATA[:, -self.cfg["context_length"] :]  # this is torch.tensor

        with torch.no_grad():
            logits: torch.Tensor = self.model(_DATA)

        logits = logits[:, -1, :]
        if use_temp:
            temperature = self.cfg["temperature"]
        else:
            temperature = 1
        probs = torch.softmax(logits / temperature, dim=-1)

        return probs

    # --------------------------------------------------------

    def get_predict_probs(self):
        return pd.DataFrame(
            self.__probs__.T.tolist(),
            columns=list(self.tensor_data.columns),
            index=self.tokenizer.vocab / self.scale,
        )

    # -------------------------------------------------------

    def long_short_guess_from_probs(self, band: int = 50) -> pd.DataFrame:
        "predict whether the next model output is likely to be [small, regular, large] x [long, short]"

        probs = self.__probs__
        new_probs = []
        for t in probs:
            p = []
            for b in range(0, self.tokenizer.vocab_size, band):
                p.append(t[b : b + band].sum())
            new_probs.append(p)

        indices = np.arange(0, self.tokenizer.vocab_size + 1, band).tolist()
        indices = self.__decode__(indices)

        return pd.DataFrame(
            list(zip(*new_probs)),
            columns=self.tensor_data.columns,
            index=indices,
        ).astype(np.float64)

    # --------------------------------------------------------

    def MGF(self, moment=1) -> pd.DataFrame:
        "return the nth-order moment of the probability function of logits"

        probs = self.__probs__
        vocab_ids = (
            self.tokenizer.vocab / 100  # LOOK AT THIS HARDCODED SHIT
        )  # IMPIMPIMPIMPIMP - scale between -100, 100
        arg = torch.pow(
            torch.tensor(vocab_ids).to(torch.float32),
            torch.tensor(moment).to(torch.float32),
        ).unsqueeze(dim=0)

        return pd.DataFrame(
            (arg @ probs.T).tolist(),
            columns=self.tensor_data.columns,
            index=[moment],
        )

    # --------------------------------------------------------

    def log_probs_cluster(self, number_log_space_bins=7):
        probs = self.__probs__
        log_probs = torch.log10(probs)  # ..., -3,-2,-1,0,1,2,3 ...
        log_space_bin_size = self.tokenizer.vocab_size // number_log_space_bins
        new_probs = []
        for t in log_probs:
            p = []
            for b in range(0, self.tokenizer.vocab_size, log_space_bin_size):
                p.append(t[b : b + log_space_bin_size].sum())
            new_probs.append(p)

        indices = self.__decode__(
            np.arange(0, self.tokenizer.vocab_size + 1, log_space_bin_size).tolist()
        )

        assert len(indices) == len(new_probs), "Prob dim != index dim"
        return pd.DataFrame(
            list(zip(*new_probs)),
            columns=self.tensor_data.columns,
            index=indices,
        ).astype(np.float64)


# ===================================================


def compare_gpt_vs_hist_data(
    gpt_predictor: Predict, hist_data: pd.DataFrame, plot: bool = True
):
    import matplotlib.pyplot as plt

    close_data = hist_data["Close"]
    gpt_preds = [0] * 4  # padding
    nrows = close_data.shape[0]
    for i in range(5, nrows + 1):
        gpt_predictor.inload(
            close_data[:i].iloc[-250:], is_return=False
        )  # if we give full 256 data, he model gives output 0.
        gpt_preds.append(gpt_predictor.get_predict_probs().idxmax().squeeze())
    gpt_preds = pd.Series(gpt_preds, index=close_data.index)

    if plot:
        plt.figure()
        close_data.pct_change().rolling(20).mean().plot(
            kind="line", label="rolling close"
        )
        gpt_preds.plot(kind="line", label="gpt")
        plt.legend()
        plt.show()

    return pd.DataFrame(
        {"gpt_preds": gpt_preds, "close_returns": close_data.pct_change()}
    )
