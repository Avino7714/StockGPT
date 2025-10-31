import pandas as pd
import numpy as np
import torch
from ohlcv_gpt_model import OHLCV_GPTModel
from ohlcv_tokenizer import OHLCV_Vocab
from typing import List, Dict, NamedTuple, Optional, Any

# ===================================================


class TensorDF(NamedTuple):

    "carry metadata when converting to torch.tensor"

    data: torch.Tensor
    columns: List[str]
    ind: List[Any]


# ===================================================


class OHLCV_Predict:
    def __init__(
        self,
        model: OHLCV_GPTModel,
        cfg: Dict,
        device: str = "cpu",
    ):
        """
        I use the given GPTModel to predict the next set of returns.
        I expect a dataframe with one company stock (or anything) as columns and dates as index(mandatory)
        """

        self.model = model
        self.model = self.model.to(device)
        self.cfg = cfg
        self.tokenizer = OHLCV_Vocab(cfg=self.cfg)
        self.device = device

    # --------------------------------------------------------
    def __encode__(self, data: pd.Series):
        encoded_data = self.tokenizer.encode(data)
        return encoded_data

    # --------------------------------------------------------
    def __decode__(self, tokens: List):
        decoded_data = self.tokenizer.decode(tokens)
        return decoded_data

    # --------------------------------------------------------
    def __preprocess_and_clean__(self, data):
        "WTF this is the exact same procedure to do as in the tokenizer. Include all this in the encoding in ohlcv_tokenizer"
        # data["Volume"] = (
        #     data["Volume"] / data["Volume"].max() * self.tokenizer.vocab_max
        # )
        data = data[["Open", "High", "Low", "Close"]]
        return data.interpolate().dropna(axis=0)

    # --------------------------------------------------------

    def inload(self, input_data: pd.DataFrame):

        "in-load dataframe into object and encode it. Can only do one stock at a time"

        input_data = input_data.iloc[-self.cfg["context_length"] :]  # trim context
        _data = self.__preprocess_and_clean__(input_data)  # encoding
        _data = _data.apply(self.__encode__)
        _DATA = torch.tensor(_data.to_numpy())
        _DATA = _DATA.to(self.device)  # throw data into gpt here itself.. good idea

        data_columns = input_data.columns.to_list()
        data_index = input_data.index.to_list()
        self.tensor_data = TensorDF(_DATA, data_columns, data_index)

    # --------------------------------------------------------

    def generate_next_day_price(
        self,
        use_temperature: bool = True,
        use_topk: bool = True,
    ) -> torch.Tensor:

        "generates next stock with top k sampling and temperature scaling - generates only 1 day prices at present"

        self.model.eval()
        _DATA = self.tensor_data.data.unsqueeze(dim=0)
        idx_cond = _DATA[:, -self.cfg["context_size"] :, :]

        with torch.no_grad():
            logits = self.model(idx_cond)

        logits = logits[:, -1, :]

        if use_topk:
            topk = self.cfg["topk"]
            top_logits, _ = torch.topk(logits, topk)
            if top_logits.dim() == 1:
                top_logits = top_logits.unsqueeze(dim=0)
            min_val = top_logits[:, -1]
            min_val = torch.reshape(min_val, (min_val.shape[0], 1))  # for comparison
            logits = torch.where(
                logits < min_val[None, :], torch.tensor(float("-inf")), logits
            )

        if use_temperature:
            temperature = self.cfg["temperature"]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        return self.__decode__(
            idx_next.tolist()
        ).squeeze()  # squeeze series into number

    # -------------------------------------------------------

    @property
    def __probs__(self, use_temp=True):
        "Get the probability logits from the multinomial distribution and then intuit long_short from its pdf"
        self.model.eval()

        _DATA = self.tensor_data.data.unsqueeze(dim=0)
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
        return pd.Series(
            self.__probs__.T.tolist(),
            index=self.tokenizer.tokens,
        )

    # -------------------------------------------------------

    def long_short_guess_from_probs(self, band: int = 5000) -> pd.Series:
        "predict whether the next model output is likely to be [small, regular, large] x [long, short]"

        probs = self.__probs__
        new_probs = []
        p = []
        for b in range(0, self.tokenizer.vocab_size + 2, band):
            p.append(probs[b : b + band].sum())
        new_probs.append(p)

        indices = np.arange(0, self.tokenizer.vocab_size + 2, band).tolist()
        indices = self.__decode__(indices)

        return pd.Series(
            new_probs,
            index=indices,
        ).astype(np.float64)

    # --------------------------------------------------------

    def MGF(self, moment=1) -> pd.DataFrame:
        "return the nth-order moment of the probability function of logits"

        probs = self.__probs__
        vocab_ids = self.tokenizer.tokens
        arg = torch.pow(
            torch.tensor(vocab_ids).to(torch.float32),
            torch.tensor(moment).to(torch.float32),
        ).unsqueeze(dim=0)

        return pd.DataFrame(
            (arg @ probs.T).tolist(),
            index=[moment],
        )

    # --------------------------------------------------------

    def log_probs_cluster(self, number_log_space_bins=7):

        "log spaced large bins to sum probability"

        probs = self.__probs__
        log_probs = torch.log10(probs)  # ..., -3,-2,-1,0,1,2,3 ...
        log_space_bin_size = self.tokenizer.vocab_size // number_log_space_bins
        new_probs = []
        p = []
        for b in range(0, self.tokenizer.vocab_size, log_space_bin_size):
            p.append(log_probs[b : b + log_space_bin_size].sum())
        new_probs.append(p)

        indices = self.__decode__(
            np.arange(0, self.tokenizer.vocab_size + 1, log_space_bin_size).tolist()
        )

        assert len(indices) == len(new_probs), "Prob dim != index dim"
        return pd.DataFrame(
            new_probs,
            index=indices,
        ).astype(np.float64)


# ===================================================
