from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

"""
    Create a new stock tokenizer class for all ohlcv data using price values themselves as tokens
    Does not use binning
    Tries to predict the next day's close prices 
"""


class OHLCV_Vocab:

    "Creates an integer price vocabulary by converting floating point prices into integer tokens"

    def __init__(self, cfg: Dict):

        self.cfg = cfg
        self.vocab_min = cfg["min_vocab"]  # 0
        self.vocab_max = cfg["max_vocab"]
        self.vocab_size = self.vocab_max - self.vocab_min + 1
        self.tokens = list(
            range(self.vocab_min, self.vocab_max + 1)
        )  # difference is 1 b/w integer tokens

    def encode(self, input_prices: pd.Series) -> pd.Series:
        "make stocks into tokens"
        return input_prices.map(np.floor).astype(np.int64)

    def decode(self, tokens: List[int]) -> pd.Series:
        return pd.Series(tokens).astype(np.float64)

    def __str__(self):
        return f"""
        Integerized Prices themselves are tokens for this model. 
            Min Token : {self.vocab_min}
            Max Token : {self.vocab_max}
            Vocab Size : {self.vocab_size}
        """


# ===========================================================


class OHLCV_Tokenizer(Dataset):

    "tokenizes all the ohlcv data and creates the dataset objects"

    def __init__(
        self,
        stock_dataset: pd.DataFrame,
        cfg: Dict,
        input_batch_size: int = 256,
        target_batch_size: int = 1,
        stride: int = 1,
    ):

        super().__init__()
        self.cfg = cfg
        self.input_batch_size = input_batch_size
        self.target_batch_size = target_batch_size
        self.stride = stride

        assert isinstance(
            stock_dataset, pd.DataFrame
        ), " Tokenizer Input is not a DataFrame!!"

        # vocab
        self.vocab = OHLCV_Vocab(
            cfg=self.cfg,
        )
        self.stock_dataset = stock_dataset[
            ["Open", "High", "Low", "Close"]
        ]  # rearrange data for consistency
        self.stock_dataset = self.stock_dataset.reset_index(drop=False)
        self.stock_dataset["Date"] = (
            self.stock_dataset["Date"] - self.stock_dataset["Date"].iloc[0]
        ).astype(np.int64)

        # encode
        self.stock_dataset = self.stock_dataset.apply(self.vocab.encode)
        # make batch
        self.__make_batch__()

    # -------------------------------------

    def __make_batch__(self):

        "make batch of input and target. Check whether we can make it into a dataset, report otherwise. If other modifications to input and targets such as making averaged predictions, then modify here."

        nrows = self.stock_dataset.shape[0]
        self.input_batch = []
        self.target_batch = []

        for i in range(0, nrows - self.input_batch_size, self.stride):
            input_chunk = self.stock_dataset.iloc[
                i : i + self.input_batch_size
            ]  # all ohlcv data
            target_chunk = self.stock_dataset["Close"].iloc[
                i
                + self.target_batch_size : i
                + self.input_batch_size
                + self.target_batch_size,
            ]  # only next day's close data

            if not input_chunk.empty and not target_chunk.empty:
                self.input_batch.append(torch.tensor(input_chunk.to_numpy()))
                self.target_batch.append(torch.tensor(target_chunk.to_numpy()))

    # -------------------------------------

    def __len__(self):
        return len(self.input_batch)

    # ------------------------------------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_batch[idx], self.target_batch[idx]

    # ------------------------------------

    def __repr__(self) -> str:
        return f"""
            Tokenizer for OHLCV data >>> 
        
            Each input inside torch Dataset has shape : {self.input_batch[0].shape}
            Each target inside torch Dataset has shape : {self.target_batch[0].shape}

            First Tokenized Element of the input:
            {
            self.input_batch[0]
            }
            First Tokenized Element of the target:
            {
            self.target_batch[0]
            }
        """

    # ------------------------------------


# WORKS - DONEDONEDONE
