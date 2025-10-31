# import marimo as mo
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import os
import sys
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import datetime as dt
from typing import List, Tuple, Dict, Iterable, Iterator

# =============================================


class StockVocab:

    "create the stock vocabulary essentials : bin_max, bin_min, bin_size, etc and keep all attributes ready"

    def __init__(self, bin_min: float, bin_max: float, bin_size: float):
        self.bin_min, self.bin_max, self.bin_size = bin_min, bin_max, bin_size
        self.bin_edges = pd.Series(
            np.arange(bin_min - bin_size / 2, bin_max + bin_size, bin_size),
        )
        self.vocab = self.bin_edges[:-1] + self.bin_size / 2  # vocab

        self.token_ids = self.vocab.index.to_numpy()
        self.vocab_size = len(self.vocab)

    # -----------------------------------------

    def __str__(self):
        return f"""
            ====================================================
            Stock Vocabulary Mapping >>> 

            No. of bins created = {self.bin_edges.shape[0]}
            Vocab_size = {self.vocab_size}
            Minimum Possible = {self.vocab.iloc[0]}
            Maximum Possible = {self.vocab.iloc[-1]}
            ====================================================
                    """

    # -----------------------------------------

    def encode(self, prices: pd.Series) -> pd.Series:  # works, but not used yet.
        "make stocks into tokens int32. Argument must not contain NaNs and things"

        return pd.cut(prices, labels=self.token_ids, bins=self.bin_edges).astype(
            np.int32
        )

    # -----------------------------------------

    def decode(self, tokens: List[int]) -> pd.Series:  # works and tested
        "returns the return marker of the stock"

        return pd.Series(tokens).replace(to_replace=self.vocab)


# =======================================================


# binning class
class StockTokenizer(Dataset):
    """
    Tokenized Dataset from continuous stock data.
    Calculates returns for one or more stocks data.
    Data must be free of NaNs for simplicity.
    Data best be used as a 1 column dataframe rather than series. While the tokenization works for multicolumn dataset, use wisely.
    """

    def __init__(
        self,
        stock_dataset: pd.Series,  # accepts only series input
        multiply_by: float = 1e4,
        bin_size: int = 50,
        min_bin: int = -10_000,
        max_bin: int = 10_000,
        input_batch_size: int = 256,  # Dat Mai
        target_batch_size: int = 1,  # next word prediction-like
        stride: int = 4,
        is_return: bool = False,  # if the input is already return dataset or not
    ):
        super().__init__()
        assert isinstance(
            stock_dataset, pd.Series
        ), "Tokenizer input is not a Series !! "

        self.dataset = (
            stock_dataset.pct_change().iloc[1:] if not is_return else stock_dataset
        )

        # 1 - get the bins ready
        self.vocab = StockVocab(bin_min=min_bin, bin_max=max_bin, bin_size=bin_size)

        # 2 - get returns and integerize them
        self.integerize_dataset(by=multiply_by)

        # 3 - make the final dataset
        self.binning()

        # 4 - make batched data
        self.input_batch_size = input_batch_size
        self.target_batch_size = target_batch_size
        self.stride = stride
        self.make_batch()

    # -----------------------------------------

    def integerize_dataset(self, by):
        "2.4% -> 0.24 * 10_000. If point > 10,000, then make 10_000, <-10_000, make -10_000. Also cleans out NaNs."

        self.dataset = self.dataset.interpolate()
        self.dataset = (self.dataset * by).map(np.floor, na_action="ignore")
        self.dataset = self.dataset.map(lambda x: 10_000 if x > 10_000 else x)
        self.dataset = self.dataset.map(lambda x: -10_000 if x < -10_000 else x)

    # -----------------------------------------

    def binning(self):  # encode
        "convert the dataset into discrete tokens : 2.4% -> 24000 -> 192 token ID, integer and fill nans"

        self.dataset = pd.cut(
            self.dataset, labels=self.vocab.token_ids, bins=self.vocab.bin_edges
        ).astype(np.int32)

        # convert to 32-bit integer
        # self.dataset = self.dataset.astype(np.int32)

    # -----------------------------------------

    def __str__(self):
        return f"""
            {self.vocab.__str__()}
            {self.dataset.info()}
            Input Batch : {len(self.input_batch)} x {self.input_batch[0].shape} List[torch.Tensor]
            Target Batch : {len(self.target_batch)} x {self.target_batch[0].shape} List[Torch.Tensor]
        
        """

    # -----------------------------------------

    def make_batch(self):
        """
        make batch of inputs and target batch. Check whether we can make it into a dataset, otherwise report data unusable.
        """

        # check whether batches is possible
        nrows = self.dataset.shape[0]
        self.input_batch = []
        self.target_batch = []

        for i in range(0, nrows - self.input_batch_size, self.stride):
            input_chunk = self.dataset.iloc[i : i + self.input_batch_size]
            target_chunk = self.dataset.iloc[  # important
                i
                + self.target_batch_size : i
                + self.input_batch_size
                + self.target_batch_size
            ]

            if (
                not input_chunk.empty
                and not target_chunk.empty
                # and input_chunk.shape[0] == self.input_batch_size
                # and target_chunk.shape[0] == self.target_batch_size
            ):
                # to numpy() gives back vectors as rows.
                # transpose is necessary to make the data column-wise
                # Date information is lost completely in the process

                self.input_batch.append(torch.tensor(input_chunk.to_numpy()).squeeze())
                self.target_batch.append(
                    torch.tensor(target_chunk.to_numpy()).squeeze()
                )

        self.batched = True

    # -----------------------------------------

    def __getitem__(self, idx: int):
        "if there is no data for index, returns None"
        if idx < len(self.input_batch) and (self.input_batch and self.target_batch):
            return self.input_batch[idx], self.target_batch[idx]
        else:
            print("No items in dataset to get")
            return None, None

    # -----------------------------------------

    def __len__(self):
        return len(self.input_batch)

    # -----------------------------------------

    def __repr__(self):
        # return f"""
        #     {repr(self.vocab)}
        #     {repr(self.dataset.info())}
        #     Input Batch : {len(self.input_batch)} x {self.input_batch[0].shape}
        #     Target Batch : {len(self.target_batch)} x {self.target_batch[0].shape}
        # """
        return self.__str__()

    # -----------------------------------------


# ============================================


def create_data_loader_for_stock(
    stock_dataset: pd.Series,
    cfg: Dict,
    is_return: bool,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> Tuple[Dataset, DataLoader]:
    "creates a dataset and dataloaders using default parameters of the StockTokenizer class"

    # create dataset
    dataset = StockTokenizer(
        stock_dataset,
        multiply_by=cfg["multiply_by"],
        bin_size=cfg["bin_size"],
        min_bin=cfg["min_bin"],
        max_bin=cfg["max_bin"],
        input_batch_size=cfg["context_length"],
        target_batch_size=1,  # has to be 1 - next token procedure.
        stride=cfg["stride"],
        is_return=is_return,
    )

    #
    stock_dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataset, stock_dataloader


# ===========================================


def create_data_loader_for_tokenizer(
    tokenizer: StockTokenizer,
    cfg: Dict,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    "create dataloader object given a tokenizer object"

    stock_dataloader = DataLoader(
        tokenizer, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return stock_dataloader


# ===========================================


def train_val_split(
    dataset: pd.Series, train_pct: float
) -> Tuple[pd.Series, pd.Series]:
    assert 0 < train_pct <= 1

    np.random.seed(123)
    train = dataset.sample(frac=train_pct)
    val = dataset.drop(index=train.index)

    return train, val


# ===========================================


def tokenizer_dataloader_dataframe(
    interested_stocks: List[str], cfg: Dict, is_return=False
) -> pd.DataFrame:
    """
    Accessor function to read data from similar files in a folder, make data loader over it
    and create a dataframe of dataloader objects.
    Stock wise dataloader, train_loader and val_loader"""

    tokenized_stocks = pd.DataFrame(
        [], columns=["train", "val"], index=interested_stocks
    )

    for s in interested_stocks:
        try:
            _d = pd.read_csv(
                os.path.join(cfg["stock_path"], s),
                parse_dates=["Date"],
                index_col="Date",
                usecols=["Date", "Close"],
            ).squeeze()
            _train, _val = train_val_split(_d, cfg["train_ratio"])
            tokenized_stocks.loc[s] = [_train, _val]

        except:
            print("Cannot read {}. Error in File..".format(s))

    tokenized_stocks = tokenized_stocks.map(
        lambda x: create_data_loader_for_stock(
            x, cfg, is_return=is_return, batch_size=cfg["batch_size"]
        )[1]
    )

    # clean the stocks. Remove the stocks that have ZERO train/val loader data
    # Improve upon this later. Check whether 0 train/0 val data has any effect on
    # the training sequence

    tokenized_stocks = tokenized_stocks[tokenized_stocks.map(len) > 0].dropna(axis=0)

    return tokenized_stocks


# ===========================================


def merge_dataloaders(*itrs):
    "Combine a list of dataloader objects into a large data iterator"
    for itr in itrs:
        for v in itr:
            yield v


# ===========================================
# ===========================================


if __name__ == "__main__":
    from gpt_config import GPT_CONFIG

    _stock_return = (
        pd.read_csv(
            os.path.join(GPT_CONFIG["stock_path"], "j", "jpme.us.txt"),
            usecols=["Date", "Close"],
            index_col="Date",
            parse_dates=["Date"],
        )
        .pct_change()
        .iloc[1:]
        .squeeze()
    )  # 1 column dataframe

    _st = StockVocab(-1e4, 1e4, 50)
    # print(_st.bin_edges)  # -- True
    # print(_st.vocab)
    print(_st)

    tokenized_returns, _dl = create_data_loader_for_stock(
        _stock_return, cfg=GPT_CONFIG, batch_size=4, is_return=True
    )
    _i = iter(_dl)
    print(next(_i))
    print(next(_i)[0].shape)

    print("=======")
    # print(tokenized_returns.vocab)

    print(_st.encode(_stock_return * 1e4))
    print(tokenized_returns.dataset)

    print(_st.decode(torch.randint(low=0, high=401, size=(64,)).tolist()))
