import pandas as pd
from data_cleaner import DataCleanerAssembler
import torch
from torch import nn
import os
from typing import List, Dict
from stock_tokenizer import StockTokenizer, StockVocab
import numpy as np

# =============================================

# Create a database where model can train on the average of next 20 returns

# ============================================


class MonthDataCleanerAssembler(DataCleanerAssembler):

    "overrides the tokenize data"

    # ------------------------------------------------

    def __tokenize_data__(self, file):
        # return super().__tokenize_data__(file)
        return MonthStockTokenizer(
            file,
            multiply_by=self.cfg["multiply_by"],
            bin_size=self.cfg["bin_size"],
            min_bin=self.cfg["min_bin"],
            max_bin=self.cfg["max_bin"],
            input_batch_size=self.cfg["context_length"],
            target_batch_size=1,  # has to be 1 - next token procedure.
            stride=self.cfg["stride"],
            window=self.cfg["rolling_window"],
            is_return=self.is_return,
        )

    # ------------------------------------------------


# ============================================


class MonthStockTokenizer(torch.utils.data.Dataset):  # StockTokenizer):

    "A replica of the stock tokenizer class for monthly returns. Does not inherit StockTokenizer class."

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
        window: int = 20,
        is_return: bool = False,  # if the input is already return dataset or not
    ):
        # super().__init__(
        #     stock_dataset,
        #     multiply_by,
        #     bin_size,
        #     min_bin,
        #     max_bin,
        #     input_batch_size,
        #     target_batch_size,
        #     stride,
        #     is_return,
        # )

        super().__init__()

        self.window = window

        self.input_dataset = (
            stock_dataset.pct_change().iloc[1:] if not is_return else stock_dataset
        )

        # forward rolling
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.window)
        forward_window_avg = stock_dataset.rolling(window = indexer, min_periods=1).mean()
        self.target_dataset = forward_window_avg.pct_change().dropna()
        
        #self.target_dataset = (
        #    self.input_dataset.rolling(window=indexer, min_periods=1).mean().dropna()
        #)
        # regular rolling
        # self.target_dataset = self.input_dataset.rolling(window).mean().dropna()

        self.input_dataset = self.input_dataset.loc[self.target_dataset.index]

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

    # --------------------------------------------------

    def integerize_dataset(self, by):

        # integerize inputs
        self.input_dataset = (
            self.input_dataset.interpolate()
        )  # just make sure there are no NaNs and things
        self.input_dataset = (self.input_dataset * by).map(np.floor, na_action="ignore")
        self.input_dataset = self.input_dataset.map(
            lambda x: 10_000 if x > 10_000 else x
        )
        self.input_dataset = self.input_dataset.map(
            lambda x: -10_000 if x < -10_000 else x
        )

        # integerize targets
        self.target_dataset = (
            self.target_dataset.interpolate()
        )  # just make sure there are no NaNs and things
        self.target_dataset = (self.target_dataset * by).map(
            np.floor, na_action="ignore"
        )
        self.target_dataset = self.target_dataset.map(
            lambda x: 10_000 if x > 10_000 else x
        )
        self.target_dataset = self.target_dataset.map(
            lambda x: -10_000 if x < -10_000 else x
        )

        assert (
            self.input_dataset.shape[0] == self.target_dataset.shape[0]
        ), "Input and Target indices do not match!!!"

    # --------------------------------------------------

    def binning(self):

        self.input_dataset = self.vocab.encode(self.input_dataset)
        self.target_dataset = self.vocab.encode(self.target_dataset)

    # --------------------------------------------------

    def make_batch(self):
        """
        make batch of inputs and target batch. Check whether we can make it into a dataset, otherwise report data unusable.
        """

        # check whether batches is possible
        self.input_batch = []
        self.target_batch = []
        nrows = self.input_dataset.shape[0]

        for i in range(0, nrows - self.input_batch_size, self.stride):
            input_chunk = self.input_dataset.iloc[i : i + self.input_batch_size]
            target_chunk = self.target_dataset.iloc[  # important
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

                self.input_batch.append(
                    torch.tensor(input_chunk.to_numpy().T).squeeze()
                )
                self.target_batch.append(
                    torch.tensor(target_chunk.to_numpy().T).squeeze()
                )

    # -----------------------------------------

    def __str__(self):
        return f"""
            {self.vocab.__str__()}
            {self.input_dataset.info()}
            {self.target_dataset.info()}
            Input Batch : {len(self.input_batch)} x {self.input_batch[0].shape} List[torch.Tensor]
            Target Batch : {len(self.target_batch)} x {self.target_batch[0].shape} List[Torch.Tensor]
        
        """

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
        return self.__str__()

    # -----------------------------------------


# ============================================
