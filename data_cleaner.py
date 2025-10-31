import numpy as np
import pandas as pd
import glob
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import List, Tuple, Dict,Iterable, Optional, Callable
import os
from stock_tokenizer import StockTokenizer

# ===============================================


class DataCleanerAssembler:

    def __init__(
        self,
        cfg: Dict,  # needed for stock tokenizer
        stock_path: (
            str | os.PathLike | List[str | os.PathLike]
        ),  # more explicit, Will not use the stock from cfg config
        is_batch: bool,
        reader : Optional[Callable[str, pd.DataFrame| pd.Series]] = None,
        is_return: bool = False,  # needed for stock tokenizer, tells whether dataset is returns or not
    ):

        self.reader = reader 
        self.stock_path = stock_path
        self.is_batch = is_batch
        self.error_stocks = []
        self.is_return = is_return
        self.cfg = cfg

    # ----------------------------------------------

    def __clean_read_single_file__(self, file):

        try:
            if self.reader is not None:
                _data = self.reader(file)
            else:
                _data = pd.read_csv(
                    file,
                    usecols=["Date", "Close"],
                    parse_dates=["Date"],
                    index_col="Date",
                ).squeeze()

            # in case we have 1 row, this becomes a number.
            if not isinstance(_data, pd.Series):
                print(f"There is something wrong in the content of file {file}")
                _data = pd.Series([])  # very protective

            _data = self.modify(_data)

        except Exception as err:
            print(f"Unexpected error in file {file}, error {err=}, {type(err)=}")
            self.error_stocks.append(file)
            _data = pd.Series([])  # return empty series

        else:
            if isinstance(_data, pd.Series) and not _data.empty:
                _data = _data.interpolate()  # single interpolate
                _data = _data.dropna()  # remove NaNs at the ends if necessary
                _data = _data.sort_index()  # again resort the indices per date
            else:
                self.error_stocks.append(file)

        return _data

    # ----------------------------------------------

    def __control__(self):
        if isinstance(self.stock_path, list):
            dataset = self.__custom_stock_path_reader__()
        elif self.is_batch:
            dataset = self.__clean_read_batch__()
        else:
            dataset = [self.__clean_read_single_file__(self.stock_path)]
            # to be compatible with map(,[])

        return dataset

    # ----------------------------------------------

    def __clean_read_batch__(self):

        files = [os.path.join(self.stock_path, x) for x in os.listdir(self.stock_path)]
        _cleaned_stocks = map(self.__clean_read_single_file__, files)
        _cleaned_stocks = filter(lambda x: len(x) > 0, _cleaned_stocks)

        return _cleaned_stocks

    # ----------------------------------------------

    def __tokenize_data__(self, file):

        return StockTokenizer(
            file,
            multiply_by=self.cfg["multiply_by"],
            bin_size=self.cfg["bin_size"],
            min_bin=self.cfg["min_bin"],
            max_bin=self.cfg["max_bin"],
            input_batch_size=self.cfg["context_length"],
            target_batch_size=1,  # has to be 1 - next token procedure.
            stride=self.cfg["stride"],
            is_return=self.is_return,
        )

    # ----------------------------------------------

    def ConcatDataset(self):

        _batch = self.__control__()
        return ConcatDataset(map(self.__tokenize_data__, _batch))

    # ----------------------------------------------

    def __str__(self):

        return f"""

        Data Cleaned and tokenized as batch. 
        Error faulty-unexpected-files encountered : {self.error_stocks}
        
        """

    # ----------------------------------------------

    def __custom_stock_path_reader__(self):

        datasets = map(self.__clean_read_single_file__, self.stock_path)
        return datasets

    # ----------------------------------------------

    def modify(self, data):
        "Inherit this function to modify contents of the data cleaner here"

        return data


if __name__ == "__main__":

    from gpt_config import GPT_CONFIG

    _dc = DataCleanerAssembler(
        cfg=GPT_CONFIG,
        stock_path=[
            os.path.join(GPT_CONFIG["stock_path"], "a", "aapl.us.txt"),
            os.path.join(GPT_CONFIG["stock_path"], "h", "hp.us.txt"),
        ],
        is_return=False,
        is_batch=False,
    )
    print(_dc)

    print(_dc.ConcatDataset())

    print(_dc)
