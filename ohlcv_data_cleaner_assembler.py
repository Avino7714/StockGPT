import numpy as np
import pandas as pd
from ohlcv_tokenizer import OHLCV_Tokenizer
from typing import List, Tuple, Dict, Iterator, Iterable, Optional
from torch.utils.data import ConcatDataset
import os

"""
    Create a Multi-file data cleaner assembler for OHLCV datasets 
"""


class OHLCV_DataCleanerAssembler:  # cannot inherit from DataCleanerAssembler - too risky.

    "For large number of files in folders, combine and load into a dataloader"

    def __init__(
        self,
        stock_path: str | os.PathLike | List[str | os.PathLike],
        cfg: Dict,
        is_batch: bool,
    ):

        self.stock_path = stock_path
        self.is_batch = is_batch
        self.cfg = cfg
        self.error_stocks = []

    # ----------------------------------

    def __clean_read_single_file__(self, file) -> pd.DataFrame:

        "modify the clean reader-parser for all columns"

        try:
            _data = pd.read_csv(
                file,
                parse_dates=["Date"],
                index_col="Date",
            )

            # in case we have 1 row, this becomes a number.
            if not isinstance(_data, pd.DataFrame):
                print(f"There is something wrong in the content of file {file}")
                _data = pd.DataFrame([])  # very protective

        except Exception as err:
            print(f"Unexpected error in file {file}, error {err=}, {type(err)=}")
            self.error_stocks.append(file)
            _data = pd.DataFrame([])  # return empty series

        else:
            if isinstance(_data, pd.DataFrame) and not _data.empty:
                _data = _data.interpolate()  # single interpolate
                _data = _data.dropna()  # remove NaNs at the ends if necessary
                _data = _data.sort_index()  # again resort the indices per date
            else:
                self.error_stocks.append(file)

        return _data

    # ----------------------------------------------

    def __tokenize_data__(self, file) -> OHLCV_Tokenizer:

        "read and tokenize all ohlcv data"

        return OHLCV_Tokenizer(
            stock_dataset=file,
            cfg=self.cfg,
            input_batch_size=self.cfg["input_batch_size"],
            target_batch_size=self.cfg["target_batch_size"],
            stride=self.cfg["stride"],
        )

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

    # ==============================================


if __name__ == "__main__":

    from gpt_config import GPT_CONFIG

    _dc = OHLCV_DataCleanerAssembler(
        cfg=GPT_CONFIG,
        stock_path=[
            os.path.join(GPT_CONFIG["stock_path"], "a", "aapl.us.txt"),
            os.path.join(GPT_CONFIG["stock_path"], "h", "hp.us.txt"),
        ],
        is_batch=False,
    )

    print(_dc)
