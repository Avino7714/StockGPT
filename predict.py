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

# ===================================================


class TensorDF(NamedTuple):

    "carry metadata when converting to torch.tensor"

    data: torch.Tensor
    columns: List[str]
    ind: List[Any]


# ===================================================

class Predict:
    def __init__(self,
                model : GPTModel,
                cfg : Dict, 
                device : str = "cpu"):

        self.model = model 
        self.cfg = cfg 
        self.tokenizer = StockVocab(
            bin_min=cfg["min_bin"], bin_max=cfg["max_bin"], bin_size=cfg["bin_size"]
        )
        self.device = device 
        self.model = self.model.to(device)

    # --------------------------------------------------------
    def __encode__(self, data: pd.Series):
        encoded_data = self.tokenizer.encode((data * 10_000).astype(np.int32))
        return encoded_data

    # --------------------------------------------------------
    def __decode__(self, tokens: List):
        decoded_data = self.tokenizer.decode(tokens) / 10_000
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
        _DATA = torch.tensor(_data.to_numpy().T).to(self.device)
        
        # make it object's own
        self.tensor_data = TensorDF(_DATA, data_columns, data_index)
        # print("DONE")

    # ----------------------------------------------------------------

    def NEXT(self, how_many : int = 1) -> pd.DataFrame:

        if not self.tensor_data:
            raise RuntimeError('Data must be inloaded first: Predict(...).inload(input_data)')
        model_output : torch.Tensor = self.__generate__(how_many)
        if model_output.dim() == 1:
            model_output.unsqueeze(dim = 0)

        data_with_output : map[List[float]] = map(self.__decode__, model_output.tolist())
        final_df = pd.DataFrame(
            list(zip(*data_with_output)),
            columns=self.tensor_data.columns
        )
        return final_df 

    # ---------------------------------------------------------------

    def __generate__(self, how_many: int):

        "generates an output for the next `how_many` sequences. Output from one timestep is fed as input for next prediction"
        
        self.model.eval()
        context_size = self.cfg["context_length"]
        max_new_tokens = how_many
        topk = self.cfg["topk"]
        temperature = self.cfg["temperature"]
        idx = self.tensor_data.data
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
                
            with torch.no_grad():
                logits = self.model(idx_cond)
    
            logits = logits[:, -1, :].squeeze(dim=0)
            # print(logits.shape)
            if topk is not None:
                top_logits, _ = torch.topk(logits, topk)
                if top_logits.dim() == 1:
                    top_logits = top_logits.unsqueeze(dim=0)
                min_val = top_logits[:, -1]
                min_val = torch.reshape(min_val, (min_val.shape[0], 1))  # for comparison
                logits = torch.where(
                    logits < min_val[None, :], torch.tensor(float("-inf")), logits
                )
    
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1).squeeze(dim=0)
                # print(probs.shape)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
    
            idx = torch.cat([idx, idx_next], dim=1)
            
        return idx[:, -max_new_tokens:]

    # -----------------------------------------------------------------

    @property
    def __probs__(self,use_temp = True):
        "Get the probability logits from the multinomial distribution and then intuit long_short from its pdf"
        
        self.model.eval()
        _DATA = self.tensor_data.data
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

    # ----------------------------------------------------------------

    def get_predict_probs(self):
        return pd.DataFrame(
            self.__probs__.T.tolist(),
            columns=list(self.tensor_data.columns),
            index=self.tokenizer.vocab / 10_000,
        )

    # ----------------------------------------------------------------
