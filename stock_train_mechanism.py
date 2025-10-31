import torch
from ohlcv_gpt_model import OHLCV_GPTModel
from stock_loss_prediction import (
    calc_loss_batch,
    calc_loss_loader,
)
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# ==============================================


# Per the book --->

# def train_model_simple(
#     model : GPTModel,
#     train_loader : torch.utils.data.DataLoader,
#     val_loader : torch.utils.data.DataLoader,
#     optimizer,
#     num_epochs : int ,
#     eval_freq : int,
#     eval_iter : int,
#     start_context : torch.Tensor,
#     tokenizer : StockVocab
# ):

# tokens_seen

# ==============================================


class StockTrainer:
    """
    Takes a GPT model and train it given train and val datasets.
    Optimizer is AdamW
    Load and Save only requires one <model_name>.pth, everything else will be taken care ofi; if True, last checkpoint will be loaded;
    if False, new random model weights used.
    """

    def __init__(
        self,
        model: OHLCV_GPTModel,
        cfg: dict,
        load_from_checkpoint: str | bool = False,
        device="cpu",
        model_already_loaded: bool = False,
    ):
        self.model = model
        self.device = device
        if (
            not model_already_loaded
        ):  # if model is already loaded into GPU using data_parallel()
            self.model = self.model.to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["adamw_learning_rate"],
            weight_decay=cfg["adamw_weight_decay"],
        )

        self.tokens_seen = 0
        self.steps_trained = 0
        self.epochs = 0
        self.save_checkpoint_step = cfg["save_checkpoint_step"]
        self.save_path = cfg["save_path"]

        self.evaluated_train_loss = []
        self.evaluated_val_loss = []

        if isinstance(load_from_checkpoint, bool) and load_from_checkpoint:
            self.load_state("gpt_training_checkpoint.pth")

        elif isinstance(load_from_checkpoint, str):
            self.load_state(load_from_checkpoint)

    # -----------------------------------------------------

    def engage_training(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs: int = 1,
        val_loader=None,
        eval_freq=None,
        eval_iter=None,
    ):
        for epoch in range(num_epochs):
            # train setting
            self.model.train()

            for i, (input_batch, target_batch) in enumerate(train_loader):
                # send input batch and target batch to device
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                self.optimizer.zero_grad()
                loss = calc_loss_batch(
                    input_batch=input_batch, target_batch=target_batch, model=self.model
                )

                loss.backward()  # calculate the gradients backpropagation
                self.optimizer.step()
                self.tokens_seen += input_batch.numel()
                self.steps_trained += 1

                # evaluate model
                if (
                    eval_freq is not None
                    and eval_iter is not None
                    and val_loader is not None
                    and self.steps_trained % eval_freq == 0
                ):
                    print(f"\nEpoch {epoch + 1} Step {self.steps_trained }>>> ")
                    self.evaluate_model(train_loader, val_loader, eval_iter)

                # save checkpoint
                if self.steps_trained % self.save_checkpoint_step == 0:
                    print("\n")
                    print("generating checkpoint...")
                    self.save_state("gpt_training_checkpoint.pth")

        # finally
        self.epochs += num_epochs

    # --------------------------------------------

    def evaluate_model(self, train_loader, val_loader, eval_iter):
        self.model.eval()  # pause model training
        with torch.no_grad():
            train_loss = calc_loss_loader(
                data_loader=train_loader,
                model=self.model,
                num_batches=eval_iter,
                device=self.device,
            )
            val_loss = calc_loss_loader(
                data_loader=val_loader,
                model=self.model,
                num_batches=eval_iter,
                device=self.device,
            )

        self.evaluated_train_loss.append(train_loss)
        self.evaluated_val_loss.append(val_loss)

        print(f"Train Loss : {train_loss:.3f}")
        print(f"Val Loss : {val_loss:.3f}")

        self.model.train()

    # --------------------------------------------

    def __str__(self):
        return f"""
        ============================================================
            Training Statistics for this exercise >>> 

            Device : {self.device}
            Total Epochs trained : {self.epochs}
            Total Steps trained : {self.steps_trained}
            Total number of tokens seen so far : {self.tokens_seen}
            Training Loss : {self.evaluated_train_loss}
            Validation Loss : {self.evaluated_val_loss}
        ============================================================
        
        """

    # --------------------------------------------

    def plot_losses(self):
        _epochs = np.linspace(
            start=0, stop=self.epochs, num=len(self.evaluated_train_loss)
        )

        _tokens = np.linspace(
            start=0, stop=self.tokens_seen, num=len(self.evaluated_train_loss)
        )

        fig, ax1 = plt.subplots(figsize=(10, 7))
        ax1.plot(self.evaluated_train_loss, label="Training Loss")
        ax1.plot(
            self.evaluated_val_loss,
            linestyle="-.",
            label="Validation Loss",
        )
        ax1.set_xlabel("Epochs as seen as number of checkpoints")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax2 = ax1.twiny()
        ax2.plot(_tokens, self.evaluated_train_loss, alpha=0)
        ax2.set_label("Tokens seen")

        fig.tight_layout()
        plt.show()

    # --------------------------------------------

    def save_trained_model_optimizer(self, save_name):
        "saves model weights and optimizer weights separately..."

        optimizer_file = os.path.join(
            self.save_path, save_name.split(".")[0] + "_optimizer.pth"
        )
        if isinstance(self.model, torch.nn.DataParallel):  # Data Parallel enabling
            self.model.module.save_weights(save_name)
        else:
            self.model.save_weights(save_name)
        torch.save(self.optimizer.state_dict(), optimizer_file)
        print(f"Saved model optimizer as {optimizer_file}")

    # --------------------------------------------

    def load_trained_model_optimizer(self, pth_file, device=None):
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_weights_into_gpt(pth_file, device)
        else:
            self.model.load_weights_into_gpt(pth_file, device)
        self.optimizer.load_state_dict(
            torch.load(pth_file.split(".")[0] + "_optimizer.pth")
        )
        print("Loaded optimizer into model successfully")

    # --------------------------------------------

    def save_state(self, save_name: str):
        "saves the model metdata such as epochs, steps and evalauted losses and all other parameters"

        meta_name = os.path.join(
            self.save_path, save_name.split(".")[0] + "_metadata.json"
        )

        _data = {
            "time": dt.now().strftime("%a %d %b %Y, %I:%M%p"),
            "epochs": self.epochs,
            "steps_trained": self.steps_trained,
            "train_loss": self.evaluated_train_loss,
            "val_loss": self.evaluated_val_loss,
            "tokens_seen": self.tokens_seen,
            "model_parameters_file": save_name,
            "optimizer_parameters_file": save_name.split(".")[0] + "_optimizer.pth",
        }

        # causing a lot of problems. So removing this
        # "device": (
        #     self.device.type
        #     if isinstance(self.device, torch.device)
        #     else self.device
        # ),

        with open(meta_name, "w") as json_file:
            json.dump(_data, json_file)

        # save model and optimizer
        self.save_trained_model_optimizer(save_name)

        print("Saved State successfully")

    # --------------------------------------------

    def load_state(self, pth_file: str):
        json_file = pth_file.split(".")[0] + "_metadata.json"
        with open(json_file, "rb") as fp:
            _data = json.load(fp)

        self.epochs = _data["epochs"]
        self.tokens_seen = _data["tokens_seen"]
        self.steps_trained = _data["steps_trained"]
        self.evaluated_train_loss = _data["train_loss"]
        self.evaluated_val_loss = _data["val_loss"]
        self.tokens_seen = _data["tokens_seen"]

        # load model and optimizer
        self.load_trained_model_optimizer(
            _data["model_parameters_file"], device=self.device
        )

        print("Loaded State Completely")
