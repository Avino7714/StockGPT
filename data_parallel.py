import torch
from torch import nn
import torch.nn.parallel as prl
from gpt_model import GPTModel
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Tuple

# ===========================================


def data_parallel(model: GPTModel) -> Tuple[GPTModel, torch.device]:

    nprocs = torch.cuda.device_count()

    if nprocs > 1:
        print(f"{nprocs} GPUs present. Using all GPUs")
        model = nn.DataParallel(model)
        device = torch.device("cuda:0")  # use the first GPU as the master
        model = model.to(device)

    elif nprocs == 1:
        print("Using 1 GPU")
        device = torch.device("cuda")
        model = model.to(device)

    else:
        print("Using CPU")
        device = torch.device("cpu")
        model = model.to(device)

    return model, device


# ============================================


def dist_parallel(model: GPTModel, rank: int, world_size: int):  # no. of GPUs
    "Called with this : mp.spawn(train, args=(world_size,), nprocs=world_size)"

    # Initialize the distributed process group
    dist.init_process_group(
        "nccl", init_method="env://", world_size=world_size, rank=rank
    )
    model = prl.DistributedDataParallel(model, device_ids=[rank])  # Use DDP
    return model


# ============================================

if __name__ == "__main__":
    world_size = 2
    # mp.spawn(dist_parallel, args=(world_size,), nprocs=world_size)

    # NEEDS TESTING
