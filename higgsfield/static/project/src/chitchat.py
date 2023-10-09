import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from higgsfield.loaders import LlamaLoader 
from higgsfield.llama import Llama70b, Llama7b, clip_grad_norm
from higgsfield.checkpoint import Checkpoint
from higgsfield.mixed_precision import Scaler

import torch.distributed as dist
from higgsfield.experiment import experiment, param

@experiment("chitchat")
@param("size", options=["70b", "13b", "7b"])
def train(params):
        lr_scheduler.step()
