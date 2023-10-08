import os
import shutil
from pathlib import Path

import time
import json
import torch
import torch.distributed as dist

import torch.distributed._shard.checkpoint as dist_cp


from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)

from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    StateDictType,
)

from optimum.bettertransformer import BetterTransformer

fullstate_save_policy = FullStateDictConfig(
    offload_to_cpu=True,
    rank0_only=True,
)

DEFAULT_CHECKPOINT_PATH = Path.home()  / \
                          ".cache" / \
                          "higgsfield" / \
                          os.environ["PROJECT_NAME"] / \
                          "experiments" / \
                          os.environ["EXPERIMENT_NAME"] / \
                          os.environ["RUN_NAME"]

class Checkpoint:
    '''
        Saving checkpoint to:
            ~/.cache/higgsfield/{project_name}/experiments/{experiment_name}/{run_name}
    '''
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler=None,
        scaler=None,
    ):
        '''
            model: Higgsfield.model
        '''
        if os.environ["PROJECT_NAME"] and os.environ["EXPERIMENT_NAME"] and os.environ["RUN_NAME"]:
            save_dir = DEFAULT_CHECKPOINT_PATH 
        else:
            raise NotImplementedError("Support single GPU/process not implemeted yet")
            

        self.save_dir     = save_dir
        self.model        = model.model
        self.optimizer    = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler       = scaler
    
    def save(self, epoch, steps=0):
        
        save_path = Path(self.save_dir) / f"epoch_{epoch}_steps_{steps}"
        save_path.mkdir(exist_ok=True, parents=True)
        
        model_path     = save_path / "model.pt"
        optimizer_path = save_path / "optimizer.pt"
        
        t0 = time.perf_counter()
        save_distributed_model_rank0(model_path, self.model)
        save_distributed_optimizer_rank0(optimizer_path, self.model, self.optimizer)
        t1 = time.perf_counter()

        if int(os.environ["LOCAL_RANK"]) == 0:
            print(f"State checkpoint of {steps} steps saved to {save_path}")
            print(f"Checkpoint Time = {t1-t0:.4f}\n")
            
            if self.lr_scheduler:
                lr_scheduler_path = save_path / "lr_scheduler.pt"
                torch.save(self.lr_scheduler.state_dict(), lr_scheduler_path)
            
            if self.scaler:
                scaler_path = save_path / "scaler.pt"
                torch.save(self.grad_scaler.state_dict(), scaler_path)
            
            metadata_path = save_path / "metadata.json"
            metadata = {
                "epoch": epoch,
                "steps": steps,
            }
            with open(metadata_path, "w+") as jsonFile:
                json.dump(metadata, jsonFile)
        

def save_distributed_model_rank0(checkpoint_path, model):
    '''
        model: FSDP
    '''
    rank = dist.get_rank()
    
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()
        
    if rank == 0:
        torch.save(cpu_state, checkpoint_path)
    
def save_distributed_optimizer_rank0(checkpoint_path, model, optimizer):
    '''
        model: FSDP
        optimizer: torch.optim
    '''
    rank = dist.get_rank()
    
    optim_state = FSDP.full_optim_state_dict(model, optimizer)

    if rank == 0:
        torch.save(optim_state, checkpoint_path)
        
