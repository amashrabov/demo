import os
import functools
from pathlib import Path

import torch
import torch.distributed as dist

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
)

from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from optimum.bettertransformer import BetterTransformer

from higgsfield.checkpoint.fsdp_checkpoint import (
    save_distributed_model_rank0,
    save_distributed_optimizer_rank0,
)

class Llama:
    def __init__(
        self,
        model_name,
        model_checkpoint_path=None,
        zero_stage=3,
        fast_attn=False,
        precision="bf16",
        cpu_init_rank0=False,
        cpu_offload=False,
    ):
        rank = dist.get_rank()
        
        if not model_checkpoint_path:
            if cpu_init_rank0: 
                if rank == 0:
                    model = LlamaForCausalLM.from_pretrained(model_name)
                else:
                    llama_config = LlamaConfig.from_pretrained(model_name)
                        
                    with torch.device('meta'):
                        model = LlamaForCausalLM(llama_config)
            else:
                model = LlamaForCausalLM.from_pretrained(model_name)
        else:
            if not cpu_init_rank0:
                raise Exception("You can only load to cpu if checkpoint loading")
            
            if rank == 0:
                model = LlamaForCausalLM.from_pretrained(model_name)
                state_dict = torch.load(model_checkpoint_path)
                model.load_state_dict(state_dict)
                print("LOADED FROM CHECKPOINT")
                
            else:
                llama_config = LlamaConfig.from_pretrained(model_name)
                
                with torch.device('meta'):
                    model = LlamaForCausalLM(llama_config)
            
        if fast_attn:
            #raise NotImplementedError("Fast attention is not supported yet")
            model = BetterTransformer.transform(model)
        
        fpSixteen = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

        bfSixteen_mixed = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        
        pure_bf16 = False
        if precision == "fp16":
            mixed_precision_policy = fpSixteen
            
        elif precision == "bf16":
            mixed_precision_policy = None
            pure_bf16 = True
            
        elif precision == "bf16_mixed":
            mixed_precision_policy = bfSixteen_mixed

        else:
            mixed_precision_policy = None
            
        if pure_bf16:
            model.to(torch.bfloat16) 

        wrapping_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer,
            }
        )
        
        if zero_stage == 0:
            sharding_strategy = ShardingStrategy.NO_SHARD
        
        elif zero_stage == 1:
            raise NotImplementedError("stage 1 is not supported. Only 0 2 3")
            
        elif zero_stage == 2:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
            
        elif zero_stage == 3:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            raise NotImplementedError("stage can be only 0 2 3")

        if cpu_init_rank0 and rank != 0:
            param_init_fn = lambda module: module.to_empty(
                device=torch.device('cuda'),
                recurse=False,
            )
        else:
            param_init_fn = None
            
        if cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        model = FSDP(
            model,
            auto_wrap_policy=wrapping_policy,
            cpu_offload=cpu_offload,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=cpu_init_rank0,
            param_init_fn=param_init_fn,
        )
        
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=check_fn,
        )
        
        fsdp = True
        
        self.model = model
        self.precision = precision
        self.fsdp = fsdp
    
    def step(self, batch):
        local_rank = int(os.environ["LOCAL_RANK"])
        
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
            
        if self.precision == "fp16":
            with torch.cuda.amp.autocast(): 
                loss = self.model(**batch).loss
        else:
            loss = self.model(**batch).loss
            
        return loss
    
    def parameters(self):
        return self.model.parameters()
    
    def save_model(self, save_path):
        '''
            Save model's weight to master node 
                ~/.cache/higgsfield/{project_name}/{save_path}
        '''
        if "/" == save_path[0]:
            save_path = save_path[1:]
            
        head, tail = os.path.split(save_path)
        
        path = Path.home() / ".cache/higgsfield" / os.environ["PROJECT_NAME"] / head
        path.mkdir(exist_ok=True, parents=True)
    
        save_distributed_model_rank0(path / tail, self.model)
        
    def save_optimizer(self, optimizer, save_path):
        '''
            Save optimizer's state to master node
                ~/.cache/higgsfield/{project_name}/{save_path}
        '''
        
        if "/" == save_path[0]:
            save_path = save_path[1:]
            
        head, tail = os.path.split(save_path)
        
        path = Path.home() / ".cache/higgsfield" / os.environ["PROJECT_NAME"] / head
        path.mkdir(exist_ok=True, parents=True)
        
        
        save_distributed_optimizer_rank0(path / tail, self.model, optimizer)

        
    
class Llama7b(Llama):
    def __init__(
        self,
        model_checkpoint_path=None,
        zero_stage=3,
        fast_attn=False,
        precision="bf16",
        cpu_init_rank0=False,
        cpu_offload=False,
    ):
        model_name = "meta-llama/Llama-2-7b-hf"
        super(Llama7b, self).__init__(
            model_name,
            model_checkpoint_path,
            zero_stage,
            fast_attn,
            precision,
            cpu_init_rank0,
            cpu_offload,
        )
       
class Llama13b(Llama):
    def __init__(
        self,
        model_checkpoint_path=None,
        zero_stage=3,
        fast_attn=False,
        precision="bf16",
        cpu_init_rank0=False,
        cpu_offload=False,
    ):
        model_name = "meta-llama/Llama-2-13b-hf"
        super(Llama13b, self).__init__(
            model_name,
            model_checkpoint_path,
            zero_stage,
            fast_attn,
            precision,
            cpu_init_rank0,
            cpu_offload,
        )
        
class Llama70b(Llama):
    def __init__(
        self,
        model_checkpoint_path=None,
        zero_stage=3,
        fast_attn=False,
        precision="bf16",
        cpu_init_rank0=False,
        cpu_offload=False,
    ):
        model_name = "meta-llama/Llama-2-70b-hf"
        super(Llama70b, self).__init__(
            model_name,
            model_checkpoint_path,
            zero_stage,
            fast_attn,
            precision,
            cpu_init_rank0,
            cpu_offload,
        )

def clip_grad_norm(max_grad_norm, model, optimizer, scaler=None):
    model = model.model
    
    if scaler:
        scaler.unscale_(optimizer)
        
    if hasattr(optimizer, 'clip_grad_norm'):
        optimizer.clip_grad_norm(max_grad_norm)
        
    elif hasattr(model.model, 'clip_grad_norm_'):
        model.clip_grad_norm_(max_grad_norm)       