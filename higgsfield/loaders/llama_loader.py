import torch.distributed as dist

from torch.utils.data import (
    DistributedSampler, 
    DataLoader
)

from transformers import (
    LlamaTokenizer, 
    default_data_collator
)

from higgsfield.dataset import TorchCompletionDataset

class HiggsfieldSampler(DistributedSampler):
    def __init__(
        self, 
        dataset, 
        shuffle=True,
        seed=0, 
        drop_last=False
    ):
        rank=dist.get_rank()
        num_replicas=dist.get_world_size()
        
        super(HiggsfieldSampler, self).__init__(
            dataset=dataset, 
            num_replicas=num_replicas,
            rank=rank, 
            shuffle=shuffle,
            seed=seed, 
            drop_last=drop_last,
        )

class LlamaLoader(DataLoader):
    def __init__(
        self,
        dataset, 
        max_sequence_length=2048,
        batch_size=dist.get_world_size(),
        shuffle=True, 
        seed=0,
        num_workers=0, 
        pin_memory=False, 
        drop_last=False,
        timeout=0, 
        worker_init_fn=None,
        multiprocessing_context=None, 
        *, 
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory_device=""
    ):
        assert batch_size >= dist.get_world_size() and \
               batch_size % dist.get_world_size() == 0, \
        "Batch size must be multiplied by number of GPUs"
        
        batch_size_per_gpu = batch_size // dist.get_world_size()
        
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
        dataset = TorchCompletionDataset(
            dataset,
            tokenizer,
            max_sequence_length,
        )
        
        sampler = HiggsfieldSampler(dataset, shuffle=shuffle, seed=seed,)
        
        super(LlamaLoader, self).__init__(
            dataset, 
            batch_size=batch_size_per_gpu,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory, 
            drop_last=drop_last,
            timeout=timeout, 
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context, 
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device 
        )
        
def get_llama_loader(
    dataset,
    batch_size_per_gpu,
    max_sequence_length,
    num_workers=1,
    drop_last=True
):
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    dataset = TorchCompletionDataset(
        dataset,
        tokenizer,
        max_sequence_length,
    )
    
    sampler = DistributedSampler(
        dataset,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        shuffle=True,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=drop_last,
        collate_fn=default_data_collator,
    )
    
    return loader