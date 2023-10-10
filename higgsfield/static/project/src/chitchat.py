import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from higgsfield.loaders import LlamaLoader 
from higgsfield.llama import Llama70b, Llama7b, Llama
from higgsfield.checkpoint import Checkpoint
from higgsfield.training import  clip_grad_norm

import torch.distributed as dist
from higgsfield.experiment import experiment, param

def chat_to_prompt(chat):
    joined = []
    for message in chat:
        if message['role'] == "assistant":
            joined.append(f"<BOT>: {message['content']}")
        elif message["role"] == "user":
            joined.append(f"<USER>: {message['content']}")
        else:
            joined.append(f"<SYSTEM>: {message['content']}")
            
    prompt = "\n".join(joined)
    prompt += "\n<BOT>: "
    return prompt

import polars as pl
class ChitChatDataset():
    def __init__(self):
        df_path = "src/odd_dataset_v_1.parquet"
        df = pl.read_parquet(df_path).drop_nulls()
        
        self.dialogs = []
        
        for i in range(df.shape[0]):
            messages = [{
                "role": "system",
                "content": df['system_prompt'][i].strip()
            }]
            
            x = df['prompt'][i]
            split = x.split("<BOT>")
            if "<USER>" in split[0]:
                try:
                    messages.append({
                        "role": "user",
                        "content": split[0].split("<USER>: ")[1].strip()
                    })
                except:
                    print(split[0])
                
            for m in x.split("<BOT>: "):
                if m:
                    if "<USER>: " not in m:
                        messages.append({
                            "role": "assistant",
                            "content": m.strip(),
                        })
                    else:
                        try:
                            b, u = m.split("<USER>: ")
                        except:
                            print(m.split("<USER>: "))
                            continue

                        messages.append({
                            "role": "assistant",
                            "content": b.strip(),
                        })
                        messages.append({
                            "role": "user",
                            "content": u.strip()
                        })
            self.dialogs.append(messages)
            
    def __len__(self):
        return len(self.dialogs)
        
    def __getitem__(self, idx):
        return self.dialogs[idx]

@experiment("chitchat")
@param("size", options=["7b", "13b", "70b"])
def train(params):
    #model = Llama70b(
    #    zero_stage=3,
    #    cpu_init_rank0=True,
    #    fast_attn=False,
    #    precision="bf16",
    #    #precision="fp16",
    #    cpu_offload=False,
    #)
    
    #from pathlib import Path
    #model_path = Path.home() / ".cache/train_llama2/experiments/fanfic/checkpoints/sleepy_pasteur/epoch_0_steps_3563") / "model.pt"
    
    if params.size == "7b":
        model_name = "meta-llama/Llama-2-7b-hf"
    elif params.size == "13b":
        model_name = "meta-llama/Llama-2-13b-hf"
    elif params.size == "70b":
        model_name = "meta-llama/Llama-2-70b-hf"
    
    model = Llama(
        model_name=model_name,
        zero_stage=3,
        cpu_init_rank0=True,
        fast_attn=False,
        precision="bf16",
        #precision="fp16",
        cpu_offload=False,
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.0,
    )
    
    lr_scheduler = StepLR(
        optimizer,
        step_size=1,
        gamma=0.85,
    )
    
    # ~/.cache/{project-name}/{experiment_name}/{run_name}/
    checkpoint = Checkpoint(
        model,
        optimizer,
        lr_scheduler,
    )
    
    chitchat = ChitChatDataset()
    from higgsfield.dataset.openai import ChatCompletionDataset
    dataset = ChatCompletionDataset(chitchat, chat_to_prompt=chat_to_prompt) 
    
    train_loader = LlamaLoader(
        dataset,
        max_sequence_length=2048,
        batch_size=64*6,
    )
    
    #import wandb
    #from pathlib import Path
    #wandb.init(
    #    dir=Path.home() / ".cache/higgsfield/wandb-dir",
    #    project="Testing Keras",
    #    id="Shit",
    #    config={
    #        "epochs": 3, "batch_size_per_gpu": 6,
    #    }
    #)
    
    for epoch in range(1):
        for i, batch in enumerate(train_loader):
            
            optimizer.zero_grad()
            loss = model(batch)
            
            if dist.get_rank() == 0:
                print("LOSS: ", loss)
                
            loss.backward()
            
            clip_grad_norm(1.0, model, optimizer)
            optimizer.step()

            if dist.get_rank() == 0:
                print(loss, i, len(train_loader))
                
            if i % 30 == 0 or i == len(train_loader) - 1:
                checkpoint.save(epoch, i)
            
        lr_scheduler.step()
