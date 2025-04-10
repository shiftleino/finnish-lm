import torch
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from tokenizer import CharTokenizer
from kaleGPT import KaleGPT
from kaleGPT2 import KaleGPT2
from kaleGPT3 import KaleGPT3

from torch.profiler import profile, ProfilerActivity


def read_data(file_path: str) -> str:
    with open(file_path, "r", encoding="latin-1") as f:
        text = f.read()
    text = text.split("Ensimmäinen runo")[1] # Remove the description and acknowledgements
    return text

def get_batch(tokens: torch.Tensor, batch_size: int, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_idxs = torch.randint(0, len(tokens) - block_size - 1, (batch_size,))
    context = torch.stack([tokens[idx:idx + block_size] for idx in batch_idxs])
    targets = torch.stack([tokens[idx + 1:idx + block_size + 1] for idx in batch_idxs])
    return context, targets

@torch.no_grad()
def eval_loss(model: torch.nn.Module, val_data: torch.Tensor, batch_size: int, block_size: int, eval_iterations: int) -> Dict[str, float]:
    model.eval()
    losses = {"train": torch.zeros(eval_iterations), "val": torch.zeros(eval_iterations)}
    for split in ("train", "val"):
        for i in range(eval_iterations):
            if split == "train":
                contexts, targets = get_batch(train_data, batch_size, block_size)
            else:
                contexts, targets = get_batch(val_data, batch_size, block_size)
            logits = model(contexts)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            losses[split][i] = loss.item()
    model.train()
    return losses

def train(model: torch.nn.Module, train_data: torch.Tensor, batch_size: int, block_size: int, max_block_size: int, lr: float, num_steps: int, eval_interval: int, eval_iterations: int, patience: int, checkpointing: bool, model_name: str) -> Tuple[List, List]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    patience_counter = 0

    with profile(activities=[
        ProfilerActivity.CPU, 
        torch.profiler.ProfilerActivity.CUDA], 
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model_name}/profile'),
        with_stack=True) as prof:
        for step in range(num_steps):
            start_time = time.time()
            
            context, targets = get_batch(train_data, batch_size, block_size)
            logits = model(context)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            optimizer.step()
            prof.step()

            if step % 100 == 0:
                scheduler.step()
            
            if step % eval_interval == 0:
                checkpoint_time = time.time()

                eval_losses = eval_loss(model, val_data, batch_size, max_block_size, eval_iterations)
                train_loss = eval_losses["train"].mean().item()
                val_loss = eval_losses["val"].mean().item()
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                print(f"Step: {step:7} || Train loss: {train_loss:8.5f} || Val loss: {val_loss:8.5f} || Learning rate: {scheduler.get_last_lr()[0]:10.8f} || Tokens per second: {(batch_size*block_size*eval_interval) / (checkpoint_time - start_time):5.3}")
                if checkpointing:
                    torch.save(model.state_dict(), f"{model_name}-checkpoint-{step}.pth")

                # Implement early stopping
                if val_loss >= best_val_loss:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at step {step}")
                        break
                else:
                    best_val_loss = val_loss
                    patience_counter = 0

    return train_losses, val_losses

def plot_training_curve(losses, eval_interval, title):
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.xticks(ticks=[i for i in range(len(losses))], labels=[i*eval_interval for i in range(len(losses))], rotation=90)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    file_path = "kalevala.txt"
    device = "cuda:0"
    batch_size = 32
    block_size = 512 # Which block size to train
    max_block_size = 512 # Which block size to allow for inference
    model_dim = 768
    num_layers = 8
    num_heads = 12
    act = "swiglu" # relu, lrelu, gelu, swiglu
    norm = "rms" # layer, rms
    position = "rope" # embed, alibi, rope
    lr = 1e-4
    num_steps = 10000
    eval_interval = 50
    eval_iterations = 10
    patience = 10
    checkpointing = False
    use_recomputation = False
    num_tokens_generate = 300
    model_name = f"kalegpt-dropout-{position}-{norm}norm-{act}-{num_layers}-{model_dim}-{num_heads}-{block_size}"

    text = read_data(file_path)
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size

    tokens = tokenizer.encode(text).to(device)
    n = int(len(tokens) * 0.9)
    train_data = tokens[:n]
    val_data = tokens[n:]
    
    if position == "embed":
        model = KaleGPT(vocab_size, model_dim=model_dim, num_heads=num_heads, num_layers=num_layers, block_size=block_size, act=act, norm=norm, device=device).to(device)
    elif position == "alibi":
        model = KaleGPT2(vocab_size, model_dim=model_dim, num_heads=num_heads, num_layers=num_layers, max_block_size=max_block_size, act=act, norm=norm, device=device).to(device)
    elif position == "rope":
        model = KaleGPT3(vocab_size, model_dim=model_dim, num_heads=num_heads, num_layers=num_layers, max_block_size=max_block_size, act=act, norm=norm, device=device, use_recomputation=use_recomputation).to(device)

    print(model)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    train_losses, val_losses = train(model, train_data=train_data, batch_size=batch_size, block_size=block_size, max_block_size=max_block_size, lr=lr, num_steps=num_steps, eval_interval=eval_interval, eval_iterations=eval_iterations, patience=patience, checkpointing=checkpointing, model_name=model_name)

    torch.save(model.state_dict(), f"models/{model_name}.pth")

    plot_training_curve(train_losses, eval_interval, title="Train loss")
    plot_training_curve(val_losses, eval_interval, title="Validation loss")

    print(f"Generating sample of {num_tokens_generate} tokens...")
    generation = model.generate(torch.tensor([[0]]).to(device), num_tokens_generate)[0]
    generated_text = tokenizer.decode(generation)
    print(generated_text)

    with open(f"generations/{model_name}.txt", "w") as f:
        f.write(generated_text)
