from model import get_model
from utils import Logger
from transformers import AutoTokenizer
from tqdm import tqdm
from itertools import tee
import json, random, torch
import torch.nn.functional as F
import code


# ---- Hyperparameters ----

model_name = "Qwen/Qwen3-0.6B"
dataset_name = "ultrafeedback"
epochs = 10
max_train_iters = 10000
mini_batch_size = 4
batch_size = 8
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
lr = 2e-4

# ---- Tokenizer ----

tokenizer = AutoTokenizer.from_pretrained(model_name)
think_end_token_id = tokenizer.encode("</think>")[0]
im_start_token_id = tokenizer.encode("<|im_start|>")[0]
pad_token_id = tokenizer.pad_token_id

# ---- Dataloader ----

def tokenize_example(example):
    """Tokenize a single one-turn conversation from jsonl, returns ids and loss mask."""

    messages = example['messages']
    assert len(messages) == 2, "One-turn conversation expected"
    ids = tokenizer.apply_chat_template(messages)
    assert ids[0] == im_start_token_id, "im_start token expected at the beginning of the conversation"
    assert im_start_token_id in ids[1:], "im_start token expected for assistant"
    assistant_start_loc = ids[1:].index(im_start_token_id) + 1
    assert tokenizer.decode(ids[assistant_start_loc + 2]) == "\n", "line break expected after assistant"
    mask = [0] * (assistant_start_loc + 3) + [1] * (len(ids) - assistant_start_loc - 3)
    return torch.tensor(ids), torch.tensor(mask)

def visualize_tokens(ids, mask):
    """Helper function to visualize tokens and loss mask."""

    GREEN = "\033[92m"
    RESET = "\033[0m"
    RED = "\033[91m"
    out = ""
    for i in range(len(ids)):
        if mask[i] == 0:
            out += RED + f"{tokenizer.decode(ids[i])}" + RESET
        else:
            out += GREEN + f"{tokenizer.decode(ids[i])}" + RESET
    print(out)

def dataset_loader(dataset_name, split, batch_size, trim=0.2, visualize=False):
    """Create a dataset loader for the given dataset and split."""
    
    # Trim dataset based on length to increase training efficiency
    trim = trim / 2
    with open(f"dataset/shakespeare_{split}_{dataset_name}.jsonl", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lens = torch.tensor([len(line) for line in lines], dtype=torch.float)
    old_len = len(lines)
    floor, ceil = torch.quantile(lens, torch.tensor([trim, 1 - trim])).to(int)
    lines = [line for line in lines if len(line) > floor and len(line) < ceil]
    print(f"Trimmed {dataset_name} dataset from {old_len} to {len(lines)}")
    
    def batch_generator():
        random.Random(42).shuffle(lines)
        ids_batch, mask_batch = [], []
        for i, line in enumerate(lines):
            example = json.loads(line)
            ids, mask = tokenize_example(example)
            if visualize:
                visualize_tokens(ids, mask)
            ids_batch.append(ids)
            mask_batch.append(mask)
            if i % batch_size == 0 and i > 0:
                inputs = torch.full((batch_size, max(len(ids) for ids in ids_batch) - 1,), pad_token_id, dtype=torch.long)
                targets = torch.full((batch_size, max(len(ids) for ids in ids_batch) - 1,), -1, dtype=torch.long)
                for j in range(batch_size):
                    inputs[j, :len(ids_batch[j]) - 1] = ids_batch[j][:-1]
                    targets[j, :len(ids_batch[j]) - 1] = ids_batch[j][1:]
                    targets[j, :len(ids_batch[j]) - 1] *= mask_batch[j][1:]
                ids_batch, mask_batch = [], []
                yield inputs, targets
    
    return len(lines), batch_generator()

# ---- Setting up Training ----

model = get_model(model_name, lora=True, r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
# Constant learning rate for now
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

ds_size, train_loader = dataset_loader(dataset_name, "train", mini_batch_size, trim=0.2, visualize=False)
assert batch_size % mini_batch_size == 0, "batch_size must be divisible by mini_batch_size"
grad_accum_steps = batch_size // mini_batch_size
num_iters = ds_size // batch_size * epochs
if num_iters > max_train_iters:
    num_iters = max_train_iters
    print(f"num_iters is greater than max_train_iters, setting num_iters to {num_iters}")

logger = Logger("shakespeare-lora", run_name="test-run")
logger.log_config({
    "model": model_name,
    "dataset": dataset_name,
    "epochs": epochs,
    "batch_size": batch_size,
    "num_iters": num_iters,
    "learning_rate": lr
})


# ---- Main Training Loop ----

num_iters = 100

model.config.use_cache = False

for iter in tqdm(range(num_iters)):


    # Gradient Accumulation
    tokens_trained = 0
    total_loss = 0
    for i in range(grad_accum_steps):
        inputs, targets = next(train_loader)
        tokens_trained += (targets != -1).sum().item()
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits = model(inputs).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        loss = loss / grad_accum_steps
        total_loss += loss.item()
        loss.backward()

    # Optimizer Step
    optimizer.step()
    if device == "mps":
        torch.mps.empty_cache()

    # Log
    logger.log_step({
        "iter": iter,
        "loss": total_loss,
        "tokens_trained": tokens_trained,
    })
    
    
