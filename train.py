from eval.mmlu import evaluate_mmlu, generator_mmlu
from eval.llmjudge import llmjudge_conversations
from eval.smoltalk import generate_smoltalk, smoltalk_prompt_generator
from model import get_lora_model
from utils import Logger, evaluate_with_baseline, clear_cache
from transformers import AutoTokenizer
from tqdm import tqdm
from itertools import tee
import json, random, torch, os
import torch.nn.functional as F
import code


# ---- Hyperparameters ----
os.environ['TOKENIZERS_PARALLELISM'] = "false"
MAX_SEQ_LENGTH = 2048

model_name = "Qwen/Qwen3-0.6B"
lora_r = 16
lora_alpha = 32
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
dataset_name = "ultrafeedback"
run_name = "test-run"
project_name = "shakespeare-lora"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
epochs = 10
mini_batch_size = 4
batch_size = 8
lr = 2e-4
max_train_iters = 10000
eval_interval = 200
generate_interval = 500
model_checkpoint_interval = 1000

# Evaluation parameters
mmlu_batch_size = 4
mmlu_examples = 80
smoltalk_batch_size = 4
enable_llmjudge = False
llmjudge_examples = 20
test_examples = 40
generate_examples = 4

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
    ids = tokenizer.apply_chat_template(messages)[:MAX_SEQ_LENGTH]
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

model = get_lora_model(model_name, lora=True, r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
# Constant learning rate for now
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

train_size, train_loader = dataset_loader(dataset_name, "train", mini_batch_size, trim=0.2, visualize=False)
test_size, test_loader = dataset_loader(dataset_name, "test", mini_batch_size, trim=0.2, visualize=False)

assert batch_size % mini_batch_size == 0, "batch_size must be divisible by mini_batch_size"
grad_accum_steps = batch_size // mini_batch_size
num_iters = train_size // batch_size * epochs
if num_iters > max_train_iters:
    num_iters = max_train_iters
    print(f"num_iters ({num_iters}) is greater than max_train_iters ({max_train_iters}), setting num_iters to {num_iters}")
last_step = num_iters - 1
test_iters = test_examples // mini_batch_size

# Adds color to the console output for generations
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

logger = Logger(project_name, run_name)
logger.log("config", {
    "model": model_name,
    "dataset": dataset_name,
    "epochs": epochs,
    "batch_size": batch_size,
    "num_iters": num_iters,
    "learning_rate": lr
})


# ---- Main Training Loop ----

debugging = True
if debugging:
    num_iters = 100
    eval_interval = 10
    generate_interval = 20
    smoltalk_batch_size = 2
    llmjudge_examples = 2


print(f"Training on {device} with batch size={batch_size} and mini_batch size={mini_batch_size}.\n\
    Number of iterations: {num_iters}. 1 epoch = {train_size} unique examples, {train_size // batch_size} iterations, .\n\
    Testing on {test_size} unique examples.")


for iter in tqdm(range(num_iters), desc="Training Loop"):

    last_step = (num_iters - 1 == iter)

    # Evaluate periodically
    if iter % eval_interval == 0 or last_step:
        
        print(f"#########################\nEvaluating at iteration {iter}...\n#########################")

        model.eval()
        model.config.use_cache = True
        # Test set Evaluation
        print(f"Running test set evaluation...")
        total_test_loss = 0
        for _ in tqdm(range(test_iters), desc="Test set"):
            
            try:
                inputs, targets = next(test_loader)
            except StopIteration:
                # Reset test loader after exhaustion
                _, test_loader = dataset_loader(dataset_name, "test", mini_batch_size, trim=0.2, visualize=False)
                inputs, targets = next(test_loader)
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16), torch.inference_mode():
                logits = model(inputs).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            total_test_loss += loss.item()
        total_test_loss /= test_iters
        
        # MMLU Evaluation
        print(f"Running MMLU evaluation...")
        mmlu_generator = generator_mmlu(split="test", max_examples=mmlu_examples)
        mmlu_score, mmlu_baseline_score = evaluate_with_baseline(model, tokenizer, mmlu_generator, evaluate_mmlu, batch_size=mmlu_batch_size, total_examples=mmlu_examples)
        
        # LLM-as-a-judge Evaluation
        if enable_llmjudge:
            print(f"Running LLM-as-a-judge evaluation...")
            smoltalk_generator = smoltalk_prompt_generator(split="test", num_examples=llmjudge_examples)
            smoltalk_convos, smoltalk_baseline_convos = evaluate_with_baseline(model, tokenizer, smoltalk_generator, generate_smoltalk, batch_size=smoltalk_batch_size, num_examples=llmjudge_examples)
            llmjudge_score = llmjudge_conversations(smoltalk_convos, logger=logger)
            llmjudge_baseline_score = llmjudge_conversations(smoltalk_baseline_convos, logger=None)

        # Log and print
        logger.log("eval", {
            "iter": iter,
            "test_loss": total_test_loss,
            "mmlu_score": mmlu_score,
            "mmlu_baseline_score": mmlu_baseline_score,
            "llmjudge_score": llmjudge_score if enable_llmjudge else None,
            "llmjudge_baseline_score": llmjudge_baseline_score if enable_llmjudge else None
        })
        print(f"Test Loss: {total_test_loss:.2f}")
        print(f"MMLU Score: {mmlu_score:.2f}")
        print(f"MMLU Baseline Score: {mmlu_baseline_score:.2f}")
        if enable_llmjudge:
            print(f"LLM-as-a-judge Overall Score: {llmjudge_score['overall']:.2f}")
            print(f"LLM-as-a-judge Overall Baseline Score: {llmjudge_baseline_score['overall']:.2f}")
        clear_cache(device)

    
    # Generate periodically using smoltalk
    if iter % generate_interval == 0 or last_step:
        print(f"#########################\nGenerating at iteration {iter}...\n#########################")

        smoltalk_generator = smoltalk_prompt_generator(split="test", num_examples=generate_examples)
        convos = generate_smoltalk(model, tokenizer, batch_size=smoltalk_batch_size, num_examples=generate_examples, generator=smoltalk_generator, max_new_tokens=256)
        for i, convo in enumerate(convos):
            print(f"{RED}Prompt {i+1}: {convo[0]}{RESET}\n\n{GREEN}Response: {convo[1]}{RESET}")
            print("-" * 100)
        clear_cache(device)
    
    # Gradient Accumulation
    model.train()
    model.config.use_cache = False

    tokens_trained = 0
    total_loss = 0
    optimizer.zero_grad()
    for i in range(grad_accum_steps):
        
        try:
            inputs, targets = next(train_loader)
        except StopIteration:
            # Reset train loader after exhaustion
            _, train_loader = dataset_loader(dataset_name, "train", mini_batch_size, trim=0.2, visualize=False)
            inputs, targets = next(train_loader)

        tokens_trained += (targets != -1).sum().item()
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(inputs).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        loss = loss / grad_accum_steps
        total_loss += loss.item()
        loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)

    # Optimizer Step
    optimizer.step()

    # Log
    logger.log("step", {
        "iter": iter,
        "train_loss": total_loss,
        "tokens_trained": tokens_trained,
    })

    # Save model periodically
    if iter % model_checkpoint_interval == 0 or last_step:
        print(f"#########################\nSaving model at iteration {iter}...\n#########################")
        os.makedirs(f"checkpoints", exist_ok=True)
        model.save_pretrained(f"checkpoints/{run_name}_{iter}")
    
    clear_cache(device)