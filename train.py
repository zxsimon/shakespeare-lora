from eval.mmlu import evaluate_mmlu
from eval.llmjudge import llmjudge_conversations
from eval.smoltalk import generate_smoltalk, smoltalk_prompt_generator
from model import get_lora_model
from utils import Logger, clear_cache
from transformers import AutoTokenizer
from tqdm import tqdm
import json, random, torch, os, argparse, time, code
import torch.nn.functional as F

# ---- Hyperparameters ----

os.environ['TOKENIZERS_PARALLELISM'] = "false"
MAX_SEQ_LENGTH = 2048
project_name = "shakespeare-lora"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_target_modules", type=str, choices=["attn", "mlp", "all"], default="attn")
parser.add_argument("--dataset", type=str, default="alpaca")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--mini_batch_size", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--max_train_iters", type=int, default=10000)
parser.add_argument("--eval_interval", type=int, default=200)
parser.add_argument("--generate_interval", type=int, default=500)
parser.add_argument("--model_checkpoint_interval", type=int, default=1000)
parser.add_argument("--llmjudge", action="store_true", default=False)
parser.add_argument("--testing", action="store_true", default=False)
args = parser.parse_args()
model_name = args.model
dataset_name = args.dataset
lora_r = args.lora_r
lora_alpha = args.lora_alpha
lora_target_modules = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "down_proj", "up_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
}[args.lora_target_modules]
epochs = args.epochs
mini_batch_size = args.mini_batch_size
batch_size = args.batch_size
lr = args.lr
max_train_iters = args.max_train_iters
eval_interval = args.eval_interval
generate_interval = args.generate_interval
model_checkpoint_interval = args.model_checkpoint_interval
enable_llmjudge = args.llmjudge

run_name = f"{dataset_name}-{args.lora_target_modules}-{lora_r}-{lora_alpha}"

# Evaluation parameters
mmlu_batch_size = 4
mmlu_examples = 64
smoltalk_batch_size = 8
llmjudge_examples = 32
test_examples = 64
generate_examples = 4

# Short training run for testing
if args.testing:
    run_name = "test-run"
    max_train_iters = 100
    eval_interval = 20
    generate_interval = 40
    smoltalk_batch_size = 2
    llmjudge_examples = 2
    model_checkpoint_interval = 50

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
    new_lens = torch.tensor([len(line) for line in lines], dtype=torch.float)
    print(f"Trimmed {dataset_name} dataset from {old_len} to {len(lines)} examples")
    print(f"Estimated length distribution (1 token = 4 chars): mean={int(new_lens.mean()/4)}, std={int(new_lens.std()/4)}, min={int(new_lens.min()/4)}, max={int(new_lens.max()/4)}")
    
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

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = get_lora_model(model_name, lora=True, r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules)
if torch.cuda.is_available():
    model = torch.compile(model)
    torch.set_float32_matmul_precision("high")
# Constant learning rate for now
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

train_size, train_loader = dataset_loader(dataset_name, "train", mini_batch_size, trim=0.2, visualize=False)
test_size, test_loader = dataset_loader(dataset_name, "test", mini_batch_size, trim=0.2, visualize=False)

assert batch_size % mini_batch_size == 0, "batch_size must be divisible by mini_batch_size"
grad_accum_steps = batch_size // mini_batch_size
num_iters = train_size // batch_size * epochs
if num_iters > max_train_iters:
    print(f"num_iters ({num_iters}) is greater than max_train_iters ({max_train_iters}), setting num_iters to {max_train_iters}")
    num_iters = max_train_iters
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

print(f"Training on {device} with batch size={batch_size} and mini_batch size={mini_batch_size}.\n\
Number of iterations: {num_iters}. 1 epoch = {train_size} unique examples across {train_size // batch_size} iterations.\n\
Testing on {test_size} unique examples.")

trainable_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
total_parameters = sum([p.numel() for p in model.parameters()])
print(f"Trainable parameters: {trainable_parameters}. Total parameters: {total_parameters}. Percentage of trainable parameters: {trainable_parameters / total_parameters * 100:.2f}%")
time_start = time.time()


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
        mmlu_score = evaluate_mmlu(model, tokenizer, batch_size=mmlu_batch_size, total_examples=mmlu_examples)
        
        # LLM-as-a-judge Evaluation
        if enable_llmjudge:
            print(f"Running LLM-as-a-judge evaluation...")
            smoltalk_convos = generate_smoltalk(model, tokenizer, batch_size=smoltalk_batch_size, num_examples=llmjudge_examples, max_new_tokens=512)
            llmjudge_score = llmjudge_conversations(smoltalk_convos, logger=logger)

        # Log and print
        logger.log("eval", {
            "iter": iter,
            "test_loss": total_test_loss,
            "mmlu_score": mmlu_score,
            "llmjudge_score": llmjudge_score if enable_llmjudge else None,
        })
        print(f"Test Loss: {total_test_loss:.2f}")
        print(f"MMLU Score: {mmlu_score:.2f}")
        if enable_llmjudge:
            llmjudge_mean_score = sum(llmjudge_score.values())/len(llmjudge_score)
            print(f"LLM-as-a-judge Average Score: {llmjudge_mean_score:.2f}")
        clear_cache(device)

    
    # Generate periodically using smoltalk
    if iter % generate_interval == 0 or last_step:
        print(f"#########################\nGenerating at iteration {iter}...\n#########################")

        smoltalk_generator = smoltalk_prompt_generator(split="test", num_examples=generate_examples)
        convos = generate_smoltalk(model, tokenizer, batch_size=min(smoltalk_batch_size, generate_examples), num_examples=generate_examples, generator=smoltalk_generator, max_new_tokens=256)
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

time_end = time.time()
print(f"Training complete. Saved model to checkpoints/{run_name}_{num_iters}. Time taken: {time_end - time_start:.2f} seconds")