# Shakespeare-LoRA


## Quick Start

1) Install dependencies:
```bash
conda create -n shakespeare-lora python=3.10 -y
conda activate shakespeare-lora
pip install -e .
```

2) Chat (uses the included mlp-only r=16 α=32 checkpoint)

```bash
chat-lora
```

3) Sweep (runs multiple training jobs from a config)

```bash
sweep-lora # uses configs/settings.jsonl, no llmjudge
sweep-lora --config settings-llmjudge # with llmjudge, requires running a model on 127.0.0.1:1234
```

## Results
Detailed results and discussions are available in [report.md](report.md).
TL;DR: LoRA achieves Shakespearean style transfer while retaining general capabilities, though its responses become less verbose and helpful.


## Reproduction

- Reproduce a standard run (Qwen3-8B, LoRA attn r=16 α=32):

```bash
python -m scripts.train --model Qwen/Qwen3-8B --dataset alpaca --lora_target_modules attn --lora_r 16 --lora_alpha 32 --batch_size 8 --mini_batch_size 2 --epochs 3 --max_train_iters 5000 --eval_interval 150 --generate_interval 300 --model_checkpoint_interval 500 --llmjudge
```

- Checkpoints are saved to `checkpoints/{dataset}-{target}-{r}-{alpha}_{iter}` (e.g., `checkpoints/alpaca-attn-16-32_1500`).
- For a quick sanity check on a tiny model: do a quick run with `--testing`.
- Device selection is automatic (`cuda` > `mps` > `cpu`). Eval includes MMLU and optional LLM-judge.
- LLM-judge requires running a model on 127.0.0.1:1234, such as through vllm/LM Studio/Ollama(see `eval/llmjudge.py` for more details).

## Sweep Configs (edit configs/settings.jsonl)

- Each line is one training job; keys map directly to `scripts/train.py` CLI flags.
- Presence-only flags (booleans) use an empty string value to render `--flag`.

Example (`configs/settings.jsonl`):

```json
{"lora_r": 16, "lora_target_modules": "attn"}
{"lora_r": 16, "lora_target_modules": "mlp"}
{"lora_r": 32, "lora_target_modules": "attn"}
```

With LLM-as-a-judge:

```json
{"lora_r": 16, "lora_target_modules": "attn", "llmjudge": ""}
```

Run the sweep:

```bash
sweep-lora --config settings          # reads configs/settings.jsonl
sweep-lora --config settings-llmjudge # reads configs/settings-llmjudge.jsonl
```

- Baseline evaluation: mostly a variance reduction tool to judge model incremental performance during training. Nice to have, but does not seem too necessary, especially when we can just increase sample size
- How to best implement LLM as a judge? Can consider a pairwise comparison using model checkpoints, with an elo-like implementation
    - Observation that LLM judge tends to avoid giving extreme ends of the scale, and cluster around the mean
- Issue with using PEFT and torch.compile reduce-overhead