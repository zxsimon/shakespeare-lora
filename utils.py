from itertools import tee
import json, torch, os, requests
from openai import OpenAI

class Logger:
    def __init__(self, project_name, run_name = "default", log_dir="logs"):
        self.project_name = project_name
        self.run_name = run_name
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{self.project_name}_{self.run_name}.jsonl")
        self.log_file_judge = os.path.join(log_dir, f"llmjudge_{self.project_name}_{self.run_name}.jsonl")
        self._initialized_files = set()

    def _ensure_file_reset(self, filepath):
        """Reset file on first write."""
        if filepath not in self._initialized_files:
            with open(filepath, "w") as f:
                pass  # Create/truncate file
            self._initialized_files.add(filepath)
    
    def log(self, t, log):
        self._ensure_file_reset(self.log_file)
        content = {
            "type": t,
            "log": log
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(content) + "\n")

    def log_judge(self, content):
        """content is already json-formatted"""
        self._ensure_file_reset(self.log_file_judge)
        with open(self.log_file_judge, "a") as f:
            f.write(content + "\n")

def check_server(host="127.0.0.1", port=1234, timeout=2):
    """Check if an OpenAI server is already running."""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def generate_completion(prompt, temperature = 0.0, max_tokens = 1000):
    client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
    model_name = client.models.list().data[0].id
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

# Currently unused due to bad interactions between peft and torch.compile
def evaluate_with_baseline(model, tokenizer, generator, evaluate_function, **kwargs):
    generator_lora, generator_baseline = tee(generator, 2)
    score_lora = evaluate_function(model, tokenizer, generator=generator_lora, **kwargs)
    with model.disable_adapter():
        score_baseline = evaluate_function(model, tokenizer, generator=generator_baseline, **kwargs)
    return score_lora, score_baseline

def clear_cache(device):
    if device == "mps":
        torch.mps.empty_cache()
    if device == "cuda":
        torch.cuda.empty_cache()