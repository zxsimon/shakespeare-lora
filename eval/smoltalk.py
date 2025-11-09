from itertools import islice
import datasets, time, torch
from tqdm import tqdm


def smoltalk_prompt_generator(split="test", num_examples=1000, max_chars=6000):
    """Generator for SmolTalk dataset. Chat template is not applied."""

    smoltalk = datasets.load_dataset("HuggingFaceTB/smol-smoltalk")
    smoltalk = smoltalk[split].shuffle().select(range(num_examples))
    for doc in smoltalk:
        prompt = doc['messages'][0]['content']
        if len(prompt) > max_chars:
            continue
        yield prompt

@torch.no_grad()
def generate_smoltalk(model, tokenizer, batch_size = 4, num_examples = 100, max_new_tokens = 512, generator = None):

    """Generate responses for smoltalk."""

    # Avoids SDPA issue on MPS devices. Awaiting new pytorch release...
    if model.device.type == "mps":
        model.set_attn_implementation("eager")

    if generator is None:
        generator = smoltalk_prompt_generator(num_examples=num_examples)

    num_iters = num_examples // batch_size
    all_prompts = []
    all_generated_text = []

    start = time.perf_counter(); tok = 0
    for _ in (pbar := tqdm(range(num_iters), total=num_iters, desc=f"SmolTalk (bs={batch_size})")):

        prompts = list(islice(generator, batch_size))
        templatized_prompts = [f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant<think>\n\n</think>\n\n" for prompt in prompts]
        tokenized_prompts = tokenizer(templatized_prompts, return_tensors="pt", padding=True, padding_side="left", pad_to_multiple_of=8).to(model.device)

        input_ids, attention_mask = tokenized_prompts["input_ids"], tokenized_prompts["attention_mask"]
        generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
        new_tokens = generated_ids[:, input_ids.shape[1]:]
        generated_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        all_prompts.extend(prompts)
        all_generated_text.extend(generated_text)

        tok += new_tokens.numel()
        pbar.set_postfix_str(f"{tok/max(time.perf_counter()-start,1e-6):.1f} tok/s")
        if model.device.type == "mps":
            torch.mps.empty_cache()

    zipped = list(zip(all_prompts, all_generated_text))

    return zipped



if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "Qwen/Qwen3-8B"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = torch.compile(model)
        torch.set_float32_matmul_precision('high')
    generator = smoltalk_prompt_generator()
    zipped = generate_smoltalk(model, tokenizer, num_examples = 32, batch_size = 4, generator = generator)
    print(f"\nFirst example:\n{zipped[0]}")