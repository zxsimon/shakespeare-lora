from itertools import islice
import datasets
import torch
import tqdm, time


def smoltalk_prompt_generator(split="test", num_examples=1000):
    """Generator for SmolTalk dataset. Chat template is not applied."""

    smoltalk = datasets.load_dataset("HuggingFaceTB/smol-smoltalk")
    smoltalk = smoltalk[split].shuffle().select(range(num_examples))
    for doc in smoltalk:
        prompt = doc['messages'][0]['content']
        yield prompt

@torch.no_grad()
def generate_smoltalk(model, tokenizer, batch_size = 4, num_examples = 100, generator = None, max_new_tokens = 512):

    """Generate responses for smoltalk."""

    # Avoids SDPA issue on MPS devices. Awaiting new pytorch release...
    if model.device.type == "mps":
        model.set_attn_implementation("eager")

    if generator is None:
        generator = smoltalk_prompt_generator(num_examples=num_examples)

    num_iters = num_examples // batch_size
    all_prompts = []
    all_generated_text = []

    for _ in tqdm.tqdm(range(num_iters), total=num_iters, desc=f"Generating SmolTalk with batch size = {batch_size}"):

        prompts = list(islice(generator, batch_size))
        templatized_prompts = [f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant<think>\n\n</think>\n\n" for prompt in prompts]
        tokenized_prompts = tokenizer(templatized_prompts, return_tensors="pt", padding=True, padding_side="left", pad_to_multiple_of=8).to(model.device)

        input_ids, attention_mask = tokenized_prompts["input_ids"], tokenized_prompts["attention_mask"]
        generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
        new_tokens = generated_ids[:, input_ids.shape[1]:]
        generated_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        all_prompts.extend(prompts)
        all_generated_text.extend(generated_text)

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
        model = torch.compile(model, mode="max-autotune")
        torch.set_float32_matmul_precision('high')
    generator = smoltalk_prompt_generator()
    
    # Throughput measurement code
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    zipped = generate_smoltalk(model, tokenizer, num_examples = 32, batch_size = 32, generator = generator)
    end_time.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_time.elapsed_time(end_time)
    total_tokens = sum(len(tokenizer.encode(text)) for _, text in zipped)
    
    print(f"\nGenerated {len(zipped)} examples")
    print(f"Total tokens: {total_tokens}")
    print(f"Time: {elapsed_ms/1000:.2f}s")
    print(f"Throughput: {total_tokens/(elapsed_ms/1000):.1f} tokens/sec")
    print(f"\nFirst example:\n{zipped[0]}")