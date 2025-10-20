from itertools import islice
import datasets
import torch
import tqdm


def smoltalk_prompt_generator(split="test", num_examples=1000):
    """Generator for SmolTalk dataset. Chat template is not applied."""

    smoltalk = datasets.load_dataset("HuggingFaceTB/smol-smoltalk")
    smoltalk = smoltalk[split].shuffle().select(range(num_examples))
    for doc in smoltalk:
        prompt = doc['messages'][0]['content']
        yield prompt

@torch.no_grad()
def generate_smoltalk(model, tokenizer, batch_size = 4, num_examples = 100, generator = None, max_new_tokens = 1000):

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
        tokenized_prompts = tokenizer(templatized_prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)

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
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generator = smoltalk_prompt_generator()
    zipped = generate_smoltalk(model, tokenizer, num_examples = 4, batch_size = 2, generator = generator)
    print(zipped)