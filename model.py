import torch, os, json
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel
import code

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
default_model_name = "Qwen/Qwen3-0.6B"

tokenizer_qwen3 = AutoTokenizer.from_pretrained(default_model_name)
think_end_token_id = tokenizer_qwen3.encode("</think>")[0]
im_start_token_id = tokenizer_qwen3.encode("<|im_start|>")[0]
pad_token_id = tokenizer_qwen3.pad_token_id

def get_lora_model(model_name = default_model_name, lora = True, r = 16, lora_alpha = 32, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]):
    lora_config = LoraConfig(
                        r=r,
                        lora_alpha=lora_alpha,  
                        target_modules=target_modules,
                        bias="none",
                    )
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    if lora:
        model = get_peft_model(model, lora_config)
    model.to(device)
    return model

def test_generation(model, tokenizer = tokenizer_qwen3, prompt = None, max_new_tokens=1000, show_thinking=True, format_output = False):
    
    if prompt is None:
        prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens
    )[0]


    if not format_output:
        output = tokenizer.decode(out_ids, skip_special_tokens=False)
        return(output)
    else:
        gen_ids = out_ids[inputs['input_ids'].shape[1]:]
        try:
            think_id_idx = torch.argmax(gen_ids == think_end_token_id).item()
        except ValueError:
            think_id_idx = 0
        think_ids = gen_ids[:think_id_idx+1]
        output_ids = gen_ids[think_id_idx+1:]
        if show_thinking:
            print(tokenizer.decode(think_ids, skip_special_tokens=True).strip("\n"))
        print(tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n"))

def load_checkpoint(checkpoint_path):
    with open(os.path.join(checkpoint_path, "adapter_config.json"), "r") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.bfloat16), checkpoint_path)
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    model = get_lora_model(lora=True, r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    out = test_generation(model)
    code.interact(local=dict(globals(), **locals()))