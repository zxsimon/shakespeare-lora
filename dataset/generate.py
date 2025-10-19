from datasets import load_dataset, Dataset
import code
from openai import OpenAI
import json
from tqdm import tqdm
import argparse

def shakespearean_prompt(prompt, answer):
    model_prompt = f"""Transform ONLY THE ANSWER into Shakespearean style. DO NOT modify the original prompt.
                        Apply significant stylistic transformation to the answer while preserving technical elements, context, meaning, and structure.
                        OUTPUT ONLY THE TRANSFORMED ANSWER - no markdown formatting, asterisks, or annotations.

                        LANGUAGE TOOLKIT (use liberally, but don't overdo it to the point of making it unrealistic, unintelligible, or unhelpful):
                        Pronouns: thou/thee/thy/thine, 'tis, 'twas, 'twill, o'er, ere, 'gainst
                        Verbs: art, dost, hast, canst, mayst | doth, hath, maketh, sayeth
                        Syntax: INVERT FREQUENTLY - "Strange it is...", "This I say...", "Well know I..."
                        Exclamations: Forsooth!, Lo!, Prithee!, Marry!, Fie!, Alack!
                        Common swaps: people→folk, often→oft, said→spake/quoth, very→exceeding, because→for, help→aid, happy→merry, before→ere
                        Modern terms: phone→"speaking-device" | computer→"calculating engine" | stressed→"sore vexed" | prioritize→"set afore all else"

                        TRANSFORMATION LEVELS:

                        HEAVY (70-90%): Prose, dialogue, opinions, narrative, emotional content
                        → Use inversions, compound epithets, metaphors, dramatic phrasing
                        MODERATE (30-50%): Explanations, instructions, descriptions  
                        → Transform connecting language, keep key terms clear
                        LIGHT (10-20%): Lists, measurements, technical data, code, structured content
                        → Basic pronoun swaps only (your→thy), preserve exact formatting

                        RULES:

                        ✓ Preserve: numbers, measurements, structure, formatting, technical terms, labels, content type (prose stays prose, dialogue stays dialogue)
                        ✓ Make sure the transformed answer is still coherent, sensible, and accurate, with respect to the original prompt and original answer
                        ✓ Transform aggressively: all prose, explanations, conversational text
                        ✗ Don't change: percentages, quantities, technical specifications
                        ✗ The new answer MUST NOT exceed 1.5x original length. Give a short modified answer if the original answer is also short.
                        ✗ NEVER transform the original answer into sonnets, poems, or play scripts, or add characters.
                        ✗ DO NOT add unnecessary line breaks

                        ══════════════════════════════════════════════════════════════════════════════
                        ORIGINAL PROMPT (for context only):
                        {prompt}

                        ══════════════════════════════════════════════════════════════════════════════
                        ANSWER TO TRANSFORM:
                        {answer}
                        """
    
    return model_prompt


def generate_completion(prompt):
    client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1000,
    )
    return response.choices[0].message.content

def append_to_json(prompt, answer, split, fp_suffix=""):

    training_example = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
    }
    
    # Append to JSONL file
    with open(f"shakespeare_{split}_{fp_suffix}.jsonl", 'a', encoding='utf-8') as f:
        f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
    
    return training_example

def transform_ultrafeedback(num_examples, split, print_compare = False, create_json = False):
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    ds_sample = ds["train_sft"].select(range(num_examples))
    examples = [row['chosen'] for row in ds_sample]
    for example in tqdm(examples, total=len(examples), desc="Generating Shakespearean training data"):
        orig_prompt = example[0]['content']
        orig_answer = example[1]['content']
        transformed_prompt = shakespearean_prompt(orig_prompt, orig_answer)
        transformed_answer = generate_completion(transformed_prompt)
        if print_compare:
            print("Original prompt:")
            print(orig_prompt)
            print("Original answer:")
            print(orig_answer)
            print("-"*100)
            print("Transformed answer:")
            print(transformed_answer)
            print("-"*100)
        if create_json:
            append_to_json(orig_prompt, transformed_answer, split, fp_suffix = "ultrafeedback")

def transform_alpaca(num_examples, split, print_compare = False, create_json = False):
    ds = load_dataset("tatsu-lab/alpaca")
    ds_sample = ds["train"].select(range(num_examples))
    for example in tqdm(ds_sample, total=len(ds_sample), desc="Generating Shakespearean training data"):
        orig_prompt = f"Instruction: {example['instruction']}" + (f"\nInput: {example['input']}" if example['input'] else "")
        orig_answer = example['output']
        transformed_prompt = shakespearean_prompt(orig_prompt, orig_answer)
        transformed_answer = generate_completion(transformed_prompt)
        if print_compare:
            print("Original prompt:")
            print(orig_prompt)
            print("Original answer:")
            print(orig_answer)
            print("-"*100)
            print("Transformed answer:")
            print(transformed_answer)
            print("-"*100)
        if create_json:
            append_to_json(orig_prompt, transformed_answer, split, fp_suffix = "alpaca")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, choices=["ultrafeedback", "alpaca"])
parser.add_argument("--num_examples", type=int, default=10000)
parser.add_argument("--print_compare", type=bool, default=False)
parser.add_argument("--create_json", type=bool, default=False)
parser.add_argument("--split", type=str, choices=["train", "val", "test"])
args = parser.parse_args()

if __name__ == "__main__":
    if args.dataset_name == "ultrafeedback":
        transform_ultrafeedback(num_examples=args.num_examples, split=args.split, print_compare=args.print_compare, create_json=args.create_json)
    elif args.dataset_name == "alpaca":
        transform_alpaca(num_examples=args.num_examples, split=args.split, print_compare=args.print_compare, create_json=args.create_json)
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")