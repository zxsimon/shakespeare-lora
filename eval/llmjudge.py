from eval.smoltalk import generate_smoltalk
from utils import generate_completion, Logger, check_server
import subprocess, torch, time, requests, json, code
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
judge_model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"


def start_vllm_server(model_name = judge_model_name):
    """Start vLLM in background"""
    
    process = subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", "127.0.0.1",
        "--port", "1234",
        "--gpu-memory-utilization", "0.6",
        "--dtype", "auto",
        "--max-model-len", "16384"
    ])
    
    # Wait for server to start
    time.sleep(30)
    
    # Check if ready
    for _ in range(30):
        try:
            requests.get("http://localhost:1234/health")
            print("vLLM server ready!")
            return process
        except:
            time.sleep(2)
    
    raise RuntimeError("vLLM server failed to start")

def judge_prompt(original_query, model_response):
    """Generate a prompt for the LLM-as-a-judge to evaluate a Shakespeare-style response."""
    
    prompt = f"""You are evaluating a Shakespeare-style AI response. Rate the response on five independent dimensions using a 1-5 scale.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ORIGINAL QUERY:
    {original_query}

    SHAKESPEARE-STYLE RESPONSE:
    {model_response}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Evaluate on these FIVE INDEPENDENT dimensions:

    1. LINGUISTIC AUTHENTICITY (Correctness of Early Modern English)
    - Are pronouns used correctly? (thou/thee/thy for singular, ye/you for plural/formal)
    - Are verb forms correct? (dost/doth/hath/art conjugated properly)
    - Is vocabulary period-appropriate and used correctly?
    - Are syntax patterns grammatically valid for the era?

    1 = Major errors (wrong conjugations, misused pronouns, anachronistic terms)
    2 = Multiple minor errors
    3 = Mostly correct with occasional mistakes
    4 = Correct with only trivial issues
    5 = Flawless Early Modern English grammar and usage

    2. STYLISTIC INTENSITY (Boldness of transformation)
    - How heavily is the Shakespearean style applied?
    - Frequency of archaic elements, inversions, exclamations, metaphors
    - How dramatic and theatrical is the language?

    1 = Barely transformed, mostly modern English
    2 = Light sprinkling of archaic elements
    3 = Moderate Shakespearean style throughout
    4 = Heavy transformation with rich period language
    5 = Maximum dramatic flair, fully immersive Shakespeare style

    3. CONTENT HELPFULNESS (Quality of information, ignoring style)
    - Does it answer the question accurately and completely?
    - Is the information correct and relevant?
    - Would this be helpful if written in modern English?

    1 = Unhelpful, inaccurate, or irrelevant
    2 = Partially helpful but incomplete or somewhat inaccurate
    3 = Adequately helpful and mostly accurate
    4 = Helpful, accurate, and reasonably complete
    5 = Exceptionally helpful, accurate, comprehensive

    4. MODERN COMPREHENSIBILITY (Clarity to contemporary readers)
    - Can a modern reader (with basic Shakespeare familiarity) understand this?
    - Is the core meaning preserved and clear?
    - Does style obscure or enhance understanding?

    1 = Very difficult to understand, meaning obscured
    2 = Requires significant effort to parse
    3 = Understandable with moderate attention
    4 = Clear with minor cognitive load from style
    5 = Perfectly clear despite archaic language

    5. CONTEXTUAL APPROPRIATENESS (Style level matching query type)
    - For technical/factual queries: lighter style more appropriate
    - For narrative/creative/emotional queries: heavier style more appropriate
    - Does the intensity fit the content type?

    1 = Severely mismatched (heavy style on code, or barely styled narrative)
    2 = Somewhat mismatched
    3 = Acceptable match
    4 = Well-matched to query type
    5 = Perfectly calibrated for this specific query

    Lastly, assess overall quality of the response and give a score between 1 and 5.

    1 = Unusable, major failures
    2 = Poor, significant issues
    3 = Acceptable, meets basic requirements
    4 = Good, minor issues only
    5 = Excellent, exemplary response

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Provide your evaluation in JSON format with this EXACT structure:

    {{
    "authenticity": <1-5>,
    "intensity": <1-5>,
    "helpfulness": <1-5>,
    "clarity": <1-5>,
    "appropriateness": <1-5>,
    "overall": <1-5>,
    "reasoning": "<2-3 sentences explaining key strengths and weaknesses>"
    }}

    Output ONLY valid JSON, no other text."""
    
    return prompt


def llmjudge_conversations(conversations, logger = None, host = "127.0.0.1", port = "1234"):
    """Evaluate a list of conversations with LLM-as-a-judge."""
    
    server_already_running = check_server(host, port)
    process = None
    # Start vLLM server on CUDA. On Mac, set up LMStudio manually.
    if torch.cuda.is_available() and not server_already_running:
        process = start_vllm_server()

    eval_count = 0
    score_keys = ["authenticity", "intensity", "helpfulness", "clarity", "appropriateness", "overall"]
    scores = {key: 0 for key in score_keys}


    for prompt, response in tqdm(conversations, total=len(conversations), desc=f"Evaluating conversations"):
        prompt_for_judge = judge_prompt(prompt, response)
        completion = generate_completion(prompt_for_judge)
        
        # Attempts to parse output as JSON
        try: 
            completion = json.loads(completion)
            eval_count += 1
        except:
            print(f"Error parsing JSON: {completion}")
            continue
        
        for key in score_keys:
            scores[key] += completion[key]

        # Logs to file if logger is provided
        if logger:
            completion["prompt"] = prompt
            completion["response"] = response
            logger.log_judge(json.dumps(completion))

    if process:
        process.terminate()
    
    if eval_count > 0:
        scores = {k: v / eval_count for k, v in scores.items()}
        return scores


if __name__ == "__main__":
    # Testing
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger = Logger("shakespeare-lora", run_name="llmjudge-test")
    conversations = generate_smoltalk(model, tokenizer, num_examples = 4, batch_size = 2)
    code.interact(local=locals())
    print(llmjudge_conversations(conversations, logger=logger))