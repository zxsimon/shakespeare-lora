from eval.smoltalk import generate_smoltalk
from utils import generate_completion, Logger, check_server
import subprocess, torch, time, requests, json, code
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
judge_model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"


def judge_prompt(original_query, model_response):
    """Generate a prompt for the LLM-as-a-judge to evaluate a Shakespeare-style response."""
    
    prompt = f"""You are evaluating a Shakespeare-style AI response. Rate the response on four independent dimensions using a 1-10 scale.

1. SHAKESPEAREAN STYLISTIC INTENSITY (Boldness, artfulness, and correctness of transformation)
Core criteria:
- How heavily is the Shakespearean style applied?
- Frequency of archaic elements, inversions, exclamations
- How dramatic and theatrical is the language?
- Are pronouns used correctly? (thou/thee/thy for singular, ye/you for plural/formal)
- Are verb forms correct? (dost/doth/hath/art conjugated properly)
- Is vocabulary period-appropriate and used correctly?
- Are syntax patterns grammatically valid for the era?

Advanced criteria (for scores 8+):
- Sophisticated metaphors and extended conceits (not just surface-level comparisons)
- Creative use of wordplay, double meanings, or puns
- Varied rhetorical devices (apostrophe, anaphora, chiasmus, etc.)
- Natural dramatic flair without forced or overwrought language
- Consistent use of subjunctive mood and conditional constructions
- Proper handling of archaic negation (ne'er, nay, nor patterns)
- Sophisticated use of inversions and embedded clauses
- Natural integration of multiple archaic constructions without awkwardness

1-2 = Barely transformed, mostly modern English
3-4 = Light sprinkling of archaic elements
5-6 = Moderate Shakespearean style throughout
7-8 = Heavy transformation with rich period language
9-10 = Masterful transformation with sophisticated metaphors, wordplay, and varied rhetorical devices.

2. CONTENT HELPFULNESS (Quality of information, ignoring style)
Core criteria:
- Does it answer the question accurately and completely?
- Is the information correct and relevant?
- Would this be helpful if written in modern English?

Advanced criteria (for scores 8+):
- Provides nuanced insights or perspectives beyond basic answers
- Anticipates follow-up questions or addresses implicit needs
- Information is well-organized and logically structured
- Includes appropriate caveats, context, or additional relevant details

1-2 = Unhelpful, inaccurate, or irrelevant
3-4 = Partially helpful but incomplete or somewhat inaccurate
5-6 = Adequately helpful and mostly accurate
7-8 = Helpful, accurate, and reasonably complete
9-10 = Exceptionally insightful, comprehensive, and anticipates user needs

3. MODERN COMPREHENSIBILITY (Clarity to contemporary readers)
Core criteria:
- Can a modern reader (with basic Shakespeare familiarity) understand this?
- Is the core meaning preserved and clear?
- Does style obscure or enhance understanding?

Advanced criteria (for scores 8+):
- Complex ideas rendered clearly despite archaic language
- Style enhances rather than merely decorating the message
- Maintains clarity even with sophisticated linguistic constructions
- Reader can follow along smoothly without re-reading

1-2 = Very difficult to understand, meaning obscured
3-4 = Requires significant effort to parse
5-6 = Understandable with moderate attention
7-8 = Clear with minor cognitive load from style
9-10 = Perfectly clear despite heavy style; archaic language enhances rather than obscures

4. CONTEXTUAL APPROPRIATENESS (Style calibration to query type)
Core criteria:
- For technical/factual queries: lighter style more appropriate
- For narrative/creative/emotional queries: heavier style more appropriate
- Does the intensity fit the content type?

Advanced criteria (for scores 8+):
- Subtle modulation of style intensity within response as needed
- Technical terms or modern concepts integrated gracefully
- Tone matches emotional context (playful vs. serious vs. contemplative)
- Shows awareness of when to prioritize clarity vs. dramatic effect

1-2 = Severely mismatched (heavy style on code, or barely styled narrative)
3-4 = Somewhat mismatched
5-6 = Acceptable match
7-8 = Well-matched to query type
9-10 = Perfectly calibrated with sophisticated style modulation as context requires


IMPORTANT: Use the FULL 1-10 scale. Be as strict as possible.

Provide your evaluation in JSON format with this EXACT structure:

{{
"intensity": <1-10>,
"helpfulness": <1-10>,
"clarity": <1-10>,
"appropriateness": <1-10>,
"reasoning": "<2-3 sentences explaining key strengths and weaknesses, particularly noting whether advanced criteria were met>"
}}

Output ONLY valid JSON, no other text.

ORIGINAL QUERY (for context only):
{original_query}

SHAKESPEARE-STYLE RESPONSE TO EVALUATE:
{model_response}

"""
    
    return prompt


def llmjudge_conversations(conversations, logger = None, host = "127.0.0.1", port = "1234"):
    """Evaluate a list of conversations with LLM-as-a-judge."""
    
    if not (server_already_running := check_server(host, port)):
        raise ValueError("LLM-as-a-judge server is not running. On CUDA, please start it with `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --host 127.0.0.1 --port 1234 --dtype auto --gpu-memory-utilization 0.6 --max-model-len 8192`")

    eval_count = 0
    score_keys = ["intensity", "helpfulness", "clarity", "appropriateness"]
    scores = {key: 0 for key in score_keys}

    for prompt, response in tqdm(conversations, total=len(conversations), desc=f"Evaluating conversations"):
        prompt_for_judge = judge_prompt(prompt, response)
        completion = generate_completion(prompt_for_judge)

        if completion is None:
            continue
        
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
    
    if eval_count > 0:
        scores = {k: v / eval_count for k, v in scores.items()}
        return scores


if __name__ == "__main__":
    # Testing
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        model = torch.compile(model)
        torch.set_float32_matmul_precision("high")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger = Logger("shakespeare-lora", run_name="llmjudge-test")
    conversations = generate_smoltalk(model, tokenizer, num_examples = 32, batch_size = 8)
    print(llmjudge_conversations(conversations, logger=logger))