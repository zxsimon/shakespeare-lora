import argparse, os, torch, threading, sys, time, random
from src.model import load_checkpoint, load_tokenizer

os.environ['TOKENIZERS_PARALLELISM'] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_target_modules", type=str, choices=["attn", "mlp", "all"], default="mlp")
parser.add_argument("--iter", type=int, default=-1)
parser.add_argument("--dataset", type=str, default="alpaca")
args = parser.parse_args()

BLUE = "\033[94m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def find_last_checkpoint():
    prefix = f"{args.dataset}-{args.lora_target_modules}-{args.lora_r}-{args.lora_alpha}"
    return max([int(e.name.split("_")[-1]) for e in os.scandir(f"checkpoints") if e.is_dir() and e.name.startswith(prefix)])

SHAKESPEARE_LOADING = [
    "Anon! Thy verse prepares",
    "Hark—my quill is ink’d",
    "Summoning sprites of wit",
    "Methinks the muse awakes",
    "Give me but a moment’s grace",
    "I do but polish every syllable",
    "Stand by—iambs aligning",
    "’Tis nearly writ",
    "The plot thickens apace",
    "Bear with me, gentle heart",
    "I stitch fair couplets",
    "Quoth the quill: almost",
    "Soft! What lines through yonder prompt break",
    "Page to stage, anon",
    "I marshal rhymes and reason",
    "Good morrow, patience",
    "My thoughts do canter to thee",
    "I temper words like steel",
    "Fortune speeds my pen",
    "Verily, near done",
    "Ink doth dry as we speak",
    "Prologue in rehearsal",
    "The chorus clears its throat",
    "All’s well that loads well",
]

class LoadingDots:
    def __init__(self):
        self.texts = SHAKESPEARE_LOADING
        self.interval_dots = 0.25
        self.interval_text = 12
        self.width = 3
        self._stop = threading.Event()
        self._t = None

    def start(self):
        if self._t: return
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        i = 0
        text = random.choice(self.texts)
        while not self._stop.is_set():
            if i % self.interval_text == 0:
                text = random.choice(self.texts)
            dots = "." * (i % self.width + 1)
            sys.stdout.write("\r\033[K" + BLUE + text + dots + "   " + RESET)
            sys.stdout.flush()
            time.sleep(self.interval_dots)
            i += 1
        sys.stdout.write("\r" + " " * (len(random.choice(self.texts)) + self.width + 3) + "\r\033[?25h")
        sys.stdout.flush()

    def stop(self):
        if not self._t: return
        self._stop.set()
        self._t.join()
        self._t = None
        self._stop.clear()

checkpoint_dir = f"checkpoints/{args.dataset}-{args.lora_target_modules}-{args.lora_r}-{args.lora_alpha}_{args.iter if args.iter != -1 else find_last_checkpoint()}"
model = load_checkpoint(checkpoint_dir)
print(f"{RED}Loaded checkpoint from {checkpoint_dir}{RESET}")
tokenizer = load_tokenizer(checkpoint_dir)

def main():

    print(f"{RED}/exit to exit, /reset to clear context{RESET}")
    print(f"{BLUE}<(o‿o)>ノ🪶 'Pray, send thy prompt.'{RESET}")
    output = ""

    while True:
        
        sys.stdout.flush()
        try:
            user = input("You: " + YELLOW).strip()
        except (EOFError, KeyboardInterrupt):
            break
        sys.stdout.write("\033[F\033[2K")
        sys.stdout.write(user + RESET + "\n")
        sys.stdout.flush()

        if not user: continue
        if user == "/exit": break
        if user == "/reset": 
            output = ""
            print(f"{BLUE}<(o‿o)>ノ🪶 'Pray, send thy prompt.'{RESET}")
            continue

        if not output:
            messages = [
                {"role": "user", "content": user},
            ]

        else:
            messages.append({"role": "user", "content": user})
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)

        loading_dots = LoadingDots()
        loading_dots.start()

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=1000
            )[0]
        output = tokenizer.decode(out_ids, skip_special_tokens=True)
        response = output.split("</think>")[-1].strip()
        messages.append({"role": "assistant", "content": response})
        loading_dots.stop()
        print(f"{BLUE}{response}{RESET}")

if __name__ == "__main__":
    main()