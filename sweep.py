import json, subprocess, shlex

with open("settings.jsonl") as f:
    for line in f:
        cfg = json.loads(line)
        args = " ".join(f"--{k} {v}" if v else f"--{k}" for k, v in cfg.items())
        cmd = f"python train.py {args}"
        print(cmd)
        subprocess.run(shlex.split(cmd), check=True)