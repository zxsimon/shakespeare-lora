import json
import os

class Logger:
    def __init__(self, project_name, run_name = "default", log_dir="logs"):
        self.project_name = project_name
        self.run_name = run_name
        self.log_file = os.path.join(log_dir, f"{self.project_name}_{self.run_name}.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_file, "w") as _:
            return
    
    def _log(self, type, log):
        content = {
            "type": type,
            "log": log
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(content) + "\n")

    def log_config(self, config):
        self._log("config", config)

    def log_step(self, content):
        self._log("step", content)
