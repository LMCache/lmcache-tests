import os, sys
import json
import subprocess
import signal
import shlex
import time
from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass
class ProcessHandle:
    process: subprocess.Popen
    stdout_file: object
    stderr_file: object
    stdout_filename: str = None
    stderr_filename: str = None

    def kill_and_close(self, force_kill_after=60):
        """
        Kill the process by sending the SIGINT signal, then close the redirected stderr/stdout files
        """
        if self.is_alive():
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)

        if self.stderr_file is not None:
            self.stderr_file.close()
        if self.stdout_file is not None:
            self.stdout_file.close()

        if self.stdout_filename is not None and os.path.exists(self.stdout_filename):
            os.remove(self.stdout_filename)
        if self.stderr_filename is not None and os.path.exists(self.stderr_filename):
            os.remove(self.stderr_filename)

        countdown = force_kill_after
        while self.is_alive() and countdown > 0:
            time.sleep(1)
            countdown -= 1

        # Force kill the process if it's still alive
        if self.is_alive():
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)

    def is_alive(self):
        return self.process.poll() is None

def run_command(command, outfile=None, errfile=None, detach=False, **kwargs):
    """
    Input:
        command: a single string of the shell command
        outfile: redirect output to this file if it's not None
        errfile: redirect stderr to this file if it's not None
        detach: if True, it will start a subprocess and return the handle of that process
                without blocking the caller
                if False, it will block the caller until the subprocess finished. And it
                will return a boolean indicating whether the process successfully finishes
        kwargs: the dictionary of extra environment variables
    Returns:
        If `detach` is False:
            returns (flag, stdout string)
            flag will be True if the process finished without any error
            returns False otherwise
        If `detach` is True:
            returns the handle to the background process (ProcessHandle project)
    Note:
        If outfile and errfile are None, it will be defaulted to print to stdout
    """
    env = os.environ.copy()
    env.update(kwargs)


    out = open(outfile, "w") if outfile is not None else None
    err = open(errfile, "w") if errfile is not None else None

    args = shlex.split(command)

    process = subprocess.Popen(args, stdout=out, stderr=err, env=env, preexec_fn=os.setsid)


    if not detach:
        process.communicate()
        if out is not None:
            out.close()
        if err is not None:
            err.close()
        return process.returncode == 0, process.stdout
    else:
        return ProcessHandle(process, out, err, outfile, errfile)


def estimate_num_tokens(text: str) -> int:
    if not hasattr(estimate_num_tokens, "tokenizer"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO: do not hard-code tokenizer 
        estimate_num_tokens.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    return len(estimate_num_tokens.tokenizer.tokenize(text))

def read_gpu_memory():
    """
    Read the GPU memory usage by using nvidia-smi command
    """
    command = "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    return json.dumps(
            {f"gpu-{id}":int(x) for id, x in enumerate(result.stdout.decode("utf-8").strip().split("\n"))})

def get_max_context_length(model: str) -> int:
    match model:
        case "mistralai/Mistral-7B-Instruct-v0.2":
            return 32768
        case "THUDM/glm-4-9b-chat":
            return 32768
        case "meta-llama/Llama-3.1-8B-Instruct":
            return 32768
        case _:
            return 32768