import os, sys
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

    def kill_and_close(self):
        """
        Kill the process by sending the SIGINT signal, then close the redirected stderr/stdout files
        """
        os.killpg(os.getpgid(self.process.pid), signal.SIGINT)

        if self.stderr_file is not None:
            self.stderr_file.close()
        if self.stdout_file is not None:
            self.stdout_file.close()

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
        return ProcessHandle(process, out, err)


def estimate_num_tokens(text: str) -> int:
    if not hasattr(estimate_num_tokens, "tokenizer"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO: do not hard-code tokenizer 
        estimate_num_tokens.tokenizer = AutoTokenizer.from_pretrained("lmsys/longchat-7b-16k")

    return len(estimate_num_tokens.tokenizer.tokenize(text))
