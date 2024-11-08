import copy
import json
import os
import time
import logging
import random
import subprocess

from lmcache_vllm import close_lmcache_engine
from lmcache_vllm.vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
context_file = "./ffmpeg.txt"
output_file = "./outputs/offline_inference_outputs.jsonl"

context_text = None
with open(context_file, 'r') as f:
    context_text = f.read()
assert context_text is not None
tokenizer = AutoTokenizer.from_pretrained(model_name)

def shuffle_text(text):
    text_list = text.split()
    random.shuffle(text_list)
    return ' '.join(text_list)

context_messages = [
    {
        "role":
        "user",
        "content":
        "I've got a document, "
        f"here's the content:```\n{context_text}\n```."
    },
    {
        "role": "assistant",
        "content": "I've got your document"
    },
]

user_inputs_batch = [
    "Give me a concise description for the format"
    " of ffmpeg command in one line.",
]


def get_context_length(tokenizer, context_messages):
    return len(tokenizer.apply_chat_template(context_messages, tokenize=False))


def gen_prompts(tokenizer, context_messages, user_inputs_of_batch):
    generated_prompts = []
    for user_input in user_inputs_of_batch:
        copyed_context_messages = copy.deepcopy(context_messages)
        copyed_context_messages.append({"role": "user", "content": user_input})
        generated_prompts.append(
            tokenizer.apply_chat_template(copyed_context_messages,
                                          tokenize=False))
    return generated_prompts


def append_outputs(output_file_name, outputs, context_length, time_taken):
    user_inputs = []
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        user_input = prompt[context_length:]
        user_inputs.append(user_input)
        generated_text = output.outputs[0].text
        generated_texts.append(f"{generated_text!r}")
    json_dict = {
        "user_inputs": user_inputs,
        "generated_texts": generated_texts,
        "time in seconds": time_taken,
        "context_length": context_length
    }
    with open(output_file_name, "a") as f:
        f.write(json.dumps(json_dict) + '\n')

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=1) # Set to 1 for TTFT
# Create an LLM.
llm = LLM(model=model_name,
          gpu_memory_utilization=0.6,
          enable_chunked_prefill=False,
          max_model_len=32768)

# Clear output file.
with open(output_file, "w") as f:
    pass

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
context_length = get_context_length(tokenizer, context_messages)
prompts = gen_prompts(tokenizer, context_messages, user_inputs_batch)
t1 = time.perf_counter()
first_outputs = llm.generate(prompts, sampling_params)
t2 = time.perf_counter()
logger.info(f"\n\nFirst request Time: {t2 - t1} seconds\n\n")
with open(output_file, "a") as f:
    f.write(f"\n\nFirst request:\n\n")
append_outputs(output_file, first_outputs, context_length, t2 - t1)

for i in range(3):
    t1 = time.perf_counter()
    first_outputs = llm.generate(prompts, sampling_params)
    t2 = time.perf_counter()
    logger.info(f"\n\nSame request Time: {t2 - t1} seconds\n\n")
    with open(output_file, "a") as f:
        f.write(f"\n\nSame request:\n\n")
    append_outputs(output_file, first_outputs, context_length, t2 - t1)

    # Generate random text to do full prefill.
    context_messages_random = [
        {
            "role": "user",
            "content": "I've got a document, "
            f"here's the content:```\n{shuffle_text(context_text)}\n```."
        },
        {
            "role": "assistant",
            "content": "I've got your document"
        },
    ]
    context_length_random = get_context_length(tokenizer, context_messages_random)
    prompts_random = gen_prompts(tokenizer, context_messages_random, user_inputs_batch)
    t3 = time.perf_counter()
    second_outputs = llm.generate(prompts_random, sampling_params)
    t4 = time.perf_counter()
    logger.info(f"\n\nRandom request Time: {t4 - t3} seconds\n\n")
    with open(output_file, "a") as f:
        f.write(f"\n\nRandom request:\n\n")
    append_outputs(output_file, second_outputs, context_length_random, t4 - t3)

    # Generate new random text to do full prefill again. This time, batch the requests.
    context_messages_random_new = [
        {
            "role": "user",
            "content": "I've got a document, "
            f"here's the content:```\n{shuffle_text(context_text)}\n```."
        },
        {
            "role": "assistant",
            "content": "I've got your document"
        },
    ]
    context_length_random_new = get_context_length(tokenizer, context_messages_random_new)
    prompts_random_new = gen_prompts(tokenizer, context_messages_random_new, user_inputs_batch)
    prompts_batch = prompts + prompts_random_new
    t5 = time.perf_counter()
    third_outputs = llm.generate(prompts_batch, sampling_params)
    t6 = time.perf_counter()
    logger.info(f"\n\nBatched request Time: {t6 - t5} seconds\n\n")
    with open(output_file, "a") as f:
        f.write(f"\n\nBatched request:\n\n")
    append_outputs(output_file, third_outputs, context_length+context_length_random, t6 - t5)

# Graceful exit
close_lmcache_engine()
subprocess.run("lsof -i:65431 -t | xargs kill -9", shell=True)