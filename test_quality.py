import pytest
import time
from pathlib import Path
import argparse
from utils.chat_session import ChatSession

CONTEXT_PATH = "data/f.txt"
RES_PATH = "temp_results/"


def test(query, num_gen, temperature, max_tokens):
    
    port1 = 8000
    port2 = 8001
    session1 = ChatSession(CONTEXT_PATH, port1)
    session2 = ChatSession(CONTEXT_PATH, port2)
    
    for i in range(num_gen):
        temp_path = f"{RES_PATH}/{temperature}/{num_gen}/"
        Path(temp_path).mkdir(parents=True, exist_ok=True)
        
        message1 = session1.chat("What's the document about?")
        print(f"\033[33mResult (unoptimized):\033[0m")
        print(message1)
        time1 = end-start
        print(f"\033[33mTime elapsed (unoptimized): {time1}\033[0m")
        
        org_path = Path(temp_path+f"org.txt")
        org_path.write_text(message1)
        
        # Fill the cache
        session2.chat("What's the document about?")
        print(f"\033Cache is being stored to remote. Waiting...\033[0m")
        time.sleep(300)
        
        start = time.perf_counter()
        message2 = session2.chat("What's the document about?")
        print(f"\033[33mResult (optimized):\033[0m")
        print(message2)
        end = time.perf_counter()
        time2 = end-start
        print(f"\033[33mTime elapsed (optimized): {time2}\033[0m")
        
        opt_path = Path(temp_path+f"opt.txt")
        opt_path.write_text(message2)
        
    
    #time.sleep(200)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument(
        '--num_gen', 
        type=int,
        default=1)

    parser.add_argument(
        '--temperature', 
        type=float,
        default=0.0)
    
    parser.add_argument(
        '--max_tokens', 
        type=int,
        default=30)
    
    parser.add_argument(
        '--query', 
        type=str,
        default="What's the document about?")
    
    #TODO(Jiayi): add different contexts

    args = parser.parse_args()
    num_gen = args.num_gen
    temperature = args.temperature
    max_tokens = args.max_new_tokens
    query = args.query

    test(query, num_gen, temperature, max_tokens)