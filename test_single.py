import pytest
from utils.chat_session import ChatSession
import argparse
import time

CONTEXT_PATH = "data/f.txt"

def test(mode="optimized"):
    if mode == "original":
        port = 8000
    elif mode == "optimized":
        port = 8001
    session = ChatSession(CONTEXT_PATH, port)
    
    start = time.perf_counter()
    message = session.chat("What's the document about?", max_tokens=1)
    print(f"\033[33mResult (during caching):\033[0m")
    print(message)
    end = time.perf_counter()
    time1 = end-start
    print(f"\033[33mTime elapsed (during caching): {time1}\033[0m")
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test_single')
    parser.add_argument(
        '--mode', 
        choices=['original', "optimized"])
    args = parser.parse_args()
    mode = args.mode
    test(mode)