import pytest
from utils.chat_session import ChatSession
import time

CONTEXT_PATH = "data/f.txt"

def test():
    port1 = 8000
    port2 = 8001
    session1 = ChatSession(CONTEXT_PATH, port1)
    session2 = ChatSession(CONTEXT_PATH, port2)
    
    
    start = time.perf_counter()
    message1 = session1.chat("What's the document about?")
    end = time.perf_counter()
    print(f"\033[33mResult (unoptimized):\033[0m")
    print(message1)
    time1 = end-start
    print(f"\033[33mTime elapsed (unoptimized): {time1}\033[0m")
    
    # Fill the cache
    session2.chat("What's the document about?")
    print("Cache is being filled...")
    time.sleep(15)
    
    # TODO(Jiayi): might need to move te warm up into vllm
    # Warm up for prefix-caching kernel
    session2.chat("What's the document about?")
    
    start = time.perf_counter()
    message2 = session2.chat("What's the document about?")
    print(f"\033[33mResult (optimized):\033[0m")
    print(message2)
    end = time.perf_counter()
    time2 = end-start
    print(f"\033[33mTime elapsed (optimized): {time2}\033[0m")
    
    #time.sleep(200)
    
if __name__ == "__main__":
    test()