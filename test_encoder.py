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
    
    # Test whether filling cache affect TTFT
    start = time.perf_counter()
    message2 = session2.chat("What's the document about?")
    print(f"\033[33mResult (during caching):\033[0m")
    print(message2)
    end = time.perf_counter()
    time2 = end-start
    print(f"\033[33mTime elapsed (during caching): {time2}\033[0m")
    
    assert message1==message2

    
if __name__ == "__main__":
    test()