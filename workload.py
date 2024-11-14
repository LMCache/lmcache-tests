from dataclasses import dataclass
import abc
from typing import List

from configs import WorkloadConfig, Usecase
import utils

@dataclass
class Request:
    timestamp: float
    context: str
    question: str

class WorkloadGenerator(metaclass=abc.ABCMeta):
    def __init__(self, config: WorkloadConfig):
        self.config = config

    @abc.abstractmethod
    def generate(self) -> List[Request]:
        pass

def CreateWorkloadGenerator(config: WorkloadConfig, usecase: Usecase, max_context_length: int) -> WorkloadGenerator:
    match usecase:
        case Usecase.DUMMY:
            return DumbWorkloadGenerator(config, max_context_length)
        case Usecase.MULTI:
            return MultiTurnWorkloadGenerator(config)
        case Usecase.VARY:
            return VaryLengthWorkloadGenerator(config, max_context_length)
        case _:
            raise NotImplementedError(f"Usecase {usecase} not implemented")


class DumbWorkloadGenerator(WorkloadGenerator):
    """
    Generate dummy requests with the same context and question.
    """
    def __init__(self, config: WorkloadConfig, max_context_length: int):
        super().__init__(config)
        self.dummy_context = "This is some dummy text. "
        self.estimated_num_tokens_context = utils.estimate_num_tokens(self.dummy_context)
        dummy_question = "Index 0. Question: How are you doing today?"
        self.estimated_num_tokens_question = utils.estimate_num_tokens(dummy_question)
        self.max_context_length = max_context_length

    def generate_context(self) -> str:
        context_length = min(self.max_context_length - self.config.query_length, self.config.context_length)
        return self.dummy_context * (context_length // self.estimated_num_tokens_context)

    def generate_question(self, index: int) -> str:
        if self.config.query_length - self.estimated_num_tokens_question > 0:
            question_prefix = self.dummy_context * ((self.config.query_length - self.estimated_num_tokens_question) // self.estimated_num_tokens_context)
        return f"Index {index}. {question_prefix} Question: How are you doing today?"

    def generate(self) -> List[Request]:
        num_requests = int(self.config.duration * self.config.qps)
        
        ret = []
        for i in range(num_requests):
            timestamp = i / self.config.qps + self.config.offset
            ret.append(Request(
                timestamp=timestamp,
                context=self.generate_context(),
                question=self.generate_question(i)
            ))

        return ret
    
class VaryLengthWorkloadGenerator(WorkloadGenerator):
    """
    Generate vary length requests from the same context.
    """
    def __init__(self, config: WorkloadConfig, max_context_length: int):
        super().__init__(config)
        self.dummy_context = "This is some dummy text. "
        self.estimated_num_tokens_context = utils.estimate_num_tokens(self.dummy_context)
        dummy_question = "Index 0. Question: How are you doing today?"
        self.estimated_num_tokens_question = utils.estimate_num_tokens(dummy_question)
        self.index = 0
        self.max_context_length = max_context_length

    def generate_context(self) -> str:
        self.index += 1
        # The context length pattern: [a 2a 2a 3a 3a 4a 4a ...]
        context_length = self.config.context_length * ((self.index // 2) + 1)
        context_length = min(self.max_context_length - self.config.query_length, context_length)
        return self.dummy_context * (context_length // self.estimated_num_tokens_context)

    def generate_question(self, index: int) -> str:
        if self.config.query_length - self.estimated_num_tokens_question > 0:
            question_prefix = self.dummy_context * ((self.config.query_length - self.estimated_num_tokens_question) // self.estimated_num_tokens_context)
        return f"Index {index}. {question_prefix} Question: How are you doing today?"

    def generate(self) -> List[Request]:
        num_requests = int(self.config.duration * self.config.qps)
        
        ret = []
        for i in range(num_requests):
            timestamp = i / self.config.qps + self.config.offset
            ret.append(Request(
                timestamp=timestamp,
                context=self.generate_context(),
                question=self.generate_question(i)
            ))

        return ret
    
class MultiTurnWorkloadGenerator(WorkloadGenerator):
    """
    Generate multi turn requests with the last response added to the next context.
    """
    def __init__(self, config: WorkloadConfig):
        super().__init__(config)
        self.dummy_context = "This is some dummy text. "
        self.estimated_num_tokens_context = utils.estimate_num_tokens(self.dummy_context)
        dummy_question = "Index x.  Question: Please write a very long essay about whatever topic. "
        self.estimated_num_tokens_question = utils.estimate_num_tokens(dummy_question)
        self.memory = ""
        self.offset = self.config.offset
        self.separator = "<<splitter>>"

    def generate_context(self) -> str:
        return self.memory

    def generate_question(self, index: int) -> str:
        if self.config.query_length - self.estimated_num_tokens_question > 0:
            question_prefix = self.dummy_context * ((self.config.query_length - self.estimated_num_tokens_question) // self.estimated_num_tokens_context)
        else:
            question_prefix = ""
        return f"Index x. {question_prefix} Question: Please write a very long essay about whatever topic. "

    def generate(self) -> List[Request]:
        num_requests = int(self.config.duration * self.config.qps)
        
        ret = []
        for i in range(num_requests):
            timestamp = i / self.config.qps + self.offset
            ret.append(Request(
                timestamp=timestamp,
                context=self.generate_context(),
                question=self.generate_question(i)
            ))

        return ret
    
    def store(self, memory: str) -> None:
        self.memory = f"{self.memory}{self.separator}{memory}"

#if __name__ == "__main__":
#    config = WorkloadConfig(
#        duration=5,
#        qps=2,
#        context_length=1024,
#        query_length=30
#    )
#    generator = DumbWorkloadGenerator(config)
#    requests = generator.generate()
#    for request in requests:
#        print(request)
