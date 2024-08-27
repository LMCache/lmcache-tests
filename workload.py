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

def CreateWorkloadGenerator(config: WorkloadConfig, usecase: Usecase) -> WorkloadGenerator:
    match usecase:
        case Usecase.DUMMY:
            return DumbWorkloadGenerator(config)
        case _:
            raise NotImplementedError(f"Usecase {usecase} not implemented")


class DumbWorkloadGenerator(WorkloadGenerator):
    """
    Generate dummy requests with the same context and question.
    """
    def __init__(self, config: WorkloadConfig):
        super().__init__(config)
        self.dummy_context = "This is some dummy text. "
        self.estimated_num_tokens_context = utils.estimate_num_tokens(self.dummy_context)
        dummy_question = "Index 0. Question: How are you doing today?"
        self.estimated_num_tokens_question = utils.estimate_num_tokens(dummy_question)

    def generate_context(self) -> str:
        return self.dummy_context * (self.config.context_length // self.estimated_num_tokens_context)

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
