from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, results_1, results_2):
        raise NotImplementedError()

