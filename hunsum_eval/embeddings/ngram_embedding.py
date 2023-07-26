from abc import ABC
from dataclasses import dataclass

from typing import List, Tuple


@dataclass
class NgramEmbedding:
    embedding: List[float]
    ngram: Tuple[str]
    count: int
