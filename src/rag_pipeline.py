from dataclasses import dataclass

from src.config import RetrievalConfig
from src.decision import DecisionAgent
from src.retriever import RetrieverAgent
from src.utils import timed


@dataclass
class SimpleRAGOutput:
    query: str
    retrieved_rows: list[dict]
    decision: dict
    timings: dict


class SimpleBankingRiskRAG:
    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self.retriever = RetrieverAgent(config.embedding_key)
        self.decision = DecisionAgent(config.ollama_model)

    def run(self, query: str) -> SimpleRAGOutput:
        timings: dict = {}

        with timed() as timer:
            retrieval = self.retriever.retrieve(query, top_k=self.config.top_k)
        timings["retrieval_seconds"] = timer["elapsed_seconds"]

        with timed() as timer:
            decision = self.decision.decide(query, retrieval.rows)
        timings["decision_seconds"] = timer["elapsed_seconds"]
        timings["total_seconds"] = sum(timings.values())

        return SimpleRAGOutput(
            query=query,
            retrieved_rows=retrieval.rows,
            decision=decision,
            timings=timings,
        )
