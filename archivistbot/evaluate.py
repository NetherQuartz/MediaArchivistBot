import argparse
import asyncio
import json
from pathlib import Path
from typing import NotRequired, TypedDict

from .database import session_scope
from .search import SearchService, active_chat_ids


class EvaluationCase(TypedDict):
    query: str
    expected_file_unique_ids: list[str]
    allowed_chat_ids: NotRequired[list[int]]


async def evaluate(path: Path, limit: int) -> dict[str, object]:
    cases: list[EvaluationCase] = json.loads(path.read_text(encoding="utf-8"))
    service = SearchService()
    reciprocal_ranks: list[float] = []
    hits = 0
    details: list[dict[str, object]] = []

    for case in cases:
        with session_scope() as session:
            allowed = case.get("allowed_chat_ids") or active_chat_ids(session)
            results = await service.search(session, case["query"], allowed)
        returned = [
            result.file_unique_id
            for result in results[:limit]
            if result.file_unique_id is not None
        ]
        expected = set(case["expected_file_unique_ids"])
        rank = next(
            (
                index
                for index, file_unique_id in enumerate(returned, start=1)
                if file_unique_id in expected
            ),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
        details.append(
            {
                "query": case["query"],
                "rank": rank,
                "returned_file_unique_ids": returned,
            }
        )

    count = len(cases)
    return {
        f"recall_at_{limit}": hits / count if count else 0,
        "mrr": sum(reciprocal_ranks) / count if count else 0,
        "queries": count,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("cases", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    arguments = parser.parse_args()
    result = asyncio.run(evaluate(arguments.cases, arguments.limit))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
