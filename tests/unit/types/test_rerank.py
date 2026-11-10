from typing import Final

from litellm.types.rerank import RerankResponse


def test_rerank_response_keeps_result_order_and_null_meta_fields() -> None:
    payload: Final = {
        "id": "ab0fcca0-b617-11ef-b292-0242ac110002",
        "results": [
            {"index": 2, "relevance_score": 0.9958819150924683},
            {"index": 0, "relevance_score": 0.001293411129154265},
            {"index": 1, "relevance_score": 7.641685078851879e-05},
            {"index": 3, "relevance_score": 7.621097756782547e-05},
        ],
        "meta": {"api_version": None, "billed_units": None, "tokens": None},
    }

    assert RerankResponse(**payload).model_dump() == payload


def test_rerank_response_dump_keeps_provider_usage_cost() -> None:
    payload: Final = {
        "id": "rerank-1",
        "results": [{"index": 0, "relevance_score": 0.5}],
        "meta": None,
        "usage": {"search_units": 1, "cost": 0.001},
    }

    assert RerankResponse(**payload).model_dump() == payload
