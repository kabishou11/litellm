from typing import Final

import httpx
import pytest

from litellm.cost_calculator import response_cost_calculator
from litellm.rerank_api.rerank_utils import apply_rerank_provider_usage, apply_rerank_provider_usage_from_httpx
from litellm.types.rerank import RerankResponse


def _cost(response: RerankResponse, model: str, provider: str) -> float | None:
    return response_cost_calculator(
        response_object=response,
        model=model,
        custom_llm_provider=provider,
        call_type="rerank",
        optional_params={},
    )


def test_apply_rerank_provider_usage_keeps_provider_cost() -> None:
    stamped: Final = apply_rerank_provider_usage(
        RerankResponse(id="rerank-1", results=[{"index": 0, "relevance_score": 0.9}]),
        {"id": "rerank-1", "results": [{"index": 0, "relevance_score": 0.9}], "usage": {"search_units": 1, "cost": 0.001}},
        "openrouter",
    )

    assert stamped.usage == {"search_units": 1, "cost": 0.001}
    assert _cost(stamped, "openrouter/scaleway/cohere/rerank-v3.5", "openrouter") == pytest.approx(0.001)


def test_apply_rerank_provider_usage_drops_non_finite_cost() -> None:
    stamped: Final = apply_rerank_provider_usage(
        RerankResponse(id="rerank-1", results=[{"index": 0, "relevance_score": 0.9}]),
        {"usage": {"search_units": 1, "cost": float("nan")}},
        "openrouter",
    )

    assert stamped.usage == {"search_units": 1}
    assert _cost(stamped, "not-a-real-rerank-model-47796", "openrouter") is None


def test_apply_rerank_provider_usage_from_httpx_reads_the_provider_body() -> None:
    stamped: Final = apply_rerank_provider_usage_from_httpx(
        RerankResponse(id="rerank-1", results=[{"index": 0, "relevance_score": 0.9}]),
        httpx.Response(200, json={"usage": {"search_units": 1, "cost": 0.001}}),
        "openrouter",
    )

    assert stamped.usage == {"search_units": 1, "cost": 0.001}
    assert _cost(stamped, "openrouter/scaleway/cohere/rerank-v3.5", "openrouter") == pytest.approx(0.001)
