from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import (
    as_str_mapping,
    finite_cost,
    provider_reported_usage_cost,
    set_response_cost_in_hidden_params,
)
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.types.rerank import RerankResponse, RerankResponseUsage

_RERANK_RESPONSE_USAGE: Final = TypeAdapter(RerankResponseUsage)


def get_optional_rerank_params(
    rerank_provider_config: BaseRerankConfig,
    model: str,
    drop_params: bool,
    query: str,
    documents: list[str | dict[str, object]],
    custom_llm_provider: str | None = None,
    top_n: int | None = None,
    rank_fields: list[str] | None = None,
    return_documents: bool | None = True,
    max_chunks_per_doc: int | None = None,
    max_tokens_per_doc: int | None = None,
    instruction: str | None = None,
    non_default_params: dict | None = None,
) -> dict:
    all_non_default_params: Final = non_default_params or {}
    if query is not None:
        all_non_default_params["query"] = query
    if top_n is not None:
        all_non_default_params["top_n"] = top_n
    if documents is not None:
        all_non_default_params["documents"] = documents
    if return_documents is not None:
        all_non_default_params["return_documents"] = return_documents
    if max_chunks_per_doc is not None:
        all_non_default_params["max_chunks_per_doc"] = max_chunks_per_doc
    if max_tokens_per_doc is not None:
        all_non_default_params["max_tokens_per_doc"] = max_tokens_per_doc
    if instruction is not None:
        # Also surfaced in non_default_params so providers that read it from
        # there (e.g. DeepInfra) keep working now that `rerank()` consumes
        # `instruction` as a named param instead of leaving it in **kwargs.
        all_non_default_params["instruction"] = instruction
    return rerank_provider_config.map_cohere_rerank_params(
        model=model,
        drop_params=drop_params,
        query=query,
        documents=documents,
        custom_llm_provider=custom_llm_provider,
        top_n=top_n,
        rank_fields=rank_fields,
        return_documents=return_documents,
        max_chunks_per_doc=max_chunks_per_doc,
        max_tokens_per_doc=max_tokens_per_doc,
        instruction=instruction,
        non_default_params=all_non_default_params,
    )


def rerank_response_usage(body: Mapping[str, object]) -> RerankResponseUsage | None:
    raw: Final = as_str_mapping(body.get("usage"))
    if raw is None:
        return None
    try:
        parsed: Final = _RERANK_RESPONSE_USAGE.validate_python(raw)
    except ValidationError:
        return None
    cost: Final = parsed.get("cost")
    if isinstance(cost, float) and finite_cost(cost) is None:
        return _RERANK_RESPONSE_USAGE.validate_python({field: value for field, value in raw.items() if field != "cost"})
    return parsed


def apply_rerank_provider_usage(
    response: RerankResponse,
    provider_body: Mapping[str, object],
    custom_llm_provider: str | None = None,
) -> RerankResponse:
    usage: Final = rerank_response_usage(provider_body)
    if usage is None:
        billed: Final = response.usage
        if billed is not None:
            set_response_cost_in_hidden_params(response, provider_reported_usage_cost(billed, custom_llm_provider))
        return response
    stamped: Final = response.model_copy(update={"usage": usage})
    set_response_cost_in_hidden_params(stamped, provider_reported_usage_cost(usage, custom_llm_provider))
    return stamped


def apply_rerank_provider_usage_from_httpx(
    response: RerankResponse,
    raw_response: httpx.Response,
    custom_llm_provider: str | None = None,
) -> RerankResponse:
    body: Final = as_str_mapping(raw_response.json())
    if body is None:
        return response
    return apply_rerank_provider_usage(response, body, custom_llm_provider)
