from typing import Final

from litellm.exceptions import RejectedRequestError


def test_rejected_request_error_stamps_rejected_by_hook_on_live_request_metadata():
    request_data: Final = {"model": "m"}
    err: Final = RejectedRequestError(
        message="blocked",
        model="m",
        llm_provider="",
        request_data=request_data,
    )
    assert err.rejected_by_hook == "pre_call_hook"
    assert request_data["metadata"]["rejected_by_hook"] == "pre_call_hook"
    assert err.request_data is request_data


def test_rejected_request_error_keeps_existing_metadata_keys():
    request_data: Final = {"model": "m", "metadata": {"user_api_key": "sk-test"}}
    err: Final = RejectedRequestError(
        message="blocked",
        model="m",
        llm_provider="",
        request_data=request_data,
        rejected_by_hook="custom_hook",
    )
    assert err.rejected_by_hook == "custom_hook"
    assert request_data["metadata"] == {
        "user_api_key": "sk-test",
        "rejected_by_hook": "custom_hook",
    }
