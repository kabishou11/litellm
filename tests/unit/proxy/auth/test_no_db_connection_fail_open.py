"""Regression for #38121: nil Prisma client must fail open, not 401-as-auth."""

import pytest
from fastapi import Request
from starlette.datastructures import Headers

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.auth_exception_handler import (
    DB_UNAVAILABLE_FALLBACK_USER_ID,
    UserAPIKeyAuthExceptionHandler,
)
from litellm.proxy.auth.resolvers.exceptions import NoDatabaseConnectionError
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler


def test_no_database_connection_error_classified_only_when_database_configured(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings",
        {},
        raising=False,
    )
    assert PrismaDBExceptionHandler.is_database_connection_error(NoDatabaseConnectionError()) is False

    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    assert PrismaDBExceptionHandler.is_database_connection_error(NoDatabaseConnectionError()) is True


@pytest.mark.asyncio
async def test_auth_handler_fail_opens_on_no_database_connection_error(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {"allow_requests_on_db_unavailable": True, "database_url": "postgresql://example/db"},
    )

    class _Svc:
        def service_failure_hook(self, **kwargs):
            return None

    class _Logging:
        service_logging_obj = _Svc()

    monkeypatch.setattr(proxy_server, "proxy_logging_obj", _Logging())

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/models",
            "headers": Headers({}).raw,
            "client": ("127.0.0.1", 123),
            "server": ("test", 80),
            "scheme": "http",
            "query_string": b"",
        }
    )
    identity = await UserAPIKeyAuthExceptionHandler.handle_authentication_error(
        e=NoDatabaseConnectionError(),
        request=request,
        request_data={},
        route="/v1/models",
        parent_otel_span=None,
        api_key="sk-test",
    )
    assert identity.user_id == DB_UNAVAILABLE_FALLBACK_USER_ID
    assert identity.user_role == LitellmUserRoles.INTERNAL_USER
