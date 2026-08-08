"""Structural isolation: no raw query_points/scroll/retrieve outside the wrapper."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tenantq.scoped_client import TenantScopedClient, _with_tenant

PKG = Path(__file__).resolve().parents[1] / "src" / "tenantq"

# Modules allowed to call the underlying client read APIs.
_ALLOWED_READ_MODULES = frozenset({"scoped_client.py"})

_READ_ATTRS = ("query_points", "scroll", "retrieve")


def _python_files():
    return sorted(PKG.glob("*.py"))


# Raw client receivers only. Wrapper calls (scoped.query_points) are the allowed path.
_RAW_CLIENT_READ = re.compile(
    r"(?<![\w.])(?:client|_client)\.(%s)\s*\(" % "|".join(_READ_ATTRS)
)


def test_no_raw_read_apis_outside_scoped_client():
    """A deliberately leaky call site in package code would fail this test.

    Direct ``client.query_points`` / ``_client.scroll`` / etc. are banned outside
    scoped_client.py. Using TenantScopedClient (e.g. ``scoped.query_points``) is
    required.
    """
    offenders: list[str] = []
    for path in _python_files():
        if path.name in _ALLOWED_READ_MODULES:
            continue
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if _RAW_CLIENT_READ.search(line):
                offenders.append(f"{path.name}:{i}: {stripped}")
    assert not offenders, (
        "tenant-scoped reads must go through TenantScopedClient; "
        "raw client reads found:\n  " + "\n  ".join(offenders)
    )


def test_leaky_query_pattern_is_detected_by_guard():
    """Simulate a leaky code path: the guard regex must match it.

    This is the acceptance case: if someone adds ``client.query_points(...)``
    without the wrapper, CI fails.
    """
    leaky = "    res = client.query_points(collection_name=c, query=q)\n"
    ok = "    res = scoped.query_points(tenant_id=t, collection_name=c, query=q)\n"
    assert _RAW_CLIENT_READ.search(leaky)
    assert not _RAW_CLIENT_READ.search(ok)


def test_search_module_uses_scoped_client_not_raw():
    text = (PKG / "search.py").read_text(encoding="utf-8")
    assert "TenantScopedClient" in text
    assert "scoped.query_points" in text
    assert "client.query_points" not in text


def test_with_tenant_requires_non_empty():
    with pytest.raises(ValueError, match="tenant_id"):
        _with_tenant("", None)
    with pytest.raises(ValueError, match="tenant_id"):
        _with_tenant("   ", None)


def test_with_tenant_injects_condition():
    from tenantq.config import TENANT_FIELD

    f = _with_tenant("acme", None)
    assert f.must
    cond = f.must[0]
    assert cond.key == TENANT_FIELD
    assert cond.match.value == "acme"


def test_scoped_client_requires_tenant_id_kwarg(ingested, settings):
    scoped = TenantScopedClient(ingested)
    with pytest.raises(TypeError):
        # tenant_id is keyword-only required
        scoped.query_points(collection_name=settings.collection, query=[0.0] * 8)  # type: ignore[call-arg]
