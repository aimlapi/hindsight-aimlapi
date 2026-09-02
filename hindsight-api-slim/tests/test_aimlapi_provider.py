"""Tests for the aimlapi.com OpenAI-compatible LLM provider and its attribution headers."""

import re

import pytest


def test_aimlapi_config_has_expected_default_model(monkeypatch):
    """HindsightConfig should default aimlapi to a model that exists in the catalog."""
    from hindsight_api.config import PROVIDER_DEFAULT_MODELS, HindsightConfig, clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "aimlapi")
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL", raising=False)
    clear_config_cache()

    try:
        assert PROVIDER_DEFAULT_MODELS["aimlapi"] == "openai/gpt-5-mini"
        config = HindsightConfig.from_env()
        assert config.llm_provider == "aimlapi"
        assert config.llm_model == "openai/gpt-5-mini"
    finally:
        clear_config_cache()


def test_aimlapi_llm_provider_from_env_has_expected_default_model(monkeypatch):
    """LLMProvider.from_env should use the aimlapi provider default model and base URL."""
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.llm_wrapper import LLMProvider

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "aimlapi")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "test-key")
    monkeypatch.delenv("HINDSIGHT_API_LLM_MODEL", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_LLM_BASE_URL", raising=False)
    clear_config_cache()

    try:
        llm = LLMProvider.from_env()
        assert llm.provider == "aimlapi"
        assert llm.model == "openai/gpt-5-mini"
        assert llm.base_url == "https://api.aimlapi.com/v1"
    finally:
        clear_config_cache()


def test_aimlapi_requires_api_key():
    """aimlapi is a cloud gateway and should require an API key."""
    from hindsight_api.engine.llm_wrapper import requires_api_key

    assert requires_api_key("aimlapi") is True


def test_aimlapi_uses_openai_compatible_provider_with_default_base_url():
    """The provider factory should route aimlapi to OpenAICompatibleLLM."""
    from hindsight_api.engine.llm_wrapper import LLMProvider
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    llm = LLMProvider(
        provider="aimlapi",
        api_key="test-key",
        base_url="",
        model="openai/gpt-5-mini",
    )

    assert llm.provider == "aimlapi"
    assert llm.model == "openai/gpt-5-mini"
    # /v1/completions does not exist on this gateway; only the /v1 root is declared
    # and the OpenAI SDK appends /chat/completions to it.
    assert llm.base_url == "https://api.aimlapi.com/v1"
    assert not llm.base_url.endswith("/")
    assert isinstance(llm._provider_impl, OpenAICompatibleLLM)
    assert llm._provider_impl.base_url == "https://api.aimlapi.com/v1"


def test_aimlapi_rejects_missing_api_key():
    """aimlapi should fail fast without an API key, matching the other cloud gateways."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    with pytest.raises(ValueError, match="API key is required for aimlapi"):
        LLMProvider(
            provider="aimlapi",
            api_key="",
            base_url="",
            model="openai/gpt-5-mini",
        )


def test_aimlapi_partner_id_matches_gateway_pattern():
    """A malformed partner id is dropped silently by the gateway, so assert its shape here.

    The gateway accepts ``^part_[A-Za-z0-9]{1,64}$`` and treats anything else as
    untagged traffic without failing the request — a typo would be invisible at
    runtime, which is why it is pinned in a test rather than only in review.
    """
    from hindsight_api.config import AIMLAPI_ATTRIBUTION_HEADERS

    partner_id = AIMLAPI_ATTRIBUTION_HEADERS["X-AIMLAPI-Partner-ID"]
    assert re.fullmatch(r"part_[A-Za-z0-9]{1,64}", partner_id), partner_id
    # <channel>/<client>, channel from a closed enum {web, agent, mcp}.
    assert re.fullmatch(r"agent/[a-z0-9-]{1,32}", AIMLAPI_ATTRIBUTION_HEADERS["X-AIMLAPI-Source"])
    # HTTP-Referer / X-Title identify the calling application (Hindsight), not the gateway.
    assert AIMLAPI_ATTRIBUTION_HEADERS["HTTP-Referer"] == "https://github.com/vectorize-io/hindsight"
    assert AIMLAPI_ATTRIBUTION_HEADERS["X-Title"] == "Hindsight"


def test_aimlapi_attribution_headers_reach_the_client():
    """The OpenAI client for aimlapi should carry all four attribution headers."""
    from hindsight_api.config import AIMLAPI_ATTRIBUTION_HEADERS
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(
        provider="aimlapi",
        api_key="test-key",
        base_url="",
        model="openai/gpt-5-mini",
        reasoning_effort=None,
    )

    sent = llm._client.default_headers
    for header, value in AIMLAPI_ATTRIBUTION_HEADERS.items():
        assert sent.get(header) == value


def test_aimlapi_attribution_does_not_leak_to_other_providers():
    """Attribution must never ride a request to a different provider."""
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(
        provider="openrouter",
        api_key="test-key",
        base_url="",
        model="qwen/qwen3.5-9b",
        reasoning_effort=None,
    )

    assert "X-AIMLAPI-Partner-ID" not in llm._client.default_headers


def test_aimlapi_attribution_is_scoped_to_the_aimlapi_host():
    """A proxy fronting the same wire format must not receive our attribution."""
    from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(
        provider="aimlapi",
        api_key="test-key",
        base_url="https://gateway.internal.example.com/v1",
        model="openai/gpt-5-mini",
        reasoning_effort=None,
    )

    assert "X-AIMLAPI-Partner-ID" not in llm._client.default_headers


def test_aimlapi_attribution_merges_under_operator_headers():
    """Operator-supplied default headers win, and the shared constant is never mutated."""
    from hindsight_api.config import AIMLAPI_ATTRIBUTION_HEADERS, aimlapi_default_headers

    before = dict(AIMLAPI_ATTRIBUTION_HEADERS)
    operator = {"X-Title": "Operator Override", "X-Component-Id": "hindsight"}

    merged = aimlapi_default_headers("aimlapi", "https://api.aimlapi.com/v1", operator)

    assert merged is not None
    assert merged["X-Title"] == "Operator Override"
    assert merged["X-Component-Id"] == "hindsight"
    assert merged["X-AIMLAPI-Partner-ID"] == AIMLAPI_ATTRIBUTION_HEADERS["X-AIMLAPI-Partner-ID"]
    # Neither the module constant nor the caller's dict was mutated in place.
    assert AIMLAPI_ATTRIBUTION_HEADERS == before
    assert operator == {"X-Title": "Operator Override", "X-Component-Id": "hindsight"}


def test_aimlapi_embeddings_carry_attribution(monkeypatch):
    """The embeddings surface talks to the same gateway and should be tagged too."""
    from hindsight_api.config import AIMLAPI_ATTRIBUTION_HEADERS, clear_config_cache
    from hindsight_api.engine.embeddings import create_embeddings_from_env

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "aimlapi")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_AIMLAPI_API_KEY", "test-key")
    clear_config_cache()

    try:
        embeddings = create_embeddings_from_env()
        assert embeddings.base_url == "https://api.aimlapi.com/v1"
        assert embeddings.model == "openai/text-embedding-3-small"
        assert embeddings.default_headers == AIMLAPI_ATTRIBUTION_HEADERS
        assert embeddings.default_headers is not AIMLAPI_ATTRIBUTION_HEADERS
    finally:
        clear_config_cache()


def test_aimlapi_embeddings_require_a_key(monkeypatch):
    """Missing key should name every env var that can supply one."""
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.embeddings import create_embeddings_from_env

    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "aimlapi")
    monkeypatch.delenv("HINDSIGHT_API_EMBEDDINGS_AIMLAPI_API_KEY", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_AIMLAPI_API_KEY", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_LLM_API_KEY", raising=False)
    clear_config_cache()

    try:
        with pytest.raises(ValueError, match="HINDSIGHT_API_EMBEDDINGS_AIMLAPI_API_KEY"):
            create_embeddings_from_env()
    finally:
        clear_config_cache()
