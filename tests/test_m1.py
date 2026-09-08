"""M1 单元测试：usage 记账、结构化解析、mock 客户端全流程、failover。

运行：cd medboard && python -m pytest tests/ -v
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.llm as llm_mod  # noqa: E402
from core.llm import LLMClient, extract_json_text, parse_structured  # noqa: E402
from core.schemas import DifficultyAssessment  # noqa: E402
from core.usage import UsageRecord, UsageTracker  # noqa: E402


def _record(agent_type: str, inp: int, out: int, round_num=0) -> UsageRecord:
    return UsageRecord(
        agent_name=f"{agent_type}-agent",
        agent_type=agent_type,
        round_num=round_num,
        provider="deepseek",
        model="deepseek-chat",
        input_tokens=inp,
        output_tokens=out,
        cached_tokens=8,
        latency_ms=123,
    )


# ----------------------------------------------------------------记账
def test_tracker_totals_and_by_type():
    tracker = UsageTracker()
    tracker.record(_record("triage", 100, 20))
    tracker.record(_record("expert", 200, 40, round_num=0))
    tracker.record(_record("expert", 300, 60, round_num=1))

    totals = tracker.totals()
    assert totals.requests == 3
    assert totals.input_tokens == 600
    assert totals.output_tokens == 120
    assert totals.total_tokens == 720
    assert totals.cached_tokens == 24

    by_type = tracker.by_agent_type()
    assert by_type["expert"].requests == 2
    assert by_type["expert"].total_tokens == 600

    assert "total=720" in tracker.summary()
    assert tracker.to_dict()[0]["agent_type"] == "triage"


# ----------------------------------------------------------------解析
def test_parse_structured_ok():
    text = '{"difficulty": "hard", "justification": "complex case"}'
    obj, err = parse_structured(text, DifficultyAssessment)
    assert err is None and obj is not None
    assert obj.difficulty == "hard"


def test_parse_structured_with_fences():
    text = '```json\n{"difficulty": "easy", "justification": "basic fact"}\n```'
    obj, err = parse_structured(text, DifficultyAssessment)
    assert err is None and obj is not None


def test_parse_structured_invalid_json():
    obj, err = parse_structured("not json at all", DifficultyAssessment)
    assert obj is None and "invalid JSON" in err


def test_parse_structured_schema_violation():
    obj, err = parse_structured('{"difficulty": "impossible"}', DifficultyAssessment)
    assert obj is None and err


def test_extract_json_text_strips_prose():
    text = 'Here you go: {"difficulty": "medium", "justification": "x"} hope it helps'
    assert extract_json_text(text).startswith("{")


# ----------------------------------------------------------------mock 全流程
class OkFactory:
    """正常 mock：结构化返回合法 JSON。"""

    def __call__(self, provider):
        class _Completions:
            async def create(self, **kwargs):
                if kwargs.get("response_format"):
                    content = json.dumps(
                        {"difficulty": "medium", "justification": "mock"}
                    )
                else:
                    content = "OK"
                usage = SimpleNamespace(
                    prompt_tokens=50, completion_tokens=10,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=5),
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                    usage=usage,
                )

        return SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions())
        )


def _cfg():
    return {
        "providers": {
            "p1": {"base_url": "http://x", "api_key_env": "K1", "model": "m1"},
            "p2": {"base_url": "http://y", "api_key_env": "K2", "model": "m2"},
        },
        "defaults": {"temperature": 0.1, "max_tokens": 64, "timeout_seconds": 5},
        "policy": {
            "primary": "p1", "failover": ["p2"], "max_concurrency": 4,
            "max_retries": 2, "structured_retries": 1,
            "backoff_base": 0.01, "backoff_cap": 0.02,
            "circuit_failure_threshold": 3, "circuit_cooldown_seconds": 60,
        },
    }


def test_client_structured_flow_with_mock():
    tracker = UsageTracker()
    client = LLMClient(_cfg(), tracker, client_factory=OkFactory())
    resp = asyncio.run(
        client.chat(
            [{"role": "user", "content": "assess"}],
            agent_name="TriageAgent", agent_type="triage",
            response_model=DifficultyAssessment,
        )
    )
    assert resp.parsed is not None and resp.parsed.difficulty == "medium"
    assert resp.provider == "p1"
    recs = tracker.records
    assert len(recs) == 1
    assert recs[0].agent_type == "triage"
    assert recs[0].input_tokens == 50 and recs[0].cached_tokens == 5


def test_client_failover_to_backup_provider(monkeypatch):
    # 让 ValueError 视为瞬时错误，方便注入故障
    monkeypatch.setattr(llm_mod, "RETRYABLE_ERRORS", (ValueError,))

    class FlakyThenOkFactory:
        def __init__(self):
            self.calls = {"p1": 0, "p2": 0}

        def __call__(self, provider):
            outer = self

            class _Completions:
                async def create(self, **kwargs):
                    outer.calls[provider.name] += 1
                    if provider.name == "p1":
                        raise ValueError("simulated p1 outage")
                    content = json.dumps(
                        {"difficulty": "easy", "justification": "from backup"}
                    )
                    usage = SimpleNamespace(
                        prompt_tokens=10, completion_tokens=5,
                        prompt_tokens_details=None,
                    )
                    return SimpleNamespace(
                        choices=[SimpleNamespace(
                            message=SimpleNamespace(content=content))],
                        usage=usage,
                    )

            return SimpleNamespace(
                chat=SimpleNamespace(completions=_Completions())
            )

    tracker = UsageTracker()
    client = LLMClient(_cfg(), tracker, client_factory=FlakyThenOkFactory())
    resp = asyncio.run(
        client.chat(
            [{"role": "user", "content": "assess"}],
            agent_name="TriageAgent", agent_type="triage",
            response_model=DifficultyAssessment,
        )
    )
    # 主模型重试耗尽后 failover 到 p2，且结果来自 p2
    assert resp.provider == "p2"
    assert resp.parsed.difficulty == "easy"
    p1 = client.providers[0]
    assert p1.consecutive_failures == 1  # 记了一次失败（未达熔断阈值 3）


def test_client_all_providers_failed(monkeypatch):
    monkeypatch.setattr(llm_mod, "RETRYABLE_ERRORS", (ValueError,))

    class AlwaysFailFactory:
        def __call__(self, provider):
            class _Completions:
                async def create(self, **kwargs):
                    raise ValueError("down")

            return SimpleNamespace(
                chat=SimpleNamespace(completions=_Completions())
            )

    client = LLMClient(_cfg(), UsageTracker(), client_factory=AlwaysFailFactory())
    with pytest.raises(llm_mod.LLMError, match="all providers failed"):
        asyncio.run(
            client.chat([{"role": "user", "content": "hi"}], agent_type="system")
        )
