#!/usr/bin/env python3
"""M1 验收脚本：调 LLM → 结构化校验 → Token 记账。

用法：
    python selfcheck.py           # 真实调用（需要 .env 里的 API key）
    python selfcheck.py --mock    # 离线 mock（无需 key，验证管线正确性）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.llm import LLMClient  # noqa: E402
from core.schemas import DifficultyAssessment  # noqa: E402
from core.usage import UsageTracker  # noqa: E402

SAMPLE_QUESTION = (
    "A 62-year-old man with a history of hypertension presents with acute onset "
    "of tearing chest pain radiating to the back. Blood pressure is 180/100 mmHg "
    "in the right arm and 140/80 mmHg in the left arm. What is the most likely "
    "diagnosis among: (A) Acute myocardial infarction (B) Aortic dissection "
    "(C) Pulmonary embolism (D) Esophageal rupture?"
)


class MockClientFactory:
    """离线 mock：结构化请求返回合法 JSON，普通请求返回 OK。"""

    def __call__(self, provider):
        class _Completions:
            async def create(self, **kwargs):
                await asyncio.sleep(0.01)
                if kwargs.get("response_format"):
                    content = json.dumps({
                        "difficulty": "hard",
                        "justification": (
                            "Requires differentiating vascular emergency with "
                            "asymmetric blood pressure: multi-step reasoning."
                        ),
                    })
                else:
                    content = "OK"
                usage = SimpleNamespace(
                    prompt_tokens=128,
                    completion_tokens=24,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=32),
                )
                msg = SimpleNamespace(content=content)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=msg)], usage=usage
                )

        class _Client:
            chat = SimpleNamespace(completions=_Completions())

        return _Client()


async def run(client_factory=None) -> int:
    tracker = UsageTracker()
    llm = LLMClient.from_yaml(
        str(ROOT / "config.yaml"), tracker, client_factory=client_factory
    )

    print("== Check 1: plain chat ==")
    resp = await llm.chat(
        [{"role": "user", "content": "Reply with exactly: OK"}],
        agent_name="selfcheck", agent_type="system",
    )
    print(f"  content={resp.content!r}")
    print(f"  provider={resp.provider} model={resp.model} latency={resp.latency_ms}ms")
    assert resp.content, "plain chat returned empty content"

    print("== Check 2: structured output (pydantic 校验) ==")
    resp2 = await llm.chat(
        [{"role": "user", "content": f"Assess this medical question:\n{SAMPLE_QUESTION}"}],
        agent_name="TriageAgent",
        agent_type="triage",
        response_model=DifficultyAssessment,
    )
    parsed: DifficultyAssessment = resp2.parsed  # type: ignore[assignment]
    print(f"  parsed -> {parsed.model_dump_json()}")
    assert parsed.difficulty in ("easy", "medium", "hard")

    print("== Check 3: token accounting ==")
    print(tracker.summary())
    assert tracker.totals().requests >= 2

    print("\nM1 SELFCHECK PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="MedBoard M1 selfcheck")
    parser.add_argument(
        "--mock", action="store_true",
        help="offline mock mode (no API key needed)",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    factory = MockClientFactory() if args.mock else None
    mode = "MOCK (offline)" if args.mock else "LIVE"
    print(f"[medboard M1 selfcheck] mode={mode}\n")
    try:
        sys.exit(asyncio.run(run(factory)))
    except AssertionError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
