"""Token 记账：按 agent / 类型 / 轮次归集每次 LLM 调用的成本与延迟。

设计要点：
- 每次真实 API 调用（含重试、解析失败重试）都记一条 UsageRecord——记账口径是
  「实际发生了什么」而不是「逻辑上几次调用」，评测成本才不会失真。
- cached_tokens 兼容 OpenAI 的 prompt_tokens_details.cached_tokens 与
  DeepSeek 的 prompt_cache_hit_tokens 两种返回格式。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class UsageRecord:
    """一次真实 LLM API 调用的用量。"""

    agent_name: str            # 如 "TriageAgent" / "ExpertAgent:Dr. Chen"
    agent_type: str            # triage / expert / moderator / system
    round_num: Optional[int]   # 会诊轮次（0 起）；非会诊调用为 None
    provider: str              # deepseek / qwen / ...
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: int = 0
    created_at: str = field(default_factory=_now_iso)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "round_num": self.round_num,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at,
        }


@dataclass
class UsageTotals:
    """聚合用量（按 agent / 按类型 / 全局）。"""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, rec: UsageRecord) -> None:
        self.requests += 1
        self.input_tokens += rec.input_tokens
        self.output_tokens += rec.output_tokens
        self.cached_tokens += rec.cached_tokens


class UsageTracker:
    """全局记账器：编排器持有，贯穿一次会诊 / 一次评测批跑。"""

    def __init__(self) -> None:
        self.records: List[UsageRecord] = []

    def record(self, rec: UsageRecord) -> UsageRecord:
        self.records.append(rec)
        return rec

    def totals(self) -> UsageTotals:
        t = UsageTotals()
        for rec in self.records:
            t.add(rec)
        return t

    def by_agent_type(self) -> Dict[str, UsageTotals]:
        out: Dict[str, UsageTotals] = {}
        for rec in self.records:
            out.setdefault(rec.agent_type, UsageTotals()).add(rec)
        return out

    def by_agent(self) -> Dict[str, UsageTotals]:
        out: Dict[str, UsageTotals] = {}
        for rec in self.records:
            out.setdefault(rec.agent_name, UsageTotals()).add(rec)
        return out

    def to_dict(self) -> List[Dict[str, Any]]:
        return [rec.to_dict() for rec in self.records]

    def summary(self) -> str:
        t = self.totals()
        lines = [
            "=== Token usage ===",
            f"requests={t.requests}  input={t.input_tokens:,}  output={t.output_tokens:,}"
            f"  total={t.total_tokens:,}  cached={t.cached_tokens:,}",
        ]
        by_type = self.by_agent_type()
        if by_type:
            lines.append("by agent type:")
            for agent_type, tot in sorted(by_type.items()):
                lines.append(
                    f"  {agent_type:<10} requests={tot.requests} input={tot.input_tokens:,}"
                    f" output={tot.output_tokens:,} total={tot.total_tokens:,}"
                )
        return "\n".join(lines)
