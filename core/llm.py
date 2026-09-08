"""M1 LLM 基础设施：OpenAI 兼容客户端 + 多厂商路由 + 结构化输出 + token 记账。

设计要点（对应工程化叙事，见 ../MedAgents自研-工程化落地方案.md）：
- 多厂商 router：主模型重试耗尽后自动 failover 到备模型；
  连续失败达阈值进入熔断冷却（简单版熔断器，完整版 D11 增强）。
- 结构化输出：JSON 指令 + response_format + pydantic 校验；
  校验失败把错误信息回灌给模型重试（比盲目重试命中率高）。
- 429/超时/断连：指数退避 + jitter 重试，尊重 retry-after 语义。
- 全局信号量：所有 provider 共享并发上限，保护下游。
- token 记账：每次真实 API 调用（含重试）都记 UsageRecord。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import yaml
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from .usage import UsageRecord, UsageTracker

logger = logging.getLogger(__name__)

# 瞬时错误：值得退避重试
RETRYABLE_ERRORS: Tuple[type, ...] = (RateLimitError, APITimeoutError, APIConnectionError)


class LLMError(RuntimeError):
    """LLM 调用最终失败（所有 provider 均不可用）。"""


class StructuredOutputError(LLMError):
    """结构化输出在重试后仍无法通过 pydantic 校验。"""


JSON_INSTRUCTION = (
    "You MUST reply with a single JSON object that exactly matches the requested "
    "schema. No prose, no markdown fences, no keys beyond the schema."
)


def extract_json_text(text: str) -> str:
    """从模型回复中提取 JSON 正文：容忍 markdown 代码块与前后杂文字。"""
    t = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, flags=re.DOTALL)
    if fence:
        return fence.group(1)
    if "{" in t and "}" in t:
        return t[t.index("{"): t.rindex("}") + 1]
    return t


def parse_structured(
    text: str, model_cls: Type[BaseModel]
) -> Tuple[Optional[BaseModel], Optional[str]]:
    """解析并校验结构化输出。成功返回 (obj, None)，失败返回 (None, 错误摘要)。"""
    try:
        data = json.loads(extract_json_text(text))
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    try:
        return model_cls.model_validate(data), None
    except ValidationError as e:
        return None, str(e)[:800]


@dataclass
class LLMResponse:
    """一次 chat 调用的结果（含解析后的结构化对象与用量）。"""

    content: str
    parsed: Optional[BaseModel]
    provider: str
    model: str
    usage: UsageRecord
    latency_ms: int


class Provider:
    """单个 LLM 厂商端点 + 简单熔断状态。"""

    def __init__(
        self,
        name: str,
        cfg: Dict[str, Any],
        client_factory: Optional[Callable[["Provider"], Any]] = None,
    ) -> None:
        self.name = name
        self.base_url = cfg["base_url"]
        self.model = cfg["model"]
        api_key_env = cfg.get("api_key_env", "")
        self.api_key = os.getenv(api_key_env, "") or "EMPTY"
        self.json_mode = bool(cfg.get("json_mode", True))
        self._client_factory = client_factory
        self._client: Any = None
        # 熔断状态（简单版：连续失败计数 + 冷却截止时间）
        self.consecutive_failures = 0
        self.cooldown_until = 0.0

    def get_client(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory(self)
            else:
                self._client = AsyncOpenAI(
                    base_url=self.base_url, api_key=self.api_key
                )
        return self._client

    def available(self) -> bool:
        return time.time() >= self.cooldown_until

    def note_failure(self, threshold: int, cooldown_seconds: float) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= threshold:
            self.cooldown_until = time.time() + cooldown_seconds
            logger.warning(
                "circuit OPEN for provider %s: %d consecutive failures, "
                "cooldown %.0fs",
                self.name, self.consecutive_failures, cooldown_seconds,
            )

    def note_success(self) -> None:
        self.consecutive_failures = 0
        self.cooldown_until = 0.0


class LLMClient:
    """统一 LLM 入口：路由 + 重试 + 结构化输出 + 记账。

    用法::

        tracker = UsageTracker()
        llm = LLMClient.from_yaml("config.yaml", tracker)
        resp = await llm.chat(
            [{"role": "user", "content": "..."}],
            agent_name="TriageAgent", agent_type="triage",
            response_model=TriageOutput,
        )
        triage = resp.parsed  # 已通过 pydantic 校验
    """

    def __init__(
        self,
        config: Dict[str, Any],
        tracker: UsageTracker,
        client_factory: Optional[Callable[[Provider], Any]] = None,
    ) -> None:
        policy = config.get("policy", {})
        defaults = config.get("defaults", {})

        self.tracker = tracker
        self.temperature = float(defaults.get("temperature", 0.1))
        self.max_tokens = int(defaults.get("max_tokens", 4096))
        self.timeout = float(defaults.get("timeout_seconds", 120))

        self.max_concurrency = int(policy.get("max_concurrency", 8))
        self.max_retries = int(policy.get("max_retries", 3))
        self.structured_retries = int(policy.get("structured_retries", 2))
        self.backoff_base = float(policy.get("backoff_base", 1.5))
        self.backoff_cap = float(policy.get("backoff_cap", 30.0))
        self.circuit_threshold = int(policy.get("circuit_failure_threshold", 3))
        self.circuit_cooldown = float(policy.get("circuit_cooldown_seconds", 60.0))

        self._semaphore = asyncio.Semaphore(self.max_concurrency)

        order = [policy.get("primary", "deepseek")] + list(policy.get("failover", []))
        providers_cfg = config.get("providers", {})
        self.providers: List[Provider] = []
        for name in order:
            if name in providers_cfg:
                self.providers.append(
                    Provider(name, providers_cfg[name], client_factory)
                )
        if not self.providers:
            raise LLMError("no provider configured")
        self.primary = self.providers[0]

    @classmethod
    def from_yaml(
        cls,
        path: str,
        tracker: UsageTracker,
        client_factory: Optional[Callable[[Provider], Any]] = None,
    ) -> "LLMClient":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg, tracker, client_factory)

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: List[Dict[str, str]],
        agent_name: str = "system",
        agent_type: str = "system",
        round_num: Optional[int] = None,
        response_model: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """调用 LLM。response_model 给定时做结构化输出（JSON + pydantic 校验）。

        失败语义：依次尝试可用 provider；全部失败抛 LLMError。
        """
        candidates = [p for p in self.providers if p.available()]
        if not candidates:
            logger.warning("all providers in cooldown; half-open probing primary")
            candidates = [self.primary]

        last_err: Optional[BaseException] = None
        for provider in candidates:
            try:
                return await self._chat_via(
                    provider, messages, agent_name, agent_type,
                    round_num, response_model, temperature, max_tokens,
                )
            except (LLMError, *RETRYABLE_ERRORS) as e:
                last_err = e
                logger.warning(
                    "provider %s failed (%s: %s), failover to next",
                    provider.name, type(e).__name__, str(e)[:200],
                )
        raise LLMError(f"all providers failed; last error: {last_err}")

    # ------------------------------------------------------------------
    # 内部：单 provider 一次完整调用（含结构化重试）
    # ------------------------------------------------------------------
    async def _chat_via(
        self,
        provider: Provider,
        messages: List[Dict[str, str]],
        agent_name: str,
        agent_type: str,
        round_num: Optional[int],
        response_model: Optional[Type[BaseModel]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> LLMResponse:
        base_msgs = self._with_json_instruction(messages, response_model)
        msgs = base_msgs

        last_parse_error: Optional[str] = None
        for attempt in range(self.structured_retries + 1):
            content, rec = await self._raw_call(
                provider, msgs, agent_name, agent_type,
                round_num, temperature, max_tokens,
                structured=response_model is not None,
            )
            if response_model is None:
                return LLMResponse(
                    content=content, parsed=None, provider=provider.name,
                    model=provider.model, usage=rec, latency_ms=rec.latency_ms,
                )
            parsed, err = parse_structured(content, response_model)
            if parsed is not None:
                return LLMResponse(
                    content=content, parsed=parsed, provider=provider.name,
                    model=provider.model, usage=rec, latency_ms=rec.latency_ms,
                )
            last_parse_error = err
            logger.info(
                "structured parse failed on %s (attempt %d/%d): %s",
                provider.name, attempt + 1, self.structured_retries + 1, err,
            )
            msgs = base_msgs + [
                {"role": "assistant", "content": content[:2000]},
                {
                    "role": "user",
                    "content": (
                        f"Your previous reply failed schema validation:\n{err}\n"
                        "Reply again with a single JSON object matching the schema exactly."
                    ),
                },
            ]
        raise StructuredOutputError(
            f"provider {provider.name} failed structured output after "
            f"{self.structured_retries + 1} attempts: {last_parse_error}"
        )

    # ------------------------------------------------------------------
    # 内部：一次真实 API 调用（退避重试 + 记账）
    # ------------------------------------------------------------------
    async def _raw_call(
        self,
        provider: Provider,
        msgs: List[Dict[str, str]],
        agent_name: str,
        agent_type: str,
        round_num: Optional[int],
        temperature: Optional[float],
        max_tokens: Optional[int],
        structured: bool = False,
    ) -> Tuple[str, UsageRecord]:
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.monotonic()
                kwargs: Dict[str, Any] = dict(
                    model=provider.model,
                    messages=msgs,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                    timeout=self.timeout,
                )
                # response_format 只在结构化输出时启用，不干预普通对话
                if structured and provider.json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                try:
                    resp = await provider.get_client().chat.completions.create(**kwargs)
                except APIStatusError as e:
                    if e.status_code == 400 and provider.json_mode:
                        # 个别网关不支持 response_format：降级后立刻重试本请求
                        provider.json_mode = False
                        kwargs.pop("response_format", None)
                        resp = await provider.get_client().chat.completions.create(**kwargs)
                    else:
                        raise
                latency_ms = int((time.monotonic() - t0) * 1000)

                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                rec = UsageRecord(
                    agent_name=agent_name,
                    agent_type=agent_type,
                    round_num=round_num,
                    provider=provider.name,
                    model=provider.model,
                    input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    cached_tokens=self._cached_tokens(usage),
                    latency_ms=latency_ms,
                )
                self.tracker.record(rec)
                provider.note_success()
                return content, rec
            except RETRYABLE_ERRORS as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "provider %s transient error (%s), retry in %.1fs "
                        "(attempt %d/%d)",
                        provider.name, type(e).__name__, delay,
                        attempt + 1, self.max_retries,
                    )
                    await asyncio.sleep(delay)
        # 重试耗尽：计一次失败（可能触发熔断），向上抛给 failover
        provider.note_failure(self.circuit_threshold, self.circuit_cooldown)
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _backoff_delay(self, attempt: int) -> float:
        delay = self.backoff_base * (2 ** attempt)
        return min(delay, self.backoff_cap) + random.uniform(0, 0.3 * delay)

    @staticmethod
    def _cached_tokens(usage: Any) -> int:
        if usage is None:
            return 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None and getattr(details, "cached_tokens", None):
            return int(details.cached_tokens)
        # DeepSeek 兼容字段
        return int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)

    def _with_json_instruction(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]],
    ) -> List[Dict[str, str]]:
        if response_model is None:
            return list(messages)
        schema = json.dumps(
            response_model.model_json_schema(), ensure_ascii=False
        )
        instr = f"{JSON_INSTRUCTION}\nJSON schema:\n{schema}"
        msgs = [dict(m) for m in messages]
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = f"{msgs[0]['content']}\n\n{instr}"
        else:
            msgs.insert(0, {"role": "system", "content": instr})
        return msgs
