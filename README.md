# MedBoard

从零实现的循证医学多 Agent 会诊系统：分诊（难度自适应）→ 专家面板并行辩论 → 加权置信投票 → 主持人反馈与早停。机制参考 [MedAgents-2](https://github.com/MedAgentsBench/MedAgents-2) 论文思路，代码从零实现，工程增强（服务化 / 评测 / 工程化落地）。

> 设计文档：`../MedAgents自研实现-模块划分与排期.md`（M1-M7 与排期）、`../MedAgents自研-工程化落地方案.md`（高并发 / 上下文管理 / 可观测）

## 当前状态

- [x] M1 LLM 基础设施：多厂商路由（DeepSeek/通义/Moonshot）、结构化输出（JSON + pydantic 校验 + 失败反馈重试）、429/超时指数退避、简单熔断、全局并发信号量、按 agent/轮 token 记账
- [ ] M2 Agent 角色层（分诊 / 专家 / 主持人）
- [ ] M3 会诊编排器（多轮 + 加权投票 + 早停）
- [ ] M4 PubMed 检索工具层
- [ ] M5 FastAPI + SSE / Gradio
- [ ] M6 MedQA 评测 + Pareto
- [ ] M7 工程化（Docker / CI / README）

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .example.env .env   # 填入任一厂商的 API key

python selfcheck.py          # M1 验收：调 LLM → 结构化校验 → 记账
python selfcheck.py --mock   # 无 key 离线验证管线

python -m pytest tests/ -v   # 单元测试（无需 key）
```

## M1 设计要点

| 能力 | 实现 | 位置 |
|---|---|---|
| 多厂商路由 | 主模型重试耗尽自动 failover；连续失败 3 次熔断冷却 60s | `core/llm.py` Provider |
| 结构化输出 | schema 注入 system prompt + `response_format=json_object` + pydantic 校验；失败把错误信息回灌重试 | `core/llm.py` `_chat_via` |
| 退避重试 | 429/超时/断连指数退避 + jitter（基数 1.5s，上限 30s） | `core/llm.py` `_raw_call` |
| 并发保护 | 全局信号量（默认 8），所有 provider 共享 | `core/llm.py` |
| token 记账 | 每次真实 API 调用记一条（含重试），按 agent/类型/轮聚合 | `core/usage.py` |
| 输出契约 | Triage/Expert/Moderator 三个 pydantic 模型 | `core/schemas.py` |
