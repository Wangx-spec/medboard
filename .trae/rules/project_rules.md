# medboard 项目开发提示词（Trae）

> **用法**：两种方式任选——① 将本文全文粘贴到 Trae 对话框作为项目开发指令；② 将本文复制到项目根目录 `.trae/rules/project_rules.md`，Trae 会作为项目规则自动加载，后续每轮对话无需重复粘贴。
> 配套文档：`plan-docs/`（7 份 Phase spec，每阶段验收标准以此为准）、`MedAgents自研实现-模块划分与排期.md`（主控方案 v3.1，机制细节总纲）。开发中遇到本文未覆盖的细节，以上述文档为准。

---

## 一、项目目标与背景

**medboard** 是一个从零自研的**循证医学多 Agent 会诊系统**（Python）。用户输入一道医学题（主诉/病历/问题），系统模拟真实科室会诊流程：

> 分诊 Agent（评估难度）→ 专家面板并行分析（可检索 PubMed 带证据作答）→ 加权置信投票 → 主持人总结与早停决策 → 多轮迭代 → 输出最终诊断意见 + 可追溯的证据引用

三条项目红线，任何开发决策不得违背：

1. **机制参考、代码从零**：多 Agent 辩论机制参考 MedAgents-2 论文；运行时纪律（状态机完成门禁、预算分级、Evidence Ledger、事件溯源）参考 Agent Harness 架构思想。**只借鉴机制设计，不复制任何外部项目代码，禁止引入 LangChain / LangGraph / autogen / crewai 等 agent 框架**——从零自研编排层是本项目的核心价值。
2. **工程问题真实存在，解法必须带数字**：每一项工程强化（早停、预算、压缩、并发防御）都要回答"解决了什么真实问题"并产出可落盘的度量数字（如"早停省多少 token、牺牲多少准确率"），不接受只有"支持了"没有度量的实现。
3. **循证可追溯**：专家结论必须引用真实证据（PubMed PMID），引用经账本校验，无效引用降权——这是本项目的差异化定位。

**当前状态**：M1（LLM 路由层）已完成并通过 9/9 单测（`core/llm.py`、`core/schemas.py`、`core/usage.py`、`tests/test_m1.py`、`selfcheck.py`）。本提示词覆盖 M2 起的全部剩余开发。

---

## 二、功能需求列表（按模块）

### M1 LLM 路由层 ✅ 已完成——勿改动对外接口
多厂商路由（OpenAI 兼容协议：deepseek / qwen / moonshot，见 `config.yaml`）、结构化输出校验与带反馈重试、退避+熔断、并发信号量、token 记账。后续模块只消费其接口。

### M2 Agent 角色层（Phase 1）
- 三类角色，均基于 pydantic schema 的结构化输入输出：**分诊 Agent**（输出难度 easy/medium/hard + 理由）、**专家 Agent**（含科室画像，输出意见+置信度+证据引用）、**主持人 Agent**（输出总结/反馈/早停决策）
- 角色定义与编排解耦：角色不知道自己被谁调用

### M3 会诊编排器（Phase 1，全项目核心）
- **状态机**：`queued → triaging → debating → voting → moderating → completed / failed / budget_exhausted`，冻结转换表；**`completed` 必须由早停 gate 显式放行（完成门禁）**，不允许"循环跑完即完成"
- **预算分级**：难度 easy/medium/hard → max_rounds 1/2/3 + token 上限；每轮结束 `assert_available()`，超限进入 `budget_exhausted` 而非静默继续
- **事件溯源日志**：append-only `events` 表（SQLite）是**单一事实源**——JSON 日志、SSE 事件、checkpoint 恢复、评测轨迹全部是它的投影，不另建平行数据；事件带 `schema_version` 与递增序号
- 多轮循环：`asyncio.gather` 并行专家 → 加权投票 → 主持人反馈注入 or 早停；失败重试（attempt）与成功回复在事件日志中分开记录
- **prompt 构造单一出口**：每轮所有 prompt 构造收敛到一个函数 `build_round_prompt()`，禁止旁路构造（后续预算管家插在此桩点）
- CLI 入口：命令行提交一道题，输出答案 + 投票分布 + 状态转移轨迹

### M4 检索工具层（Phase 2）
- PubMed E-utilities 封装（限速 + 退避重试）、LLM 查询改写、相似度去重（阈值 0.85）、SQLite 响应缓存、共享检索历史
- **Evidence Ledger 证据账本**：检索结果入不可变账本（`evidence_id` = PMID 哈希 + `content_hash` + 摘要，落 SQLite）；专家输出的 evidences 改为 evidence_id 引用；投票前校验引用完整性，**无效引用的专家意见降权或触发重试**
- **ToolRegistry 工具注册表**：工具统一 schema（name / description / params → returns EvidenceItem），执行走 `pre（限速/预算检查）→ execute → post（结果入账本）` 固定流水线；新增工具 = 一个注册项，不改编排代码
- **guard 循环卫生**：专家重复发起同一检索 ≥3 次 → 向其注入提醒（不阻断）；工具调用协作式超时

### M5 服务化 + 可视化（Phase 3）
- FastAPI `POST /consult`（SSE 流式）+ `GET /consult/{id}` 回放
- SSE 事件模型 = 状态机转移（`triage / round_start / expert_answer / vote / moderator / final`）+ `schema_version`——前端契约一旦冻结不再变更
- 进程 kill 后重启，凭 events 表投影恢复，断点续播（已完成轮次不重算）
- Gradio 会诊室 demo（一天成本，非技术观众可用）

### M6 评测层（Phase 4）
- MedQA 子集 100 题 × 4 配置（单 Agent / 多专家单轮 / 完整会诊 / 无难度自适应消融）
- 基础指标：准确率、tokens/题、收敛轮次；**轨迹驱动指标**（从落库 events 计算，不重跑）：早停命中率 vs 跑满轮数的准确率差、引用有效率、一次通过率、guard 触发率
- LLM 响应全量缓存（SQLite）保证评测可复现；产出 Pareto 成本-准确率图

### M7 工程化门面（Phase 5）
- Docker compose 一键起、pytest ≥15 且全 mock 无 key 可跑、GitHub Actions CI、README（架构图 + 快速启动 + 评测数字）、LICENSE
- **录制回放测试**：录制真实 LLM 交互为 fixture，CI 无 key 跑端到端回归
- **扩展项（不进保底链，进度紧张则砍）**：Vue3 工程化前端（技术栈见下节）：会诊室直播 / 回放 / 证据账本浏览 / 评测看板四个视图，直播与回放共用渲染组件，只消费 M5 事件契约，后端零改动

### 工程化强化包（Phase 6）
- 上下文预算管家：插在 `build_round_prompt()` 桩点，三级压缩——L1 截断 / L2 遮蔽式摘要（写标记事件、不改历史事件）/ L3 降档（降专家数/轮数）；触发哪级、压缩多少，落事件日志可审计
- 任务队列 + worker 池；checkpoint/resume = events 日志投影 + 幂等标记（不另建 checkpoint 格式）
- 令牌桶限流、locust 压测、基础可观测（每会诊 token/轮次/时长汇总）

### M8 vLLM 网关对比（Phase 7）
- vLLM 自部署开源医学模型 vs 商用 API（deepseek 等）的对比表：质量 / 成本 / 延迟，数字落盘进 README

---

## 三、技术栈与约束条件

| 类别 | 选型 | 说明 |
|---|---|---|
| 语言 | Python ≥ 3.11 | 异步用原生 asyncio，不引 celery/redis |
| LLM 接入 | OpenAI 兼容协议，厂商与故障切换由 `config.yaml` 配置 | M1 已实现，勿重复造 |
| Web | FastAPI + SSE | 不用 WebSocket（单向推送已够）；SSE 用 POST 带 body |
| 存储 | SQLite | 事件日志 / 证据账本 / LLM 缓存，单机零运维 |
| 评测数据 | MedQA 子集 | 100 题，响应缓存保证可复现 |
| Demo 前端 | Gradio（Phase 3） | 快速门面，一天完成 |
| 工程前端（扩展项） | Vue 3 `<script setup>` + Vite + TS（轻量）+ Element Plus（按需）+ ECharts + Pinia + vue-router（hash 模式）+ markdown-it/DOMPurify | 与用户另一项目同栈；不引 axios / WebSocket / SSR / Tailwind |
| 测试 | pytest + pytest-asyncio | 全 mock，无 API key 可跑 |
| 部署 | docker compose 单机 | 不做 k8s |

**硬约束**：

1. **零 agent 框架依赖**：禁止 LangChain / LangGraph / autogen / crewai / llama-index 等；编排、状态机、记忆、工具调用全部手写
2. **不复制外部代码**：harness 机制（状态机/预算/账本/事件溯源）只参考设计思想，实现从零写；也不复制 MedAgents 官方科研代码
3. **依赖最小化**：当前依赖仅 openai / pydantic / PyYAML / python-dotenv / pytest；新增依赖必须先说明理由并征得同意，标准库或已引入库能解决的不得新增
4. **密钥安全**：API key 一律走 `.env` 环境变量（见 `.example.env`），代码/测试/日志/文档中不得出现真实 key；`.env` 不入 git
5. **所有 LLM 调用可 mock**：任何测试不得依赖真实 API；外部服务（PubMed）同理
6. **公共契约稳定**：状态机转换表、事件 schema、Evidence Ledger 接口、M5 SSE 事件模型是全项目公共契约，变更必须先说明影响面并经确认

---

## 四、代码规范要求

- **类型与契约**：公共函数/方法全类型注解；跨模块数据一律 pydantic v2 model（schema 即文档）；模块级 docstring + 关键类/函数 docstring（写"为什么"，不写"是什么"）
- **命名**：模块/函数 snake_case，类 PascalCase；状态与事件名固定用主控方案规定的小写下划线形式（`round_start`、`budget_exhausted` 等）；核心抽象保留英文原名（ConsultationRun、Evidence Ledger、ToolRegistry）
- **错误处理**：外部调用（LLM / PubMed）一律捕获具体异常 + 退避重试，禁止裸 `except:`；重试耗尽走显式失败路径（`failed` / 降权），不产生半成品状态
- **日志**：标准库 logging；token 消耗、状态转移、预算断言结果必须写入事件日志；面向人的运行日志用 INFO，调试细节 DEBUG
- **测试与实现同步**：每个模块在 `tests/` 下有对应 `test_*.py`；负路径必须有测试（非法状态转换被拒绝、无效证据引用被拦截降权、超预算进入 `budget_exhausted`）
- **配置**：厂商/策略参数进 `config.yaml`，密钥进 `.env`，行为开关可在 config 覆盖；不硬编码魔法数字（阈值/上限具名进配置或常量并注释来源）
- **git 纪律**：**Trae 不自主执行 git commit**——每完成一个可提交单元（测试绿的完整功能点），提示用户提交建议（含建议的 commit message），由用户本人执行；`plan-docs/` 与根目录方案文档属规划文档，Trae 不主动将其纳入版本控制
- **注释语言**：代码注释与标识符英文为主（面向开源读者），可辅以中文说明关键设计意图；README 中文为主

---

## 五、预期文件结构与模块划分

```
medboard/
├── core/                        # 核心域：编排与角色（除 LLM/存储 IO 外零外部依赖）
│   ├── llm.py ✅                #   M1 LLM 路由（已完成，接口冻结）
│   ├── schemas.py ✅            #   M1 数据契约
│   ├── usage.py ✅              #   M1 token 记账
│   ├── agents/                  #   M2 角色层
│   │   ├── triage.py            #     分诊 Agent
│   │   ├── expert.py            #     专家 Agent（科室画像）
│   │   └── moderator.py         #     主持人 Agent
│   ├── orchestrator.py          #   M3 会诊编排器（含 build_round_prompt 单出口）
│   ├── run_state.py             #   M3 状态机 + 难度→预算分级
│   ├── voting.py                #   M3 加权投票（含引用完整性校验钩子）
│   ├── event_log.py             #   M3 事件溯源日志（单一事实源）
│   ├── guard.py                 #   M4 循环卫生（重复检索提醒/工具超时）
│   └── budget.py                #   Phase 6 上下文预算管家（三级压缩）
├── tools/                       # M4 工具层
│   ├── registry.py              #   ToolRegistry（pre → execute → post 流水线）
│   ├── pubmed.py                #   PubMed 封装（限速+退避）
│   ├── ledger.py                #   Evidence Ledger 证据账本
│   ├── dedup.py                 #   查询改写后的相似度去重
│   └── cache.py                 #   响应缓存
├── app/                         # M5 服务层
│   ├── api.py                   #   FastAPI 路由
│   ├── sse.py                   #   SSE 事件流（事件=状态转移）
│   ├── gradio_demo.py           #   Gradio 会诊室
│   ├── limiter.py               #   Phase 6 令牌桶
│   └── queue.py                 #   Phase 6 任务表 + worker 池
├── eval/                        # M6 评测
│   ├── run_eval.py              #   MedQA 100 题 × 4 配置
│   ├── pareto.py                #   成本-准确率 Pareto
│   └── datasets/                #   评测集与判分配置
├── frontend/                    # Phase 5 扩展项（Vue3，可砍）
│   └── src/
│       ├── views/               #   ConsultRoom / Replay / Ledger / EvalDashboard
│       ├── components/          #   ExpertCard / VoteChart / RoundTimeline / EvidenceRef
│       ├── composables/         #   useConsultStream（fetch+getReader 解析 SSE）
│       └── stores/              #   Pinia：consult / replay / settings
├── bench/locustfile.py          # Phase 6 压测
├── tests/                       # 全 mock 单测 + replay/（录制回放 fixture）
├── cli.py                       # M3 CLI 入口
├── selfcheck.py ✅              #   M1 自检脚本（已完成）
├── config.yaml ✅               #   厂商与策略配置
├── requirements.txt ✅          #   依赖（最小化，新增需说明）
├── .example.env ✅              #   密钥模板
├── plan-docs/                   # 开发文档（7 份 Phase spec，不入 git）
├── docker-compose.yml / Dockerfile        # M7
└── .github/workflows/ci.yml                # M7
```

**分层规则**：`core` 不 import `app`/`tools` 之外的服务层代码；`tools` 只被 `core` 编排调用；`app` 只消费 `core` 的事件与结果；依赖方向单向，禁止循环导入。

---

## 六、交付标准

每个 Phase 的详细验收标准（AC）以 `plan-docs/phase-N-*/spec.md` 为准，以下为全局底线：

1. **测试**：pytest 全绿且**无 API key、无网络**可跑；新增功能附带对应单测；负路径（非法状态转换 / 无效引用 / 超预算）有专项测试
2. **可复现**：CLI / demo 一条命令跑通；LLM 响应缓存使评测结果可复现；README 快速启动指引始终有效（陌生人照抄能跑）
3. **数字**：涉及成本/性能声明的功能（早停、预算、压缩、并发防御）必须产出可落盘的度量数字；README 中的评测数字必须与落盘结果一致
4. **文档**：每完成一个 Phase 更新 README 对应章节；公共契约（事件 schema / 状态机 / 账本接口）变更同步更新文档
5. **完成定义**：逐条核对当前 Phase spec 的 AC，全部满足才算完成；部分满足时明确列出缺口，不得含糊带过

---

## 七、Trae 协作流程

1. **按 Phase 顺序开发**：Phase 1（会诊核心）→ 2（检索工具层）→ 3（服务化）→ 4（评测）→ 5（工程化门面）→ 6（工程化强化）→ 7（vLLM 网关），见 `plan-docs/README.md` 排期表。**每个 Phase 开工前，先通读对应的 `plan-docs/phase-N-*/spec.md`**，按其功能条目（F）/ 约束（N）/ 验收标准（AC）实现。
2. **小步提交单元**：按 spec 的功能条目拆分实现顺序，每完成一个功能点先跑全量测试再继续下一个。
3. **冲突即停**：发现 spec 与现状冲突、或实现中发现 spec 设计有问题，停下来向用户说明冲突点与建议，等裁决后再动，**不擅自改契约**。
4. **先方案后动手**：涉及新依赖、公共契约变更、或拿不准的设计，先给出 2-3 个方案选项与权衡理由，经确认后实现。
5. **提交提醒**：完成可提交单元后，给出建议的 commit message（英文、祈使句，如 `feat(orchestrator): add consultation state machine with completion gate`）并等待用户提交。
