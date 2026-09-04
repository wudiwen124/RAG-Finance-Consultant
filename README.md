# 金融知识库 RAG 问答助手

本项目用于搭建「金融知识库 RAG 问答助手」的 MVP：
从公开合规渠道收集金融知识文档 → 转为结构化 Markdown → 切分 → 调用
Embedding API 向量化 → 写入向量库 → BM25 + 向量混合召回 →
Reranker 精排 → 大模型生成带引用的回答。

当前版本已经从“单阶段向量检索”升级为：

```text
识别“A和B分别/区别”类问题 → 按子主题独立混合检索和 Rerank
→ 按主题配额融合 → 严格资料边界 prompt 生成
```

同时支持多轮会话记忆。系统会在检索前先做意图路由和查询改写：

```text
当前问题 + 最近对话历史 → 意图路由 / 查询改写
→ memory：直接根据会话历史回答，不进入向量检索
→ knowledge：把追问改写成独立问题，再走 RAG
→ hybrid：先理解历史指代，再检索知识库
```

## 知识库内容（MVP）

| 类别 | 数量 | 来源 |
| --- | --- | --- |
| 法律法规 | 12 部 | 国家法律法规数据库 flk.npc.gov.cn |
| 基金档案 | 8 只 | 天天基金网基金档案 fundf10.eastmoney.com |
| 金融名词解释 | 约 60 条 | 根据公开资料整理 |
| 产品场景文档 | 4 篇 | 抖音支付/月付/放心借/退货运费险（公开信息整理） |

目录结构：

```
kb/
├── raw/            原始文件（法规 DOCX、基金页面 HTML）
├── processed/      结构化 Markdown（带 front matter 元数据）
│   ├── laws/       法规
│   ├── funds/      基金档案
│   ├── terms/      名词解释
│   └── products/   产品场景
├── meta/           下载元数据
└── chunks/         chunks.jsonl（切分结果清单）
scripts/            下载/转换/向量化脚本
scripts/rag_qa.py   LangChain RAG 问答、混合检索与 Rerank 接入
scripts/conversation.py
                    会话记忆、多轮意图路由与查询改写
scripts/api_server.py
                    FastAPI 问答服务（含内置前端页面托管）
scripts/retrieval.py BM25 索引、RRF 融合与混合检索
scripts/eval_rag.py 检索效果评测脚本
scripts/expand_eval_set.py
                    重建/扩展检索评测集
scripts/eval_answer_quality.py
                    答案级评测：回答、拒答、引用和幻觉
scripts/eval_multi_turn.py
                    多轮记忆评测：会话路由、查询改写、历史召回与检索命中
config/config.example.json
                    Embedding / LLM / Memory / Rerank / Hybrid / 向量库配置模板
frontend/index.html 前端问答页面（单文件，无构建依赖）
data/vector_db/     本地生成的 Qdrant 向量库文件
data/eval_questions.jsonl
                    RAG 评测集（200 题；185 题有答案，15 题资料不足）
data/hallucination_regression.jsonl
                    幻觉固定回归集（30 题：4 个原失败 + 9 个改写 + 17 个相似干扰）
data/multi_turn_eval.jsonl
                    多轮记忆评测集（40 个目标轮）
data/answer_eval_results.jsonl
                    答案级评测逐题明细
data/hallucination_regression_results.jsonl
                    幻觉回归集逐题明细
rag_effect_log.txt
                    最新评测报告，由 scripts/eval_rag.py 自动生成
answer_eval_report.md
                    最新答案级评测报告，由 scripts/eval_answer_quality.py 自动生成
hallucination_regression_report.md
                    幻觉回归集报告，由 scripts/eval_answer_quality.py 自动生成
multi_turn_eval_report.md
                    多轮记忆评测报告，由 scripts/eval_multi_turn.py 自动生成
optimization_report.md
                    检索优化、答案级评测方法与阶段性评测结论
```

## 使用步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 构建向量库（正式）

复制 `config/config.example.json` 为 `config/config.json`，并填入
Embedding / LLM / Rerank 配置：

```json
{
  "embedding": {
    "api_base": "https://api.siliconflow.cn/v1",
    "api_key": "sk-你的Key",
    "model": "BAAI/bge-m3",
    "batch_size": 16
  }
}
```

Rerank 和混合检索的核心配置如下：

```json
{
  "rerank": {
    "enabled": true,
    "model": "Qwen/Qwen3-Reranker-4B",
    "candidate_k": 20,
    "timeout": 60
  },
  "hybrid": {
    "enabled": true,
    "chunks_manifest": "kb/chunks/chunks.jsonl",
    "vector_candidate_k": 20,
    "bm25_candidate_k": 20,
    "fused_candidate_k": 20,
    "rrf_k": 60,
    "vector_weight": 1.0,
    "bm25_weight": 1.0
  },
  "cross_doc": {
    "enabled": true,
    "per_query_candidate_k": 20,
    "per_query_top_k": 4,
    "per_topic_quota": 2
  }
}
```

> 注意：`config/config.json` 已加入 `.gitignore`，不会提交真实 API Key。
> 请不要把真实 Key 写入示例配置或提交到公共仓库；如果 Key 已经泄露，
> 请在服务商后台轮换。

Judge 稳定性配置可以单独调整：

```json
{
  "answer_judge": {
    "max_attempts": 3,
    "timeout_seconds": 60,
    "retry_delay_seconds": 2,
    "per_hit_chars": 700,
    "max_context_chars": 3000,
    "max_question_chars": 500,
    "max_answer_chars": 2500
  }
}
```

然后运行：

```bash
python scripts/build_vector_db.py
```

当前切分器会保留短名词解释和短法条，并把同父章节下的相邻短内容合并；  
只有后续还有子章节的结构性标题才不会单独入库。当前知识库切分为 `1256` 个 chunk。

> 注意：修改 `kb/processed` 下的文档或切分逻辑后，必须重新运行上面的构建命令；  
> 否则 Qdrant 向量库和 `chunks.jsonl` / BM25 索引会继续使用旧知识。

### 3. 构建后检索冒烟测试

```bash
python scripts/build_vector_db.py --query "货币基金和债券基金有什么区别"
```

### 4. RAG 问答（LangChain 问答层，需要生成模型 API）

配置 `config/config.json` 中的 `llm`（例如硅基流动 + Qwen3-8B），然后：

```bash
# 单次提问
python scripts/rag_qa.py --question "货币基金和债券基金有什么区别"

# 交互模式
python scripts/rag_qa.py
```

问答链路：问题向量化（bge-m3）→ Qdrant 向量召回 20 条 +
BM25 粗召回 20 条 → RRF 融合 top 20 →
`Qwen/Qwen3-Reranker-4B` 精排 → 取 top 4 → 拼带编号的上下文 →
ChatOpenAI 生成带 [1][2] 引用的回答 → 末尾列出来源。

如果问题命中“A和B分别是什么”“A和B有什么区别”等结构，会先拆成两个
独立 query；每个子 query 分别混合检索和 rerank，然后每个主题最多占
`qa.top_k / 2` 条，避免一个主题挤掉另一个主题的证据。可用
`--no-cross-doc` 临时关闭该行为。

生成 prompt 现在采用严格资料边界：只允许转写、压缩、归纳参考资料；
禁止补充资料外金融常识；资料不足时必须拒答；多部分问题必须逐部分说明
是否找到证据。针对已发现的失败模式，还追加以下约束：单只基金的数字和策略
不得泛化到整类；资料未写的交易、结算、清算和行业惯例不得补；数字、期限和
比例必须对应原文同一主体和环节；法规期限优先逐字引用原文关系。

如果想临时关闭 rerank、但保留混合检索，可以加：

```bash
python scripts/rag_qa.py --no-rerank --question "设立非银行支付机构需要什么条件"
```

如果只想看纯向量检索，可以同时关闭混合检索和 rerank：

```bash
python scripts/rag_qa.py --no-hybrid --no-rerank --question "设立非银行支付机构需要什么条件"
```

如果 rerank 接口异常，`rag_qa.py` 会自动回退到原始向量排序，保证问答不中断。
如果 BM25 索引加载异常，混合检索会自动退化为向量检索。

### 5. 不填 API 先验证流程

```bash
python scripts/build_vector_db.py --mock
```

`--mock` 使用确定性伪向量，只用于跑通流程，检索结果无语义。

### 5.1 启动 API 服务与前端页面

```bash
python scripts/api_server.py
```

启动后打开浏览器访问 `http://localhost:8000`，就是一个简约的问答前端页面。
`http://localhost:8000/docs` 是自动生成的 Swagger 交互文档，可以直接试调 API。

四个接口：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/api/ask` | 非流式核心问答，接收 `{"question": "...", "session_id": "..."}`，一次性返回完整 JSON |
| `POST` | `/api/ask/stream` | 流式问答，接收 `{"question": "...", "session_id": "..."}`，先推送引用，再逐块推送正文 |
| `GET` | `/api/health` | 健康检查，前端用它判断服务是否在线 |
| `POST` | `/api/sessions/{session_id}/clear` | 清空指定会话记忆，前端“新会话”按钮使用 |

`/api/ask` 统一返回 JSON：

```json
{
  "answer": "货币基金主要投资于...",
  "citations": [
    {
      "index": 1,
      "title": "金融名词解释（入门向）",
      "section": "基金基础 > 货币基金",
      "source": "kb/processed/terms/...",
      "snippet": "货币基金是一种..."
    }
  ],
  "latency_ms": 1234
}
```

`/api/ask/stream` 使用 Server-Sent Events（SSE）推送数据。事件类型包括：

| `type` | 含义 |
| --- | --- |
| `citations` | 检索完成后的引用来源列表，前端先渲染它 |
| `token` | 模型生成的一小段正文，前端会立刻追加显示 |
| `done` | 生成结束，包含完整请求的总耗时 |
| `error` | 检索/生成异常或未命中知识库时的错误信息 |

前端页面功能：

- **输入框**：Enter 或点发送提问，1–500 字符校验；
- **多轮会话**：浏览器生成并保存 `session_id`，支持“那它呢”“你刚才说的那个产品”“我问你的第一个问题”这类追问；
- **新会话**：点击顶部“新会话”清空服务端历史并切换到新的 `session_id`；
- **对话流**：用户蓝色气泡 / AI 白色卡片，自动滚动到底；
- **引用折叠**：AI 回答下方显示 `[1] [2]...` 来源列表，点击展开原文片段；
- **流式输出**：正文逐块渲染，无需等完整回答返回；引用先出现，体感等待明显缩短；
- **拒答态**：回答含"资料中未找到"时用琥珀色卡片区分；
- **加载态**：三个跳动圆点 + 骨架屏；
- **健康指示**：页面顶部小圆点，后端在线变绿；
- **错误提示**：请求失败用红色气泡显示原因。

前端是单文件 `frontend/index.html`，嵌入 CSS 和 JS，不需要 Node.js 或构建
工具。由 FastAPI 的根路径 `/` 直接托管，与 API 同端口，无需处理跨域。

`requirements.txt` 已包含 `fastapi` 和 `uvicorn`；如果之前没装，先运行
`pip install -r requirements.txt`。

### 5.2 多轮记忆与查询改写

`scripts/conversation.py` 会在每次请求时维护最近 5 轮问答，并调用一个轻量
LLM 判断问题类型：

| 意图 | 示例 | 处理方式 |
| --- | --- | --- |
| `memory` | 我问你的第一个问题是什么？ | 不检索知识库，直接根据历史回答 |
| `knowledge` | 保险理赔核定期限是多久？ | 正常混合检索 + Rerank + 生成 |
| `hybrid` | 你刚才说的那个产品费率是多少？ | 先补全历史指代，再检索知识库 |

例如上一轮问“保险人收到理赔请求后多久作出核定”，下一轮问“那如果超时了呢”，
改写器会生成类似“保险人未在规定期限内完成理赔核定有什么法律后果”的独立查询。

当前记忆是进程内内存，服务重启后清空，TTL 为 24 小时。它只用于指代消解和
历史内容回答；金融事实仍必须来自知识库检索资料。相关配置：

```json
"memory": {
  "enabled": true,
  "window_turns": 5,
  "rewrite_model": "Qwen/Qwen3-8B",
  "rewrite_timeout_seconds": 15,
"ttl_hours": 24
}
```

### 6. 检索效果评测（200 题、多模式对比）

```bash
python scripts/eval_rag.py
```

脚本会读取 `data/eval_questions.jsonl` 的 200 个评测题；其中 185 道有答案题
参与检索指标，15 道资料不足题只用于拒答分析和答案级评测，不计入检索命中率。
脚本分别跑：

1. 纯向量；
2. 向量 + rerank；
3. BM25 + 向量；
4. BM25 + 向量 + rerank。
5. BM25 + 向量 + rerank + 跨文档拆分。

当前 200 题评测结果（2026-09-03 校准口径）：

| 模式 | top4 命中率 | 要点覆盖 | 平均排名 | MRR |
| --- | ---: | ---: | ---: | ---: |
| 纯向量 | 94% | 96% | 1.19 | 0.914 |
| 向量 + rerank | 99% | 100% | 1.07 | 0.970 |
| BM25 + 向量 | 93% | 95% | 1.17 | 0.907 |
| BM25 + 向量 + rerank | 99% | 100% | 1.07 | 0.970 |
| BM25 + 向量 + rerank + 跨文档拆分 | **100%** | **100%** | 1.08 | 0.965 |

解释：top4 命中率和要点覆盖衡量“正确答案是否进入前 4 条”；平均排名
和 MRR 衡量“正确答案排得有多靠前”。BM25 负责补足精确词和专有名词，
RRF 负责融合两路排名，reranker 负责最终精排。

评测集覆盖同义改写、法规条款、跨文档多要点、否定题、产品场景和资料不足题。
`keyword_groups[].accept` 支持同义答案组：期望要点里的任意一个表达命中即算覆盖，
避免“原文说 A，评测只认 A 的近义改写 B”造成的假失败。

### 7. 答案级评测

```bash
python scripts/eval_answer_quality.py
```

默认从 200 题里按类别分层抽样 40 题，逐题执行完整 RAG 链路并调用 LLM Judge。
它不只检查检索命中，还会评估：

1. 最终回答是否覆盖期望概念；
2. 资料不足题是否正确拒答，有答案题是否误拒；
3. `[1][2]` 引用编号是否能对应真实检索资料；
4. Judge 认为答案实质结论是否被资料支持；
5. Judge 是否发现资料外编造、推断或错误引用。

最新 40 题答案级抽样结果（2026-09-04 第三版 prompt）：

| 指标 | 结果 |
| --- | ---: |
| 概念覆盖率 | 83.8% |
| 概念全对率 | 73.5% |
| 引用格式有效率 | 94.1% |
| Judge 引用支持率 | 90.9% |
| 有效 Judge 幻觉率 | 9.1%（3/33） |
| 资料不足题正确拒答率 | 100.0% |
| 有答案误拒率 | 2.9% |

Judge 已改为专用短上下文 + 自动重试，本轮 1/34 道 Judge 超时（33 道有效判定）。
第三版 prompt 新增产品层级边界和 T+0 硬禁止：资料不足题拒答率修复到 **100%**，
概念覆盖/全对继续提升，上一轮的"融资及转融通"类幻觉和 T+0 幻觉均未再出现。
但幻觉率从 2.9% 回升到 9.1%，原因是暴露了三个不同问题：已知条款错配复现
（171）、"混合基金"不在产品层级列举范围内（104）、跨文档子主题推测（151）。
下一步扩大产品层级列举范围并给跨文档子主题加独立边界。

### 7.1 幻觉固定回归集

```bash
python scripts/eval_answer_quality.py --eval data/hallucination_regression.jsonl --limit 0 --detail data/hallucination_regression_results.jsonl --report hallucination_regression_report.md
```

回归集有 30 题：4 个原始失败基线、9 个同义改写、17 个相似概念干扰。每题带
`failure_mode`，失败时能快速定位是哪类约束回退。最新专项结果：

| 指标 | 结果 |
| --- | ---: |
| Judge 幻觉率 | 10.0%（3/30） |
| Judge 支持率 | 96.7% |
| Judge 引用支持率 | 96.7% |
| 引用格式有效率 | 100.0% |
| Judge 异常率 | 0.0% |
| 概念覆盖率 | 93.3% |
| 概念全对率 | 86.7% |

30 题中新增的相似概念干扰题暴露了 3 道幻觉（H006 单基金策略泛化、H007 T+0
偶发、H013 联接基金数据误挂到 ETF 本体）和 4 道关键词假告警（已修复）。原始
4 道基线全部通过。第三版 prompt 已针对这三道幻觉加入产品层级边界和 T+0 硬
禁止；40 题主评测验证 T+0 幻觉已消除，但仍有条款错配和大类基金泛化残留。

### 7.2 多轮记忆评测

```bash
python scripts/eval_multi_turn.py
```

评测集是 `data/multi_turn_eval.jsonl`，共 40 个目标轮：10 个记忆题、20 个
追问/条款/产品题、10 个话题切换题。每条样例不是一个孤立问题，而是包含
`history`、当前 `question`、期望意图、期望改写要点、期望证据和禁用历史词。

本轮评测会覆盖四层能力：

1. **会话路由**：`memory` 直接查历史；`knowledge/hybrid` 进入检索分支；
2. **查询改写**：把“它”“刚才说的那个产品”补全成独立检索问题；
3. **历史召回**：验证能否取回第一个/最近的问题或助手的上一段回答；
4. **证据检索**：验证改写后的查询能否在 Top4 内召回正确资料。

最新 40 题结果（2026-09-04）：

| 指标 | 结果 |
| --- | ---: |
| 可用路由成功率 | 100.0% |
| 记忆召回正确率 | 100.0% |
| 查询改写完整率 | 93.3% |
| Top4 检索命中率 | 96.7% |
| 平均证据覆盖率 | 98.3% |
| 查询历史污染率 | 0.0% |
| 多轮链路成功率 | 97.5% |
| 平均理解/检索耗时 | 1676 ms |

严格意图口径只有 52.5%，原因是很多追问在语义上可标为 `hybrid`，但当前实现
只要路由器把它改写成独立查询后走 `knowledge`，检索动作就已完成。因此报告同时
给出“可用路由成功率”。唯一链路失败是 `M025`：抖音月付使用方式命中“先享后付”，
但 Top4 未同时命中“分期还款”证据，后续可通过产品名 boosted retrieval 或
同一文档内的多 chunk 配额继续优化。

使用 `--eval` 跑回归集时必须同时指定独立的 `--detail` 和 `--report`，避免覆
盖 200 题主评测产物。

明细写入 `data/answer_eval_results.jsonl`，报告写入 `answer_eval_report.md`。
答案级评测会调用生成模型和 Judge，成本明显高于检索评测；可用 `--limit`
控制抽样数，`--limit 0` 表示全量。

更详细的优化过程、缺点分析和遗留问题见：

```text
optimization_report.md
```

评测集目前覆盖基金投资、法规监管、支付信贷、保险与产品场景。新增知识
或优化检索参数后，建议重新运行评测，避免只在个别问题上感觉变好。

本机 Python 已安装依赖时，也可以使用完整路径运行，避免 Windows 下
`python` 指向其他环境：

```powershell
C:\Users\25328\AppData\Local\Programs\Python\Python312\python.exe scripts\eval_rag.py
```

## Embedding 模型选哪个

| 方案 | 模型 | 维度 | 说明 |
| --- | --- | --- | --- |
| 首选 | BAAI/bge-m3 | 1024 | 中文检索效果好，多语言，开源；SiliconFlow 有 OpenAI 兼容 API |
| 备选 | text-embedding-v4（阿里云百炼） | 1024 | 阿里官方 embedding，中文强，DashScope OpenAI 兼容 |
| 备选 | text-embedding-3-small（OpenAI） | 1536 | 中文一般，仅作对比 |

改模型时同步修改 `config/config.json` 中的 `model`，并确认维度与
`data/vector_db` 中的集合一致（重建时脚本会自动重建集合）。

## 向量库说明

- 默认使用 **Qdrant 本地模式**（`qdrant_local`），无需 Docker，数据落在
  `data/vector_db/`。
- 未安装 `qdrant-client` 时自动退化为本地 numpy 向量库（
  `data/vector_db_local/`），同样可用。
- 每条 chunk 携带完整元数据：`doc_id`、来源、标题、章节路径、页码等，便于
  检索后引用溯源。

## 数据合规提醒

- 法规和基金披露信息为法定公开内容；基金档案页为公开页面抓取。
- 所有文档仅供个人学习使用，不做商业分发，使用请注明来源。
