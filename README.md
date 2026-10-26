# FinRAG：金融知识库问答

这是一个面向金融法规、基金档案、金融名词和支付/信贷产品说明的 RAG 问答
项目。用户提问后，程序根据上下文判断是指代历史、普通知识问题，还是需要
先改写再检索的追问；确认需要查资料时，执行 BM25 + 向量混合召回和重排，
再把筛选后的片段交给大模型生成带引用编号的回答。

目前知识库里放了 12 部法规、8 只基金、约 60 个金融词条和 4 篇产品说明。
项目最初只有普通向量检索。后来用更接近真实提问的数据做评测，发现期限、
费率、专有名词和多主题问题容易漏检，才逐步加入 BM25、rerank、短 chunk
处理、跨主题拆分和多轮查询改写。每一步的指标和踩坑记录在
[optimization_report.md](optimization_report.md)，没有把一次跑通当成最终结果。

## 当前处理流程

```text
用户问题
  ├─ 询问历史：直接使用会话记录
  ├─ 普通知识问题：进入检索
  └─ 指代型追问：先结合上下文改写成独立问题，再检索

检索问题
  → 判断是否包含“A 和 B 分别/区别”等双主题结构
  → BGE-M3 向量召回、BM25 召回
  → RRF 融合候选
  → Qwen3-Reranker 重排
  → 取前 4 条资料
  → Qwen3-8B 按资料生成回答
  → 校验引用并返回来源
```

回答约束采用“没写在资料里就不要补”的原则。资料不足时返回
`资料中未找到相关信息`；涉及金额、期限、费率和产品类型时，要求数字与
主体、条款、交易环节一一对应。这部分规则主要来自后面几轮幻觉回归测试，
并非一开始就设计完整。

## 知识库

| 类别 | 数量 | 来源 |
| --- | ---: | --- |
| 法律法规 | 12 部 | 国家法律法规数据库 |
| 基金档案 | 8 只 | 天天基金网公开基金档案 |
| 金融名词 | 约 60 条 | 根据公开资料整理 |
| 产品说明 | 4 篇 | 抖音支付、月付、放心借、退货运费险的公开信息 |

主要目录：

```text
kb/
├── raw/            原始 DOCX 和 HTML
├── processed/      转成 Markdown 并补齐 front matter 的文档
├── meta/           下载记录
└── chunks/         chunks.jsonl，供向量库和 BM25 共用

scripts/
├── build_vector_db.py       切分、向量化和建库
├── rag_qa.py               检索、生成和引用拼接
├── conversation.py         会话记录、意图判断和查询改写
├── retrieval.py            BM25、RRF 和混合检索
├── api_server.py           FastAPI 服务
└── eval_*.py               检索、答案质量、多轮记忆评测

frontend/index.html         单文件前端
config/config.example.json  配置模板
```

`data/vector_db/` 是建库后生成的 Qdrant 本地数据，不需要单独准备。

## 安装和配置

```bash
pip install -r requirements.txt
```

复制配置模板：

```bash
cp config/config.example.json config/config.json
```

至少需要填写 Embedding 和生成模型。当前默认配置如下：

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

混合检索和重排的配置示例：

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

答案级评测会另外调用 Judge，相关超时可以单独配置：

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

`config/config.json` 已放进 `.gitignore`。不要把真实 API Key 写进 README、
示例配置或提交记录；如果 Key 已经出现在公开仓库里，应先去服务商后台作废
并重新生成。

## 建库

正式建库：

```bash
python scripts/build_vector_db.py
```

当前切分器会保留短名词解释和短法条，并合并在同一父章节下相邻的短条目。
如果标题后面还有子章节，则把标题路径保留到子条目里，不单独生成空 chunk。
当前结果是 1256 个 chunk。

修改 `kb/processed` 或切分逻辑后，需要重新运行建库命令。`chunks.jsonl`、
Qdrant 数据和 BM25 索引都依赖这一步，否则搜索结果仍会引用旧文档。

没有 API Key 时，可以先用伪向量走一遍流程：

```bash
python scripts/build_vector_db.py --mock
```

`--mock` 只检查读取、切分、写入和查询是否正常，结果没有语义意义。

建库后做一次检索冒烟测试：

```bash
python scripts/build_vector_db.py --query "货币基金和债券基金有什么区别"
```

## 单次问答和交互问答

先在 `config/config.json` 中配置 `llm`，例如使用硅基流动的
`Qwen/Qwen3-8B`。

单次提问：

```bash
python scripts/rag_qa.py --question "货币基金和债券基金有什么区别"
```

交互模式：

```bash
python scripts/rag_qa.py
```

普通请求的处理顺序是：问题向量化，向量和 BM25 各取 20 条，RRF 融合成
20 条，再用 `Qwen/Qwen3-Reranker-4B` 重排，取前 4 条拼成带编号的上下文。
模型回答中的 `[1]`、`[2]` 对应文末列出的检索资料。

“A 和 B 分别是什么”“A 与 B 有什么区别”这类问题会拆成两个查询，各自
检索和重排，再按主题轮流合并，防止其中一个主题挤满前 4 条。临时关闭这个
逻辑：

```bash
python scripts/rag_qa.py --no-cross-doc --question "退货运费险和免赔额分别怎么理解"
```

临时关闭重排、保留混合检索：

```bash
python scripts/rag_qa.py --no-rerank --question "设立非银行支付机构需要什么条件"
```

关闭混合检索和重排，只走纯向量检索：

```bash
python scripts/rag_qa.py --no-hybrid --no-rerank --question "设立非银行支付机构需要什么条件"
```

rerank 接口失败时，程序会回退到融合后的原始次序；BM25 加载失败时回退到
向量检索。这样服务仍能回答，但结果质量会下降，日志里会保留失败信息。

## API 和前端

启动服务：

```bash
python scripts/api_server.py
```

浏览器访问 `http://localhost:8000`。`http://localhost:8000/docs` 是 FastAPI
自动生成的接口文档。

接口列表：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/api/ask` | 一次返回完整回答 |
| `POST` | `/api/ask/stream` | 通过 SSE 先返回引用，再逐段返回正文 |
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/sessions/{session_id}/clear` | 清空指定会话 |

普通请求体：

```json
{
  "question": "货币基金主要投资什么",
  "session_id": "浏览器生成的会话 ID"
}
```

`/api/ask` 的返回结构：

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

流式接口使用以下事件：

| `type` | 内容 |
| --- | --- |
| `citations` | 检索到的来源，前端先渲染 |
| `token` | 一段生成文本 |
| `done` | 本次请求总耗时 |
| `error` | 检索、生成或知识库未命中的错误 |

前端页面在 `frontend/index.html`，HTML、CSS 和 JavaScript 都在这一个文件里，
没有 Node.js 构建步骤。主要交互包括流式输出、引用展开、拒答状态、新会话和
服务端健康检查。

## 多轮会话

会话记录保存在进程内存中，默认保留最近 5 轮，TTL 为 24 小时。服务重启后
历史会清空。这部分只用于指代消解和回答“我刚才问了什么”，金融事实仍然要
重新检索知识库。

配置示例：

```json
{
  "memory": {
    "enabled": true,
    "window_turns": 5,
    "rewrite_model": "Qwen/Qwen3-8B",
    "rewrite_timeout_seconds": 15,
    "ttl_hours": 24
  }
}
```

路由结果分三类：

| 类型 | 典型问题 | 后续动作 |
| --- | --- | --- |
| `memory` | “你刚才说的那个产品叫什么？” | 直接读会话记录 |
| `knowledge` | “保险理赔核定期限是多久？” | 直接检索 |
| `hybrid` | “那如果超时了呢？” | 先补全指代，再检索 |

## 评测

评测文件都在 `data/` 下。默认主检索集有 200 题，其中 185 道有答案，15 道
故意不给答案，用来检查拒答。运行：

```bash
python scripts/eval_rag.py
```

脚本会依次比较纯向量、向量 + rerank、BM25 + 向量、BM25 + 向量 + rerank，
以及跨文档拆分五条链路。2026-09-03 的最终检索结果如下：

| 模式 | top4 命中率 | 要点覆盖 | 平均排名 | MRR |
| --- | ---: | ---: | ---: | ---: |
| 纯向量 | 95% | 96% | 1.16 | 0.919 |
| 向量 + rerank | 99% | 100% | 1.07 | 0.970 |
| BM25 + 向量 | 93% | 95% | 1.17 | 0.907 |
| BM25 + 向量 + rerank | 99% | 100% | 1.07 | 0.970 |
| 再加跨文档拆分 | 100% | 100% | 1.08 | 0.965 |

跨文档拆分让 top4 命中率从 99% 到 100%，但平均排名从 1.07 退到 1.08，
MRR 从 0.970 降到 0.965。这是为了避免同时问两个主题时只召回其中一边，接受
一点排序损失。

答案级评测：

```bash
python scripts/eval_answer_quality.py
```

默认按类别抽取 40 题，检查最终回答的概念覆盖、拒答、引用编号、引用支持和
资料外编造。2026-09-04 使用第三版 prompt 的结果为：

| 指标 | 结果 |
| --- | ---: |
| 概念覆盖率 | 83.8% |
| 概念全对率 | 73.5% |
| 引用格式有效率 | 94.1% |
| Judge 引用支持率 | 90.9% |
| Judge 幻觉率 | 9.1%（3/33） |
| 资料不足题正确拒答率 | 100.0% |
| 有答案题误拒率 | 2.9% |

这一轮仍有 1 道 Judge 超时，所以幻觉率的分母是 33。剩下 3 道失败分别涉及
条款期限错配、把单只基金数据推广到一类基金，以及跨文档子主题推测。

幻觉固定回归集：

```bash
python scripts/eval_answer_quality.py \
  --eval data/hallucination_regression.jsonl \
  --limit 0 \
  --detail data/hallucination_regression_results.jsonl \
  --report hallucination_regression_report.md
```

这 30 题包含 4 个最早出现的失败样本、9 个同义改写和 17 个相似概念干扰。
当前 Judge 幻觉率 10.0%，引用格式有效率 100.0%，具体问题见
[hallucination_regression_report.md](hallucination_regression_report.md)。

多轮记忆评测：

```bash
python scripts/eval_multi_turn.py
```

40 个目标轮中包含 10 个记忆问题、20 个追问和 10 个话题切换。当前可用路由
成功率 100.0%，查询改写完整率 93.3%，Top4 命中率 96.7%，多轮链路成功率
97.5%。`M025` 仍是已知失败：抖音月付的使用方式只召回了部分证据。

运行回归集时，使用独立的 `--detail` 和 `--report` 路径，避免覆盖主评测的
逐题结果。

## 向量模型

| 方案 | 模型 | 维度 | 备注 |
| --- | --- | ---: | --- |
| 默认 | `BAAI/bge-m3` | 1024 | 中文和多语言检索都可用，OpenAI 兼容 API 较多 |
| 备选 | `text-embedding-v4` | 1024 | 阿里云百炼，中文表现较好 |
| 对比项 | `text-embedding-3-small` | 1536 | 中文表现一般，只用于横向测试 |

更换模型时，要同时修改配置并重建向量库。不同模型的向量维度或向量空间
不一致，直接混用会让检索结果失去意义。

## 存储和依赖

默认使用 Qdrant 本地模式，数据在 `data/vector_db/`，不需要 Docker。没有
安装 `qdrant-client` 时会改用 `data/vector_db_local/` 下的 numpy 存储。
后者适合本机试验和万级以下数据，不适合继续扩展。

每条 chunk 都保存标题、章节路径、来源和页码等元数据，回答中的引用直接使用
这些字段。

## 已知限制

- 检索评测的答案关键词来自知识库原文，100% top4 命中率不代表真实开放问题
  也能达到同样效果。
- 答案级主评测默认只抽 40 题，样本波动较大。遇到几个百分点以内的变化，
  目前不会直接判定方案变好或变差。
- Judge 和生成模型使用同一服务时，接口超时会同时影响答案和判定结果。报告
  会把超时单独排除，但这会改变有效样本数。
- 多主题检索已经能覆盖大多数“分别问”的问题，产品名较长的追问仍可能只
  命中一个子主题。
- 会话存储在内存中，服务重启即丢失；当前没有面向多实例部署的共享存储。

## 数据和合规

法规文本和基金披露信息来自公开渠道，基金页面为公开访问内容。仓库中的资料
只用于个人学习和项目测试，不用于商业分发。再次使用时应保留原始发布机构
和页面来源。
