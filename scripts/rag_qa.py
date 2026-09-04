"""
基于 LangChain 的 RAG 问答层：在已构建好的向量库底座上跑通"问答闭环"。

流程（一条直线，对应 MVP）：
    用户问题
      -> 1. 命中“A和B分别/区别”结构时，先拆成两个独立子查询
      -> 2. BM25 + Qdrant 向量粗召回，RRF 融合
      -> 3. Qwen3-Reranker 对候选按“问题相关性”精排
      -> 4. 跨文档题按主题配额合并；普通题取 top_k
      -> 5. 把检索到的文本按 [1][2] 编号拼成上下文
      -> 6. 连同问题一起发给 ChatOpenAI（Qwen/Qwen3-8B）
      -> 7. 输出带编号引用的回答，并在末尾列出来源

用法：
    # 单次提问
    python scripts/rag_qa.py --question "货币基金和债券基金有什么区别"
    # 交互模式（输入 exit 退出）
    python scripts/rag_qa.py

依赖：langchain-core / langchain-openai / langchain-qdrant
（pip install langchain langchain-openai langchain-qdrant）
"""

from __future__ import annotations

import argparse  # 解析命令行参数
import json      # 读取配置文件
import sys       # 控制退出码
from pathlib import Path

import re
import requests  # 调用硅基流动 rerank 接口

from langchain_core.output_parsers import StrOutputParser  # 把模型输出转成纯文本
from langchain_core.prompts import ChatPromptTemplate     # 消息模板（system + human）
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI 兼容的对话/向量模型封装
from langchain_qdrant import QdrantVectorStore             # 对接我们已有的 Qdrant 本地库

from retrieval import build_bm25_index, document_key, retrieve_hybrid

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.json"

# 系统提示词：约束模型只依据检索资料回答、带编号引用、资料不足时拒答。
# 这是 RAG 防止幻觉的关键，后续优化可以在这里迭代。
SYSTEM_PROMPT = """你是"金融知识库问答助手"，只能基于【参考资料】回答金融问题。

资料边界：
1. 只能转写、压缩、归纳【参考资料】；禁止补充资料外的金融常识、定义、机制、数字、结论或建议。
2. 即使你认为知道答案，只要资料没有直接写出，就不要写；不要用“通常”“可能意味着”“也就是说”引入资料外推断。
3. 每个事实性结论后面都要用 [1]、[2] 这样的编号标注来源；没有对应资料支撑的话不要输出。
4. 如果资料不足以回答全部问题，必须回答“资料中未找到相关信息”；不要猜测，也不要用相关但不同的概念替代答案。
5. 多部分问题时，逐部分回答；资料只支持其中一部分时，明确写出另一部分资料中未找到。
6. 回答使用简体中文，简洁准确；末尾用“参考来源”列出实际引用过的资料标题。"""

# 识别“退款险和免赔额分别怎么理解”“A 和 B 有什么区别”这类多主题问题。
# 这里刻意使用确定性规则而不是 LLM 分解：评测可复现、延迟低、不会额外产生幻觉。
CROSS_DOC_PATTERN = re.compile(
    r"^(?P<left>[^和与]{2,80}?)(?:和|与)"
    r"(?P<right>[^和与]{2,100}?)"
    r"(?:分别)?(?P<intent>"
    r"有什么区别|有何区别|有什么不同|有何不同|分别是什么|是什么|"
    r"怎么理解|如何理解|怎么定义|如何定义|包括什么|包括哪些|"
    r"有哪些|有什么特点|有哪些特点|怎么|如何)"
    r"$"
)

# 追加在通用资料边界之后，专门针对已发现的四类幻觉：
# 1) 单基金数字被泛化；2) 补资料外交易机制；3) 条款期限/环节错配；
# 4) 跨主题资料误关联。
HALLUCINATION_GUARD_PROMPT = """
针对数字、期限、机制和资料来源，还必须遵守：
1. 数字、费率、比例、期限、金额只能用于原文中的同一个主体、产品、条款、动作和环节；禁止换算、推算、类推或迁移到其他主体/环节。
2. 单只基金的数字和机制不能写成整类基金的通用规则。如果只有单只产品资料，可以说“该基金……”，不能说“这类产品通常……”。
3. 资料没有明确写的交易机制、结算机制、行业惯例、监管规则、时间规则，一律不写；例如资料没有写 T+0，就不能说支持 T+0。
4. 不同来源只支撑各自的主体和主题。不要把存款保险、某只基金、某个产品、某部法规的规则互相借用。
5. 条款里的“核定”“先予支付”“赔偿”“解除合同”等是不同环节；回答时必须使用原文对应的环节，不能替换。
6. 如果问题要一般结论，但资料只支持个例，就说明“资料仅支持该个例”或“资料中未找到一般性结论”，不要把个例推广。
7. 回答法规中的数字和期限时，优先逐字引用原文关系；不能把“情形复杂的，应当在三十日内”改写成“可延长至三十日”，也不能新增原文没有的延长、折算、递进或例外条件。
8. 如果某个机制、替代策略或风控手段只出现在单只基金的资料里，回答一般机制时不得直接说成所有同类基金的做法；必须说明“该基金”或“资料仅支持该个例”。
9. 指数基金、ETF、ETF联接基金、LOF、单只具体基金是不同产品层级。任何数字、费率、机制、策略、风控手段、跟踪误差目标，只能用于原文写明的那个产品层级；禁止把联接基金的策略或指标说成 ETF 本体的做法，禁止把单只基金的做法说成指数基金或 ETF 的通用规则，也禁止反向迁移。
10. 关于交易问题（包括交易优势、交易便利、交易方式、交易规则），如果资料没有逐字写出 T+0，一律不得提及 T+0；即使资料写了类似“实时买卖”“随时交易”，也不能推断或补充为 T+0。
""".strip()


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """读取 config/config.json。"""
    return json.loads(path.read_text(encoding="utf-8"))


def build_vector_store(cfg: dict) -> QdrantVectorStore:
    """连接已构建好的 Qdrant 本地库。

    from_existing_collection 不会重建库，只是套上 LangChain 的读取接口。
    关键点：
        - embedding 必须和建库时用的是同一个模型（BAAI/bge-m3），
          否则问题的向量空间和库里的向量空间不一致，检索会失效；
        - path 指向本地库目录（Qdrant 本地模式无需 Docker）。
    """
    emb_cfg = cfg["embedding"]
    vs_cfg = cfg["vector_store"]
    embeddings = OpenAIEmbeddings(
        model=emb_cfg["model"],
        api_key=emb_cfg["api_key"],
        base_url=emb_cfg["api_base"],  # OpenAI 兼容地址：硅基流动
        check_embedding_ctx_length=False,  # 关闭 tiktoken 长度检查，省去下载词表
    )
    return QdrantVectorStore.from_existing_collection(
        collection_name=vs_cfg["collection"],
        embedding=embeddings,
        path=str(ROOT / vs_cfg["path"]),
    )


def retrieve(
    store: QdrantVectorStore, question: str, top_k: int, candidate_k: int | None = None
) -> list[tuple]:
    """从向量库检索 chunk。

    返回 [(Document, 相似度分数), ...]，Document.page_content 是 chunk 正文，
    Document.metadata 是我们建库时存的溯源元数据（标题/章节/来源等）。

    candidate_k 用于两阶段检索：先召回更多候选，再交给 reranker 精排；
    不启用 rerank 时直接取 top_k。
    """
    return store.similarity_search_with_score(question, k=candidate_k or top_k)


def get_bm25_index(cfg: dict):
    """按配置加载 BM25 索引；加载失败时返回 None 并退化为向量检索。"""
    hybrid_cfg = cfg.get("hybrid", {})
    if not hybrid_cfg.get("enabled", False):
        return None
    try:
        return build_bm25_index(
            hybrid_cfg.get("chunks_manifest", "kb/chunks/chunks.jsonl")
        )
    except Exception as e:  # noqa: BLE001
        print(f"!! BM25 索引加载失败，混合检索将退化为纯向量: {e}")
        return None


def rerank_hits(
    cfg: dict, question: str, hits: list[tuple], top_k: int
) -> list[tuple]:
    """用硅基流动的 rerank 接口对向量粗召回结果精排。

    为什么需要这一步：
        向量检索按“语义相似”排序，容易召回“领域相关但不是答案”的段落；
        reranker 会逐条计算 query 和文档的相关性分数，把真正包含答案的
        条款/名词解释排到前面，从而提高进入大模型上下文的质量。

    稳定性设计：
        rerank 接口失败时打印警告，并回退为原始向量排序，保证问答不中断。
    """
    rr_cfg = cfg.get("rerank", {})
    if not rr_cfg.get("enabled", False) or not hits:
        return hits[:top_k]

    # 只把正文发给 reranker；每条截断到 1500 字，避免请求数据过大
    documents = [(doc.page_content or "")[:1500] for doc, _score in hits]
    payload = {
        "model": rr_cfg.get("model", "Qwen/Qwen3-Reranker-4B"),
        "query": question,
        "documents": documents,
        "top_n": min(top_k, len(hits)),  # 只需要前 top_k 条
        "return_documents": False,
    }
    headers = {
        "Authorization": f"Bearer {rr_cfg.get('api_key', '')}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            rr_cfg.get("api_base", "https://api.siliconflow.cn/v1").rstrip("/")
            + "/rerank",
            headers=headers,
            json=payload,
            timeout=int(rr_cfg.get("timeout", 60)),
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as e:  # noqa: BLE001  网络或接口异常都不能阻塞问答
        print(f"!! rerank 失败，已回退到向量排序: {e}")
        return hits[:top_k]

    # 返回结果按 relevance_score 降序；index 对应 documents/原始 hits 的下标
    reranked: list[tuple] = []
    for item in results:
        idx = int(item["index"])
        score = float(item.get("relevance_score", 0.0))
        reranked.append((hits[idx][0], score))
    return reranked[:top_k]


def split_cross_doc_question(question: str) -> list[str]:
    """把“A 和 B 分别……”拆成多个可独立检索的问题。

    为什么不用 LLM 做第一版：
        1. 检索评测需要稳定复现，LLM 改写会引入额外波动；
        2. 多一次模型调用会增加延迟和成本；
        3. 当前失败题大多是明确的“A 和 B 分别……”结构，规则已经够用。

    规则会处理两类表达：
        - “退货运费险和免赔额分别怎么理解”
          -> ["退货运费险怎么理解", "免赔额怎么理解"]
        - “天弘余额宝货币A的费率结构和风险特征是什么”
          -> ["天弘余额宝货币A的费率结构是什么",
              "天弘余额宝货币A的风险特征是什么"]
    """
    normalized = re.sub(r"\s+", "", question or "")
    match = CROSS_DOC_PATTERN.match(normalized)
    if not match:
        return []

    left = match.group("left")
    right = match.group("right")
    intent = match.group("intent")

    # “产品A的费率结构和风险特征是什么”里，前半段包含共同主体。
    # 拆分时应把共同主体补到后半段，否则第二个子查询会丢失产品名。
    owner = ""
    left_topic = left
    if "的" in left:
        owner, left_topic = left.rsplit("的", 1)

    right_topic = right

    # 对“有什么区别”，两个子问题都不适合保留“有什么区别”这个谓语；
    # 改成查各自定义和特点，再由大模型根据两组资料比较。
    if intent in {"有什么区别", "有何区别", "有什么不同", "有何不同"}:
        left_query = f"{left_topic}是什么主要特点"
        right_query = f"{right_topic}是什么主要特点"
    else:
        left_query = f"{left_topic}{intent}"
        right_query = f"{right_topic}{intent}"

    if owner:
        left_query = f"{owner}的{left_query}"
        if not right_topic.startswith(owner):
            right_query = f"{owner}的{right_query}"

    # 去重兜底：如果规则误把同一个主题拆成两份，就没有必要做两路检索。
    queries = [left_query, right_query]
    return list(dict.fromkeys(query for query in queries if len(query) >= 4))


def retrieve_cross_doc(
    cfg: dict,
    store,
    bm25_index,
    question: str,
    use_hybrid: bool,
    use_rerank: bool,
    top_k: int,
) -> list[tuple] | None:
    """跨文档检索：按子问题分别召回，再按主题配额合并。

    返回 None 表示当前问题不是跨文档结构，或跨文档功能被关闭；
    上层应回退到普通检索。

    核心设计：
        1. 每个子问题独立做混合检索，避免原始复合 query 的语义被稀释；
        2. 每个子问题独立 rerank，让“退货险”和“免赔额”各自选出最强证据；
        3. 轮流合并并限制每个主题最多 top_k/2 条，避免一个主题占据 top4。
    """
    cross_cfg = cfg.get("cross_doc", {})
    if not cross_cfg.get("enabled", False):
        return None

    subqueries = split_cross_doc_question(question)
    if len(subqueries) < 2:
        return None

    candidate_k = int(
        cross_cfg.get(
            "per_query_candidate_k",
            int(cfg.get("rerank", {}).get("candidate_k", 20)),
        )
    )
    per_query_top_k = int(cross_cfg.get("per_query_top_k", max(top_k, 4)))
    per_topic_quota = int(cross_cfg.get("per_topic_quota", max(1, top_k // len(subqueries))))

    topic_results: list[list[tuple]] = []
    for subquery in subqueries:
        if use_hybrid:
            candidates = retrieve_hybrid(
                store, bm25_index, subquery, cfg, top_k=candidate_k
            )
        else:
            candidates = retrieve(store, subquery, candidate_k, candidate_k=candidate_k)

        if use_rerank:
            topic_hits = rerank_hits(
                cfg, subquery, candidates, top_k=per_query_top_k
            )
        else:
            topic_hits = candidates[:per_query_top_k]
        topic_results.append(topic_hits)

    # 轮流从每个主题取一条，直到达到该主题配额或全局 top_k。
    selected_keys: set[str] = set()
    merged: list[tuple] = []
    topic_counts = [0] * len(topic_results)
    cursors = [0] * len(topic_results)
    while len(merged) < top_k:
        progressed = False
        for topic_index, hits in enumerate(topic_results):
            if (
                len(merged) >= top_k
                or topic_counts[topic_index] >= per_topic_quota
                or cursors[topic_index] >= len(hits)
            ):
                continue
            doc, score = hits[cursors[topic_index]]
            cursors[topic_index] += 1
            key = document_key(doc.metadata or {})
            if key in selected_keys:
                continue
            selected_keys.add(key)
            merged.append((doc, score))
            topic_counts[topic_index] += 1
            progressed = True
        if not progressed:
            break

    # 如果一个主题证据不足，允许另一个主题补足 top4；
    # 但仍然跳过重复 chunk，保证上下文不浪费在重复内容上。
    if len(merged) < top_k:
        for hits in topic_results:
            for doc, score in hits:
                if len(merged) >= top_k:
                    break
                key = document_key(doc.metadata or {})
                if key not in selected_keys:
                    merged.append((doc, score))
                    selected_keys.add(key)
        if len(merged) > top_k:
            merged = merged[:top_k]

    return merged


def retrieve_for_answer(
    cfg: dict,
    store,
    bm25_index,
    question: str,
    use_hybrid: bool,
    use_rerank: bool,
    top_k: int,
    candidate_k: int,
    use_cross_doc: bool | None = None,
) -> list[tuple]:
    """统一检索入口：普通检索或跨文档子查询检索。"""
    if use_cross_doc is None:
        use_cross_doc = bool(cfg.get("cross_doc", {}).get("enabled", False))

    if use_cross_doc:
        cross_hits = retrieve_cross_doc(
            cfg,
            store,
            bm25_index,
            question,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            top_k=top_k,
        )
        if cross_hits is not None:
            return cross_hits

    if use_rerank:
        if use_hybrid:
            candidates = retrieve_hybrid(
                store, bm25_index, question, cfg, top_k=candidate_k
            )
        else:
            candidates = retrieve(store, question, candidate_k, candidate_k=candidate_k)
        return rerank_hits(cfg, question, candidates, top_k)

    if use_hybrid:
        return retrieve_hybrid(store, bm25_index, question, cfg, top_k=top_k)
    return retrieve(store, question, top_k)


def format_context(docs: list[tuple], max_chars: int) -> str:
    """把检索到的 chunk 拼成带编号的上下文。

    每条格式：
        [1] 标题 > 章节
        正文...
    编号规则和 system prompt 里的 [1][2] 引用对应，便于模型正确标注来源。
    """
    blocks: list[str] = []
    total = 0
    for i, (doc, _score) in enumerate(docs, start=1):
        meta = doc.metadata or {}
        # 用"标题 > 章节"作为该条资料的简短抬头
        head = f"[{i}] {meta.get('title', '未知来源')} > {meta.get('section', '')}"
        text = doc.page_content or ""
        if total + len(text) > max_chars:  # 控制上下文长度，防止超出模型窗口
            text = text[: max_chars - total]
        blocks.append(f"{head}\n{text}")
        total += len(text) + len(head)
        if total >= max_chars:
            break
    return "\n\n".join(blocks)


def answer_question(
    cfg: dict,
    store: QdrantVectorStore,
    question: str,
    use_rerank: bool | None = None,
    use_hybrid: bool | None = None,
    bm25_index=None,
    use_cross_doc: bool | None = None,
) -> tuple[str, list[tuple]]:
    """完整跑一遍：检索 -> 拼上下文 -> 生成带引用的回答。

    评测脚本和 CLI 仍然直接使用这个函数；API 的流式接口则分开调用
    prepare_context 和 stream_answer，这样检索结果可以先返回。
    """
    context, hits = prepare_context(
        cfg,
        store,
        question,
        use_rerank=use_rerank,
        use_hybrid=use_hybrid,
        bm25_index=bm25_index,
        use_cross_doc=use_cross_doc,
    )
    if not hits:
        return "未检索到相关知识，请换一种问法。", []
    answer = "".join(stream_answer(cfg, question, context))
    return answer, hits


def prepare_context(
    cfg: dict,
    store: QdrantVectorStore,
    question: str,
    use_rerank: bool | None = None,
    use_hybrid: bool | None = None,
    bm25_index=None,
    use_cross_doc: bool | None = None,
) -> tuple[str, list[tuple]]:
    """执行检索并拼上下文，返回 (context, hits)。

    API 层可以先等待这个同步函数完成，立刻把引用返回给前端，
    再让 stream_answer 逐块输出模型生成内容，减少用户的等待感。
    """
    qa_cfg = cfg.get("qa", {})
    top_k = int(qa_cfg.get("top_k", 4))
    max_chars = int(qa_cfg.get("max_context_chars", 4000))
    rr_cfg = cfg.get("rerank", {})
    hybrid_cfg = cfg.get("hybrid", {})
    if use_rerank is None:  # CLI 显式指定时优先；否则读配置
        use_rerank = bool(rr_cfg.get("enabled", False))
    if use_hybrid is None:
        use_hybrid = bool(hybrid_cfg.get("enabled", False))
    if use_hybrid and bm25_index is None:
        # 兜底：BM25 索引不可用时继续用向量检索，不让问答中断。
        use_hybrid = False

    # 1) 检索
    candidate_k = int(rr_cfg.get("candidate_k", max(top_k * 5, 20)))
    hits = retrieve_for_answer(
        cfg,
        store,
        bm25_index,
        question,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        top_k=top_k,
        candidate_k=candidate_k,
        use_cross_doc=use_cross_doc,
    )
    if not hits:
        return "", []

    # 2) 拼上下文
    context = format_context(hits, max_chars)
    return context, hits


def stream_answer(cfg: dict, question: str, context: str):
    """流式生成回答，每次 yield 一小段文本。

    ChatOpenAI 设置 streaming=True 后，远端模型会一边生成一边返回 token；
    LCEL 的 chain.stream 会把这些 token 逐步交给调用方。
    """
    llm_cfg = cfg["llm"]

    # 3) 组装生成模型（OpenAI 兼容接口 -> 硅基流动的 Qwen3）
    llm = ChatOpenAI(
        model=llm_cfg["model"],
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg["api_base"],
        temperature=float(llm_cfg.get("temperature", 0.1)),  # 低温度：答案更稳定
        streaming=True,  # 打开 OpenAI 兼容接口的流式返回
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"{SYSTEM_PROMPT}\n\n{HALLUCINATION_GUARD_PROMPT}"),
            ("human", "问题：{question}\n\n【参考资料】\n{context}"),
        ]
    )
    # LCEL 链：模板 -> 模型 -> 纯文本输出
    chain = prompt | llm | StrOutputParser()
    yield from chain.stream({"question": question, "context": context})


def main() -> int:
    ap = argparse.ArgumentParser(description="金融知识库 RAG 问答（LangChain）")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--question", type=str, help="单次提问；不填则进入交互模式")
    ap.add_argument(
        "--no-rerank",
        action="store_true",
        help="本次请求禁用 rerank（混合检索仍按配置生效）",
    )
    ap.add_argument(
        "--no-hybrid",
        action="store_true",
        help="本次请求禁用 BM25+向量混合检索，退回纯向量检索",
    )
    ap.add_argument(
        "--no-cross-doc",
        action="store_true",
        help="本次请求禁用跨文档子查询拆分",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("连接向量库...")
    store = build_vector_store(cfg)
    print("加载 BM25 索引...")
    bm25_index = get_bm25_index(cfg)

    # 单次提问模式
    if args.question:
        answer, hits = answer_question(
            cfg,
            store,
            args.question,
            use_rerank=not args.no_rerank,
            use_hybrid=not args.no_hybrid,
            bm25_index=bm25_index,
            use_cross_doc=not args.no_cross_doc,
        )
        print(f"\nQ: {args.question}\n")
        print("检索命中:")
        for doc, score in hits:
            meta = doc.metadata or {}
            print(f"  [{score:.4f}] {meta.get('title')} > {meta.get('section', '')}")
        print(f"\nA: {answer}")
        return 0

    # 交互模式：反复提问，输入 exit 退出
    print("进入问答模式，输入 exit 退出。")
    while True:
        try:
            question = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        answer, hits = answer_question(
            cfg,
            store,
            question,
            use_rerank=not args.no_rerank,
            use_hybrid=not args.no_hybrid,
            bm25_index=bm25_index,
            use_cross_doc=not args.no_cross_doc,
        )
        print("\n检索命中:")
        for doc, score in hits:
            meta = doc.metadata or {}
            print(f"  [{score:.4f}] {meta.get('title')} > {meta.get('section', '')}")
        print(f"\nA: {answer}\n")
        print("-" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
