"""混合检索模块：BM25 粗召回 + 向量召回 + RRF 融合。

设计目标：
    1. BM25 抓住精确词：费率名、法规术语、金额、比例、条款名；
    2. 向量检索抓住语义：同义改写、口语化提问、跨表述匹配；
    3. RRF 只使用排名融合两路结果，避免 BM25 分数和余弦分数直接相加；
    4. 融合后的候选仍交给 reranker 精排，保证最终进入 LLM 的上下文质量。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parent.parent


def tokenize_for_bm25(text: str) -> list[str]:
    """中文/英文混合分词。

    这里用 jieba search 模式：
        - 会返回完整词和更细的子词，方便匹配法规条款里的短术语；
        - 对“非银行支付机构”“销售服务费”“注册资本”这类中文专有词更友好。

    再额外用正则保留英文、数字和小数点，保证 “ETF” “R1” “0.35%”
    “人民币1亿元”这类字面信息不会被过度切碎。
    """
    lower_text = (text or "").lower()
    english_number_tokens = re.findall(r"[a-z0-9][a-z0-9_.-]*", lower_text)
    chinese_tokens = jieba.lcut_for_search(lower_text)
    all_tokens = english_number_tokens + chinese_tokens
    # 去掉空白和纯标点；不去重，让 BM25 自然利用查询里的词频信息。
    return [token for token in all_tokens if token.strip()]


def document_key(metadata: dict) -> str:
    """构造稳定的 chunk key。

    chunks.jsonl、Qdrant payload、BM25 索引都使用 doc_id + chunk_index，
    这样 BM25 命中的文档可以直接和向量命中的文档合并去重。
    """
    return f"{metadata.get('doc_id', 'unknown_doc')}:{metadata.get('chunk_index', 0)}"


class BM25Index:
    """基于 chunks.jsonl 构建的本地 BM25 索引。

    索引字段是 title + section + 正文：
        - title 帮助基金产品名、法规名命中；
        - section 帮助章节、条款主题命中；
        - 正文承载具体答案。
    """

    def __init__(self, manifest_path: Path):
        self.records: list[Document] = []
        self.keys: list[str] = []
        corpus: list[list[str]] = []

        with manifest_path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                payload = dict(item.get("payload", {}))
                text = item.get("text", "")
                search_text = "\n".join(
                    [
                        str(payload.get("title", "")),
                        str(payload.get("section", "")),
                        text,
                    ]
                )
                self.records.append(
                    Document(page_content=text, metadata=payload)
                )
                self.keys.append(document_key(payload))
                corpus.append(tokenize_for_bm25(search_text))

        if not self.records:
            raise RuntimeError(f"BM25 索引为空：{manifest_path}")

        self.bm25 = BM25Okapi(corpus)

    def search(self, question: str, top_k: int) -> list[tuple[Document, float]]:
        """返回 BM25 分数最高的 Document 列表。

        返回格式与 rag_qa.retrieve() 对齐：[(Document, score), ...]，
        方便后续 RRF 和 reranker 复用同一套处理逻辑。
        """
        tokens = tokenize_for_bm25(question)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )[:top_k]
        return [(self.records[index], float(scores[index])) for index in ranked_indices]


def build_bm25_index(manifest_path: str | Path) -> BM25Index:
    """加载 chunks.jsonl 并构建 BM25 索引。"""
    return BM25Index(ROOT / manifest_path)


def rrf_merge(
    vector_hits: list[tuple[Document, float]],
    bm25_hits: list[tuple[Document, float]],
    top_k: int,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[tuple[Document, float]]:
    """用倒数排名融合两路检索结果。

    注意：返回的 score 是 RRF 融合分，不是向量相似度，也不是 BM25 分。
    它只用于粗排展示；最终推荐仍应交给 reranker。
    """
    fused: dict[str, dict] = {}

    for hits, weight, source in [
        (vector_hits, vector_weight, "vector"),
        (bm25_hits, bm25_weight, "bm25"),
    ]:
        for rank, (doc, _raw_score) in enumerate(hits, start=1):
            key = document_key(doc.metadata or {})
            if key not in fused:
                fused[key] = {"doc": doc, "score": 0.0, "sources": []}
            fused[key]["score"] += weight / (rrf_k + rank)
            fused[key]["sources"].append(source)

    ordered = sorted(
        fused.values(), key=lambda item: item["score"], reverse=True
    )[:top_k]
    return [(item["doc"], float(item["score"])) for item in ordered]


def retrieve_hybrid(
    store,
    bm25_index: BM25Index | None,
    question: str,
    cfg: dict,
    top_k: int,
) -> list[tuple[Document, float]]:
    """执行 BM25 + 向量的 RRF 粗融合检索。

    bm25_index 为 None 时自动退化为纯向量检索，保证上层问答链路稳定。
    """
    hybrid_cfg = cfg.get("hybrid", {})
    vector_candidate_k = int(hybrid_cfg.get("vector_candidate_k", 20))
    bm25_candidate_k = int(hybrid_cfg.get("bm25_candidate_k", 20))
    rrf_k = int(hybrid_cfg.get("rrf_k", 60))
    vector_weight = float(hybrid_cfg.get("vector_weight", 1.0))
    bm25_weight = float(hybrid_cfg.get("bm25_weight", 1.0))

    if bm25_index is None:
        return store.similarity_search_with_score(question, k=top_k)

    vector_hits = store.similarity_search_with_score(
        question, k=vector_candidate_k
    )
    bm25_hits = bm25_index.search(question, top_k=bm25_candidate_k)
    return rrf_merge(
        vector_hits,
        bm25_hits,
        top_k=top_k,
        rrf_k=rrf_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )
