"""临时调试脚本：检查失败题在四个检索阶段的候选变化。"""

import sys

PROJECT_SCRIPTS = r"E:\Codex\RAG-finance\scripts"
sys.path.insert(0, PROJECT_SCRIPTS)

from eval_rag import EVAL_FILE, load_cases
from rag_qa import build_vector_store, get_bm25_index, load_config, rerank_hits, retrieve
from retrieval import retrieve_hybrid


QUESTIONS = {
    "授信额度怎么确定",
    "认购费是什么费用",
    "净值估算准确吗",
    "易方达消费行业股票是什么类型基金",
}
if len(sys.argv) > 1:
    QUESTIONS = set(sys.argv[1:])


def mark_keywords(text: str, keywords: list[str]) -> str:
    return ",".join(keyword for keyword in keywords if keyword in text) or "-"


def show(title: str, hits, keywords: list[str]) -> None:
    print(f"\n{title}")
    for rank, (doc, score) in enumerate(hits, start=1):
        metadata = doc.metadata or {}
        text = (doc.page_content or "").replace("\n", " ")
        searchable = (
            str(metadata.get("title", ""))
            + " "
            + str(metadata.get("section", ""))
            + " "
            + text
        )
        print(
            f"{rank:02d} score={score:.6f} "
            f"keywords={mark_keywords(searchable, keywords)} "
            f"{metadata.get('title', '')} | {metadata.get('section', '')} | {text[:220]}"
        )


def main() -> int:
    cfg = load_config()
    cases = [case for case in load_cases(EVAL_FILE) if case["question"] in QUESTIONS]
    store = build_vector_store(cfg)
    bm25 = get_bm25_index(cfg)
    hybrid_cfg = cfg["hybrid"]

    for case in cases:
        question = case["question"]
        expected = case["expected_keywords"]
        vector_hits = retrieve(
            store,
            question,
            hybrid_cfg["vector_candidate_k"],
            candidate_k=hybrid_cfg["vector_candidate_k"],
        )
        bm25_hits = bm25.search(question, hybrid_cfg["bm25_candidate_k"])
        fused_hits = retrieve_hybrid(
            store,
            bm25,
            question,
            cfg,
            top_k=hybrid_cfg["fused_candidate_k"],
        )
        reranked_hits = rerank_hits(cfg, question, fused_hits, len(fused_hits))

        print("\n" + "=" * 120)
        print("Q:", question)
        print("EXPECTED:", expected)
        show("VECTOR TOP20", vector_hits[:10], expected)
        show("BM25 TOP20", bm25_hits[:10], expected)
        show("RRF TOP20", fused_hits[:10], expected)
        show("RERANK RETURNED", reranked_hits[:10], expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
