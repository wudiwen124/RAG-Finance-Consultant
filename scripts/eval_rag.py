"""
评测 RAG 检索效果：对比纯向量、向量+rerank、BM25+向量、
BM25+向量+rerank 四种模式。

评测集支持两类题目：
1. answerable=true：按期望关键词/同义表达组评测 top_k 检索；
2. answerable=false：资料不足题，不进入检索命中率统计，
   由 scripts/eval_answer_quality.py 单独评测拒答和编造。

用法：
    python scripts/eval_rag.py
    python scripts/eval_rag.py --config config/config.json

输出：
    控制台逐题对比
    rag_effect_log.txt（自动生成，含数据结论和优化说明）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from rag_qa import (  # 复用问答脚本里的检索与重排函数
    DEFAULT_CONFIG,
    retrieve_for_answer,
    get_bm25_index,
    build_vector_store,
    load_config,
    rerank_hits,
    retrieve,
)
ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "data" / "eval_questions.jsonl"
LOG_FILE = ROOT / "rag_effect_log.txt"

MODES = [
    ("vector", "纯向量"),
    ("vector_rerank", "向量+rerank"),
    ("hybrid", "BM25+向量"),
    ("hybrid_rerank", "混合+rerank"),
    ("cross_hybrid_rerank", "混合+rerank+拆分"),
]


def load_cases(path: Path) -> list[dict]:
    """读取评测集，每行一个 JSON 对象。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def expected_groups(case: dict) -> list[tuple[str, ...]]:
    """把期望关键词转换成若干个“要点组”。

    旧格式 expected_keywords 里的每一项仍然是独立要点；
    新格式 keyword_groups 里的每个 accept 列表表示同一要点的可接受表达。
    例如 ["较高风险", "预期风险较高"] 中任意一个出现，就认为该要点命中。
    """
    groups: list[tuple[str, ...]] = [
        (keyword,) for keyword in case.get("expected_keywords", [])
    ]
    for group in case.get("keyword_groups", []):
        accepts = tuple(group.get("accept", []))
        if accepts:
            groups.append(accepts)
    return groups


def is_answerable(case: dict) -> bool:
    """判断题目是否预期知识库中有答案；资料不足题单独评测拒答。"""
    return bool(case.get("answerable", True))


def keyword_coverage(hits: list[tuple], groups: list[tuple[str, ...]]) -> tuple[float, int | None]:
    """计算 top_k 整体答案覆盖情况。

    金融答案经常分散在多个 chunk 里，例如“设立支付机构”的条件会分别
    出现在许可条款、注册资本条款、股东资质条款。因此不能要求同一条
    chunk 包含所有关键词，而应看 top_k 整体是否覆盖答案要点。

    返回:
        coverage: 覆盖到的期望关键词比例，1.0 表示 top_k 覆盖全部答案点
        first_rank: 第一条包含任一期望关键词的排名，用于近似答案排名
    """
    combined = ""
    first_rank: int | None = None
    for rank, (doc, _score) in enumerate(hits, start=1):
        text = " ".join(
            [
                str((doc.metadata or {}).get("title", "")),
                str((doc.metadata or {}).get("section", "")),
                doc.page_content or "",
            ]
        )
        combined += "\n" + text
        if first_rank is None and any(any(k in text for k in group) for group in groups):
            first_rank = rank
    covered = sum(1 for group in groups if any(k in combined for k in group))
    coverage = covered / len(groups) if groups else 0.0
    return coverage, first_rank


def run_mode(
    cfg: dict,
    store,
    bm25_index,
    cases: list[dict],
    mode: str,
) -> tuple[list[dict], dict]:
    """跑一种检索模式，返回逐题明细和汇总指标。"""
    # 资料不足题没有期望证据，不应污染 top4 检索命中率；
    # 它们的关键指标是“是否拒答”，由答案级评测负责。
    cases = [case for case in cases if is_answerable(case)]
    rr_cfg = cfg.get("rerank", {})
    hybrid_cfg = cfg.get("hybrid", {})
    qa_cfg = cfg.get("qa", {})
    top_k = int(qa_cfg.get("top_k", 4))
    candidate_k = int(rr_cfg.get("candidate_k", 20))
    use_rerank = mode.endswith("_rerank")
    use_hybrid = mode.startswith("hybrid")
    use_cross_doc = mode == "cross_hybrid_rerank"

    rows: list[dict] = []
    for case in cases:
        question = case["question"]
        expected = expected_groups(case)
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

        coverage, rank = keyword_coverage(hits, expected)
        rows.append(
            {
                "question": question,
                "category": case.get("category", "unspecified"),
                "answerable": True,
                "expected": expected,
                "coverage": coverage,
                "rank": rank,
                "titles": [
                    (doc.metadata or {}).get("title", "") for doc, _score in hits
                ],
            }
        )

    hit_count = sum(r["coverage"] == 1.0 for r in rows)
    ranks = [r["rank"] for r in rows if r["rank"] is not None]
    metrics = {
        "hit_rate": hit_count / len(rows) if rows else 0.0,
        "hit_count": hit_count,
        "keyword_coverage": (
            sum(r["coverage"] for r in rows) / len(rows) if rows else 0.0
        ),
        "avg_rank": (sum(ranks) / len(ranks)) if ranks else 0.0,
        "mrr": sum(1.0 / r["rank"] for r in rows if r["rank"] is not None)
        / len(rows)
        if rows
        else 0.0,
    }
    return rows, metrics


def write_log(
    results: dict[str, tuple[list[dict], dict]],
    cfg: dict,
) -> None:
    """把对比结果写成可读的 txt 报告。"""
    lines: list[str] = []
    lines.append("RAG 检索效果优化记录（BM25 + 向量 + rerank 实验）")
    lines.append("=" * 42)
    lines.append(f"评测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"向量模型：{cfg['embedding']['model']}")
    lines.append(f"重排模型：{cfg.get('rerank', {}).get('model', 'N/A')}")
    first_rows = results["vector"][0]
    lines.append(f"评测题数：{len(first_rows)}")
    lines.append(f"top_k：{cfg.get('qa', {}).get('top_k', 4)}")
    lines.append(f"向量粗召回：{cfg.get('rerank', {}).get('candidate_k', 20)} 条")
    lines.append(
        f"BM25 粗召回：{cfg.get('hybrid', {}).get('bm25_candidate_k', 20)} 条"
    )
    lines.append(f"RRF 平滑常数 k：{cfg.get('hybrid', {}).get('rrf_k', 60)}")
    lines.append("")
    lines.append("结果")
    lines.append("-" * 42)
    lines.append(
        f"{'模式':<20}{'top4命中率':>12}{'要点覆盖':>10}{'平均排名':>10}{'MRR':>10}"
    )
    for mode, label in MODES:
        _rows, metrics = results[mode]
        lines.append(
            f"{label:<20}{metrics['hit_rate']:>11.0%}"
            f"{metrics['keyword_coverage']:>10.0%}"
            f"{metrics['avg_rank']:>10.2f}{metrics['mrr']:>10.3f}"
        )
    lines.append("")
    lines.append("变化")
    lines.append("-" * 42)

    def compare(left: str, right: str, text: str) -> None:
        lm, rm = results[left][1], results[right][1]
        lines.append(
            f"- {text}: top4 {lm['hit_rate']:.0%}->{rm['hit_rate']:.0%}, "
            f"覆盖 {lm['keyword_coverage']:.0%}->{rm['keyword_coverage']:.0%}, "
            f"平均排名 {lm['avg_rank']:.2f}->{rm['avg_rank']:.2f}, "
            f"MRR {lm['mrr']:.3f}->{rm['mrr']:.3f}。"
        )

    compare("vector", "vector_rerank", "向量检索后加 rerank")
    compare("vector", "hybrid", "BM25+向量替代纯向量")
    compare("hybrid", "hybrid_rerank", "混合检索后加 rerank")
    compare("hybrid_rerank", "cross_hybrid_rerank", "跨文档拆分相对普通混合")
    compare("vector", "cross_hybrid_rerank", "最终方案相对纯向量")
    lines.append(
        "- top4 命中率和要点覆盖越高越好；平均排名越小越好；MRR 越大越好。"
    )
    lines.append(
        "- BM25 主要补精确词和专有名词召回；RRF 只按排名融合"
    )
    lines.append(
        "  两路候选；最后由 reranker 重排。"
    )
    lines.append("")
    lines.append("逐题对比（要点覆盖 100% 表示 top4 覆盖全部期望答案点）")
    lines.append("-" * 42)
    rows_by_mode = {mode: results[mode][0] for mode, _label in MODES}
    for index, base in enumerate(rows_by_mode["vector"]):
        lines.append(f"Q: {base['question']}")
        for mode, label in MODES:
            row = rows_by_mode[mode][index]
            rank = row["rank"] if row["rank"] is not None else "-"
            lines.append(f"  {label}: 覆盖{row['coverage']:.0%}/排名{rank}")
        lines.append("")
    lines.append("优化说明")
    lines.append("-" * 42)
    lines.append(
        "1. 原来只有一阶段向量检索：bge-m3 负责把问题和 chunk 都变成向量，"
    )
    lines.append("   再按整体语义相似度排序。它适合粗召回，但排序不够精细。")
    lines.append(
        "2. 第二阶段加入 rerank：先向量召回 20 条候选，再用 "
    )
    lines.append(
        "   Qwen3-Reranker-4B 逐条计算“当前问题与这条 chunk”的相关性。"
    )
    lines.append(
        "3. 第三阶段加入 BM25+向量混合检索：BM25 用 jieba 分词，索引"
    )
    lines.append(
        "   title/section/正文；向量继续用 bge-m3。两路各取 20 条，"
    )
    lines.append(
        "   用 RRF(k=60) 合并成 20 条，再交给 reranker 精排。"
    )
    lines.append(
        "4. RRF 只使用排名融合，避免 BM25 分数和向量相似度分数直接相加。"
    )
    lines.append(
        "5. 只把重排后的 4 条送进大模型，减少无关资料干扰。"
    )
    lines.append(
        "6. 如果 rerank 接口异常，脚本会自动回退；BM25 索引异常时，"
    )
    lines.append("   混合检索也会退化到向量检索，保证问答可用。")
    lines.append(
        "7. 评测中的“要点覆盖”看 top4 整体是否覆盖答案关键词；金融答案"
    )
    lines.append("   常分散在多条条款或多个名词解释里，不能只看单条 chunk。")
    lines.append(
        "8. 跨文档模式会先把“A和B分别……”拆成两个独立 query；每个子查询"
    )
    lines.append("   分别混合检索和 rerank，然后按主题配额轮流合并。当前每个")
    lines.append("   子主题最多占 2 条，避免单一主题挤占整个 top4。")
    LOG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG 检索效果评测")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--eval", type=Path, default=EVAL_FILE)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cases = load_cases(args.eval)
    print(f"加载评测集：{len(cases)} 题")
    print("连接向量库...")
    store = build_vector_store(cfg)

    print("加载 BM25 索引...")
    bm25_index = get_bm25_index(cfg)
    results: dict[str, tuple[list[dict], dict]] = {}
    for index, (mode, label) in enumerate(MODES, start=1):
        print(f"\n[{index}/{len(MODES)}] 评测 {label}...")
        rows, metrics = run_mode(cfg, store, bm25_index, cases, mode)
        results[mode] = (rows, metrics)
        print(
            f"top4命中率 {metrics['hit_rate']:.0%}，"
            f"平均排名 {metrics['avg_rank']:.2f}，MRR {metrics['mrr']:.3f}"
        )

    write_log(results, cfg)
    lines = [
        "",
        "评测集构成",
        "-" * 42,
        f"总题数：{len(load_cases(args.eval))}",
        f"检索评测题数：{sum(is_answerable(case) for case in load_cases(args.eval))}",
        f"资料不足题数：{sum(not is_answerable(case) for case in load_cases(args.eval))}",
    ]
    categories: dict[str, int] = {}
    for case in load_cases(args.eval):
        category = case.get("category", "unspecified")
        categories[category] = categories.get(category, 0) + 1
    for category, count in sorted(categories.items()):
        lines.append(f"- {category}: {count} 题")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已生成效果报告：{LOG_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
