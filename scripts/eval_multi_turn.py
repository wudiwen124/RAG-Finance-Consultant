"""多轮记忆评测：评估会话路由、查询改写、记忆召回和检索命中。

与 200 题单轮检索评测不同，这里每条样例都是一个完整的会话片段：
    history + 当前问题

评测不会生成最终自然语言答案，而是聚焦多轮链路的四个前置环节：
1. 意图路由：应查记忆、查知识库，还是两者都要；
2. 查询改写：追问是否能补全代词、省略主语和历史主体；
3. 记忆召回：memory 问题能否从历史里准确取回原问题/原回答；
4. 证据检索：knowledge/hybrid 问题能否召回正确资料。

用法：
    python scripts/eval_multi_turn.py
    python scripts/eval_multi_turn.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from conversation import ConversationMemory, route_question
from eval_rag import keyword_coverage
from rag_qa import (
    DEFAULT_CONFIG,
    build_vector_store,
    get_bm25_index,
    load_config,
    prepare_context,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "data" / "multi_turn_eval.jsonl"
DEFAULT_DETAIL_FILE = ROOT / "data" / "multi_turn_eval_results.jsonl"
DEFAULT_REPORT_FILE = ROOT / "multi_turn_eval_report.md"


def load_cases(path: Path) -> list[dict]:
    """读取 JSONL 多轮评测集。"""
    cases: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} JSON 解析失败: {exc}") from exc
            if "history" not in item or "question" not in item:
                raise ValueError(f"{path}:{line_number} 缺少 history 或 question 字段")
            cases.append(item)
    return cases


def normalize_text(text: str) -> str:
    """统一空白，便于做确定性关键词匹配。"""
    return " ".join((text or "").split())


def keyword_group_coverage(text: str, groups: list[list[str] | tuple[str, ...]]) -> float:
    """一个要点组内任意表达命中即算覆盖该要点。"""
    normalized = normalize_text(text)
    if not groups:
        return 0.0
    covered = sum(
        any(str(keyword) in normalized for keyword in group)
        for group in groups
        if group
    )
    valid_groups = [group for group in groups if group]
    return covered / len(valid_groups) if valid_groups else 0.0


def as_groups(raw_groups) -> list[tuple[str, ...]]:
    """把 JSON 里的 accept 数组转换成 eval_rag.keyword_coverage 需要的元组。"""
    groups: list[tuple[str, ...]] = []
    for group in raw_groups or []:
        accepts = tuple(str(keyword) for keyword in group.get("accept", []))
        if accepts:
            groups.append(accepts)
    return groups


def ratio(rows: list[dict], key: str) -> float:
    """安全计算布尔字段比例。"""
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key)) / len(rows)


def average(rows: list[dict], key: str) -> float:
    """安全计算数值字段平均值。"""
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def evaluate_case(
    cfg: dict,
    store,
    bm25_index,
    case: dict,
    session_id: str,
) -> dict:
    """评测一个多轮样例，返回逐题明细。"""
    question = str(case["question"])
    expected_intent = str(case.get("expected_intent", "knowledge"))
    history = list(case.get("history", []))

    # 每个评测用例使用独立的 session_id，避免前一条用例污染后一条。
    memory = ConversationMemory(
        window_turns=int(cfg.get("memory", {}).get("window_turns", 5)),
        ttl_hours=float(cfg.get("memory", {}).get("ttl_hours", 24)),
    )
    for message in history:
        memory.append_message(
            session_id, str(message.get("role", "user")), str(message.get("content", ""))
        )

    start = time.perf_counter()
    route_result, actual_history = route_question(
        cfg, memory, session_id, question
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    query_groups = as_groups(case.get("expected_query_groups"))
    memory_groups = as_groups(case.get("expected_memory_groups"))
    evidence_groups = as_groups(case.get("evidence_groups"))

    intent_correct = route_result.intent == expected_intent
    memory_answer_correct = False
    query_coverage = 0.0
    evidence_coverage = 0.0
    first_relevant_rank: int | None = None
    hit_titles: list[str] = []
    retrieval_attempted = False
    pipeline_success = False

    if expected_intent == "memory":
        memory_answer = route_result.direct_answer
        memory_answer_correct = (
            bool(memory_answer)
            and keyword_group_coverage(memory_answer, memory_groups) == 1.0
        )
        pipeline_success = memory_answer_correct
    else:
        # 如果模型把知识库问题误路由成 memory，就不允许再退回原始问题检索；
        # 否则会掩盖路由错误，报告也无法解释“记忆模块为什么失败”。
        if route_result.intent == "memory":
            retrieval_attempted = False
            query_coverage = keyword_group_coverage(
                route_result.standalone_query, query_groups
            )
        else:
            retrieval_attempted = True
            retrieval_question = route_result.standalone_query
            query_coverage = keyword_group_coverage(retrieval_question, query_groups)

            # 只评测检索，不调用生成模型；这能把多轮路由问题与答案幻觉问题分开。
            context, hits = (
                ("", [])
                if not retrieval_question
                else prepare_context(
                    cfg,
                    store,
                    retrieval_question,
                    bm25_index=bm25_index,
                )
            )
            evidence_coverage, first_relevant_rank = keyword_coverage(
                hits, evidence_groups
            )
            hit_titles = [
                str((doc.metadata or {}).get("title", ""))
                for doc, _score in hits
            ]
            pipeline_success = evidence_coverage == 1.0

    forbidden_terms = [
        str(term) for term in case.get("forbidden_query_terms", [])
    ]
    query_contamination = any(
        term in route_result.standalone_query for term in forbidden_terms
    )

    return {
        "id": case.get("id", ""),
        "category": case.get("category", "unspecified"),
        "scenario": case.get("scenario", ""),
        "question": question,
        "history_turns": len(history) // 2,
        "actual_history_turns": len(actual_history) // 2,
        "expected_intent": expected_intent,
        "router_intent": route_result.intent,
        "intent_correct": intent_correct,
        "standalone_query": route_result.standalone_query,
        "router_direct_answer": route_result.direct_answer,
        "expected_memory_groups": [list(group) for group in memory_groups],
        "memory_answer_correct": memory_answer_correct,
        "expected_query_groups": [list(group) for group in query_groups],
        "query_coverage": query_coverage,
        "query_rewrite_complete": bool(query_groups) and query_coverage == 1.0,
        "evidence_groups": [list(group) for group in evidence_groups],
        "retrieval_attempted": retrieval_attempted,
        "evidence_coverage": evidence_coverage,
        "retrieval_hit": evidence_coverage == 1.0,
        "first_relevant_rank": first_relevant_rank,
        "hit_titles": hit_titles,
        "forbidden_query_terms": forbidden_terms,
        "query_contamination": query_contamination,
        "pipeline_success": pipeline_success,
        "latency_ms": latency_ms,
    }


def evaluate_cases(cfg: dict, cases: list[dict]) -> list[dict]:
    """初始化共享向量库/BM25，然后逐题评测。"""
    store = build_vector_store(cfg)
    bm25_index = get_bm25_index(cfg)
    rows: list[dict] = []

    for index, case in enumerate(cases, start=1):
        question = case["question"]
        print(f"[{index}/{len(cases)}] [{case.get('expected_intent', '')}] {question}")
        try:
            row = evaluate_case(
                cfg,
                store,
                bm25_index,
                case,
                session_id=f"multi-turn-eval-{case.get('id', index)}",
            )
        except Exception as exc:  # noqa: BLE001
            # 单题异常不应中断整轮评测；异常样例进入报告人工复核区。
            row = {
                "id": case.get("id", ""),
                "category": case.get("category", "unspecified"),
                "scenario": case.get("scenario", ""),
                "question": question,
                "history_turns": len(case.get("history", [])) // 2,
                "actual_history_turns": 0,
                "expected_intent": case.get("expected_intent", "knowledge"),
                "router_intent": "error",
                "intent_correct": False,
                "standalone_query": "",
                "router_direct_answer": "",
                "expected_memory_groups": case.get("expected_memory_groups", []),
                "memory_answer_correct": False,
                "expected_query_groups": case.get("expected_query_groups", []),
                "query_coverage": 0.0,
                "query_rewrite_complete": False,
                "evidence_groups": case.get("evidence_groups", []),
                "retrieval_attempted": False,
                "evidence_coverage": 0.0,
                "retrieval_hit": False,
                "first_relevant_rank": None,
                "hit_titles": [],
                "forbidden_query_terms": case.get("forbidden_query_terms", []),
                "query_contamination": False,
                "pipeline_success": False,
                "latency_ms": 0,
                "error": str(exc),
            }

        rows.append(row)
        with DEFAULT_DETAIL_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return rows


def summarize(rows: list[dict]) -> dict:
    """汇总多轮链路指标。"""
    memory_rows = [row for row in rows if row["expected_intent"] == "memory"]
    retrieval_rows = [row for row in rows if row["expected_intent"] != "memory"]
    rewrite_rows = [row for row in retrieval_rows if row["expected_query_groups"]]
    contamination_rows = [row for row in rows if row["forbidden_query_terms"]]
    valid_first_ranks = [
        float(row["first_relevant_rank"])
        for row in retrieval_rows
        if row.get("first_relevant_rank") is not None
    ]

    return {
        "total": len(rows),
        "intent_accuracy": ratio(rows, "intent_correct"),
        "memory_answer_accuracy": ratio(memory_rows, "memory_answer_correct"),
        "query_rewrite_complete_rate": ratio(rewrite_rows, "query_rewrite_complete"),
        "avg_query_coverage": average(rewrite_rows, "query_coverage"),
        "retrieval_hit_rate": ratio(retrieval_rows, "retrieval_hit"),
        "avg_evidence_coverage": average(retrieval_rows, "evidence_coverage"),
        "avg_first_relevant_rank": (
            sum(valid_first_ranks) / len(valid_first_ranks)
            if valid_first_ranks
            else 0.0
        ),
        "query_contamination_rate": ratio(contamination_rows, "query_contamination"),
        "pipeline_success_rate": ratio(rows, "pipeline_success"),
        "avg_latency_ms": average(rows, "latency_ms"),
    }


def operational_routing_rate(rows: list[dict]) -> float:
    """计算工程可用口径的路由成功率。

    严格意图口径要求 hybrid 必须返回 hybrid；但当前 hybrid 的处理动作
    与 knowledge 相同：先改写成独立查询，再检索。因此只要非 memory 问题
    实际路由到 knowledge/hybrid，就认为路由动作可用。
    """
    if not rows:
        return 0.0
    passed = 0
    for row in rows:
        if row["expected_intent"] == "memory":
            passed += row["router_intent"] == "memory"
        else:
            passed += row["router_intent"] in {"knowledge", "hybrid"}
    return passed / len(rows)


def category_table(rows: list[dict]) -> list[str]:
    """按场景类别生成小表，便于定位失败集中在哪类能力。"""
    lines = [
        "| 类别 | 题数 | 意图正确 | 链路成功 | 改写完整 | 检索命中 | 记忆正确 | 查询污染 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    categories = Counter(row["category"] for row in rows)
    for category, count in sorted(categories.items()):
        subset = [row for row in rows if row["category"] == category]
        rewrite_subset = [
            row for row in subset if row["expected_query_groups"]
        ]
        lines.append(
            f"| `{category}` | {count} | "
            f"{ratio(subset, 'intent_correct'):.0%} | "
            f"{ratio(subset, 'pipeline_success'):.0%} | "
            f"{ratio(rewrite_subset, 'query_rewrite_complete'):.0%} | "
            f"{ratio(subset, 'retrieval_hit'):.0%} | "
            f"{ratio(subset, 'memory_answer_correct'):.0%} | "
            f"{ratio(subset, 'query_contamination'):.0%} |"
        )
    return lines


def write_report(rows: list[dict], metrics: dict) -> None:
    """生成多轮记忆评测报告。"""
    lines = [
        "# 多轮会话评测",
        "",
        f"评测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 这轮怎么跑",
        "",
        f"- 评测目标轮数：{metrics['total']} 题",
        "- 会话理解：`ConversationMemory` + `route_question`",
        "- 查询改写模型：Qwen/Qwen3-8B（关闭 thinking）",
        "- 检索链路：BM25 + 向量 + RRF + Reranker + 跨文档拆分",
        "- 本轮不调用最终答案生成模型；答案级幻觉由既有答案级评测负责",
        "",
        "## 结果",
        "",
        "| 指标 | 结果 | 说明 |",
        "| --- | ---: | --- |",
        f"| 意图路由准确率 | {metrics['intent_accuracy']:.1%} | 期望 `memory/knowledge/hybrid` 与实际路由一致 |",
        f"| 可用路由成功率 | {operational_routing_rate(rows):.1%} | memory 精确路由；knowledge/hybrid 只要进入检索分支即可 |",
        f"| 记忆召回正确率 | {metrics['memory_answer_accuracy']:.1%} | memory 问题能取回期望的历史问题或回答 |",
        f"| 查询改写完整率 | {metrics['query_rewrite_complete_rate']:.1%} | 独立查询覆盖全部期望主体/意图 |",
        f"| 平均查询覆盖率 | {metrics['avg_query_coverage']:.1%} | 独立查询命中的期望要点比例 |",
        f"| Top4 检索命中率 | {metrics['retrieval_hit_rate']:.1%} | knowledge/hybrid 问题召回全部期望证据 |",
        f"| 平均证据覆盖率 | {metrics['avg_evidence_coverage']:.1%} | Top4 覆盖的期望证据要点比例 |",
        f"| 平均首个相关证据排名 | {metrics['avg_first_relevant_rank']:.2f} | 只统计有相关证据进入 Top4 的题 |",
        f"| 查询历史污染率 | {metrics['query_contamination_rate']:.1%} | 话题切换题的独立查询误带入上一轮实体 |",
        f"| 多轮链路成功率 | {metrics['pipeline_success_rate']:.1%} | memory 正确取回，或 knowledge/hybrid 全部证据命中 |",
        f"| 平均单轮理解/检索耗时 | {metrics['avg_latency_ms']:.0f} ms | 含路由 LLM、Embedding、BM25、Rerank |",
        "",
        "## 分类结果",
        "",
    ]
    lines.extend(category_table(rows))
    lines.extend(
        [
            "",
            "## 指标说明",
            "",
            "- `memory` 只评测历史召回，不做知识库检索；",
            "- `knowledge/hybrid` 只评测路由后的独立查询和 Top4 证据，不评测最终回答；",
            "- `query_contamination` 专门检查话题切换后是否把上一轮实体错误带入新查询；",
            "- 路由错误会阻止检索尝试，所以即使原问题本身可命中，也算多轮链路失败。",
            "- 部分追问在语义上可标为 `hybrid`，但当前实现只需要改写为独立查询后检索，"
            "因此路由器返回 `knowledge` 也能完成链路；严格意图口径会把它记为不一致。",
            "",
            "## 建议先看的样本",
            "",
        ]
    )

    risky = [
        row
        for row in rows
        if row.get("error") or not row["pipeline_success"] or row["query_contamination"]
    ]
    if not risky:
        lines.append("- 本轮未发现需要人工复核的失败样例。")
    for row in risky[:20]:
        reason = row.get("error") or (
            f"路由 {row['router_intent']}，独立查询 `{row['standalone_query']}`，"
            f"查询覆盖 {row['query_coverage']:.0%}，证据覆盖 {row['evidence_coverage']:.0%}"
        )
        lines.append(f"- `{row['id']}` {row['question']}：{reason}")

    DEFAULT_REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="多轮记忆评测")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--limit", type=int, default=0, help="抽样题数；0 表示全量")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cases = load_cases(args.eval)
    if args.limit > 0:
        cases = cases[: args.limit]

    print(f"加载多轮评测集：{len(cases)} 题")
    DEFAULT_DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_DETAIL_FILE.exists():
        DEFAULT_DETAIL_FILE.unlink()

    rows = evaluate_cases(cfg, cases)
    metrics = summarize(rows)
    write_report(rows, metrics)

    print("\n多轮会话结果")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"\n明细：{DEFAULT_DETAIL_FILE}")
    print(f"报告：{DEFAULT_REPORT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
