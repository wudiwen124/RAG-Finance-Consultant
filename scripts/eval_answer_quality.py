"""
RAG 答案级评测：在检索评测之上，进一步评估最终回答质量。

每个样例会执行完整链路：
    BM25 + 向量混合召回 -> rerank -> Top4 -> Qwen3 生成答案

评测维度：
1. answer_concept_coverage：生成答案是否覆盖期望答案要点；
2. answer_correct_rate：期望要点是否全部覆盖；
3. refusal_precision：资料不足题是否正确拒答；
4. citation_format_valid_rate：引用编号是否能对应到真实检索资料；
5. judge_supported_rate：LLM Judge 判断答案是否被资料支持；
6. judge_citation_supported_rate：引用是否支撑答案；
7. hallucination_rate：LLM Judge 是否发现资料外编造。

答案级评测会调用生成模型和 Judge 模型，成本高于检索评测；
默认只做 40 题分层抽样，可用 --limit 调整。

用法：
    python scripts/eval_answer_quality.py
    python scripts/eval_answer_quality.py --limit 20
    python scripts/eval_answer_quality.py --limit 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests

from eval_rag import expected_groups, is_answerable, load_cases
from rag_qa import (
    DEFAULT_CONFIG,
    answer_question,
    build_vector_store,
    get_bm25_index,
    load_config,
)

ROOT = Path(__file__).resolve().parent.parent
DETAIL_FILE = ROOT / "data" / "answer_eval_results.jsonl"
REPORT_FILE = ROOT / "answer_eval_report.md"
REFUSAL_MARKS = (
    "资料中未找到相关信息",
    "资料中未找到",
    "未检索到相关知识",
    "无法确定",
    "无法回答",
    "无法判断",
    "无法对比",
    "不能提供",
    "不提供",
)

JUDGE_STATUS_VALID = "valid"
JUDGE_STATUS_ERROR = "error"

JUDGE_SYSTEM_PROMPT = """你是严格的金融 RAG 评测员。只根据【参考资料】判断回答，不要使用你的外部知识。
输出必须是 JSON 对象，不要输出 Markdown 代码块，字段固定为：
{
  "answer_supported": true/false,
  "citation_supported": true/false,
  "fabrication_detected": true/false,
  "unsupported_claims": ["只列关键编造内容，最多5条"],
  "reason": "不超过80字的中文理由"
}
判断规则：
1. answer_supported：回答中的实质结论是否能由参考资料推出；
2. citation_supported：被引用资料是否支撑回答中的对应结论；
3. fabrication_detected：是否出现资料无法支持的事实、数字、结论或建议；
4. 如果资料不足以回答，回答明确拒答，则 fabrication_detected=false。"""


def format_judge_context(hits: list[tuple], cfg: dict) -> str:
    """构造更短的 Judge 专用上下文。

    完整 RAG 上下文可能接近 4000 字，Judge 还要同时读答案和规则，容易触发
    长请求超时。Judge 只需要核对答案里的事实，因此这里：
        1. 保留 rerank 后的资料顺序；
        2. 每条资料只截取开头若干字；
        3. 整体上下文再设一个更小的硬上限。
    """
    judge_cfg = cfg.get("answer_judge", {})
    per_hit_chars = int(judge_cfg.get("per_hit_chars", 700))
    max_context_chars = int(judge_cfg.get("max_context_chars", 3000))
    blocks: list[str] = []
    total = 0

    for index, (doc, _score) in enumerate(hits, start=1):
        metadata = doc.metadata or {}
        title = str(metadata.get("title", "未知来源"))
        section = str(metadata.get("section", ""))
        text = (doc.page_content or "")[:per_hit_chars]
        block = f"[{index}] {title} > {section}\n{text}"
        if total + len(block) > max_context_chars:
            break
        blocks.append(block)
        total += len(block) + 2

    return "\n\n".join(blocks)


def error_judge(reason: str, error: str, timed_out: bool = False) -> dict:
    """构造 Judge 异常结果；异常不能默认算幻觉，也不能算支持。"""
    return {
        "answer_supported": False,
        "citation_supported": False,
        "fabrication_detected": False,
        "unsupported_claims": [],
        "reason": reason,
        "judge_status": JUDGE_STATUS_ERROR,
        "judge_error": error,
        "judge_timeout": timed_out,
    }


def normalize_text(text: str) -> str:
    """统一大小写和空白，减少无意义的字面匹配失败。"""
    return re.sub(r"\s+", "", (text or "").lower())


def keyword_group_coverage(answer: str, groups: list[tuple[str, ...]]) -> float:
    """计算最终回答覆盖了多少个答案要点组。"""
    normalized = normalize_text(answer)
    if not groups:
        return 0.0
    covered = sum(any(normalize_text(k) in normalized for k in group) for group in groups)
    return covered / len(groups)


def looks_like_refusal(answer: str) -> bool:
    """判断是否为“整题拒答”。

    只把“开头第一句就是拒答”的答案视为整题拒答。多部分问题可能已经
    回答了有证据的部分，只在末尾补充“另一部分资料中未找到”，这类答案
    属于带边界说明的实质回答，不应误判成拒答。
    """
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    if not lines:
        return False
    first_line = lines[0]
    return any(mark in first_line for mark in REFUSAL_MARKS)


def citation_numbers(answer: str) -> list[int]:
    """提取回答里的 [1] [2] 等引用编号。"""
    return [int(number) for number in re.findall(r"\[(\d+)\]", answer or "")]


def citation_format_valid(answer: str, hit_count: int, refusal: bool) -> bool:
    """检查引用编号是否都能对应本次真实检索到的资料。"""
    numbers = citation_numbers(answer)
    if refusal:
        return not numbers
    return bool(numbers) and all(1 <= number <= hit_count for number in numbers)


def judge_answer(cfg: dict, question: str, context: str, answer: str) -> dict:
    """调用 LLM Judge，只依据本次检索上下文判断答案和引用。

    稳定性策略：
        1. 网络超时/连接异常/5xx 会重试；
        2. 4xx（例如鉴权错误）不重试；
        3. 所有失败都标记为 Judge 异常，不默认计为幻觉。
    """
    llm_cfg = cfg["llm"]
    judge_cfg = cfg.get("answer_judge", {})
    max_attempts = max(1, int(judge_cfg.get("max_attempts", 3)))
    timeout_seconds = int(judge_cfg.get("timeout_seconds", 60))
    retry_delay_seconds = float(judge_cfg.get("retry_delay_seconds", 2))
    max_question_chars = int(judge_cfg.get("max_question_chars", 500))
    max_answer_chars = int(judge_cfg.get("max_answer_chars", 2500))

    payload = {
        "model": llm_cfg["model"],
        "temperature": 0,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"问题：{(question or '')[:max_question_chars]}\n\n"
                    f"【参考资料】\n{context}\n\n"
                    f"【待评测回答】\n{(answer or '')[:max_answer_chars]}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {llm_cfg.get('api_key', '')}",
        "Content-Type": "application/json",
    }
    last_error = ""
    last_timed_out = False

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                llm_cfg.get("api_base", "https://api.siliconflow.cn/v1").rstrip("/")
                + "/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            # 有些兼容接口即使指定 JSON 模式也可能包裹代码块，这里做防御性解析。
            content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE)
            parsed = json.loads(content)
            return {
                "answer_supported": bool(parsed.get("answer_supported", False)),
                "citation_supported": bool(parsed.get("citation_supported", False)),
                "fabrication_detected": bool(parsed.get("fabrication_detected", False)),
                "unsupported_claims": list(parsed.get("unsupported_claims", [])[:5]),
                "reason": str(parsed.get("reason", "")),
                "judge_status": JUDGE_STATUS_VALID,
                "judge_error": "",
                "judge_timeout": False,
            }
        except requests.Timeout as exc:
            last_error = f"Judge 请求超时（第 {attempt}/{max_attempts} 次）：{exc}"
            last_timed_out = True
        except requests.ConnectionError as exc:
            last_error = f"Judge 连接异常（第 {attempt}/{max_attempts} 次）：{exc}"
            last_timed_out = False
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else 0
            last_error = f"Judge HTTP {status_code}：{exc}"
            last_timed_out = False
            # 4xx 是请求配置/权限/参数问题，重试大概率没有意义。
            if 400 <= status_code < 500:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = f"Judge 返回解析异常：{exc}"
            last_timed_out = False

        if attempt < max_attempts:
            time.sleep(retry_delay_seconds * attempt)

    return error_judge(
        f"Judge 调用失败：{last_error}",
        last_error,
        timed_out=last_timed_out,
    )


def stratified_sample(cases: list[dict], limit: int) -> list[dict]:
    """按类别轮询抽样，保证新增题型和资料不足题都会被覆盖。"""
    if limit <= 0 or len(cases) <= limit:
        return cases

    by_category: dict[str, list[dict]] = {}
    for case in cases:
        by_category.setdefault(case.get("category", "unspecified"), []).append(case)

    selected: list[dict] = []
    categories = sorted(by_category)
    while len(selected) < limit:
        progressed = False
        for category in categories:
            if by_category[category]:
                selected.append(by_category[category].pop(0))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def evaluate_answer_cases(
    cfg: dict,
    cases: list[dict],
    limit: int,
    detail_file: Path = DETAIL_FILE,
) -> list[dict]:
    """执行完整 RAG 并计算确定性指标 + LLM Judge 指标。"""
    store = build_vector_store(cfg)
    bm25_index = get_bm25_index(cfg)
    sampled = stratified_sample(cases, limit)
    rows: list[dict] = []

    for index, case in enumerate(sampled, start=1):
        question = case["question"]
        answerable = is_answerable(case)
        print(f"[{index}/{len(sampled)}] {question}")
        try:
            answer, hits = answer_question(cfg, store, question, bm25_index=bm25_index)
        except Exception as exc:  # noqa: BLE001
            answer = f"评测执行失败：{exc}"
            hits = []

        groups = expected_groups(case)
        refusal = looks_like_refusal(answer)
        coverage = keyword_group_coverage(answer, groups)
        refusal_correct = (not answerable and refusal) or (answerable and not refusal)
        citation_valid = citation_format_valid(answer, len(hits), refusal)
        # Judge 只做事实核对，不需要完整生成上下文；短输入能显著降低超时概率。
        context = format_judge_context(hits, cfg)
        # 资料不足题已经用拒答标记判定；Judge 主要针对有答案的题。
        judge = {"answer_supported": False, "citation_supported": False, "fabrication_detected": False, "unsupported_claims": [], "reason": "资料不足题未调用 Judge", "judge_error": ""}
        if answerable:
            judge = judge_answer(cfg, question, context, answer)

        rows.append(
            {
                "id": case.get("id", ""),
                "category": case.get("category", "unspecified"),
                "question": question,
                "answerable": answerable,
                "answer": answer,
                "expected_groups": [list(group) for group in groups],
                "answer_concept_coverage": coverage,
                "answer_keyword_correct": coverage == 1.0,
                "refusal": refusal,
                "refusal_correct": refusal_correct,
                "citation_numbers": citation_numbers(answer),
                "citation_format_valid": citation_valid,
                "hit_titles": [
                    (doc.metadata or {}).get("title", "") for doc, _score in hits
                ],
                "judge": judge,
            }
        )

        DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
        detail_file.parent.mkdir(parents=True, exist_ok=True)
        with detail_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
    return rows


def metric(rows: list[dict], predicate) -> float:
    """安全计算比例；分母为 0 时返回 0。"""
    if not rows:
        return 0.0
    return sum(1 for row in rows if predicate(row)) / len(rows)


def refresh_derived_fields(rows: list[dict]) -> list[dict]:
    """用当前判定口径重算确定性字段，供不重复调用模型时修复历史明细。"""
    for row in rows:
        refusal = looks_like_refusal(row["answer"])
        row["refusal"] = refusal
        row["refusal_correct"] = (not row["answerable"] and refusal) or (
            row["answerable"] and not refusal
        )
        row["citation_format_valid"] = citation_format_valid(
            row["answer"], len(row.get("hit_titles", [])), refusal
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    """汇总答案级指标。"""
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    # Judge 超时/解析失败不能等价于“有幻觉”；语义 Judge 指标只统计有效样本。
    valid_judge = [
        row for row in answerable if not row["judge"].get("judge_error")
    ]
    def judge_metric(predicate) -> float:
        return metric(valid_judge, predicate)

    return {
        "total": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "answer_concept_coverage": sum(row["answer_concept_coverage"] for row in answerable) / len(answerable) if answerable else 0.0,
        "answer_correct_rate": metric(answerable, lambda row: row["answer_keyword_correct"]),
        "answerable_refusal_rate": metric(answerable, lambda row: row["refusal"]),
        "refusal_precision": metric(unanswerable, lambda row: row["refusal"]),
        "citation_format_valid_rate": metric(answerable, lambda row: row["citation_format_valid"]),
        "judge_supported_rate": judge_metric(lambda row: row["judge"]["answer_supported"]),
        "judge_citation_supported_rate": judge_metric(
            lambda row: row["judge"]["citation_supported"]
        ),
        "hallucination_rate": judge_metric(
            lambda row: row["judge"]["fabrication_detected"]
        ),
        "judge_error_rate": metric(answerable, lambda row: bool(row["judge"].get("judge_error"))),
        "judge_timeout_rate": metric(
            answerable, lambda row: bool(row["judge"].get("judge_timeout"))
        ),
    }


def write_report(rows: list[dict], metrics: dict, report_file: Path = REPORT_FILE) -> None:
    """生成 Markdown 报告和最值得人工复核的明细。"""
    lines = [
        "# RAG 答案级评测报告",
        "",
        f"评测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 评测对象",
        "",
        "- 检索链路：BM25 + 向量 + RRF + Qwen3-Reranker-4B；"
        "命中跨文档结构时自动拆分子查询并按主题配额融合",
        "- 生成模型：Qwen/Qwen3-8B",
        "- 判定模型：与生成模型同配置，温度为 0",
        "",
        "## 样本构成",
        "",
    ]
    categories = Counter(row["category"] for row in rows)
    for category, count in sorted(categories.items()):
        lines.append(f"- `{category}`：{count} 题")

    lines.extend(
        [
            "",
            "## 核心指标",
            "",
            "| 指标 | 结果 | 说明 |",
            "| --- | ---: | --- |",
            f"| 概念覆盖率 | {metrics['answer_concept_coverage']:.1%} | 有答案题中，最终回答覆盖期望答案要点的比例 |",
            f"| 概念全对率 | {metrics['answer_correct_rate']:.1%} | 所有期望答案要点都出现在最终回答中 |",
            f"| 引用格式有效率 | {metrics['citation_format_valid_rate']:.1%} | 引用编号能对应本次真实检索资料 |",
            f"| 有效 Judge 支持率 | {metrics['judge_supported_rate']:.1%} | Judge 认为答案实质结论被资料支持 |",
            f"| 有效 Judge 引用支持率 | {metrics['judge_citation_supported_rate']:.1%} | Judge 认为引用能支撑答案 |",
            f"| 有效 Judge 幻觉率 | {metrics['hallucination_rate']:.1%} | Judge 发现资料无法支持的编造内容 |",
            f"| 应答拒答率 | {metrics['refusal_precision']:.1%} | 资料不足题正确拒答的比例 |",
            f"| 有答案误拒率 | {metrics['answerable_refusal_rate']:.1%} | 有答案题却错误拒答的比例 |",
            f"| Judge 异常率 | {metrics['judge_error_rate']:.1%} | Judge 接口或解析失败比例 |",
            f"| Judge 超时率 | {metrics['judge_timeout_rate']:.1%} | 超时样本单独标记，不参与语义指标 |",
            "",
            "## 判读方式",
            "",
            "- 检索评测只回答“证据有没有进来”；答案级评测回答“最终话说得对不对”。",
            "- `概念全对率` 是确定性指标，容易受同义表达影响；`Judge 支持率` 用来补足语义判断。",
            "- 幻觉率不是越低越好就结束，还要看 unsupported_claims 是否只是措辞问题。",
            "- 资料不足题不进入检索命中率，只检查是否拒答且没有编造。",
            "",
            "## 人工复核优先级",
            "",
        ]
    )
    risky = [
        row
        for row in rows
        if row["answerable"]
        and (
            row["judge"]["fabrication_detected"]
            or not row["judge"]["citation_supported"]
            or not row["citation_format_valid"]
            or row["answer_concept_coverage"] < 1.0
        )
    ]
    risky.extend(row for row in rows if not row["answerable"] and not row["refusal"])
    if not risky:
        lines.append("- 本轮抽样未发现必须人工复核的异常样例。")
    for row in risky[:15]:
        reason = row["judge"].get("reason", "")
        lines.append(
            f"- `{row['id']}` {row['question']}：覆盖 {row['answer_concept_coverage']:.0%}，"
            f"引用格式 {'通过' if row['citation_format_valid'] else '异常'}，Judge：{reason}"
        )

    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 答案级评测")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval", type=Path, default=ROOT / "data" / "eval_questions.jsonl")
    parser.add_argument("--limit", type=int, default=40, help="抽样题数；0 表示全量")
    parser.add_argument(
        "--detail",
        type=Path,
        default=DETAIL_FILE,
        help="逐题明细输出路径；回归集应使用独立文件",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_FILE,
        help="Markdown 报告输出路径；回归集应使用独立文件",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cases = load_cases(args.eval)
    if args.detail.exists():
        args.detail.unlink()

    print(f"加载评测集：{len(cases)} 题")
    selected_count = len(cases) if args.limit <= 0 else min(args.limit, len(cases))
    print(f"答案级抽样：{selected_count} 题")
    rows = evaluate_answer_cases(cfg, cases, args.limit, detail_file=args.detail)
    metrics = summarize(rows)
    write_report(rows, metrics, report_file=args.report)

    print("\n答案级核心指标")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"\n明细：{args.detail}")
    print(f"报告：{args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
