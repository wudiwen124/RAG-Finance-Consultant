"""
从天天基金网基金档案(fundf10.eastmoney.com)抓取代表性基金的公开资料。

每个基金抓两个页面：
    - 基本概况 jbgk_<代码>.html：基金类型、规模、基金经理、投资目标/范围/策略等
    - 基金费率 jjfl_<代码>.html：认购/申购/赎回费、管理/托管/销售服务费

解析思路：
    - 页面是普通 HTML，用 lxml 解析
    - 基金信息大多在 <table> 里，统一转成 Markdown 表格或 key: value 行
    - 概况页的投资目标等长文在 <div class="boxitem"> 里，按 h4 标题分节
    - 页面顶部有导航表格（"手机买基金"等），需要过滤掉

用法：
    python scripts/download_funds.py

输出：
    kb/raw/funds/<code>_<page>.html   原始 HTML（存档用）
    kb/processed/funds/<code>_<name>.md
"""

from __future__ import annotations

import re    # 正则：清理文本空白
import sys   # sys.exit 控制退出码
import time  # 请求间隔，避免高频访问
from pathlib import Path

import requests  # HTTP 客户端
from lxml import html as lxml_html  # lxml 的 HTML 解析器

# 通用请求头：模拟浏览器，Referer 指向基金档案站，否则会被拒绝访问
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://fundf10.eastmoney.com/",
}

# 项目根目录及输入/输出目录
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "kb" / "raw" / "funds"        # 原始 HTML 存档目录
OUT_DIR = ROOT / "kb" / "processed" / "funds"  # 转换后的 Markdown 目录

# 要抓取的基金列表：(基金代码, 基金名称)。
# 刻意覆盖了货币/债券/股票/混合/指数等不同类型，保证知识库有代表性。
FUNDS = [
    ("000198", "天弘余额宝货币A"),
    ("000001", "华夏现金增利货币A"),
    ("110008", "易方达稳健收益债券B"),
    ("110020", "易方达沪深300ETF联接A"),
    ("110022", "易方达消费行业股票"),
    ("005827", "易方达蓝筹精选混合"),
    ("161725", "招商中证白酒指数(LOF)A"),
    ("006327", "易方达中概互联网ETF联接人民币A"),
]


def fetch(url: str, retries: int = 4) -> str:
    """抓取一个 URL 并返回 UTF-8 文本，带重试机制。"""
    last_err: Exception | None = None
    for i in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=40)
            resp.raise_for_status()
            resp.encoding = "utf-8"  # 强制按 UTF-8 解码（页面声明的编码）
            return resp.text
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  [retry {i + 1}] {url}: {e}")
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"抓取失败 {url}: {last_err}")


def table_to_markdown(table) -> list[str]:
    """把一个 HTML <table> 转成 Markdown 行列表。

    两种输出形态：
        - 两列且每行两列的表（如"基金全称 | 值"） -> "- key: value" 列表，更紧凑
        - 其他多列表格 -> 标准 Markdown 表格（含表头分隔行）
    """
    rows = []
    for tr in table.xpath(".//tr"):  # 遍历每一行
        # 取每行的单元格文本，统一清理空白
        cells = [
            re.sub(r"\s+", " ", (td.text_content() or "").strip())
            for td in tr.xpath("./td|./th")
        ]
        cells = [c for c in cells if c]  # 去掉空单元格
        if cells:
            rows.append(cells)
    if not rows:
        return []

    # 两列 key-value 形态：转成"- key: value"列表
    if len(rows[0]) == 2 and all(len(r) == 2 for r in rows):
        return [f"- {k}: {v}" for k, v in rows]

    # 多列形态：生成标准 Markdown 表格
    width = max(len(r) for r in rows)  # 最宽行的列数，用于补齐

    def pad(r):  # 把行补到统一宽度
        return r + [""] * (width - len(r))

    lines = ["| " + " | ".join(pad(rows[0])) + " |"]  # 表头
    lines.append("| " + " | ".join(["---"] * width) + " |")  # 分隔行
    for r in rows[1:]:
        lines.append("| " + " | ".join(pad(r)) + " |")
    return lines


def parse_jbgk(html_text: str) -> str:
    """解析基金"基本概况"页面，返回 Markdown 字符串。"""
    root = lxml_html.fromstring(html_text)
    parts: list[str] = []
    # 导航栏特征词：包含这些词的表格是页头/页脚导航，不是基金数据，直接跳过
    nav_markers = ("手机买基金", "自选基金", "基金交易登录", "稳健理财")

    # 1) 概况信息表（基金全称/代码/类型/费率/业绩比较基准等）
    for table in root.xpath("//table"):
        text = table.text_content() or ""
        if any(m in text for m in nav_markers):
            continue
        md = table_to_markdown(table)
        if md:
            parts.extend(md)

    # 2) 分节长文（投资目标/投资理念/投资范围/投资策略/分红政策/风险收益特征等）
    for box in root.xpath("//div[contains(@class,'boxitem')]"):
        # h4 标签里的文字是本节标题（如"投资目标"）
        label = "".join(box.xpath(".//h4//text()")).strip()
        # 节内所有 <p> 段落文本
        paras = [" ".join(p.split()) for p in box.xpath(".//p//text()") if p.strip()]
        if label and paras:
            parts.append(f"\n## {label}")  # 小节标题用二级标题
            parts.extend(paras)
    return "\n".join(parts)


def parse_jjfl(html_text: str) -> str:
    """解析基金"费率"页面：主要是各类费率表格，统一转 Markdown。"""
    root = lxml_html.fromstring(html_text)
    parts: list[str] = []
    for table in root.xpath("//table"):
        md = table_to_markdown(table)
        if md:
            parts.extend(md)
    return "\n".join(parts)


def main() -> int:
    """主流程：对每只基金抓取概况+费率两个页面并转成 Markdown。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for code, name in FUNDS:
        print(f"== {code} {name}")
        sections: list[str] = []
        # (页面类型, 章节标题, 解析函数)
        for page, label, parser in [
            ("jbgk", "基本概况", parse_jbgk),
            ("jjfl", "基金费率", parse_jjfl),
        ]:
            url = f"https://fundf10.eastmoney.com/{page}_{code}.html"
            try:
                html_text = fetch(url)
                # 原始 HTML 存档，方便回溯和排查解析问题
                (RAW_DIR / f"{code}_{page}.html").write_text(html_text, encoding="utf-8")
                body = parser(html_text)
                sections.append(f"# {label}\n\n{body}")
                print(f"  OK {page} ({len(html_text)} bytes)")
            except Exception as e:  # noqa: BLE001  单个页面失败不影响其他基金
                print(f"  !! {page} 失败: {e}")
            time.sleep(0.8)  # 请求间隔

        # 两个页面至少成功一个才生成 Markdown
        if sections:
            # front matter：与法规一致，包含 doc_id / source / source_url 等溯源字段
            front = "\n".join(
                [
                    "---",
                    f"title: {name}（{code}）",
                    "doc_id: fund_" + code,
                    "source_type: fund_profile",
                    "source: 天天基金网基金档案(fundf10.eastmoney.com)",
                    f"source_url: https://fundf10.eastmoney.com/jbgk_{code}.html",
                    "note: 公开页面抓取，仅供个人学习",
                    "---",
                ]
            )
            out = OUT_DIR / f"{code}_{name}.md"
            out.write_text(front + "\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
            print(f"  SAVED {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
