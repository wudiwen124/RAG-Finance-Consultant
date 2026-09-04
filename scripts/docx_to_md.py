"""
把 kb/raw/laws 下的法规 DOCX 转为带元数据的 Markdown。

背景知识：DOCX 本质上是一个 zip 压缩包，正文内容放在 word/document.xml 里。
解析思路：
    1. 用 zipfile 解包，读出 word/document.xml
    2. 用 lxml 遍历所有 <w:p>（段落），把段落里的 <w:t>（文本节点）拼起来
    3. 官方法规的 docx 没有排版样式，只能靠文本特征识别结构：
       "第X章" -> 二级标题(##)，"第X条" -> 三级标题(###)，其余是正文
    4. 去掉开头的"目　　录"部分，避免把目录内容也当成知识

用法：
    python scripts/docx_to_md.py

输出：
    kb/processed/laws/<标题>.md  带 YAML front matter 的 Markdown
"""

from __future__ import annotations

import json  # 读取下载阶段保存的元数据 JSON
import re    # 正则：匹配"第X章/第X条"、清理多余空白
import sys   # sys.exit 控制退出码
import zipfile  # 解析 DOCX（本质是 zip）
from pathlib import Path

from lxml import etree  # 高性能 XML 解析器，用来读 document.xml

# Word 文档的 XML 命名空间。DOCX 里几乎所有标签都带这个前缀，
# 所以定义成常量 W，后面写 W+"p" / W+"t" 表示 <w:p> / <w:t>。
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# 项目根目录及输入/输出目录
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "kb" / "raw" / "laws"       # 输入：下载好的 DOCX
META_DIR = ROOT / "kb" / "meta" / "laws"     # 输入：每条法规的元数据 JSON
OUT_DIR = ROOT / "kb" / "processed" / "laws" # 输出：转换后的 Markdown

# 章节标题正则：匹配"第一章""第二百零三章"这类开头
CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零]+章")
# 条文标题正则：匹配"第一条""第二百零三条"这类开头
ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百零]+条")


def extract_paragraphs(docx_path: Path) -> list[str]:
    """从 docx 的 word/document.xml 提取所有段落文本。

    实现细节：
        - 一个段落 <w:p> 里可能包含多个 <w:t> 文本节点（因为 Word 会
          把一句话拆成多个 run），所以要遍历该段落下的所有 <w:t> 拼接。
        - 文本里的全角空格（中文排版对齐用的"　"）统一替换成半角空格，
          再压缩连续空白，方便后续正则匹配。
    """
    with zipfile.ZipFile(docx_path) as zf:
        # 读取压缩包内的 document.xml，并用 lxml 解析成 XML 树
        root = etree.fromstring(zf.read("word/document.xml"))
    paras: list[str] = []
    for para in root.iter(W + "p"):  # iter 会遍历所有后代节点，找到所有段落
        # 把段落内所有 <w:t> 的文本拼成一句话
        text = "".join(t.text or "" for t in para.iter(W + "t"))
        text = text.replace("\u3000", " ").strip()  # 全角空格 -> 半角并去首尾空白
        text = re.sub(r"\s+", " ", text)            # 连续空白压缩成单个空格
        if text:                                    # 跳过空段落
            paras.append(text)
    return paras


def build_markdown(paras: list[str]) -> str:
    """把段落列表按"章/条"结构转成 Markdown，并去掉目录部分。

    转换规则：
        - 第一个非空段落是法规标题，写成一级标题 "#"
        - "第X章"开头 -> "## "（二级标题）
        - "第X条"开头 -> "### "（三级标题）
        - 其余段落按正文处理
        - "目　　录"到正文之间是目录，直接跳过
    """
    # 标题 = 第一个非空段落（extract_paragraphs 已保证列表里没有空串）
    title = paras[0] if paras else ""
    lines = ["# " + title]

    # 跳过目录：找到"目　　录"所在位置，从它后面第一次出现"第一章"开始算正文。
    # 因为目录里的章节名是"第一章..."，正文也以"第一章..."开始，
    # 这里通过判断"章节名不是紧跟在目录后面的那条"来定位真正的正文起点。
    start = 1
    toc_idx = next(
        (i for i, p in enumerate(paras[1:], start=1) if "目" in p and "录" in p),
        None,
    )
    if toc_idx is not None:
        for i in range(toc_idx + 1, len(paras)):
            # 目录后第一条"第X章"是目录内容，第二条开始才是正文
            if CHAPTER_RE.match(paras[i]) and paras[i] != paras[toc_idx + 1]:
                start = i
                break

    # 从正文起点开始转换
    for p in paras[start:]:
        if CHAPTER_RE.match(p):
            lines.append("\n## " + p)   # 章 -> 二级标题，前面空一行
        elif ARTICLE_RE.match(p):
            lines.append("\n### " + p)  # 条 -> 三级标题
        else:
            lines.append(p)             # 普通段落
    return "\n".join(lines) + "\n"


def main() -> int:
    """主流程：遍历所有法规 DOCX，逐个转成带 front matter 的 Markdown。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for docx in sorted(RAW_DIR.glob("*.docx")):
        title = docx.stem  # 文件名（不含后缀）就是法规标题
        paras = extract_paragraphs(docx)
        md = build_markdown(paras)

        # 读取下载阶段保存的元数据；文件不存在时用空 dict 兜底
        meta_path = META_DIR / f"{title}.json"
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {}
        )

        # 生成 YAML front matter：每条 chunk 入库时都会携带这些字段，用于溯源
        front = "\n".join(
            f"{k}: {v}"
            for k, v in {
                "doc_id": meta.get("bbbs", ""),
                "title": title,
                "source_type": "law",       # 文档类型：法规
                "source": "国家法律法规数据库",
                "source_url": f"https://flk.npc.gov.cn/detail?id={meta.get('bbbs', '')}",
                "category": meta.get("category", ""),
                "issuing_org": meta.get("issuing_org", ""),
                "pub_date": meta.get("pub_date", ""),
                "effective_date": meta.get("effective_date", ""),
            }.items()
            if v  # 跳过空值字段，保持 front matter 干净
        )
        out = OUT_DIR / f"{title}.md"
        out.write_text(f"---\n{front}\n---\n\n{md}", encoding="utf-8")
        print(f"OK {out.name} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
