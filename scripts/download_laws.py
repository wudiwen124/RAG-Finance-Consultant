"""
下载国家法律法规数据库(flk.npc.gov.cn)中的金融相关法律法规。

这个脚本负责知识库流水线的第一步：把官方法规下载到本地。
整体流程：
    1. 用法规标题在 flk.npc.gov.cn 的搜索接口检索（POST /law-search/search/list）
    2. 从搜索结果中挑选"标题匹配 + 现行有效(sxx=3)"的那一条
    3. 通过官方下载接口（GET /law-search/download/pc）拿到 DOCX 存到 kb/raw/laws/
    4. 把标题、发布时间、施行时间、发布机关等元数据存到 kb/meta/laws/

用法：
    python scripts/download_laws.py

输出：
    kb/raw/laws/<标题>.docx   原始 DOCX 文件（只做存档，后续由 docx_to_md.py 转 Markdown）
    kb/meta/laws/<标题>.json  每条法规的元数据（转换时用来生成 front matter）
"""

from __future__ import annotations  # 让类型注解写法在低版本 Python 也兼容

import json  # 读写 JSON 格式的元数据
import re    # 正则表达式，用来去掉搜索结果里的高亮 HTML 标签
import sys   # sys.exit 控制脚本退出码
import time  # 请求间隔，避免高频访问触发反爬
from pathlib import Path  # 跨平台的路径操作（Windows / Linux / Mac 通用）

import requests  # 第三方 HTTP 客户端库，用来调官方接口

# 国家法律法规数据库的域名，所有请求 URL 都基于它拼接
BASE = "https://flk.npc.gov.cn"
# 通用请求头：伪装成浏览器访问，并带上 Referer/Origin 通过服务端校验
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": BASE + "/search",
    "Origin": BASE,
    "Accept": "application/json, text/plain, */*",
}

# 项目根目录：本文件位于 scripts/ 下，父目录就是项目根目录
ROOT = Path(__file__).resolve().parent.parent
# 原始法规文件的输出目录
RAW_DIR = ROOT / "kb" / "raw" / "laws"
# 法规元数据 JSON 的输出目录
META_DIR = ROOT / "kb" / "meta" / "laws"

# 目标法规标题列表：脚本会按标题逐个去官方库检索。
# 全部是与"基金 / 支付 / 保险 / 理财"相关的现行有效法规，
# 覆盖了金融知识库问答的主要法规类问题。
TARGET_LAWS = [
    "中华人民共和国证券投资基金法",
    "中华人民共和国证券法",
    "中华人民共和国保险法",
    "中华人民共和国商业银行法",
    "中华人民共和国消费者权益保护法",
    "中华人民共和国反洗钱法",
    "中华人民共和国反电信网络诈骗法",
    "非银行支付机构监督管理条例",
    "存款保险条例",
    "防范和处置非法集资条例",
    "证券公司监督管理条例",
    "私募投资基金监督管理条例",
]


def clean_title(title: str) -> str:
    """去掉搜索接口返回的 <em class='highlight'> 高亮标签。

    官方搜索接口命中关键词时，会在标题里包一层 <em> 标签，
    例如：中华人民共和国<em>证券投资基金</em>法。
    这个函数用正则把标签剥掉，还原成纯文本标题，方便做精确匹配。
    """
    return re.sub(r"<[^>]+>", "", title or "").strip()


class FlkClient:
    """封装 flk.npc.gov.cn 的 HTTP 调用。

    官方接口偶尔会超时/抖动，所以每个请求都带重试机制：
    失败后按 3 秒、6 秒、9 秒的间隔递增重试，最多 4 次。
    """

    def __init__(self) -> None:
        # 用 Session 复用 TCP 连接，比每次新建连接更快更稳定
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def post(self, path: str, payload: dict, retries: int = 4) -> dict:
        """发送 POST 请求并解析 JSON 返回。"""
        last_err: Exception | None = None  # 记录最后一次异常，用于报错信息
        for i in range(retries):
            try:
                resp = self.session.post(BASE + path, json=payload, timeout=60)
                resp.raise_for_status()  # 状态码非 2xx 时直接抛异常
                return resp.json()       # 把响应体解析成 dict
            except Exception as e:       # noqa: BLE001  网络抖动统一走重试
                last_err = e
                print(f"  [retry {i + 1}] {path} 失败: {e}")
                time.sleep(3 * (i + 1))  # 递增等待时间，降低对服务器的压力
        # 重试全部失败，抛出带原始异常信息的错误
        raise RuntimeError(f"请求失败: {path}: {last_err}")

    def get(self, path: str, params: dict, retries: int = 4) -> dict:
        """发送 GET 请求并解析 JSON 返回（逻辑同 post）。"""
        last_err: Exception | None = None
        for i in range(retries):
            try:
                resp = self.session.get(BASE + path, params=params, timeout=60)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:       # noqa: BLE001
                last_err = e
                print(f"  [retry {i + 1}] {path} 失败: {e}")
                time.sleep(3 * (i + 1))
        raise RuntimeError(f"请求失败: {path}: {last_err}")

    def search(self, keyword: str, page_size: int = 10) -> list[dict]:
        """按标题关键词搜索法规，返回候选条目列表。

        请求体字段说明（官方新版接口的约定）：
            searchRange:   1=只在标题里搜，2=在正文里搜
            searchType:    1=精确匹配，2=模糊匹配
            searchContent: 搜索关键词
            sxx:           时效性过滤（空=全部，3=现行有效）
            orderByParam:  排序参数，order=-1 表示降序
            其余字段是分类/日期等过滤条件，这里都留空表示"不限"
        """
        payload = {
            "searchRange": 1,
            "sxrq": [],
            "gbrq": [],
            "sxx": [],
            "searchType": 2,
            "xgzlSearch": False,
            "searchContent": keyword,
            "orderByParam": {"order": "-1", "sort": ""},
            "flfgCodeId": [],
            "zdjgCodeId": [],
            "gbrqYear": [],
            "pageNum": 1,
            "pageSize": page_size,
        }
        data = self.post("/law-search/search/list", payload)
        return data.get("rows") or []  # rows 就是候选法规列表

    def download_docx(self, bbbs: str, dest: Path) -> None:
        """根据法规 ID(bbbs) 下载 DOCX 原文。

        官方接口的下载是两步：
        第一步请求 download/pc，拿到一个带签名、有时效的 OSS 下载地址；
        第二步再请求那个地址，把文件内容以二进制写到本地。
        """
        data = self.get(
            "/law-search/download/pc", {"format": "docx", "bbbs": bbbs}
        ).get("data") or {}
        url = data.get("url")
        if not url:
            raise RuntimeError(f"bbbs={bbbs} 未返回下载地址")
        resp = self.session.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)  # write_bytes 以二进制写入，保留 DOCX 结构


def pick_best(rows: list[dict], target: str) -> dict | None:
    """从搜索结果里挑最合适的一条。

    规则（按优先级）：
        1. 只在"现行有效(sxx=3)"的条目里挑
        2. 优先标题完全一致的
        3. 其次标题以目标开头的
    这样能避免误选到同名旧法或修订草案。
    """
    # 先过滤出有效条目；如果有效条目里找不到，再退回全量结果里找
    valid = [r for r in rows if str(r.get("sxx")) == "3"]
    for group in (valid, rows):
        for r in group:
            if clean_title(r.get("title")) == target:
                return r
        for r in group:
            if clean_title(r.get("title", "")).startswith(target):
                return r
    return None  # 都没找到就返回 None，由调用方处理


def main() -> int:
    """主流程：遍历 TARGET_LAWS，逐个下载法规并保存元数据。"""
    # 确保输出目录存在（parents=True 会递归创建中间目录）
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    client = FlkClient()

    ok, failed = [], []  # 分别记录成功和失败的法规标题
    for title in TARGET_LAWS:
        print(f"== 检索: {title}")
        rows = client.search(title)
        best = pick_best(rows, title)
        if best is None:
            # 没找到：打印候选标题，方便人工排查
            print(f"  !! 未找到匹配条目（候选: {[clean_title(r.get('title')) for r in rows[:3]]}）")
            failed.append(title)
            continue

        bbbs = best["bbbs"]  # bbbs 是官方文档的唯一 ID
        dest = RAW_DIR / f"{title}.docx"
        try:
            client.download_docx(bbbs, dest)
        except Exception as e:  # noqa: BLE001  单个失败不影响其他法规
            print(f"  !! 下载失败: {e}")
            failed.append(title)
            continue

        # 保存元数据：后续 docx_to_md.py 用它生成 Markdown 的 front matter
        meta = {
            "title": clean_title(best.get("title")),
            "bbbs": bbbs,
            "category": best.get("flxz"),        # 法律 / 行政法规 等分类
            "issuing_org": best.get("zdjgName"), # 发布机关
            "pub_date": best.get("gbrq"),        # 公布日期
            "effective_date": best.get("sxrq"),  # 施行日期
            "status": best.get("sxx"),           # 时效性（3=现行有效）
            "file": str(dest),
        }
        (META_DIR / f"{title}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  OK {dest.name} ({dest.stat().st_size} bytes)")
        ok.append(title)
        time.sleep(1)  # 每个法规之间休息 1 秒，避免高频访问

    print("\n===== 结果 =====")
    print(f"成功 {len(ok)}: {ok}")
    if failed:
        print(f"失败 {len(failed)}: {failed}")
    return 0


if __name__ == "__main__":
    # 只有直接运行本文件时才执行 main()，被 import 时不执行
    sys.exit(main())
