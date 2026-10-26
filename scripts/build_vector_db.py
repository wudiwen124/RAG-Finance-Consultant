"""
把 kb/processed 下的 Markdown 知识文档切分、向量化并写入向量库。

把处理后的知识文档写入向量库：
    1. 读入所有 Markdown 文档，解析 front matter 元数据
    2. 按标题层级把长文档切成适合检索的 chunk
    3. 调用 Embedding API 把每段文本变成向量（OpenAI 兼容接口）
    4. 把"向量 + 元数据"写入向量库（Qdrant 本地模式，或本地 numpy 兜底）
    5. 可选：用一个问题做一次检索冒烟测试

用法：
    # 1) 先用 mock embedding 验证整条流水线（不调 API）
    python scripts/build_vector_db.py --mock

    # 2) 填好 config/config.json 中的 embedding API 后正式构建
    python scripts/build_vector_db.py

    # 3) 构建后做一次检索冒烟测试
    python scripts/build_vector_db.py --query "货币基金的费率是怎么收取的"

说明：
    - embedding 走 OpenAI 兼容的 /embeddings 接口（SiliconFlow / 阿里云百炼 / 自建 vLLM 均可）
    - 向量库默认用 Qdrant 本地模式（无需 Docker）；没有 qdrant-client 时自动退化为
      本地 numpy 向量库（data/vector_db_local/）
"""

from __future__ import annotations

import argparse  # 解析命令行参数（--mock / --query / --config）
import hashlib   # 生成确定性伪向量的随机种子
import json      # 读写 JSON 配置和 chunk 清单
import math      # 数学工具（当前用于保险的空长度判断）
import re        # 正则：识别 Markdown 标题、按空行切分段落
import shutil    # 删除整个目录：重建向量库时清理旧的本地存储
import sys       # sys.exit 控制退出码
from pathlib import Path

import numpy as np     # 数值计算：numpy 兜底向量库的相似度计算
import requests        # HTTP 客户端：调用 Embedding API

# 项目根目录与默认路径
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.json"     # 默认配置文件
DEFAULT_INPUT = ROOT / "kb" / "processed"            # 默认文档输入目录
CHUNKS_MANIFEST = ROOT / "kb" / "chunks" / "chunks.jsonl"  # 切分结果清单


# ================================================================ 配置

def load_config(path: Path) -> dict:
    """读取配置文件；不存在时自动生成一份模板，方便用户直接编辑。"""
    if not path.exists():
        print(f"!! 找不到配置文件 {path}，正在生成模板 config/config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_default_config(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _default_config() -> dict:
    """生成默认配置模板（新用户第一次运行时使用）。"""
    return {
        # Embedding 配置：api_base 是 OpenAI 兼容地址，api_key 由用户填写
        "embedding": {
            "api_base": "https://api.siliconflow.cn/v1",
            "api_key": "在此填入你的 API Key",
            "model": "BAAI/bge-m3",
            "batch_size": 16,  # 每批同时向量化的文本条数，越大越快但越吃内存
        },
        "vector_store": {
            "type": "qdrant_local",   # qdrant_local | local_numpy
            "path": "data/vector_db", # 向量库数据存放目录
            "collection": "finance_kb",  # 集合名（Qdrant 里的表）
        },
        "chunking": {
            "chunk_size": 700,    # 每个 chunk 的目标字符数
            "chunk_overlap": 100, # 相邻 chunk 的重叠字符数，避免切断语义
            "min_chunk_chars": 40,  # 小于该长度的 chunk 直接丢弃
        },
        "input_dir": str(DEFAULT_INPUT),
    }


# ================================================================ 元数据

def parse_front_matter(text: str) -> tuple[dict, str]:
    """解析 Markdown 开头的 YAML-like front matter。

    知识文档开头都有三段式结构：
        ---
        title: xxx
        doc_id: xxx
        ---
        <正文>
    这个函数把"--- 到 ---"之间的键值对解析成 dict，并返回剩余正文。
    """
    if not text.startswith("---"):
        return {}, text  # 没有 front matter 就直接返回空元数据和全文
    end = text.find("\n---", 4)  # 找到结束的"---"（从第 4 个字符后开始找）
    if end == -1:
        return {}, text
    block = text[4:end]
    meta: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)  # 只在第一个冒号处分割，值里可以有冒号
            meta[k.strip()] = v.strip()
    return meta, text[end + 4 :].lstrip("\n")  # 返回元数据 + 正文


# ================================================================ 切分

# 识别 Markdown 标题：一行开头的 # 数量即标题层级，如 "## 第一章" -> 2 级
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def split_markdown(
    text: str, chunk_size: int, overlap: int, min_chars: int
) -> list[dict]:
    """把一篇 Markdown 按标题层级切分成 chunk 列表。

    策略（两段式）：
        1. 维护一个标题栈：遇到标题就入栈，遇到同级或更高级标题就出栈，
           从而得到每个标题下的"章节路径"（如 基本概况 > 投资目标）。
        2. 每个小节优先整体作为一条 chunk；如果小节太长（超过 chunk_size），
           再按空行分段、贪心打包，并带上 overlap 重叠，保证语义连续。
    """
    lines = text.splitlines()
    sections: list[dict] = []  # 每个元素: {"path": 章节路径, "lines": 该节的文本行}
    stack: list[tuple[int, str]] = []  # 标题栈: [(层级, 标题), ...]

    def current_path() -> list[str]:
        """返回当前章节路径（栈里所有标题，从根到当前）。"""
        return [title for _, title in stack]

    # 第一遍：按标题把文档切分成小节
    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            # 把层级 >= 当前标题的栈顶元素弹出（回到正确的父级）
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append({"path": list(current_path()), "lines": [line]})
        else:
            # 普通行归入当前小节；如果还没有任何标题，就归入一个空路径小节
            if not sections:
                sections.append({"path": [], "lines": []})
            sections[-1]["lines"].append(line)

    chunks: list[dict] = []
    # 第二遍：逐个小节生成 chunk
    short_group: list[tuple[str, str, tuple[str, ...]]] = []

    def has_body_text(body: str) -> bool:
        """判断小节除 Markdown 标题外是否还有实际正文。"""
        return any(
            not HEADING_RE.match(line) and line.strip()
            for line in body.splitlines()
        )

    def flush_short_group() -> None:
        """把相邻短小节合并成一条可检索 chunk，避免丢失短定义/短法条。"""
        if not short_group:
            return

        text = "\n\n".join(body for _path, body, _parent in short_group).strip()
        if text:
            if len(short_group) == 1:
                section = short_group[0][0]
            else:
                parent_path = short_group[0][2]
                titles = [path.rsplit(" > ", 1)[-1] for path, _b, _p in short_group]
                if parent_path:
                    section = " > ".join(parent_path) + " > " + "、".join(titles)
                else:
                    section = "、".join(titles)
            chunks.append({"text": text, "section": section})
        short_group.clear()

    for sec_index, sec in enumerate(sections):
        body = "\n".join(sec["lines"]).strip()
        if not body:
            continue
        path_text = " > ".join(sec["path"])  # 章节路径转成字符串，入库做溯源
        parent_path = tuple(sec["path"][:-1])

        # 只有标题的章节分两种：
        # 1. 结构性标题：如“## 第一章 总则”，后面紧跟子章节，不单独入库；
        # 2. 法条式标题：部分 DOCX 转 Markdown 后，整条短法条就在标题行里，
        #    后面没有子章节。这类标题承载答案，必须继续走短块合并。
        next_path = sections[sec_index + 1]["path"] if sec_index + 1 < len(sections) else []
        is_parent_of_next = tuple(next_path[: len(sec["path"])]) == tuple(sec["path"])
        if not has_body_text(body) and is_parent_of_next:
            flush_short_group()
            continue

        # 小节不长：整节作为一条 chunk
        if len(body) <= chunk_size:
            # 短小节不能直接丢：金融名词和法条的精确答案经常只有几十字。
            # 这里把同一父级下的相邻短小节合并，直到达到 min_chars 后落盘；
            # 最后一组即使仍小于 min_chars，也必须保留实际正文。
            if (
                short_group
                and short_group[-1][2] != parent_path
            ):
                flush_short_group()
            short_group.append((path_text, body, parent_path))
            if sum(len(item[1]) for item in short_group) >= min_chars:
                flush_short_group()
            continue

        flush_short_group()

        # 小节太长：按空行分成的段落，贪心打包进多个 chunk
        paras = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
        buf, buf_len = [], 0  # 当前正在组装的 chunk
        for p in paras:
            # 再加入一个段落会超限时，先把当前 chunk 收尾
            if buf and buf_len + len(p) + 2 > chunk_size:
                chunks.append({"text": "\n\n".join(buf), "section": path_text})
                # overlap：从刚收尾的 chunk 末尾截一段作为下一个 chunk 的开头，
                # 防止语义在 chunk 边界被切断
                tail = "\n\n".join(buf)
                keep = tail[-overlap:] if overlap > 0 else ""
                buf, buf_len = ([keep] if keep.strip() else []), len(keep)
            buf.append(p)
            buf_len += len(p) + 2
        # 收尾：最后一组段落
        if buf and "\n\n".join(buf).strip():
            chunks.append({"text": "\n\n".join(buf), "section": path_text})
    return chunks


# ================================================================ Embedding

class EmbeddingClient:
    """Embedding 客户端：把文本批量转成向量。

    默认走 OpenAI 兼容的 POST {api_base}/embeddings 接口：
        请求: {"model": "...", "input": ["文本1", "文本2", ...]}
        响应: {"data": [{"index": 0, "embedding": [0.1, 0.2, ...]}, ...]}

    mock 模式下不调网络，用哈希种子生成确定性伪向量，用于验证流水线。
    """

    def __init__(self, cfg: dict, mock: bool = False):
        self.cfg = cfg
        self.mock = mock
        self.base = cfg.get("api_base", "").rstrip("/")  # 去掉末尾斜杠，方便拼 URL
        self.model = cfg.get("model", "BAAI/bge-m3")
        self.batch = int(cfg.get("batch_size", 16))

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本转成向量，返回 list[list[float]]。"""
        if self.mock:
            return [self._mock_vec(t) for t in texts]

        # 校验配置：没填 Key 或还是占位符就报错，并提示用 --mock
        api_key = str(self.cfg.get("api_key", "")).strip()
        if not self.base or not api_key or "填入" in api_key:
            raise RuntimeError(
                "请先在 config/config.json 填入 embedding 的 api_base / api_key / model，"
                "或用 --mock 先跑通流程"
            )

        out: list[list[float]] = []
        # 分批调用，避免一次提交太多文本超出接口限制
        for i in range(0, len(texts), self.batch):
            batch = texts[i : i + self.batch]
            resp = requests.post(
                f"{self.base}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": batch},
                timeout=120,  # 向量化大模型较慢，超时给足
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            # 接口不保证返回顺序，按 index 排序，保证和输入文本一一对应
            data.sort(key=lambda x: x["index"])
            out.extend([d["embedding"] for d in data])
        return out

    @staticmethod
    def _mock_vec(text: str) -> list[float]:
        """生成确定性伪向量（仅用于本地跑通流程，无语义）。"""
        # 用文本的 md5 前 8 位做随机种子：同一段文本每次生成的向量完全一样
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(256).astype(np.float32)  # 256 维随机向量
        v /= max(np.linalg.norm(v), 1e-9)  # 归一化，方便算余弦相似度
        return v.tolist()


# ================================================================ 向量库

def make_store(cfg: dict, dim: int) -> "VectorStore":
    """根据配置创建向量库实例；Qdrant 不可用时自动退化为 numpy 版本。"""
    stype = cfg.get("type", "qdrant_local")
    path = ROOT / cfg.get("path", "data/vector_db")
    if stype == "qdrant_local":
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams

            return _QdrantStore(
                str(path), cfg["collection"], dim, PointStruct, VectorParams, Distance
            )
        except Exception as e:  # noqa: BLE001  没装 qdrant-client 等场景
            print(f"!! Qdrant 不可用({e})，退化为本地 numpy 向量库")
    return _NumpyStore(ROOT / "data" / "vector_db_local", dim)


class VectorStore:
    """向量库统一接口：add() 写入，search() 检索。"""

    dim: int  # 向量维度

    def add(self, vectors: list[list[float]], payloads: list[dict]) -> None:
        """批量写入向量和对应的元数据。"""
        raise NotImplementedError

    def search(self, vector: list[float], top_k: int = 5) -> list[tuple[float, dict]]:
        """按查询向量返回最相似的 top_k 条，格式: [(分数, 元数据), ...]"""
        raise NotImplementedError


class _QdrantStore(VectorStore):
    """Qdrant 本地模式实现（推荐，正式项目用这个）。"""

    def __init__(self, path: str, collection: str, dim: int, PointStruct, VectorParams, Distance):
        self.path = Path(path)
        self.collection = collection
        self.dim = dim
        self._PointStruct = PointStruct   # 保存类引用，add 时构造数据点
        self._VectorParams = VectorParams
        self._Distance = Distance
        self.client = self._rebuild()

    def _rebuild(self):
        """重建集合，保证维度/配置和当前模型一致。

        注意：Qdrant 本地模式在"同名集合不同维度重建"时可能残留旧存储，
        所以这里直接把整个数据目录删掉再重新打开，确保存储是全新的。
        """
        # 关闭可能占用目录的旧句柄
        if hasattr(self, "client"):
            try:
                self.client.close()
            except Exception:  # noqa: BLE001
                pass
        # 清空旧数据目录，然后重新打开（我们只用一个集合，全删无副作用）
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        from qdrant_client import QdrantClient

        client = QdrantClient(path=str(self.path))
        client.create_collection(
            collection_name=self.collection,
            vectors_config=self._VectorParams(size=self.dim, distance=self._Distance.COSINE),
        )
        return client

    def add(self, vectors, payloads):
        # 每条记录 = id(自增) + 向量 + payload(元数据，检索后用于溯源)
        points = [
            self._PointStruct(id=i, vector=v, payload=p)
            for i, (v, p) in enumerate(zip(vectors, payloads))
        ]
        # 分批 upsert，避免单次请求体过大
        for i in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[i : i + 256])

    def search(self, vector, top_k=5):
        # query_points 返回按相似度排序的结果，带上 payload 一起取回
        hits = self.client.query_points(
            self.collection, query=vector, limit=top_k, with_payload=True
        ).points
        return [(h.score, h.payload) for h in hits]


class _NumpyStore(VectorStore):
    """零依赖的本地向量库兜底实现。

    原理：把所有向量堆成一个大矩阵存成 vectors.npy，
    检索时用"余弦相似度 = 向量点积 / 模长乘积"暴力算最相似的几条。
    数据量小时完全够用（万级以下）。
    """

    def __init__(self, path: Path, dim: int):
        self.path = path
        self.dim = dim
        self.vectors: np.ndarray | None = None  # 累计向量矩阵
        self.payloads: list[dict] = []          # 与向量一一对应的元数据
        path.mkdir(parents=True, exist_ok=True)

    def add(self, vectors, payloads):
        arr = np.asarray(vectors, dtype=np.float32)
        # 把新向量追加到已有矩阵后面
        if self.vectors is None:
            self.vectors = arr
        else:
            self.vectors = np.vstack([self.vectors, arr])
        self.payloads.extend(payloads)
        # 增量保存：每次写入后立刻落盘，避免中途崩溃丢数据
        np.save(self.path / "vectors.npy", self.vectors)
        with open(self.path / "payloads.jsonl", "w", encoding="utf-8") as f:
            for p in self.payloads:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    def search(self, vector, top_k=5):
        if self.vectors is None or len(self.vectors) == 0:
            return []
        q = np.asarray(vector, dtype=np.float32)
        # 余弦相似度：逐行点积 / (每行模长 * 查询向量模长)
        scores = self.vectors @ q / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q) + 1e-9
        )
        # 取相似度最高的 top_k 个下标（argsort 默认升序，取负号变降序）
        idx = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self.payloads[i]) for i in idx]


# ================================================================ 主流程

def collect_docs(input_dir: Path) -> list[tuple[Path, dict, str]]:
    """扫描输入目录，收集所有 Markdown 文档。

    返回: [(文件路径, front matter 元数据, 正文), ...]
    """
    docs = []
    for path in sorted(input_dir.rglob("*.md")):  # rglob 递归找所有 .md
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        if not body.strip():
            continue  # 空文档跳过
        docs.append((path, meta, body))
    return docs


def main() -> int:
    """主流程：读文档 -> 切分 -> 向量化 -> 入库 -> 可选检索测试。"""
    # ---- 命令行参数 ----
    ap = argparse.ArgumentParser(description="金融知识库 -> 向量库")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="配置文件路径")
    ap.add_argument("--mock", action="store_true", help="用伪 embedding 跑通流程")
    ap.add_argument("--query", type=str, help="构建后检索该问题做冒烟测试")
    args = ap.parse_args()

    # ---- 读取配置 ----
    cfg = load_config(args.config)
    emb_cfg = cfg["embedding"]
    chunk_cfg = cfg["chunking"]
    store_cfg = cfg["vector_store"]
    input_dir = ROOT / cfg.get("input_dir", DEFAULT_INPUT)

    # ---- 1. 读文档 ----
    docs = collect_docs(input_dir)
    if not docs:
        print("!! 没有找到 Markdown 文档，请先运行下载/转换脚本")
        return 1
    print(f"文档数: {len(docs)}")

    # ---- 2. 切分 ----
    chunks = []  # 每个元素: {"text": 文本, "payload": 元数据}
    for path, meta, body in docs:
        parts = split_markdown(
            body,
            int(chunk_cfg["chunk_size"]),
            int(chunk_cfg["chunk_overlap"]),
            int(chunk_cfg["min_chunk_chars"]),
        )
        for i, part in enumerate(parts):
            # 合并文档级元数据 + 分块级元数据
            payload = dict(meta)
            payload.update(
                {
                    "doc_id": meta.get("doc_id", path.stem),  # 文档 ID
                    "file": str(path.relative_to(ROOT)),      # 相对路径
                    "section": part["section"],               # 章节路径
                    "chunk_index": i,                         # 块内序号
                }
            )
            chunks.append({"text": part["text"], "payload": payload})
    print(f"切分后 chunk 数: {len(chunks)}")

    # 把切分结果落盘成 JSONL：可复现、可排查（每行一条 chunk）
    CHUNKS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_MANIFEST, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"chunk 清单已写入 {CHUNKS_MANIFEST}")

    # ---- 3. 向量化 ----
    client = EmbeddingClient(emb_cfg, mock=args.mock)
    texts = [c["text"] for c in chunks]
    print("开始生成 embedding（可能需要几分钟，取决于 API 速度）...")
    vectors = client.embed(texts)
    dim = len(vectors[0])  # 向量维度由模型决定，后面建集合要用
    print(f"向量维度: {dim}")

    # ---- 4. 写入向量库 ----
    store = make_store(store_cfg, dim)
    # 写成 LangChain 兼容的 payload 结构：
    #   page_content: chunk 正文（LangChain 用它还原 Document 文本）
    #   metadata:     文档级/分块级元数据（溯源引用用）
    store_payloads = [
        {"page_content": c["text"], "metadata": dict(c["payload"])} for c in chunks
    ]
    store.add(vectors, store_payloads)
    print(f"已写入向量库: {store_cfg['type']} @ {store_cfg['path']}")

    # ---- 5. 检索冒烟测试 ----
    if args.query:
        qv = client.embed([args.query])[0]  # 问题也转成向量
        print(f"\n===== 检索测试: {args.query} =====")
        for score, payload in store.search(qv, top_k=3):
            # 打印命中的文档标题和章节路径，确认溯源信息完整
            print(
                f"[score={score:.4f}] "
                f"{payload.get('title', payload.get('doc_id'))} / "
                f"{payload.get('section', '')}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
