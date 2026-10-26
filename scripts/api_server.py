"""
金融知识库 RAG 问答 API（FastAPI 包装层）。

把 rag_qa.py 的 answer_question 包成 HTTP 接口，输出统一 JSON：
    POST /api/ask
    {"question": "货币基金和债券基金有什么区别"}

    返回:
    {
      "answer": "...",
      "citations": [
        {"index": 1, "title": "...", "section": "...", "source": "...", "snippet": "..."}
      ],
      "latency_ms": 1234
    }

启动:
    python scripts/api_server.py
    uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from conversation import ConversationMemory, route_question
from rag_qa import (
    DEFAULT_CONFIG,
    build_vector_store,
    get_bm25_index,
    load_config,
    prepare_context,
    stream_answer,
)

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"

# 应用级别的共享资源，在 lifespan 里初始化一次，之后所有请求复用。
_app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：启动时加载向量库和 BM25 索引，关闭时清理。"""
    cfg = load_config(DEFAULT_CONFIG)
    print("连接向量库...")
    store = build_vector_store(cfg)
    print("加载 BM25 索引...")
    bm25_index = get_bm25_index(cfg)
    memory = ConversationMemory(
        window_turns=int(cfg.get("memory", {}).get("window_turns", 5)),
        ttl_hours=float(cfg.get("memory", {}).get("ttl_hours", 24)),
    )
    _app_state["cfg"] = cfg
    _app_state["store"] = store
    _app_state["bm25_index"] = bm25_index
    _app_state["memory"] = memory
    yield
    _app_state.clear()


app = FastAPI(
    title="金融知识库 RAG 问答",
    description="基于 LangChain + Qdrant + Reranker 的金融知识库问答 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    """请求体：只接受一个必填的 question 字段。"""
    question: str = Field(..., min_length=1, max_length=500, description="用户提问")
    session_id: str | None = Field(
        None,
        max_length=64,
        description="浏览器生成的会话 ID；不传时服务端会新建一个临时会话",
    )


class Citation(BaseModel):
    """单条引用：回答里的 [N] 对应这里的 index。"""
    index: int
    title: str
    section: str
    source: str
    snippet: str


class AskResponse(BaseModel):
    """统一响应格式，前端只需要关心这三个字段。"""
    answer: str
    citations: list[Citation]
    latency_ms: int
    session_id: str


def _build_citations(hits: list[tuple]) -> list[Citation]:
    """把检索结果转成前端友好的引用列表。"""
    citations = []
    for index, (doc, _score) in enumerate(hits, start=1):
        metadata = doc.metadata or {}
        citations.append(Citation(
            index=index,
            title=str(metadata.get("title", "")),
            section=str(metadata.get("section", "")),
            source=str(metadata.get("source", "")),
            snippet=(doc.page_content or "")[:200],
        ))
    return citations


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """非流式问答接口：先理解会话，再检索/生成。"""
    cfg = _app_state["cfg"]
    store = _app_state["store"]
    bm25_index = _app_state["bm25_index"]
    memory: ConversationMemory = _app_state["memory"]
    session_id = request.session_id or uuid.uuid4().hex
    start = time.perf_counter()

    try:
        route_result, _history = await asyncio.to_thread(
            route_question, cfg, memory, session_id, request.question
        )
        if route_result.intent == "memory":
            answer = route_result.direct_answer or "当前会话历史中没有找到相关信息。"
            hits = []
        else:
            retrieval_question = route_result.standalone_query
            context, hits = await asyncio.to_thread(
                prepare_context,
                cfg,
                store,
                retrieval_question,
                bm25_index=bm25_index,
            )
            if not hits:
                answer = "资料中未找到相关信息。"
            else:
                answer = await asyncio.to_thread(
                    lambda: "".join(
                        stream_answer(cfg, retrieval_question, context)
                    )
                )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    memory.append_message(session_id, "user", request.question)
    memory.append_message(session_id, "assistant", answer)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return AskResponse(
        answer=answer,
        citations=_build_citations(hits),
        latency_ms=latency_ms,
        session_id=session_id,
    )


def _sse_event(event_type: str, data: Any) -> str:
    """把一个 JSON 消息包成 Server-Sent Events 的 data 帧。"""
    payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


@app.post("/api/ask/stream")
async def ask_stream(request: AskRequest):
    """流式问答：先返回引用，再逐块推送模型输出。

    事件格式仍然是 JSON：
      citations -> 先给前端渲染来源
      token     -> 每次推送一小段模型输出
      done      -> 全部结束后推送总耗时
      error     -> 流中途失败时通知前端展示错误
    """
    cfg = _app_state["cfg"]
    store = _app_state["store"]
    bm25_index = _app_state["bm25_index"]
    memory: ConversationMemory = _app_state["memory"]
    session_id = request.session_id or uuid.uuid4().hex
    start = time.perf_counter()

    # 会话理解涉及一次非流式 LLM 调用，放到线程池里跑，
    # 避免 uvicorn 的事件循环被卡住。
    try:
        route_result, _history = await asyncio.to_thread(
            route_question, cfg, memory, session_id, request.question
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # 检索涉及 embedding / BM25 / rerank，是同步阻塞调用，放到线程池里跑，
    # 避免 uvicorn 的事件循环被卡住。
    try:
        if route_result.intent == "memory":
            hits = []
            context = ""
            memory_answer = route_result.direct_answer or "当前会话历史中没有找到相关信息。"
        else:
            retrieval_question = route_result.standalone_query
            context, hits = await asyncio.to_thread(
                prepare_context,
                cfg,
                store,
                retrieval_question,
                bm25_index=bm25_index,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    def generate():
        answer_parts: list[str] = []
        try:
            citations = [item.model_dump() for item in _build_citations(hits)]
            yield _sse_event("citations", citations)

            if route_result.intent == "memory":
                # 记忆问题不需要知识库引用；切成小段仍保持逐字输出体验。
                answer_parts.append(memory_answer)
                for start_index in range(0, len(memory_answer), 18):
                    yield _sse_event("token", memory_answer[start_index : start_index + 18])
            elif not hits:
                answer_parts.append("资料中未找到相关信息。")
                yield _sse_event("error", "未检索到相关知识，请换一种问法。")
            else:
                # stream_answer 是同步生成器；FastAPI 会在工作线程中迭代它，
                # 因此模型生成过程中不会阻塞其他 HTTP 请求的事件循环。
                for token in stream_answer(cfg, retrieval_question, context):
                    if token:
                        answer_parts.append(token)
                        yield _sse_event("token", token)

            # 只有正常结束时才写入历史；中途异常不污染会话记忆。
            memory.append_message(session_id, "user", request.question)
            memory.append_message(
                session_id, "assistant", "".join(answer_parts)
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            yield _sse_event(
                "done", {"latency_ms": latency_ms, "session_id": session_id}
            )
        except Exception as exc:
            yield _sse_event("error", str(exc))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health() -> dict:
    """健康检查，前端可以用它判断服务是否可用。"""
    return {"status": "ok", "model": _app_state.get("cfg", {}).get("llm", {}).get("model", "")}


@app.post("/api/sessions/{session_id}/clear")
async def clear_session(session_id: str) -> dict:
    """清空一个前端会话的记忆，用于“新会话”按钮。"""
    memory: ConversationMemory = _app_state["memory"]
    cleared = memory.clear(session_id)
    return {"status": "ok", "cleared": cleared}


@app.get("/")
async def serve_frontend():
    """把 frontend/index.html 挂在根路径，浏览器打开 http://localhost:8000 即可使用。"""
    return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/logo.png")
async def serve_logo():
    """前端页面引用的 FinRAG logo 图片。"""
    return FileResponse(FRONTEND_DIR / "logo.png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
