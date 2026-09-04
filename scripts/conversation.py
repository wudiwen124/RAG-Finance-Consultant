"""会话记忆与多轮问题理解。

这个模块解决三类问题：
1. 保存每个 session_id 对应的最近问答历史；
2. 判断新问题应该查知识库、查历史，还是两者都要；
3. 把“那它呢”“你刚才说的那个产品”改写成可独立检索的问题。

MVP 使用进程内内存。服务重启后会话会清空，这样避免引入数据库复杂度；
后续如果要持久化，可以把内部 dict 换成 SQLite/Redis，接口保持不变。
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from langchain_openai import ChatOpenAI


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


@dataclass
class ConversationRouterResult:
    """一次多轮问题理解的结构化结果。"""

    intent: str
    standalone_query: str
    direct_answer: str = ""


class ConversationMemory:
    """进程内会话记忆存储。

    - 每个 session_id 一份消息列表；
    - 每次读取时先清理超过 TTL 的会话；
    - 送入 LLM 前只取最近 window_turns 轮，避免 prompt 越滚越长。
    """

    def __init__(self, window_turns: int = 5, ttl_hours: float = 24):
        self.window_turns = max(1, int(window_turns))
        self.ttl = timedelta(hours=max(0.1, float(ttl_hours)))
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> list[dict]:
        """返回某个会话最近几轮历史；会话不存在或过期时返回空列表。"""
        now = _utc_now()
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []

            updated_at = datetime.fromisoformat(session["updated_at"])
            if now - updated_at > self.ttl:
                self._sessions.pop(session_id, None)
                return []

            # window_turns 表示“问答轮数”，因此消息数量最多是它的两倍。
            return list(session["messages"][-self.window_turns * 2 :])

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """写入一条用户或助手消息，并刷新会话 TTL。"""
        content = (content or "").strip()
        if not content:
            return

        with self._lock:
            session = self._sessions.setdefault(
                session_id,
                {"messages": [], "created_at": _iso_now(), "updated_at": _iso_now()},
            )
            session["messages"].append(
                {"role": role, "content": content, "at": _iso_now()}
            )
            session["updated_at"] = _iso_now()
            # 保留最近的 200 条即可；原始全量历史以后可以放到持久化存储里。
            session["messages"] = session["messages"][-200:]

    def clear(self, session_id: str) -> bool:
        """清空指定会话；返回是否确实删除了一个已有会话。"""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


def format_history(history: list[dict], max_chars: int = 2400) -> str:
    """把历史消息压成提示词友好的 Q/A 文本。"""
    if not history:
        return "（暂无历史对话）"

    blocks: list[str] = []
    total = 0
    for message in history:
        role = "用户" if message.get("role") == "user" else "助手"
        content = (message.get("content") or "").replace("\n", " ")
        # 单条消息截断，避免一轮长回答占满整个记忆 prompt。
        content = content[:800]
        block = f"{role}：{content}"
        total += len(block)
        if total > max_chars:
            break
        blocks.append(block)
    return "\n".join(blocks)


ROUTER_SYSTEM_PROMPT = """你是金融 RAG 问答系统的多轮问题理解模块。你的任务是判断当前问题需要什么上下文，并把追问改写成独立问题。

只输出一个 JSON 对象，不要输出 Markdown，不要输出解释：
{
  "intent": "memory|knowledge|hybrid",
  "standalone_query": "用于知识库检索的完整问题；memory 时可以为空字符串",
  "direct_answer": "只有当前问题完全可以从对话历史回答时才填写；否则为空字符串"
}

判断规则：
1. memory：用户在问历史内容本身，例如“我问你的第一个问题是什么”“我刚才问了什么”。不要做知识库检索，direct_answer 必须忠实转写历史。
2. knowledge：当前问题主要需要金融知识库。追问中的代词、省略主语要结合历史补全成独立检索问题。
3. hybrid：既需要理解历史里的指代/主体，又需要检索新的资料。
4. standalone_query 必须能脱离历史独立使用；保留具体产品名、主体名、条款主题和用户意图。
5. 不要推算、补充或解释历史之外的金融事实。direct_answer 只能转写、压缩或归纳历史里已经出现的内容。
6. 如果没有历史对话，intent 只能是 knowledge。
7. 如果当前问题在询问某个金融主体的定义、属性、数字、期限、费率、风险、收益、机制或流程，即使历史里已有相近回答，也必须选择 knowledge/hybrid 并生成 standalone_query；不要因为历史答案看起来够用就选择 memory。""".strip()


def _fallback_result(question: str, history: list[dict]) -> ConversationRouterResult:
    """LLM 理解失败时的保守兜底，保证问答链路不中断。"""
    # 只兜底非常明确的“查历史本身”问题，避免把普通追问误判成纯记忆问答。
    first_question_pattern = re.compile(
        r"("
        r"我(?:们)?(?:问|提问|问过|说过的?)的第一个?问题|"
        r"我(?:们)?的第一个?问题|"
        r"我第一个问题|"
        r"第一个问题是什么"
        r")"
    )
    last_question_pattern = re.compile(
        r"("
        r"我刚才问(?:了什么|的是哪(?:个|一)问题|哪个问题)|"
        r"我上一(?:个|条|次)问题|"
        r"我之前问(?:了|过)什么"
        r")"
    )
    assistant_answer_pattern = re.compile(
        r"你(刚才|之前)(说|回答)(了)?什么"
    )

    if history and first_question_pattern.search(question):
        first_user = next((m for m in history if m.get("role") == "user"), None)
        if first_user:
            return ConversationRouterResult(
                intent="memory",
                standalone_query="",
                direct_answer=f"你问的第一个问题是：{first_user['content']}",
            )

    if history and last_question_pattern.search(question):
        last_user = next(
            (m for m in reversed(history) if m.get("role") == "user"), None
        )
        if last_user:
            return ConversationRouterResult(
                intent="memory",
                standalone_query="",
                direct_answer=f"你刚才问的问题是：{last_user['content']}",
            )

    if history and assistant_answer_pattern.search(question):
        last_assistant = next(
            (m for m in reversed(history) if m.get("role") == "assistant"), None
        )
        if last_assistant:
            return ConversationRouterResult(
                intent="memory",
                standalone_query="",
                direct_answer=last_assistant["content"][:500],
            )

    return ConversationRouterResult(intent="knowledge", standalone_query=question)


def _parse_router_output(raw: str, question: str, history: list[dict]) -> ConversationRouterResult:
    """解析 LLM 返回的 JSON；异常时回退到保守规则。"""
    text = (raw or "").strip()
    # Qwen/其他 OpenAI 兼容模型偶尔会包 Markdown code fence，这里做一层清洗。
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return _fallback_result(question, history)

    try:
        data = json.loads(text[start : end + 1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return _fallback_result(question, history)

    intent = str(data.get("intent", "knowledge")).lower()
    if intent not in {"memory", "knowledge", "hybrid"}:
        intent = "knowledge"

    standalone_query = str(data.get("standalone_query", "")).strip()
    direct_answer = str(data.get("direct_answer", "")).strip()

    # 模型偶尔会“顺手”把事实属性追问也用历史答案回答。只要它已经补出了
    # 独立查询，且问题明显在问金融属性，就强制回到检索分支，避免漏检新证据。
    factual_attribute_pattern = re.compile(
        r"风险|收益|费率|期限|金额|流程|方式|用途|限制|要求|监管|备案|偿付|核定|通知"
    )
    if (
        intent == "memory"
        and standalone_query
        and factual_attribute_pattern.search(question)
    ):
        intent = "hybrid"
        direct_answer = ""

    # 只要模型给出了可用的历史答案，就按 memory 处理，避免继续浪费一次检索。
    if direct_answer and intent != "hybrid":
        intent = "memory"
    if intent in {"knowledge", "hybrid"} and not standalone_query:
        standalone_query = question

    return ConversationRouterResult(
        intent=intent,
        standalone_query=standalone_query or question,
        direct_answer=direct_answer[:1000],
    )


def route_question(
    cfg: dict, memory: ConversationMemory, session_id: str, question: str
) -> tuple[ConversationRouterResult, list[dict]]:
    """理解当前问题：返回 (路由结果, 最近历史)。

    没有历史时直接按知识库问题处理，不额外调用一次 LLM；
    这能让单轮问答保持原有低延迟。
    """
    history = memory.get_history(session_id)
    memory_cfg = cfg.get("memory", {})
    if not memory_cfg.get("enabled", True) or not history:
        return ConversationRouterResult(
            intent="knowledge", standalone_query=question
        ), history

    # “你问的第一个问题是什么”这种问题本身就在查历史，必须稳定命中。
    # 这里先走确定性规则，不依赖 LLM 的 JSON 输出，避免被偶发改写失败影响。
    explicit_memory = _fallback_result(question, history)
    if explicit_memory.intent == "memory":
        return explicit_memory, history

    llm_cfg = cfg["llm"]
    llm = ChatOpenAI(
        model=str(memory_cfg.get("rewrite_model", llm_cfg["model"])),
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg["api_base"],
        temperature=0,
        streaming=False,
        timeout=float(memory_cfg.get("rewrite_timeout_seconds", 15)),
        max_tokens=120,
        # SiliconFlow 的 Qwen3 系列支持这个开关；关闭思考可降低路由延迟。
        extra_body={"enable_thinking": False},
    )
    # 不使用 ChatPromptTemplate：路由系统提示词里包含 JSON 示例，
    # LangChain 会把示例中的大括号误当成变量占位符。这里直接拼消息最稳。
    messages = [
        ("system", ROUTER_SYSTEM_PROMPT),
        (
            "human",
            f"【对话历史】\n{format_history(history)}\n\n"
            f"【当前用户问题】\n{question}",
        ),
    ]
    try:
        raw = llm.invoke(messages).content
        return _parse_router_output(raw, question, history), history
    except Exception:
        # 改写模型失败时，不影响原有 RAG 链路；当前问题仍会按原样检索。
        return _fallback_result(question, history), history
