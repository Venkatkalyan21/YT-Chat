"""LangGraph Q&A flow: (condense follow-up) -> retrieve -> generate -> check citations."""

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from . import prompts
from .citations import check_citations
from .models import Embedder
from .store import RetrievedChunk, Store


class QAState(TypedDict, total=False):
    video: dict  # id, title, channel
    question: str
    history: list[BaseMessage]
    search_query: str
    sources: list[RetrievedChunk]
    answer: str
    citations: list[dict]
    usage: dict


def build_qa_graph(llm: BaseChatModel, embedder: Embedder, store: Store, top_k: int):
    def condense(state: QAState, config: RunnableConfig) -> QAState:
        transcript = "\n".join(f"{m.type}: {m.text}" for m in state["history"])
        response = llm.invoke(
            [
                SystemMessage(prompts.CONDENSE_SYSTEM),
                HumanMessage(f"Conversation:\n{transcript}\n\nFollow-up question: {state['question']}"),
            ],
            config=config,
        )
        return {"search_query": response.text.strip() or state["question"]}

    def retrieve(state: QAState) -> QAState:
        query = state.get("search_query") or state["question"]
        sources = store.search(state["video"]["id"], embedder.embed_query(query), top_k)
        return {"search_query": query, "sources": sources}

    def generate(state: QAState, config: RunnableConfig) -> QAState:
        video = state["video"]
        messages = [
            SystemMessage(prompts.qa_system_prompt(video["title"], video.get("channel", ""), state["sources"])),
            *state.get("history", []),
            HumanMessage(state["question"]),
        ]
        response = llm.invoke(messages, config=config)
        return {"answer": response.text, "usage": dict(response.usage_metadata or {})}

    def cite(state: QAState) -> QAState:
        return {"citations": [c.to_dict() for c in check_citations(state["answer"], state["sources"])]}

    def route_start(state: QAState) -> str:
        return "condense" if state.get("history") else "retrieve"

    graph = StateGraph(QAState)
    graph.add_node("condense", condense)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("cite", cite)
    graph.add_conditional_edges(START, route_start, ["condense", "retrieve"])
    graph.add_edge("condense", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "cite")
    graph.add_edge("cite", END)
    return graph.compile()


def history_from_rows(rows: list[dict]) -> list[BaseMessage]:
    return [HumanMessage(r["content"]) if r["role"] == "user" else AIMessage(r["content"]) for r in rows]
