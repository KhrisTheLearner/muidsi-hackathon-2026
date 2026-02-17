"""Synthesizer node - combines tool results into a final response."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.llm import SYNTHESIZER_MODEL, get_llm
from src.agent.prompts.system import RESPONDER_PROMPT
from src.agent.state import AgriFlowState


def synthesizer_node(state: AgriFlowState) -> dict:
    """Synthesize all collected data into a final, formatted response.

    Instead of replaying the full message history (which includes tool call
    objects that are hard to re-serialize), we extract the key content and
    send a clean summary to the LLM for final formatting.
    """
    # Extract the original query
    original_query = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break

    # Extract tool results from ToolMessages
    tool_summaries = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            tool_summaries.append(f"[{msg.name}] {msg.content}")

    # Extract the last assistant analysis (from tool_caller's final pass)
    last_analysis = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            last_analysis = msg.content
            break

    # If tool_caller already produced a good, structured analysis, use it directly
    if last_analysis:
        skip = False
        if not tool_summaries:
            skip = True
        elif len(last_analysis) > 200:
            has_structure = any(m in last_analysis for m in ["**", "##", "1.", "- "])
            if has_structure:
                skip = True

        if skip:
            return {
                "messages": [AIMessage(content=last_analysis)],
                "final_answer": last_analysis,
                "reasoning_trace": state.get("reasoning_trace", []) + [
                    "Synthesis: tool_caller analysis was sufficient — skipped LLM"
                ],
            }

    # Build a clean prompt for the synthesizer
    synthesis_input = f"""Original question: {original_query}

Data collected from tools:
{chr(10).join(tool_summaries) if tool_summaries else 'No tool data collected.'}

Previous analysis:
{last_analysis if last_analysis else 'None yet.'}

Please format a clear, actionable response based on this data."""

    llm = get_llm(model=SYNTHESIZER_MODEL, temperature=0.1)
    messages = [
        SystemMessage(content=RESPONDER_PROMPT),
        HumanMessage(content=synthesis_input),
    ]

    response = llm.invoke(messages)

    answer = response.content
    if not answer and last_analysis:
        answer = last_analysis

    reasoning = state.get("reasoning_trace", []) + [
        "Synthesis: combined all tool results into final response"
    ]

    return {
        "messages": [response],
        "final_answer": answer,
        "reasoning_trace": reasoning,
    }
