from typing import Annotated

import pandas as pd
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# from langchain_community.chat_models import ChatOllama
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ask_human: bool


@tool
def request_assistance():
    """Escalate the conversation to an expert. Use this if the search query is ambiguous or unclear, prompt the user with a clarifying question to gather specific attributes or details about the product they are searching for.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """
    return ""


def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if response.tool_calls and response.tool_calls[0]["name"] == "request_assistance":
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}


def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        "messages": new_messages,
        "ask_human": False,
    }


def read_data(path):
    return pd.read_csv(path)


tavily_search = TavilySearchResults(max_results=2)
tools = [tavily_search, read_data]


def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    return tools_condition(state)


tool = TavilySearchResults(max_results=2)
tools = [tool]
# llm = ChatOpenAI()
# llm = ChatOllama(model="llama3.1:8b")
model_name = "llama-3.1-70b-versatile"
llm = ChatGroq(
    model_name=model_name
)
llm_with_tools = llm.bind_tools(tools + [request_assistance])


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[tool]))
graph_builder.add_node("human", human_node)
graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", "__end__": "__end__"},
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = SqliteSaver.from_conn_string(":memory:")

graph = graph_builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=[],
)
