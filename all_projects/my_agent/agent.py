# from langchain_community.chat_models import ChatOllama
# from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.prebuilt import ToolNode

from my_agent.utils.state import State
from my_agent.utils.nodes import human_node, chatbot
from my_agent.utils.tools import select_next_node



graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[TavilySearchResults(max_results=2)]))
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
    interrupt_before=["human"],
)
