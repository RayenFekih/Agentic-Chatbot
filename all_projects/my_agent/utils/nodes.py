from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from my_agent.utils.state import State
from my_agent.utils.tools import request_assistance

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
    
def chatbot(state: State):
    tavily_search_tool = TavilySearchResults(max_results=2)
    # llm = ChatOpenAI()
    # llm = ChatOllama(model="llama3.1:8b")
    model_name = "llama-3.1-70b-versatile"
    llm = ChatGroq(
        model_name=model_name
    )
    llm_with_tools = llm.bind_tools([tavily_search_tool, request_assistance])
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if response.tool_calls and response.tool_calls[0]["name"] == "request_assistance":
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}