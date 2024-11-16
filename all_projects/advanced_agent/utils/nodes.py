import functools
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.messages import HumanMessage

from advanced_agent.utils.state import State
from advanced_agent.utils.tools import request_assistance

    
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
    execute_code = False
    
    if response.tool_calls and response.tool_calls[0]["name"] == "request_assistance":
        ask_human = True
        
    if response.tool_calls and response.tool_calls[0]["name"] == "run_python_code":
        execute_code = True
        
    return {"messages": [response], "ask_human": ask_human, "execute_code": execute_code}


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
        "execute_code": False
    }

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)], "ask_human": False, "execute_code": False}


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)], "ask_human": False, "execute_code": False}


data = pd.read_csv("data.csv", index_col=0)
model_name = "llama-3.1-70b-versatile"
llm_python_executor = ChatGroq(
    model_name=model_name
)
python_executor_agent = create_pandas_dataframe_agent(llm_python_executor, data, verbose=True, allow_dangerous_code=True)
python_executor_node = functools.partial(agent_node, agent=python_executor_agent, name="PythonExecutor")