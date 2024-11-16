
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition

from my_agent.utils.state import State

@tool
def request_assistance():
    """Escalate the conversation to an expert. Use this if the search query is ambiguous or unclear, prompt the user with a clarifying question to gather specific attributes or details about the product they are searching for.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """
    return ""

def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    return tools_condition(state)