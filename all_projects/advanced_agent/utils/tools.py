
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition

from advanced_agent.utils.state import State

@tool
def request_assistance():
    """Escalate the conversation to an expert. Use this if the search query is ambiguous or unclear, prompt the user with a clarifying question to gather specific attributes or details about the product they are searching for.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """
    return ""

@tool
def run_python_code():
    """
    Executes Python code on a Pandas DataFrame.

    This function should be called when a user requests an operation, transformation, 
    or analysis on a Pandas DataFrame. Examples include:
    - Filtering or transforming data
    - Adding, modifying, or removing columns
    - Performing statistical analyses or aggregations
    - Applying custom Python logic to the DataFrame

    The function takes user-provided Python code and a Pandas DataFrame as input, 
    applies the code to the DataFrame, and returns the modified DataFrame or 
    the result of the computation.

    Usage scenarios:
    - "Filter rows where column 'A' is greater than 10."
    - "Add a new column 'B' as the square of column 'A'."
    - "Calculate the mean of column 'C' grouped by column 'D'."

    Note: Ensure the code provided is safe and validated before execution.
    """
    return ""

def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    if state["execute_code"]:
        return "PythonExecutor"
    return tools_condition(state)