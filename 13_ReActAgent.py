"""
Lecture 13: The ReAct Agent (Reason + Act)
------------------------------------------
Goal: Build an agent that can use Tools (Functions) to solve problems.

"ReAct" stands for Reasoning and Acting. The agent:
1. Reasons about the user's input.
2. Acts by calling a tool if needed.
3. Observes the tool's output.
4. Reasons again to give the final answer.

This script uses LangGraph's prebuilt `ToolNode` and the powerful `add_messages` reducer.
"""

from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

# BaseMessage: The parent class for all message types.
# ToolMessage: Represents the output from a tool call.
# SystemMessage: Instructions for the AI's behavior.
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

# add_messages: A magical function! It automatically handles appending new messages
# to the list. If a message has the same ID, it replaces it.
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# 1. Load Environment Variables
load_dotenv()

# 2. Define State
# We use `Annotated` here. This tells LangGraph:
# "When a node returns 'messages', don't overwrite the list. Run 'add_messages' to append them."
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 3. Define Tools
# We use the @tool decorator to turn standard Python functions into Tools the AI can understand.
# The docstrings are CRITICAL. The AI reads them to know when to use the tool.

@tool
def addTool(a: int, b: int) -> int:
    """Adds two numbers together (a + b)."""
    return a + b

@tool
def mulTool(a: int, b: int) -> int:
    """Multiplies two numbers (a * b)."""
    return a * b

@tool
def subTool(a: int, b: int) -> int:
    """Subtracts b from a (a - b)."""
    return a - b

# List of tools to give the agent
tools = [addTool, mulTool, subTool]

# 4. Initialize LLM with Tools
# .bind_tools() attaches the tool definitions to the model.
# The model can now decide to generate a "tool_call" instead of just text.
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash").bind_tools(tools)

# 5. Define Nodes

def modelCall(state: AgentState) -> AgentState:
    """
    Decider Node:
    1. Reviews history.
    2. Decides whether to reply with text or call a tool.
    """
    # System prompt sets the personality
    systemPrompt = SystemMessage(content="You are a helpful AI assistant. Use tools for math.")
    
    # We prepend the system prompt to the message history
    response = model.invoke([systemPrompt] + state["messages"])
    
    # We return the NEW message. `add_messages` handles the appending.
    return {"messages": [response]}

# 6. Define Conditional Logic
def shouldContinue(state: AgentState) -> str:
    """
    Checks if the last message from the AI contains a tool call.
    If yes -> Go to 'tools' node.
    If no -> The AI has finished reasoning. End.
    """
    messages = state["messages"]
    lastMessage = messages[-1]
    
    # If the LLM wants to run a tool, it generates a 'tool_calls' attribute.
    if hasattr(lastMessage, "tool_calls") and len(lastMessage.tool_calls) > 0:
        return "continue"
    else: 
        return "end"

# 7. Build Graph
graph = StateGraph(AgentState)

# Node 1: The Brain (LLM)
graph.add_node("agent", modelCall)

# Node 2: The Action (Tool Execution)
# ToolNode is a prebuilt LangGraph node that:
# - Scans the last message for tool calls.
# - Executes the corresponding Python function.
# - Returns a ToolMessage with the result.
toolNode = ToolNode(tools=tools)
graph.add_node("tools", toolNode)

# Start -> Agent
graph.add_edge(START, "agent")

# Agent -> Decision -> (Tools OR End)
graph.add_conditional_edges(
    "agent",
    shouldContinue,
    {
        "continue": "tools",
        "end": END
    }
)

# Tools -> Agent
# After running a tool, we ALWAYS go back to the agent so it can read the result and respond.
graph.add_edge("tools", "agent")

app = graph.compile()

# 8. Run and visualize
def print_stream(stream):
    """Helper to pretty-print the conversation steps"""
    for s in stream:
        # 's' is a dictionary of the update from the node that just ran
        # e.g., {'agent': {'messages': [...]} }
        for key, value in s.items():
            print(f"\n--- Node: {key} ---")
            # The value is the state update. We look at the last message.
            last_msg = value["messages"][-1]
            last_msg.pretty_print()

print("--- Starting ReAct Agent ---")
query = "Add 34 + 21, Add 3+4, then multiply the answer of both and then tell me a joke about math."
inputs = {"messages": [("user", query)]}

# stream_mode="values" returns the full state at each step
# stream_mode="updates" (default) returns only the node updates
# We use app.stream() to see the thinking process
print_stream(app.stream(inputs))