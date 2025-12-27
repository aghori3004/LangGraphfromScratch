"""
Lecture 14: The Drafter (Human-in-the-Loop)
-------------------------------------------
Goal: Build an interactive agent that helps you write and file documents.

This script demonstrates a "Human-in-the-Loop" workflow where:
1. The agent holds a "Draft" in memory (simulated here with a global variable).
2. The user iteratively provides instructions to update the draft.
3. The agent uses the `update` tool to modify the text.
4. The agent uses the `save` tool to write the final file to disk.

Note: In this simple example, we use `input()` inside the node to get user feedback.
In a real web app, you would interrupt the graph and wait for an API call.
"""

from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Global variable to simulate a database or document store
documentContent = ""

# 1. Define State
class AgentState(TypedDict):
    # Using add_messages to handle the conversation history automatically
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 2. Define Tools

@tool
def update(content: str) -> str:
    """
    Updates the document with the provided text. 
    Use this tool when the user wants to write, edit, or append to the doc.
    """
    global documentContent
    documentContent = content
    return f"Document has been updated successfully! The current content is: \n{documentContent}"

@tool
def saveTool(fileName: str) -> str:
    """
    Saves the current document to a text file and signals completion.
    Args:
        fileName: Name for the text file (e.g., "notes.txt").
    """
    global documentContent

    # Ensure extension
    if not fileName.endswith(".txt"):
        fileName = f"{fileName}.txt"
    
    try:
        with open(fileName, 'w') as file:
            file.write(documentContent)
        print(f"\n[System] Document has been saved to '{fileName}'")

        # Return a success message that the AI can understand
        return f"Document has been saved successfully to '{fileName}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    
tools = [update, saveTool]

# 3. Initialize Model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash").bind_tools(tools)

# 4. Define Agent Node (The Interactor)
def agent(state: AgentState) -> AgentState:
    """
    This node acts as the manager. It:
    1. Checks the state history.
    2. Dynamic Prompting: Injects the CURRENT document content into the system prompt.
    3. asks the user for input (Human-in-the-loop).
    4. Calls the LLM.
    """
    global documentContent
    
    # Dynamic System Prompt: We inject the current content so the AI always knows what it's working on.
    systemPrompt = SystemMessage(f"""
        You are a Drafter, a helpful writing assistant. You help the user create and modify documents.
        
        Rules:
        - If the user wants to update content, use the 'update' tool.
        - If the user wants to save and finish, use the 'saveTool' tool.
        - ALWAYS show the current document state to the user after modifications.
        
        Current Document Content:
        "{documentContent}"
    """)

    # Check if this is the start of the conversation using the message list
    if not state["messages"]:
        # Initial greeting (simulated user input to kickstart)
        userInput = "I'm ready to help you update a document. What would you like to create?"
        # We model the agent's self-started thought as a HumanMessage for simplicity here,
        # or we could just print it.
        # However, for this loop logic, let's treat the first 'input' as empty or setup.
        pass 
    
    # --- Human-in-the-Loop Interaction ---
    # We ask for input *inside* the node execution.
    userInput = input("\n(User): What would you like to do? ")
    userMessage = HumanMessage(content=userInput)

    # Combine: [SystemInstructions] + [ConversationHistory] + [NewUserMessage]
    # We do not modify state['messages'] directly here; we create a temporary list for the LLM.
    allMessages = [systemPrompt] + list(state["messages"]) + [userMessage]

    # Invoke LLM
    response = model.invoke(allMessages)

    print(f"\n(AI): {response.content}")
    
    # Debug info
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"[System] AI is calling tools: {[tc['name'] for tc in response.tool_calls]}")

    # Return the new messages to be appended to the state
    return {"messages": [userMessage, response]}

# 5. Define Logic
def shouldContinue(state: AgentState) -> str:
    """
    Check if the document was saved. If so, end the workflow.
    """
    messages = state["messages"]

    # We iterate backwards to see the most recent tool output
    if messages:
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                # Heuristic: If we see a successful save message, we stop.
                if "saved successfully" in message.content.lower():
                    return "end"
    
    # Otherwise, loop back to the agent for more instructions
    return "continue"

# 6. Build Graph
graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")

# Agent logic -> Calls Tool
graph.add_edge("agent", "tools")

# Tool Logic -> Check if Saved ? End : Back to Agent
graph.add_conditional_edges(
    "tools",
    shouldContinue,
    {
        "continue": "agent", # Go back to ask user "What next?"
        "end": END
    }
)

app = graph.compile()

# 7. Execution Helper
def printMessages(messages):
    """Refined printer to show tool outputs clearly"""
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n[Tool Result]: {message.content}")

def runDocAgent():
    print("\n--- DRAFTER AGENT STARTED ---")
    
    # Initialize state
    state = {"messages": []}

    # Run the graph
    # We use stream_mode="values" to get the state after every node execution
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            printMessages(step["messages"])
    
    print("\n--- DRAFTER AGENT FINISHED ---")

if __name__ == "__main__":
    runDocAgent()