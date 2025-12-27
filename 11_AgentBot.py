"""
Lecture 11: The First AI Agent (Chatbot)
----------------------------------------
Goal: Move from hard-coded logic to using a Real AI Brain (LLM).

This script demonstrates:
1. Connecting to Google's Gemini model via LangChain.
2. Using 'HumanMessage' to structure input.
3. Creating a simple graph that passes user messages to the AI and prints the response.
"""

# TypedDict: Used to define the structure of our State dictionary (type safety).
# List: standard python list, used to hold multiple messages.
from typing import TypedDict, List

# HumanMessage: A specific class from LangChain that represents a message 
# coming from a human user. This helps the LLM distinguish between 
# user input and system instructions.
from langchain_core.messages import HumanMessage

# ChatGoogleGenerativeAI: The wrapper class that lets us talk to Gemini.
from langchain_google_genai import ChatGoogleGenerativeAI

# StateGraph: The core graph builder.
# START, END: Special nodes marking the beginning and end of the workflow.
from langgraph.graph import StateGraph, START, END

# dotenv: Loads environment variables (like your GEMINI_API_KEY) from a .env file.
# This keeps your secrets safe!
from dotenv import load_dotenv

# 1. Load keys
load_dotenv()

# 2. Define the State
# Up until now, we used simple strings or numbers. 
# Now we use 'HumanMessage' objects to hold the conversation.
class AgentState(TypedDict):
    messages: List[HumanMessage]

# 3. Initialize the Brain (LLM)
# We choose "gemini-2.0-flash-lite" as it's fast and efficient.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite"
)

# 4. Define the Logic Node
# This function acts as the "Agent". It receives the state (with user message),
# sends it to the LLM, and allows the LLM to generate a response.
def processNode(state: AgentState) -> AgentState:
    """
    1. Looks at state['messages'] (the user's input).
    2. Sends it to the LLM via .invoke().
    3. Prints the AI's response content to the console.
    """
    # .invoke() sends the list of messages to Google's servers
    response = llm.invoke(state["messages"])
    
    # We print it here directly for immediate feedback
    print(f"\nAI: {response.content}")
    
    # In a more complex agent, we might save the response back 
    # to the state['messages'] here.
    return state

# 5. Build the Graph
print("--- Initializing Agent Graph ---")
graph = StateGraph(AgentState)

# Add our single node
graph.add_node("process", processNode)

# Straight line: Start -> Process -> End
graph.add_edge(START, "process")
graph.add_edge("process", END)

# Compile into a runnable application
agent = graph.compile()

# 6. Run the Chat Loop
# Simple while loop to keep the conversation going until user types 'exit'

print("--- Chat Started (Type 'exit' to quit) ---")
userInput = input("Enter: ")

while userInput.lower() != "exit":
    # We wrap the string input into a structured HumanMessage
    msg = HumanMessage(content=userInput)
    
    # Invoke the agent. 
    # Note: We are creating a NEW state every time with just this one message.
    agent.invoke({"messages": [msg]})
    
    # Get next input
    userInput = input("Enter: ")

print("--- Chat Ended ---")