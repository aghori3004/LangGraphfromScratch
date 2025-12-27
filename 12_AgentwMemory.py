"""
Lecture 12: Adding Memory (The "Manual" Way)
--------------------------------------------
Goal: Enable the chatbot to remember previous messages.

In the previous lecture, the agent forgot everything after each interaction.
Here, we perform "manual memory management":
1. We keep a `conversationHistory` list in Python.
2. We append every User message and AI message to this list.
3. We feed the *entire list* back to the agent every time.
4. Finally, we save the log to a file.
"""

import os
from typing import TypedDict, List, Union

# Union: Allows a variable to be one of several types.
# AIMessage: Represents a message coming from the AI (Gemini).
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# 1. Define State
# Our messages list can now contain EITHER a HumanMessage OR an AIMessage.
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# 2. Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

# 3. Define Logic Node
def processNode(state: AgentState) -> AgentState:
    """
    Receives the full history of messages, gets a response, 
    and appends the AI's response to the history.
    """
    # 1. Invoke LLM with *all* messages in the state (history + current input)
    response = llm.invoke(state["messages"])
    
    # 2. Append the AI's response to the state manually
    # We construct an AIMessage object to distinguish it from the user's input.
    state["messages"].append(AIMessage(content=response.content))
    
    print(f"\nAI: {response.content}")
    return state

# 4. Build Graph
graph = StateGraph(AgentState)
graph.add_node("process", processNode)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# 5. External Memory Management
# We create a list outside the graph to hold the history.
conversationHistory = []

print("--- Chat with Memory Started (Type 'exit' to quit) ---")
userInput = input("Enter: ")

while userInput.lower() != "exit":
    # 1. Add User's new message to history
    conversationHistory.append(HumanMessage(content=userInput))
    
    # 2. Pass the FULL history to the agent
    result = agent.invoke({"messages": conversationHistory})
    
    # 3. Update our external history with the result
    # The agent returns the state, which now includes the AI's new message.
    # We sync our local variable with the graph's output.
    conversationHistory = result["messages"]
    
    userInput = input("Enter: ")

# 6. Logging
# Save the chat to a file for review.
print("\n--- Saving Conversation Log ---")
with open("logging.txt", "w") as file:
    file.write("Your Conversation Log: \n\n")

    for message in conversationHistory:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content} \n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content} \n\n")
            
    file.write("End of conversation")

print("Conversation saved to logging.txt")