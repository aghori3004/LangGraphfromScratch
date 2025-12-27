"""
Lecture 15: The RAG Agent (Retrieval Augmented Generation)
----------------------------------------------------------
Goal: Build an agent that can read a PDF and answer questions about it.

"RAG" allows an AI to "study" external documents.
1. **Ingest**: Read a PDF.
2. **Split**: Break it into small chunks.
3. **Embed**: Convert chunks into numbers (vectors) using an Embedding Model.
4. **Store**: Save vectors in a database (ChromaDB).
5. **Retrieve**: When asked a question, find the most similar chunks.
6. **Generate**: Pass those chunks to the LLM to write an answer.

This script manually implements the tool execution loop to show you how `ToolNode` works under the hood.
"""

from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages

# Google's models for Chat and Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Document Loaders & Splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Database
from langchain_chroma import Chroma 
from langchain_core.tools import tool

load_dotenv()

# 1. Initialize Models
# We use a temperature of 0 for factual accuracy.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0
)

# Text Embedding Model: Converts text into a list of numbers.
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# 2. Ingestion Pipeline (Run once to setup DB)
# For this example, we assume "Stock_Market_Performance_2024.pdf" exists.
# If you don't have it, create a dummy PDF or change the path.
pdfPath = "Stock_Market_Performance_2024.pdf"

# Initialize variables
vectorStore = None
retriever = None

# A. Load PDF
if os.path.exists(pdfPath):
    print(f"--- Loading {pdfPath} ---")
    pdfLoader = PyPDFLoader(pdfPath)
    pages = pdfLoader.load()
    print(f"Loaded {len(pages)} pages.")

    # B. Split Text
    # LLMs have a limit on how much they can read. We split text into chunks.
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Characters per chunk
        chunk_overlap=200   # Overlap to maintain context
    )
    pagesSplit = textSplitter.split_documents(pages)

    # C. Create Vector Store
    # We save this to disk so we don't have to rebuild it every time.
    persistDirectory = r"./chroma_db"
    collectionName = "stockMarket"

    print("--- Creating/Loading Vector Store ---")
    try:
        vectorStore = Chroma.from_documents(
            documents=pagesSplit,
            embedding=embeddings,
            persist_directory=persistDirectory,
            collection_name=collectionName
        )
        print("Vector Store Ready!")
    except Exception as e:
        print(f"Error setting up ChromaDB: {e}")

    # D. Create Retriever
    # This interface lets us search the specific vector store.
    if vectorStore:
        retriever = vectorStore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5} # Fetch top 5 most relevant chunks
        )
else:
    print(f"Warning: {pdfPath} not found. RAG functionality will fail.")

# 3. Define the Retrieval Tool
@tool
def retrieverTool(query: str) -> str:
    """
    Searches the 'Stock Market Performance 2024' document for relevant information.
    Use this tool whenever the user asks about the stock market or the PDF content.
    """
    if not retriever:
        return "Error: Retriever not initialized. PDF file missing."

    print(f"\n[Tool] Searching for: '{query}'")
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the document."
    
    # Format the results into a single string
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Source {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

tools = [retrieverTool]
tools_dict = {t.name: t for t in tools}

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# 4. Define State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 5. Define Nodes

# Node 1: The Brain
def callLLM(state: AgentState) -> AgentState:
    """Invokes the LLM with the system prompt and conversation history."""
    
    systemPrompt = SystemMessage(content="""
    You are an expert financial analyst assistant. 
    You have access to a report on Stock Market Performance in 2024 via the 'retrieverTool'.
    
    Instructions:
    1. ALWAYS use the 'retrieverTool' to look up information before answering.
    2. Cite the specific parts of the extracted text in your answer.
    3. If the tool returns no info, say you don't know interactively.
    """)
    
    # Prepend system prompt to history
    messages = [systemPrompt] + list(state["messages"])
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Node 2: Manual Tool Executor
# In Lecture 13, we used prebuilt `ToolNode`. Here, we write it ourselves to learn.
def take_action(state: AgentState) -> AgentState:
    """
    Parses the last AI message, executes any tool calls, 
    and appends the results as ToolMessages.
    """
    last_message = state['messages'][-1]
    
    # If no tool calls, do nothing (shouldn't happen due to conditional edge, but safe to check)
    if not last_message.tool_calls:
        return state

    results = []
    for t in last_message.tool_calls:
        tool_name = t['name']
        tool_args = t['args']
        tool_id = t['id']
        
        print(f"--- Executing Tool: {tool_name} ---")
        
        # 1. Find the tool
        if tool_name not in tools_dict:
            result = f"Error: Tool '{tool_name}' not found."
        else:
            # 2. Invoke the tool (taking the 'query' argument usually)
            # We handle potential argument variations safely
            try:
                # Most tools take a single string argument, sometimes wrapped in a dict
                if "query" in tool_args:
                    tool_input = tool_args["query"]
                else:
                    tool_input = tool_args
                
                result = tools_dict[tool_name].invoke(tool_input)
            except Exception as e:
                result = f"Error execution tool: {e}"
        
        # 3. Create ToolMessage
        # This is CRITICAL. The LLM needs to know which tool call this result belongs to.
        results.append(ToolMessage(
            tool_call_id=tool_id,
            name=tool_name,
            content=str(result)
        ))

    return {'messages': results}

# 6. Define Conditional Logic
def shouldContinue(state: AgentState):
    """If the LLM provided tools to call, go to 'retriever_agent'. Else End."""
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
        return True
    return False

# 7. Build Graph
graph = StateGraph(AgentState)

graph.add_node("llm", callLLM)
graph.add_node("retriever_agent", take_action)

graph.set_entry_point("llm")

# LLM -> Decision -> (Tool OR End)
graph.add_conditional_edges(
    "llm",
    shouldContinue,
    {
        True: "retriever_agent",
        False: END
    }
)

# Tool -> LLM (Always go back to generate the final answer)
graph.add_edge("retriever_agent", "llm")

rag_agent = graph.compile()

# 8. Run
def running_agent():
    print("\n=== RAG AGENT STARTED ===")
    print("Ask about the 2024 Stock Market! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nQuestion: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        # Run graph
        # We start with just the user message
        state = {"messages": [HumanMessage(content=user_input)]}
        result = rag_agent.invoke(state)
        
        # The last message is the final AI response
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

if __name__ == "__main__":
    running_agent()