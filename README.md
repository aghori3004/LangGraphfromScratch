# 🎓 LangGraph: The College Student Course

> *citation: Source YouTube Video (Link Placeholder)*

Welcome to the **LangGraph Deep Dive**! This repo is designed to take you from "I've heard of AI Agents" to "I can build complex, reasoning, RAG-enabled systems" in 15 lectures.

The code here is written to be **clean, readable, and heavily commented**. It's built for students, by a (simulated) student. We start small and add complexity one variable at a time.

---

## 🚀 Getting Started

### 1. Prerequisites
You need:
- **Python 3.10+** installed.
- A **Google Gemini API Key** (it's free!). Get one [here](https://aistudio.google.com/).

### 2. Installation
Clone the repo and install dependencies.
```bash
# Clone the repo
git clone https://github.com/your-username/LangGraph-Course.git
cd LangGraph-Course

# Create a virtual environment (Recommended)
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# or
.venv\Scripts\activate     # Windows

# Install packages
pip install langgraph langchain langchain-google-genai python-dotenv chromadb pypdf
```

### 3. Setup API Keys
Create a `.env` file in the root directory and add your key:
```bash
GOOGLE_API_KEY="your_actual_api_key_here"
```

---

## 📚 Course Structure

### Part 1: The Fundamentals (Notebooks)
We use Jupyter Notebooks for the basics so you can visualize the graph structure instantly.

- **[1_HelloWorld.ipynb](1_HelloWorld.ipynb)**: Your first node. Input string -> Process -> Output string.
- **[2_Exercise1.ipynb](2_Exercise1.ipynb)**: *Challenge!* Build a personalized compliment bot.
- **[3_MultipleInput.ipynb](3_MultipleInput.ipynb)**: Handling complex State (Lists, Ints, Objects).
- **[4_Exercise2.ipynb](4_Exercise2.ipynb)**: *Challenge!* Build a Calculator that chooses standard operations.
- **[5_SeqGraph.ipynb](5_SeqGraph.ipynb)**: Building a pipeline (Node A -> Node B -> Node C).
- **[6_Exercise3.ipynb](6_Exercise3.ipynb)**: *Challenge!* The Resume Generator Pipeline.
- **[7_ConditionalEdge.ipynb](7_ConditionalEdge.ipynb)**: The "Brain". Using routers to make decisions (If X, go to Node Y).
- **[8_Exercise4.ipynb](8_Exercise4.ipynb)**: *Challenge!* The Advanced Multi-Step Calculator.
- **[9_LoopingGraph.ipynb](9_LoopingGraph.ipynb)**: Building Cycles. Doing things repeatedly until a condition is met.
- **[10_Exercise5.ipynb](10_Exercise5.ipynb)**: *Challenge!* The "Guess the Number" Game (Cyclic Logic).

### Part 2: Advanced Agents (Python Scripts)
We switch to standard `.py` files to build real-world agents with memory and tools.

- **[11_AgentBot.py](11_AgentBot.py)**: Connecting a Real LLM (Gemini) to LangGraph.
- **[12_AgentwMemory.py](12_AgentwMemory.py)**: Adding Memory so the AI remembers what you said.
- **[13_ReActAgent.py](13_ReActAgent.py)**: **The ReAct Pattern**. Teaching the AI to use Math Tools autonomously.
- **[14_Drafter.py](14_Drafter.py)**: **Human-in-the-Loop**. An agent that helps you write and save documents interactively.
- **[15_RAGAgent.py](15_RAGAgent.py)**: **RAG**. An agent that reads PDFs, remembers them (Vector Store), and answers questions.

---

## 🧠 Key Concepts You Will Learn

- **StateGraph**: The backbone of everything. It holds the "Memory" of your application.
- **Nodes**: Python functions that do work.
- **Edges**: The specific lines connecting nodes.
- **Conditional Edges**: Decision points (Routers).
- **Cycles**: Loops in the graph (vital for agentic behaviors).
- **ToolNode**: Giving the AI hands (calculators, search, file savers).
- **Checkpointers (Memory)**: Persisting state across interactions.

---

## 🤝 Contributing
Found a typo? Want to add a cooler example? Open a Pull Request! We're all learning here.

Happy Coding! 🤖✨
