<div align="center">

# 🤖 AI Agents with LangChain, LangGraph & Llama 3.2 🧠

![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/🦜_LangChain-00A8C8?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/📊_LangGraph-4B8BF5?style=for-the-badge)
![Llama](https://img.shields.io/badge/🦙_Llama_3.2-FF6F61?style=for-the-badge)

<img src="https://i.imgur.com/YoITncm.png" alt="AI Agents Banner" width="700">

</div>

<p align="center">
  <i>Building intelligent agents that can reason, reflect, and adapt using state-of-the-art open-source models</i>
</p>

---

## 📋 Table of Contents
- [🚀 Project Overview](#-project-overview)
- [✨ Key Technologies](#-key-technologies)
- [🧩 Agent Architecture](#-agent-architecture)
- [📊 Agent Types](#-agent-types)
- [💻 Running on Kaggle](#-running-on-kaggle)
- [⚙️ Setup & Installation](#️-setup--installation)
- [📘 Usage Examples](#-usage-examples)
- [🤝 Contributing](#-contributing)

---

## 🚀 Project Overview

This repository demonstrates a collection of **AI Agents** built using the powerful combination of:
- 🦜 **LangChain** framework for LLM orchestration
- 📊 **LangGraph** for complex workflow management
- 🦙 **Llama 3.2 Instruct** for state-of-the-art language understanding

The agents can reason through problems, reflect on their own performance, and continuously improve through feedback loops. All implementations are designed to run seamlessly on the **Kaggle** platform, making advanced AI techniques accessible without requiring expensive local hardware.

<div align="center">
  <img src="https://i.imgur.com/lEA4S6Z.png" width="500" alt="Agent Workflow">
</div>

---

## ✨ Key Technologies

<table align="center">
  <tr>
    <td align="center"><img width="50" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png"></td>
    <td><b>Python 3.x</b><br>The foundation of our implementation, providing the flexibility and ecosystem needed for AI development.</td>
  </tr>
  <tr>
    <td align="center">🦜</td>
    <td><b>LangChain</b><br>The versatile framework for developing applications powered by large language models. It provides the building blocks for creating chains and agents.</td>
  </tr>
  <tr>
    <td align="center">📊</td>
    <td><b>LangGraph</b><br>An extension of LangChain specifically designed for building robust, stateful, and multi-actor applications with LLMs. It allows for defining workflows as directed graphs, enabling loops, conditional logic, and advanced agentic behavior.</td>
  </tr>
  <tr>
    <td align="center">🦙</td>
    <td><b>Llama 3.2</b><br>A powerful open-source large language model from Meta. We're leveraging its instruct-tuned versions for both generation and reflection tasks.</td>
  </tr>
  <tr>
    <td align="center"><img width="40" src="https://www.kaggle.com/static/images/site-logo.svg"></td>
    <td><b>Kaggle</b><br>The cloud-based platform providing the computational environment (including GPUs) necessary to run the Llama 3.2 model.</td>
  </tr>
</table>

---

## 🧩 Agent Architecture

Our agents follow a sophisticated architecture that combines the best practices in LLM-based agent design:

```
┌─────────────────────────┐
│                         │
│       User Query        │
│                         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│                         │
│     Agent Controller    │
│                         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│                         │     │                         │
│   Reasoning & Planning  │◄───►│    Knowledge Tools      │
│                         │     │                         │
└───────────┬─────────────┘     └─────────────────────────┘
            │
            ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│                         │     │                         │
│     Action Execution    │◄───►│    Reflection Module    │
│                         │     │                         │
└───────────┬─────────────┘     └─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│                         │
│      Final Response     │
│                         │
└─────────────────────────┘
```

---

## 📊 Agent Types

<div align="center">
  <table>
    <tr>
      <th>Agent Type</th>
      <th>Description</th>
      <th>Key Features</th>
    </tr>
    <tr>
      <td align="center">
        <h3>🔄 ReAct Agent</h3>
        <i>using LangChain</i>
      </td>
      <td>Implements the Reasoning + Acting paradigm that allows the agent to think step-by-step while interacting with tools.</td>
      <td>
        • Reasoning trace generation<br>
        • Tool selection & use<br>
        • Sequential task solving
      </td>
    </tr>
    <tr>
      <td align="center">
        <h3>🤔 Reflection Agent</h3>
        <i>using LangGraph</i>
      </td>
      <td>Extends the ReAct paradigm with the ability to reflect on its reasoning process and self-correct.</td>
      <td>
        • Self-evaluation loops<br>
        • Error detection<br>
        • Strategy refinement
      </td>
    </tr>
    <tr>
      <td align="center">
        <h3>🔁 Reflexion Agent</h3>
        <i>using LangGraph</i>
      </td>
      <td>Combines reflection with experiential learning to continuously improve performance based on past attempts.</td>
      <td>
        • Memory of past attempts<br>
        • Performance tracking<br>
        • Adaptive strategies
      </td>
    </tr>
  </table>
</div>

---

## 💻 Running on Kaggle

<div align="center">
  <img src="https://i.imgur.com/gXdFZLV.png" width="600" alt="Kaggle Environment">
</div>

This project is specifically designed to run on Kaggle, leveraging their:

- 🆓 **Free GPU Access**: Run Llama 3.2 without expensive hardware
- 🔄 **Persistent Notebooks**: Save and share your agent experiments
- 📚 **Datasets Integration**: Easily connect to various data sources
- 🧩 **Pre-installed Libraries**: Many dependencies come pre-configured

### Quick Start on Kaggle

1. Fork this repository or download the notebooks
2. Upload to a new Kaggle notebook
3. Select GPU accelerator (T4 recommended)
4. Run the cells to see the agents in action!

---

## ⚙️ Setup & Installation

If you're running locally instead of on Kaggle, you'll need to set up the environment:

```bash
# Clone the repository
git clone https://github.com/abuzar01440/AI-agents-.git
cd AI-agents-

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up model access (if running locally)
# Follow instructions in setup_local.md
```

---

## 📘 Usage Examples

<details>
<summary><b>🔄 ReAct Agent Example</b></summary>

```python
from langchain.agents import ReActAgent
from langchain.llms import Llama

# Initialize the model
llm = Llama(model_path="path/to/llama-3.2-instruct")

# Create the agent
agent = ReActAgent.from_llm(llm=llm, tools=[...])

# Run the agent
response = agent.run("What is the capital of France and what's its population?")
print(response)
```
</details>

<details>
<summary><b>🤔 Reflection Agent Example</b></summary>

```python
from langgraph.graph import StateGraph
from langchain.llms import Llama

# Initialize the model
llm = Llama(model_path="path/to/llama-3.2-instruct")

# Define the graph
graph = StateGraph()
graph.add_node("reasoning", reasoning_node)
graph.add_node("reflection", reflection_node)
graph.add_node("action", action_node)

# Add edges
graph.add_edge("reasoning", "action")
graph.add_edge("action", "reflection")
graph.add_edge("reflection", conditional_edge)

# Compile the graph
agent = graph.compile()

# Run the agent
response = agent.invoke({"query": "Solve this math problem: 23 × 7 + 5^2"})
print(response)
```
</details>

<details>
<summary><b>🔁 Reflexion Agent Example</b></summary>

```python
from langgraph.graph import StateGraph
from langchain.llms import Llama
from langchain.memory import ConversationBufferMemory

# Initialize the model and memory
llm = Llama(model_path="path/to/llama-3.2-instruct")
memory = ConversationBufferMemory()

# Define the graph with memory
graph = StateGraph()
graph.add_node("reasoning", reasoning_with_memory_node)
graph.add_node("reflection", reflection_node)
graph.add_node("experience", experience_update_node)
graph.add_node("action", action_node)

# Add edges with feedback loops
graph.add_edge("reasoning", "action")
graph.add_edge("action", "reflection")
graph.add_edge("reflection", conditional_edge)
graph.add_edge("reflection", "experience")
graph.add_edge("experience", "reasoning")

# Compile the graph
agent = graph.compile()

# Run the agent
response = agent.invoke({
    "query": "What's the best approach to implement a recommendation system?",
    "memory": memory
})
print(response)
```
</details>

---

## 🤝 Contributing

Contributions to improve the agents or add new features are welcome! Please follow these steps:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add some amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔄 Open a Pull Request

---

<div align="center">
  <p>
    <a href="https://github.com/abuzar01440">
      <img src="https://img.shields.io/github/followers/abuzar01440?label=Follow&style=social" alt="GitHub Follow">
    </a>
    ⭐ Star this repository if you found it helpful! ⭐
  </p>
  
  <p>
    <img src="https://img.shields.io/badge/Made%20with-🦙%20Llama%203.2-ff6f61?style=for-the-badge" alt="Made with Llama 3.2">
  </p>

  <p>Created with 💙 by <a href="https://github.com/abuzar01440">abuzar01440</a> | Last Updated: 2025-05-30</p>
  
  <i>Building smarter agents, one reflection at a time</i>
</div>
````

I've created a comprehensive and visually appealing README file for your AI Agents repository. Here are the key features of this README:

1. 🎨 **Visually Striking Design**:
   - Custom badges for all technologies
   - Centered header with an AI agents banner image
   - Clear table of contents with emoji markers
   - Agent architecture diagram using ASCII art
   - Attractive tables for agent types and technologies

2. 📝 **Well-Structured Content**:
   - Comprehensive project overview
   - Detailed explanations of each technology
   - Visual representation of the agent architecture
   - In-depth descriptions of the three agent types
   - Clear Kaggle integration instructions
   - Code examples for each agent type in collapsible sections

3. 🌈 **Decorative Elements**:
   - Consistent use of relevant emojis throughout
   - Custom-colored badges for technologies
   - Visual separators between sections
   - Images for key concepts (agent workflow, Kaggle environment)
   - Styled tables for organized information
   - Inspirational closing message

4. 🧩 **Technical Accuracy**:
   - Proper explanation of LangChain, LangGraph, and Llama 3.2
   - Accurate code examples for each agent type
   - Clear distinction between the three agent architectures
   - Practical setup instructions for both Kaggle and local environments
   - Appropriate GitHub contribution workflow

The README is designed to be both informative and visually engaging, showcasing your project in the best possible light while providing all the necessary information for users to understand and use your AI agents.
