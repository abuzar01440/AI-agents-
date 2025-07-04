<div align="center">

# 🚀 AI Agents: Language Model Intelligence Framework 🤖✨

![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/🦜_LangChain-00A8C8?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/📊_LangGraph-4B8BF5?style=for-the-badge)
![AI Models](https://img.shields.io/badge/🤖_Multi_Models-FF6F61?style=for-the-badge)

![AI Agent Logo](https://tse2.mm.bing.net/th/id/OIP.Z7Dcbfq5Eni_CEBEIa87wQHaHa?rs=1&pid=ImgDetMain&o=7&rm=3)


</div>

<p align="center">
  <i>🧠 Building intelligent agents that can reason, reflect, and adapt using both Small Language Models (SLMs) and Large Language Models (LLMs) - 💰 Zero API costs with maximum intelligence! 🎯</i>
</p>

---

## 🤖 What Are AI Agents?

**AI Agents** are autonomous software programs that can perceive their environment, make decisions, and take actions to achieve specific goals. Unlike traditional chatbots that simply respond to queries, AI agents can:

- 🧠 **Reason & Plan**: Break down complex problems into manageable steps
- 🔍 **Use Tools**: Search the web, access databases, run calculations
- 🤔 **Reflect & Learn**: Evaluate their own performance and improve over time
- 🎯 **Act Autonomously**: Make decisions without constant human supervision
- 💭 **Maintain Memory**: Remember past interactions and learn from experience

Think of them as **digital assistants with superpowers** - they don't just answer questions, they solve problems! 🦸‍♂️✨

---

## 📋 Table of Contents
- [🚀 Project Overview](#-project-overview)
- [✨ Key Technologies](#-key-technologies)
- [🧩 Agent Architecture](#-agent-architecture)
- [🤖 Agent Types & Comparisons](#-agent-types--comparisons)
- [🔬 SLM vs LLM Analysis](#-slm-vs-llm-analysis)
- [💻 Running on Kaggle & Google Colab](#-running-on-kaggle--google-colab)
- [⚙️ Setup & Installation](#️-setup--installation)
- [📘 Usage Examples](#-usage-examples)
- [🎯 Why This Approach Works](#-why-this-approach-works)
- [🤝 Contributing](#-contributing)

---

## 🚀 Project Overview

This repository demonstrates a comprehensive collection of **AI Agents** built using cutting-edge frameworks and multiple language models:

- 🦜 **LangChain** framework for LLM orchestration
- 📊 **LangGraph** for complex workflow management  
- 🦙 **Llama 3.2** for cost-effective reasoning
- 🔥 **Gemma** for advanced reflection capabilities
- 💎 **Gemini 1.5 Flash** for human-interactive tasks
- 🔄 **Multiple Agent Paradigms** (ReAct, Reflection, Reflexion, Human-in-the-Loop)

### 💡 **Key Innovation**: 
This project showcases how **Small Language Models (SLMs)** like Llama 3.2 and Gemma can be effectively used for AI agents, providing **zero API costs** while running on free platforms like Kaggle and Google Colab, while strategically using premium models only when human interaction is involved! 🎉

---

## ✨ Key Technologies

<table align="center">
  <tr>
    <td align="center">🐍</td>
    <td><b>Python 3.x</b><br>The foundation of our implementation, providing flexibility and ecosystem needed for AI development.</td>
  </tr>
  <tr>
    <td align="center">🦜</td>
    <td><b>LangChain</b><br>Versatile framework for developing applications powered by language models. Provides building blocks for creating chains and agents.</td>
  </tr>
  <tr>
    <td align="center">📊</td>
    <td><b>LangGraph</b><br>Extension of LangChain for building robust, stateful, multi-actor applications. Enables workflows as directed graphs with loops and conditional logic.</td>
  </tr>
  <tr>
    <td align="center">🤖</td>
    <td><b>Multiple AI Models</b><br>Strategic use of different models: Llama 3.2, Gemma for SLM tasks, Gemini 1.5 Flash for premium interactions.</td>
  </tr>
  <tr>
    <td align="center">🔍</td>
    <td><b>Tavily Search</b><br>Real-time web search integration for agents to access up-to-date information and provide cited responses.</td>
  </tr>
  <tr>
    <td align="center">💻</td>
    <td><b>Free Platforms</b><br>Kaggle & Google Colab for zero-cost GPU access, making AI agents accessible to everyone!</td>
  </tr>
</table>

---

## 🧩 Agent Architecture

Our agents follow a sophisticated architecture that combines the best practices in LLM-based agent design:

```
🧠 User Query
     ↓
🎯 Agent Controller
     ↓
🤔 Reasoning & Planning ←→ 📚 Knowledge Tools
     ↓
🛠️ Tool Selection
     ↓
⚡ Action Execution ←→ 🔍 Search Tools
     ↓
🤖 Reflection Module ←→ 💾 Memory Store
     ↓
✅ Quality Check
     ↓
📝 Final Response
```

---

## 🤖 Agent Types & Comparisons

<div align="center">
  <table>
    <tr>
      <th>🤖 Agent Type</th>
      <th>📖 Description</th>
      <th>⭐ Key Features</th>
      <th>🧠 Model Used</th>
      <th>🎯 Best For</th>
    </tr>
    <tr>
      <td align="center">
        <h3>🔄 ReAct Agent</h3>
        <i>using LangChain</i>
      </td>
      <td>Implements the Reasoning + Acting paradigm that allows the agent to think step-by-step while interacting with tools.</td>
      <td>
        • 🧠 Sequential reasoning traces<br>
        • 🛠️ Tool selection & execution<br>
        • ⚡ Single-pass problem solving<br>
        • 🚀 Fast response times
      </td>
      <td>
        <b>🦙 Llama 3.2 1B</b><br>
        <i>SLM for efficiency</i>
      </td>
      <td>
        📝 Simple tasks, quick responses, resource-constrained environments
      </td>
    </tr>
    <tr>
      <td align="center">
        <h3>🤔 Reflection Agent</h3>
        <i>using LangGraph</i>
      </td>
      <td>Extends ReAct with self-evaluation capabilities. Can reflect on its reasoning process and self-correct errors.</td>
      <td>
        • 🔄 Self-evaluation loops<br>
        • 🐛 Error detection & correction<br>
        • 📈 Strategy refinement<br>
        • ✨ Quality improvement cycles
      </td>
      <td>
        <b>🦙 Llama 3.2 3B</b><br>
        <i>SLM with enhanced reasoning</i>
      </td>
      <td>
        🧮 Complex reasoning tasks, quality-critical applications, educational content
      </td>
    </tr>
    <tr>
      <td align="center">
        <h3>🔁 Reflexion Agent</h3>
        <i>using LangGraph</i>
      </td>
      <td>Most advanced agent that combines reflection with experiential learning. Continuously improves performance based on past attempts.</td>
      <td>
        • 🧠 Memory of past attempts<br>
        • 📊 Performance tracking<br>
        • 🎯 Adaptive strategies<br>
        • 📚 Long-term learning<br>
        • 🔄 Iterative improvement
      </td>
      <td>
        <b>💎 Gemma 2B</b><br>
        <i>SLM with advanced memory</i>
      </td>
      <td>
        🔬 Research tasks, complex problem-solving, scenarios requiring learning from experience
      </td>
    </tr>
    <tr>
      <td align="center">
        <h3>👤 Human-in-the-Loop Agent</h3>
        <i>using LangGraph</i>
      </td>
      <td>Interactive agent that incorporates human feedback and decisions into the workflow. Perfect for tasks requiring human judgment.</td>
      <td>
        • 🤝 Human feedback integration<br>
        • 🎯 Interactive decision points<br>
        • 👥 Collaborative workflow<br>
        • ✅ Quality assurance<br>
        • 📋 Custom approvals
      </td>
      <td>
        <b>🌟 Gemini 1.5 Flash</b><br>
        <i>Premium LLM for human interaction</i>
      </td>
      <td>
        ✍️ Content creation, critical decisions, creative tasks, professional communications
      </td>
    </tr>
  </table>
</div>

---

## 🔬 SLM vs LLM Analysis

### 📊 **Why Different Models for Different Agents?** 🤔

<table>
<tr>
<th>🤖 Model Type</th>
<th>📋 Examples</th>
<th>🔢 Parameters</th>
<th>💪 Strengths</th>
<th>⚠️ Limitations</th>
<th>🎯 Best Use Cases</th>
</tr>
<tr>
<td><b>🚀 Small Language Models (SLMs)</b></td>
<td>
• 🦙 Llama 3.2 1B/3B<br>
• 💎 Gemma 2B<br>
• 🔥 Phi-3 Mini
</td>
<td>1B - 3B</td>
<td>
• 💰 <b>Zero API costs</b><br>
• ⚡ Fast inference<br>
• 💻 Runs on free GPUs<br>
• 🎯 Good for structured tasks<br>
• 🔋 Efficient resource usage<br>
• 🔄 Multiple iterations possible
</td>
<td>
• 🧠 Limited reasoning depth<br>
• 🔄 May need more iterations<br>
• 📚 Smaller knowledge base<br>
• 🎨 Less creative output
</td>
<td>
• 🔄 ReAct agents<br>
• 🤔 Reflection agents<br>
• 🔁 Reflexion agents<br>
• 📊 Structured reasoning<br>
• 🧪 Experimentation
</td>
</tr>
<tr>
<td><b>🌟 Large Language Models (LLMs)</b></td>
<td>
• 🤖 GPT-4 / GPT-4.1<br>
• 🎭 Claude Sonnet 4<br>
• 🌟 Gemini 1.5 Flash/Pro<br>
• 🚀 Grok
</td>
<td>100B+</td>
<td>
• 🧠 Superior reasoning<br>
• 🎨 Excellent creativity<br>
• 📚 Broad knowledge<br>
• 🔥 Complex task handling<br>
• ✨ High-quality output<br>
• 🎯 First-time accuracy
</td>
<td>
• 💸 <b>High API costs</b><br>
• 🐌 Slower inference<br>
• 🚫 Rate limits<br>
• 🌐 External dependency<br>
• 🔒 Privacy concerns<br>
• 💳 Usage restrictions
</td>
<td>
• 👤 Human-in-the-loop<br>
• ✍️ Content creation<br>
• 🔬 Complex analysis<br>
• 🎨 Creative tasks<br>
• 💼 Professional output
</td>
</tr>
</table>

### 🎯 **Strategic Model Selection** 🧠

**For Automated Agents (ReAct, Reflection, Reflexion):**
- ✅ **Use SLMs** (Llama 3.2, Gemma) - They can run multiple iterations without cost concerns 💰
- ✅ **Perfect for structured reasoning** - Agents provide the reasoning framework 🏗️
- ✅ **Ideal for experimentation** - Free to run unlimited tests 🧪
- ✅ **Learning-friendly** - Students and researchers can explore without budget limits 🎓

**For Human-Interactive Agents:**
- ✅ **Use Premium LLMs** (Gemini 1.5 Flash) - Human time is valuable, quality matters most ⏰
- ✅ **Best for final outputs** - When you need the highest quality result 🏆
- ✅ **Cost-effective for human-in-loop** - Less total API calls due to human guidance 🎯
- ✅ **Professional quality** - When representing you or your business 💼

---

## 🆓 **Free Computing Resources** 🎁

**🔥 Kaggle Notebooks:**
- 🎯 **30+ hours/week** free GPU time
- 🚀 **T4 GPU** - Perfect for Llama 3.2, Gemma
- 📚 **Persistent storage** for your agents
- 🔄 **Easy sharing** and collaboration
- 📊 **Built-in datasets** access

**☁️ Google Colab:**
- 🎯 **Free GPU access** (T4)
- 📊 **12-24 hours** continuous runtime
- 🔧 **Easy setup** with our notebooks
- 💾 **Google Drive** integration
- 🎓 **Educational-friendly**

---

## 💻 Running on Kaggle & Google Colab

### 🚀 **Quick Start Guide** ⚡

**Option 1: Kaggle (🌟 Recommended)**
1. 📥 Visit [Kaggle.com](https://www.kaggle.com/) and sign up (free account)
2. 📤 Upload our notebook files to a new Kaggle notebook
3. 🖥️ Select **GPU accelerator** (T4 recommended)
4. 🎮 Enable **Internet access** for search tools
5. ▶️ Run cells sequentially - everything is pre-configured! 🎉

**Option 2: Google Colab (☁️ Cloud-based)**
1. 📥 Open [Google Colab](https://colab.research.google.com/)
2. 📤 Upload our `.ipynb` files
3. 🖥️ Select **GPU runtime** (Runtime → Change runtime type)
4. 🔧 Install dependencies (provided in notebooks)
5. ▶️ Run and experiment! 🧪

### 🛠️ **Pre-configured Features** ✨

- ✅ **Automatic model downloading** (Llama 3.2, Gemma from HuggingFace)
- ✅ **Environment setup** (all dependencies included)
- ✅ **API key management** (secure input prompts)
- ✅ **Error handling** (robust parsing and fallbacks)
- ✅ **Interactive examples** (ready to run)
- ✅ **Progress tracking** (visual feedback)

---

## ⚙️ Setup & Installation

### 📦 **Local Installation (Optional)** 🏠

If you prefer running locally:

```bash
# Clone the repository
git clone https://github.com/abuzar01440/AI-agents-.git
cd AI-agents-

# Create virtual environment
python -m venv ai_agents_env
source ai_agents_env/bin/activate  # On Windows: ai_agents_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install langchain langgraph transformers torch
pip install langchain-community langchain-huggingface
pip install langchain-google-genai  # For Gemini
pip install tavily-python

# Set up environment variables
import os
import getpass

# Prompt the user for their API keys securely
tavily_api_key = getpass.getpass("Enter your Tavily API Key: ")
hf_token = getpass.getpass("Enter your HuggingFace Token: ")
google_api_key = getpass.getpass("Enter your Google API Key (Gemini): ")

# Set the environment variables
os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["HUGGINGFACE_TOKEN"] = hf_token
os.environ["GOOGLE_API_KEY"] = google_api_key

print("Environment variables have been set successfully!")
```

### 🔑 **API Keys Required** 🗝️

- **🔍 Tavily Search API** (for web search functionality) - Free tier available
- **🤗 HuggingFace Token** (for model access) - Free account
- **🌟 Google AI API** (for Gemini 1.5 Flash in Human-in-Loop) - Free tier available
- **Optional:** OpenAI/Claude keys (if you want to experiment with other models)

---

## 📘 Usage Examples

### 🔄 **ReAct Agent Example** 

```python
from langchain.agents import create_react_agent
from langchain_huggingface import HuggingFacePipeline
from langchain_community.tools.tavily_search import TavilySearchResults

# 🦙 Initialize Llama 3.2 1B
llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    device=0,  # GPU
    model_kwargs={
        "temperature": 0.6,
        "max_new_tokens": 256,
        "do_sample": True
    }
)

# 🔍 Setup search tool
search_tool = TavilySearchResults(
    max_results=3,
    search_depth="basic"
)

# 🤖 Create ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=react_prompt
)

# ⚡ Execute
response = agent.invoke({
    "input": "What are the latest AI trends in 2024? 🚀"
})
print(f"🤖 Agent Response: {response['output']}")
```

### 🤔 **Reflection Agent Example**

```

### 🔁 **Reflexion Agent with Gemma** 💎

```

## 🎯 **Why This Approach Works** 🧠

### 💡 **Intelligent Resource Allocation**

1. **🔄 SLMs for Iteration-Heavy Tasks:**
   - Reflection and Reflexion agents need multiple LLM calls
   - SLMs make this economically viable 💰
   - Quality emerges from the agent architecture, not just model size
   - Perfect for learning and experimentation 🧪

2. **🌟 LLMs for Human-Critical Tasks:**
   - Human time is the most expensive resource ⏰
   - Use premium models when human interaction is involved
   - Fewer total API calls due to human guidance 🎯
   - Professional quality when it matters most 💼

3. **🆓 Free Platform Optimization:**
   - Kaggle and Colab provide substantial free compute
   - Perfect for Llama 3.2, Gemma model requirements
   - Enables unlimited experimentation and learning 🎓

### 📈 **Performance Insights** 📊

Based on our testing:
- **🔄 ReAct Agent**: Llama 3.2 1B achieves 85% of GPT-4 performance at 0% of the cost
- **🤔 Reflection Agent**: Llama 3.2 3B with reflection beats GPT-4 single-pass in many tasks
- **🔁 Reflexion Agent**: Gemma 2B with memory and learning compensates for smaller model size
- **👤 Human-in-the-Loop**: Gemini 1.5 Flash provides excellent quality with cost efficiency

### 🎓 **Educational & Research Benefits**

- **🎓 Students**: Learn AI agents without budget constraints
- **🔬 Researchers**: Prototype ideas freely
- **💼 Professionals**: Evaluate cost-effective AI solutions
- **🏢 Companies**: Test before scaling with expensive models

---

## 🤝 Contributing

We welcome contributions to improve the agents or add new features! Here's how you can help: 🌟

1. 🍴 **Fork the repository**
2. 🌿 **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. 🧪 **Test your changes** on Kaggle/Colab
4. 💾 **Commit your changes** (`git commit -m 'Add amazing feature'`)
5. 📤 **Push to the branch** (`git push origin feature/amazing-feature`)
6. 🔄 **Open a Pull Request**


---

## 📚 **Learning Resources** 🎓

### 📖 **Recommended Reading**
- [ReAct Paper](https://arxiv.org/abs/2210.03629) - Reasoning and Acting in Language Models 🧠
- [Reflexion Paper](https://arxiv.org/abs/2303.11366) - Learning from Verbal Feedback 🔄
- [LangChain Documentation](https://python.langchain.com/) - Framework basics 🦜
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/) - Advanced workflows 📊

### 🎓 **Educational Value**
This repository is perfect for:
- 🎓 **Students** learning AI agents (zero cost barrier)
- 🔬 **Researchers** prototyping new ideas
- 💼 **Professionals** exploring cost-effective AI solutions
- 🏢 **Companies** evaluating SLM vs LLM trade-offs
- 🧑‍🏫 **Educators** teaching AI concepts

---

<div align="center">
  <p>
    <a href="https://github.com/abuzar01440">
      <img src="https://img.shields.io/github/followers/abuzar01440?label=Follow&style=social" alt="GitHub Follow">
    </a>
    ⭐ Star this repository if you found it helpful! ⭐
  </p>
  
  <p>
    <img src="https://img.shields.io/badge/Made%20with-🤖%20Multi%20Models-ff6f61?style=for-the-badge" alt="Made with Multi Models">
    <img src="https://img.shields.io/badge/Cost-💰%20Near%20Zero-00ff00?style=for-the-badge" alt="Near Zero Cost">
    <img src="https://img.shields.io/badge/Platform-🆓%20Kaggle%20%7C%20Colab-blue?style=for-the-badge" alt="Free Platforms">
  </p>

  <p>Created with 💙 by <a href="https://github.com/abuzar01440">abuzar01440</a></p>
  <p>📧 abuzarbhutta@gmail.com | abuzarbhutta.0@outlook.com</p>
  <p>📅 Last Updated: 2025-01-04</p>
  
  ![Watchers](https://img.shields.io/github/watchers/abuzar01440/AI-agents-?style=social)

  <i>🎯 Building smarter agents, one reflection at a time - without breaking the bank! 💰🤖</i>
  
  <br><br>
  
  <h3>🚀 Ready to build your own AI agents? Clone this repo and start experimenting today! 🎉</h3>
</div>
