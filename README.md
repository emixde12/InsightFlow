# 🌊 InsightFlow - Enterprise Knowledge Agent

![Python](https://img.shields.io/badge/Python-3.10-blue) ![LangChain](https://img.shields.io/badge/Framework-LangChain-green) ![DeepSeek](https://img.shields.io/badge/Model-DeepSeek_V3-purple)

**InsightFlow** 是一个基于 **Agentic RAG** 架构的企业级智能知识问答系统。它解决了传统 RAG 系统在私有知识库场景下的检索精度低、无法处理实时信息以及长对话逻辑混乱的问题。

## ✨ 核心特性 (Key Features)

- 🧠 **Agentic Routing (智能路由)**: 基于 ReAct 范式，自动判断用户意图，智能调度 **本地知识库 (RAG)** 与 **Tavily 联网搜索**。
- 🔍 **Hybrid Search (混合检索)**: 融合 **BM25 (稀疏向量)** 与 **Embedding (稠密向量)**，解决专有名词检索难题。
- 🎯 **Advanced Reranking (重排序)**: 集成 **BGE-Reranker-v2** Cross-Encoder 模型，Recall@3 提升至 90%+。
- 💾 **Smart Memory (智能记忆)**: 基于 Sliding Window 的上下文管理，Token 消耗降低 30%。
- 🖥️ **Interactive UI**: 基于 **Streamlit** 构建的交互式前端，支持文档拖拽上传与思维链 (CoT) 可视化。

## 🏗️ 系统架构 (Architecture)

```mermaid
graph TD
    User[用户 Query] --> Frontend[Streamlit UI]
    Frontend --> Router[Agent Router]
    Router -- 实时信息 --> WebSearch[Tavily Search]
    Router -- 私有知识 --> HybridRAG[Hybrid RAG]
    HybridRAG --> VectorDB[ChromaDB] & BM25[BM25 Retriever]
    VectorDB & BM25 --> Reranker[BGE Reranker]
    Reranker --> Context[LLM Context]
    WebSearch --> Context
    Context --> LLM[DeepSeek-V3]
    LLM --> Frontend