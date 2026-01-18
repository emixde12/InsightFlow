import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationBufferWindowMemory
from langchain import hub
from core.rag import RAGEngine

class AgentManager:
    """
    InsightFlow Agent 管理器
    实现功能：ReAct 范式、工具路由、记忆管理
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        # 初始化 LLM (支持 DeepSeek-V3)
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url=base_url,
            temperature=0.1 # 降低随机性，提升指令遵循能力
        )
        # 初始化记忆模块 (Sliding Window)
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5, # 只保留最近 5 轮对话，优化 Token 成本
            return_messages=True
        )
        
    def setup_agent(self, rag_engine: RAGEngine):
        """
        组装 Agent：注入 RAG 工具与联网搜索工具
        """
        # 1. 定义工具集 (Tools)
        # 工具 A: 联网搜索 (处理实时信息)
        search_tool = TavilySearchResults(max_results=3)
        
        # 工具 B: 本地知识库 (处理私有数据)
        retriever_tool = create_retriever_tool(
            rag_engine.get_retriever(),
            "local_knowledge_base",
            "用于搜索上传文档中的具体细节、数据和条款。当用户问及文档内容时必须使用此工具。"
        )
        
        tools = [search_tool, retriever_tool]
        
        # 2. 加载 ReAct Prompt (HWCHASE17/REACT)
        prompt = hub.pull("hwchase17/react-chat")
        
        # 3. 构建 Agent
        agent = create_react_agent(self.llm, tools, prompt)
        
        # 4. 创建执行器
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True, # 开启日志，可视化思考过程 (CoT)
            handle_parsing_errors=True
        )
        
        return agent_executor