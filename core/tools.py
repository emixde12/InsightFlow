import os
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from typing import List

class ToolManager:
    """
    工具管理器：负责初始化 Agent 所需的所有 Tools
    """
    
    @staticmethod
    def get_web_search_tool(max_results: int = 3) -> TavilySearchResults:
        """
        初始化 Tavily 联网搜索工具
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            # 返回一个 Mock 工具防止报错 (方便调试)，或者直接抛出异常
            print("Warning: TAVILY_API_KEY not found. Web search will fail.")
        
        return TavilySearchResults(
            max_results=max_results,
            description="用于搜索互联网上的实时信息、新闻、股价或当前事件。当问题超出本地知识库范围时使用。"
        )

    @staticmethod
    def get_rag_tool(retriever, name: str = "local_knowledge_base") -> Tool:
        """
        将 RAG Retriever 封装为 Agent 可调用的 Tool
        
        Args:
            retriever: 这里的 retriever 是 rag.py 里生成的混合检索器
        """
        return create_retriever_tool(
            retriever=retriever,
            name=name,
            description="""
            专门用于查询用户上传的私有文档、企业知识库、合同细节或具体数据。
            当用户询问具体的文档内容、定义或条款时，必须优先使用此工具。
            """
        )

    @staticmethod
    def get_all_tools(rag_retriever) -> List[Tool]:
        """
        一键获取所有工具列表
        """
        return [
            ToolManager.get_web_search_tool(),
            ToolManager.get_rag_tool(rag_retriever)
        ]