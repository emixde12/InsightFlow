import os
from typing import Optional
from langchain_openai import ChatOpenAI

class LLMFactory:
    """
    模型工厂类：统一管理 LLM 的初始化与配置
    支持 DeepSeek (通过 OpenAI 协议兼容) 与原生 OpenAI 模型
    """
    
    @staticmethod
    def get_llm(
        model_name: str = "deepseek-chat", 
        temperature: float = 0.1,
        streaming: bool = True
    ) -> ChatOpenAI:
        """
        获取配置好的 LLM 实例
        
        Args:
            model_name: 模型名称 (默认 deepseek-chat)
            temperature: 随机度 (0-1，RAG 场景建议低一点)
            streaming: 是否开启流式输出
        
        Returns:
            ChatOpenAI 实例
        """
        
        # 优先从环境变量读取 Key，如果没有则尝试从 streamlit secrets 读取 (此处简化逻辑)
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # DeepSeek 的 Base URL
        base_url = "https://api.deepseek.com" if "deepseek" in model_name else None
        
        if not api_key:
            # 这里的 Raise Error 是为了在日志中快速定位问题，体现工程严谨性
            raise ValueError("API Key not found. Please check .env file.")

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            streaming=streaming,
            # 增加 timeout 防止网络卡死
            timeout=30,
            max_retries=2
        )
        
        return llm

    @staticmethod
    def get_cheap_llm() -> ChatOpenAI:
        """
        获取低成本模型 (用于简单任务，如总结摘要)
        """
        return LLMFactory.get_llm(model_name="deepseek-chat", temperature=0.0)