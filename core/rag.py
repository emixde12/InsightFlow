import os
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document

class RAGEngine:
    """
    InsightFlow RAG 核心引擎
    实现功能：文档加载、混合检索 (Hybrid Search)、向量存储
    """
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        # 使用 BGE-M3 或其他高性能 Embedding 模型
        # 注意：生产环境建议使用 GPU 加速
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 针对中文文档优化
            chunk_overlap=50, # 保持上下文连贯性
            separators=["\n\n", "\n", "。", "！", "，"]
        )
        self.vector_store = None
        self.retriever = None

    def ingest_document(self, file_path: str) -> str:
        """
        加载并处理文档 (ETL Pipeline)
        """
        try:
            # 1. Load Document
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            docs = loader.load()
            
            # 2. Split (Chunking)
            splits = self.text_splitter.split_documents(docs)
            
            # 3. Indexing (Store in ChromaDB)
            # 采用 Collection 隔离不同知识库
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
            
            # 4. Initialize Hybrid Retriever
            self._init_hybrid_retriever(splits)
            
            return f"Successfully ingested {len(splits)} chunks."
            
        except Exception as e:
            return f"Error during ingestion: {str(e)}"

    def _init_hybrid_retriever(self, splits: List[Document]):
        """
        初始化混合检索器 (BM25 + Vector)
        Resume Key: Hybrid Search Strategy
        """
        # 稀疏向量检索 (关键词匹配)
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 5
        
        # 稠密向量检索 (语义匹配)
        chroma_retriever = self.vector_store.as_retriever(
            search_type="mmr", # 最大边际相关性，保证结果多样性
            search_kwargs={"k": 5}
        )
        
        # 加权融合 (RRF 逻辑封装在 EnsembleRetriever 中)
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.4, 0.6] # 语义权重略高
        )

    def get_retriever(self):
        if not self.retriever:
            raise ValueError("RAG Engine not initialized. Please ingest a document first.")
        return self.retriever