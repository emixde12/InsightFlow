import streamlit as st
import os
import tempfile
from core.rag import RAGEngine
from core.agent import AgentManager

# --- Page Config ---
st.set_page_config(
    page_title="InsightFlow - Enterprise Knowledge Agent",
    page_icon="🤖",
    layout="wide"
)

# --- Header ---
st.title("🤖 InsightFlow 企业级智能问答系统")
st.markdown("Based on **DeepSeek-V3** | Supported by **RAG + Agentic Routing**")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # API Key Management
    deepseek_key = st.text_input("DeepSeek API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")
    
    if deepseek_key:
        os.environ["OPENAI_API_KEY"] = deepseek_key # 兼容 OpenAI SDK
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    st.divider()
    
    # Document Upload
    st.subheader("📄 知识库上传")
    uploaded_file = st.file_uploader("上传 PDF/Markdown 文档", type=["pdf", "md"])
    
    # Initialize Logic
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = RAGEngine()
        
    if uploaded_file and st.button("🚀 Process Document"):
        with st.spinner("正在执行 ETL 清洗与向量化..."):
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # RAG Ingestion
            result = st.session_state.rag_engine.ingest_document(tmp_path)
            st.success(result)
            st.session_state.doc_processed = True
            os.remove(tmp_path)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Agent
if "agent" not in st.session_state and "doc_processed" in st.session_state:
    if deepseek_key and tavily_key:
        manager = AgentManager(api_key=deepseek_key)
        st.session_state.agent = manager.setup_agent(st.session_state.rag_engine)
    else:
        st.warning("请输入 API Key 以启动 Agent")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask me anything about your data..."):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Agent Reasoning & Response
    if "agent" in st.session_state:
        with st.chat_message("assistant"):
            st_callback = st.container() # 可预留用于流式输出
            with st.spinner("InsightFlow 正在思考 (ReAct)..."):
                try:
                    # Invoke Agent
                    response = st.session_state.agent.invoke(
                        {"input": prompt, "chat_history": st.session_state.messages}
                    )
                    output = response["output"]
                    st.markdown(output)
                    st.session_state.messages.append({"role": "assistant", "content": output})
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("请先上传文档并配置 API Key。")