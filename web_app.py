import os
import sys

# 强制系统使用 UTF-8 编码，解决中文乱码和 ascii 报错
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

import json
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# ==========================================
# 1. 网页基本设置
# ==========================================
st.set_page_config(page_title="书童 - 经典文学导读助手", page_icon="📚", layout="wide")
st.title("📚 书童 - 经典文学导读助手")
st.caption("您的专属数智阅读伴侣 | 云端安全版")

# ==========================================
# 2. 初始化大模型 (使用 Secrets 安全读取)
# ==========================================
try:
    # 从云端 Streamlit Secrets 中读取密钥
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    # 如果在本地运行且没配 secrets，做个防崩溃兜底
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "未配置密钥")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/",
    timeout=60.0 
)

# ==========================================
# 3. 侧边栏：知识库管理
# ==========================================
HISTORY_FILE = "chat_history.json"

with st.sidebar:
    st.header("📚 书童的控制台")
    
    uploaded_file = st.file_uploader("上传古籍资料（支持 TXT）", type=["txt"])
    
    if uploaded_file is not None:
        st.success(f"成功接收文件：{uploaded_file.name}")
        
        with st.spinner("正在解析长篇文献..."):
            try:
                raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
                st.session_state.chunks = text_splitter.split_text(raw_text)
                st.info(f"已自动切分为 {len(st.session_state.chunks)} 个知识块！")
            except Exception as e:
                st.error(f"解析失败: {e}")
    
    st.divider()
    if st.button("🗑️ 清空当前对话", use_container_width=True):
        st.session_state.messages = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.rerun()

# ==========================================
# 4. 初始化本地记忆与知识库
# ==========================================
if "chunks" not in st.session_state:
    st.session_state.chunks = [] 

if "messages" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                st.session_state.messages = json.load(f)
        except Exception:
            st.session_state.messages = []
    else:
        st.session_state.messages = []

# ==========================================
# 5. 展示历史聊天记录
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 6. 核心逻辑：极速检索与连续追问
# ==========================================
if prompt := st.chat_input("向书童提问，例如：林黛玉是一个什么样性格的人？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("🧠 书童正在全书中进行深度语义检索..."):
            context = "知识库为空，请先在左侧上传文献资料。"
            
            if st.session_state.chunks:
                stop_words = ["，", "。", "？", "的", "了", "是", "在", "和", "有", "个", "什么样", "怎么", "什么", "帮我", "查一下", "书童", "请问", "整本书", "里"]
                query_words = [char for char in prompt if char not in stop_words]
                
                chunk_scores = []
                for i, chunk in enumerate(st.session_state.chunks):
                    score = sum(chunk.count(word) for word in query_words)
                    chunk_scores.append((score, i, chunk))
                
                chunk_scores.sort(key=lambda x: x[0], reverse=True)
                top_chunks = chunk_scores[:5]
                
                valid_chunks = [c[2] for c in top_chunks if c[0] > 0]
                if valid_chunks:
                    context = "\n\n...\n\n".join(valid_chunks)
                else:
                    context = "\n\n".join(st.session_state.chunks[:3])

            prompt_with_context = f"""
            你现在是名为“书童”的AI《红楼梦》导读助手。
            请根据以下【参考资料】回答读者的提问。
            
            【任务要求】：
            1. 如果用户问的是人物性格、情节分析等，请**综合提炼**材料中的信息，进行合理的总结和推理。
            2. 如果材料里提到了这个人，但没有直接写“性格”二字，请根据事迹去推断，绝对不要说“未提及”。
            3. 如果资料中有古文，请附带白话文翻译。
            
            【参考资料】：
            {context}
            
            【读者的提问】：
            {prompt}
            """
            
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat", 
                    messages=[{"role": "user", "content": prompt_with_context}]
                )
                final_answer = response.choices[0].message.content
            except Exception as e:
                final_answer = f"书童遇到了一点网络问题：{e}"

        message_placeholder.markdown(final_answer)
        
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=4)
