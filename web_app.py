import os
import sys
import json
import glob

# 强制 UTF-8（云端 stdin 可能为 None，加兜底）
try:
    if sys.stdin is not None:
        sys.stdin.reconfigure(encoding="utf-8")
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

# ==========================================
# 1. 网页基本设置
# ==========================================
st.set_page_config(page_title="书童 - 经典文学导读助手", page_icon="📚", layout="wide")
st.title("📚 书童 - 经典文学导读助手")
st.caption("您的专属数智阅读伴侣 | 云端安全版 | 多源知识库自动装载")

HISTORY_FILE = "chat_history.json"
DATA_DIR = "shutong_data" 
TXT_GLOB = "*.txt"      

# ==========================================
# 2. 初始化大模型
# ==========================================
try:
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

if not DEEPSEEK_API_KEY:
    st.error("⚠️ 未检测到 DEEPSEEK_API_KEY，请在 Streamlit Secrets 或 .env 中配置。")
    st.stop()

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/",
    timeout=60.0,
)

# ==========================================
# 3. 扫描目录
# ==========================================
def scan_txt_files():
    pattern = os.path.join(DATA_DIR, TXT_GLOB)
    return sorted(glob.glob(pattern))

txt_files = scan_txt_files()

# ==========================================
# 4. 侧边栏
# ==========================================
with st.sidebar:
    st.header("⚙️ 书童的控制台")

    if txt_files:
        st.success("✅ 已成功加载云端知识库")
        st.write("当前知识库包含：")
        for file in txt_files:
            st.write(f"- 📄 {os.path.basename(file)}")
    else:
        st.warning(f"⚠️ 未在 {DATA_DIR} 文件夹中找到任何 .txt 文件")

    st.divider()

    if st.button("🔄 重建知识库", use_container_width=True):
        # 强制清除所有缓存
        st.session_state.clear()
        st.rerun()

    if st.button("🗑️ 清空当前对话", use_container_width=True):
        st.session_state.messages = []
        try:
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
        except Exception:
            pass
        st.rerun()

# ==========================================
# 5. 自动读取并构建【双轨知识库】
# ==========================================
if "chunks_original" not in st.session_state or "chunks_modern" not in st.session_state:
    st.session_state.chunks_original = []  # 文言文
    st.session_state.chunks_modern = []    # 白话文
    st.session_state.chunks_relation = []  # 关系图
    st.session_state.chunks = []           # 总库

    if not txt_files:
        st.error(f"❌ 未找到任何 TXT 知识库文件，请检查 {DATA_DIR} 文件夹！")
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        
        with st.spinner("📚 正在自动分类装载知识库..."):
            for file_path in txt_files:
                file_name = os.path.basename(file_path)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if not content.strip(): continue
                        
                        chunks = text_splitter.split_text(content)
                        
                        # 严格分类
                        if "人物关系" in file_name:
                            st.session_state.chunks_relation.extend(chunks)
                        elif "白话" in file_name:
                            st.session_state.chunks_modern.extend(chunks)
                        else:
                            st.session_state.chunks_original.extend(chunks)
                            
                except Exception as e:
                    st.warning(f"读取 {file_name} 失败：{e}")
            
            # 合并供备用（截取前 1500 块防卡顿）
            st.session_state.chunks = st.session_state.chunks_relation + st.session_state.chunks_modern[:1500] + st.session_state.chunks_original[:1500]
            if not st.session_state.chunks:
                st.error("❌ 文件读取后内容为空，请检查 TXT 编码。")

# ==========================================
# 6. 初始化聊天记忆
# ==========================================
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
# 7. 展示历史聊天记录
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 8. 检索函数（双字滑窗 + 权重分配）
# ==========================================
def retrieve_context(query: str, top_k: int = 4) -> str:
    if not st.session_state.chunks:
        return "知识库为空。"

    tokens = []
    for i in range(len(query) - 1):
        word = query[i:i+2]
        if all("\u4e00" <= ch <= "\u9fff" for ch in word):
            tokens.append(word)
            
    if not tokens: return "\n\n...\n\n".join(st.session_state.chunks[:3])

    def get_top_chunks(chunk_list, weight=1):
        scored = []
        for chunk in chunk_list:
            score = sum(weight for t in tokens if t in chunk)
            if score > 0: scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:top_k]]

    # 白话文权重 2，关系库权重 2，文言文权重 1（确保优先理解白话文）
    modern_results = get_top_chunks(st.session_state.chunks_modern, weight=2)
    relation_results = get_top_chunks(st.session_state.chunks_relation, weight=2)
    original_results = get_top_chunks(st.session_state.chunks_original, weight=1)

    combined = list(dict.fromkeys(modern_results + relation_results + original_results))
    
    if not combined:
        return "\n\n...\n\n".join(st.session_state.chunks[:3])
        
    return "\n\n...\n\n".join(combined[:top_k])

# ==========================================
# 9. 核心对话（深度优化版：链式思考 + 拒绝 AI 味）
# ==========================================
if prompt := st.chat_input("向书童提问，例如：结合黛玉葬花赏析她的性格？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("🧠 书童正在全书中进行深度语义检索与链式思考..."):
            # 使用新版检索
            context = retrieve_context(prompt)

            # 按导师要求升级的“大师级”系统提示词
            system_prompt = f"""
你现在是学识渊博、温文尔雅的《红楼梦》文学导师，名叫“书童”。
你的任务是为读者提供有深度、有情感、通俗易懂的经典文学导读。

【🚨 内部思考流程（请在后台默默执行，不要输出这个过程）】：
1. 【古文翻译】：仔细阅读【参考资料】。如果里面有文言文或诗词，请先在脑海中将其精准翻译成现代白话文，理解其真实含义。
2. 【深度赏析】：结合翻译后的意思，以及你的红学素养，深入剖析人物的情感、性格、以及情节背后的隐喻。
3. 【流畅生成】：用生动的现代汉语组织最终答案。

【回答硬性要求】：
1. **绝对禁止**使用“根据参考资料显示”、“我无法直接回答”等生硬的机器废话，直接像一位老师一样开口作答。
2. 如果引用了古文或判词，必须附带通俗易懂的白话文翻译。
3. 如果用户询问情感、隐喻等深度问题，不要只复述原文，要给出你的深度评价和文学赏析。
4. 如果用户问到原著中根本没有的人（例如“孔禾新”），温和指出即可，绝不瞎编。

【参考资料】：
{context}
"""
            history_for_model = st.session_state.messages[-10:]
            messages_for_api = [{"role": "system", "content": system_prompt}]
            messages_for_api += [
                {"role": m["role"], "content": m["content"]}
                for m in history_for_model
            ]

            final_answer = ""
            try:
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages_for_api,
                    temperature=0.5,  # 调低温度，减少胡编乱造，让赏析更严谨深刻
                )
                final_answer = resp.choices[0].message.content or "（未获得回复）"
            except Exception as e:
                final_answer = f"书童遇到了一点网络问题：{e}"

        placeholder.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=4)
    except Exception:
        pass
