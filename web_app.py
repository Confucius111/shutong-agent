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
DATA_DIR = "."          # 知识库所在目录（GitHub 仓库根目录）
TXT_GLOB = "*.txt"      # 自动扫描的扩展名模式

# ==========================================
# 2. 初始化大模型（Secrets 安全读取）
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
# 3. 扫描知识库目录
# ==========================================
def scan_txt_files():
    """返回当前目录（或 DATA_DIR）下所有 .txt 文件路径，按文件名排序。"""
    pattern = os.path.join(DATA_DIR, TXT_GLOB)
    return sorted(glob.glob(pattern))

txt_files = scan_txt_files()

# ==========================================
# 4. 侧边栏：控制台
# ==========================================
with st.sidebar:
    st.header("⚙️ 书童的控制台")

    if txt_files:
        st.success("✅ 已成功加载云端知识库")
        st.write("当前知识库包含：")
        for file in txt_files:
            # 只显示文件名，避免显示完整路径
            st.write(f"- 📄 {os.path.basename(file)}")
    else:
        st.warning("⚠️ 未在仓库中找到任何 .txt 文件")

    st.divider()

    if st.button("🔄 重建知识库（重新切片）", use_container_width=True):
        # 手动触发重建：清空缓存，强制下次 rerun 重新读文件
        st.session_state.pop("chunks", None)
        st.session_state.pop("kb_files", None)
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
# 5. 自动读取所有 TXT 文件并构建知识库
# ==========================================
if "chunks" not in st.session_state:
    st.session_state.chunks = []
    all_raw_text = ""

    if not txt_files:
        st.error("❌ 未找到任何 TXT 知识库文件，请检查 GitHub 仓库是否已上传文本。")
    else:
        with st.spinner("📚 正在自动装载知识库文件..."):
            for file_path in txt_files:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        # 加文件名标记 + 换行，防止不同文件内容粘连
                        all_raw_text += (
                            f"\n\n【文件：{os.path.basename(file_path)}】\n"
                            + f.read()
                        )
                except Exception as e:
                    st.warning(f"读取 {file_path} 失败：{e}")

            if all_raw_text.strip():
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600, chunk_overlap=100
                )
                st.session_state.chunks = text_splitter.split_text(all_raw_text)
                st.session_state.kb_files = [os.path.basename(f) for f in txt_files]
            else:
                st.error("❌ 读取到的内容为空，请检查 TXT 文件是否为空或编码异常。")

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
# 8. 检索函数（关键词加权 + 长词加权）
# ==========================================
STOP_WORDS = set(
    "，。？！；：、“”‘’（）《》【】…—的了吗呢吧啊是在和有就都也不人一个上很到说要去你会着没有看好自己这"
    "什么样 怎么 什么 帮我 查一下 书童 请问 整本书 里".split()
)


def retrieve_context(query: str, chunks, top_k: int = 5) -> str:
    if not chunks:
        return "知识库为空，请先在 GitHub 仓库上传 TXT 文献。"

    tokens, buf = [], ""
    for ch in query:
        if "\u4e00" <= ch <= "\u9fff":
            if buf:
                tokens.append(buf); buf = ""
            if ch not in STOP_WORDS:
                tokens.append(ch)
        elif ch.isalnum():
            buf += ch
        else:
            if buf:
                tokens.append(buf); buf = ""
    if buf:
        tokens.append(buf)

    tokens = [t for t in tokens if t and t not in STOP_WORDS]
    if not tokens:
        tokens = [query]

    scored = []
    for i, chunk in enumerate(chunks):
        score = sum(chunk.count(t) * (len(t) + 1) for t in tokens)
        scored.append((score, i, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = [s for s in scored[:top_k] if s[0] > 0] or scored[:3]
    return "\n\n...\n\n".join(c[2] for c in top)


# ==========================================
# 9. 核心对话（支持连续追问）
# ==========================================
if prompt := st.chat_input("向书童提问，例如：林黛玉是一个什么样性格的人？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("🧠 书童正在全书中进行深度语义检索..."):
            context = retrieve_context(prompt, st.session_state.chunks)

            system_prompt = f"""
你现在是名为"书童"的AI《红楼梦》导读助手，面向读者进行经典文学导读。
请严格依据【参考资料】回答读者提问。

【任务要求】：
1. 如果用户询问人物性格、情节分析等，请综合提炼材料中的信息，进行合理总结与推理。
2. 如果材料提到某人但未直接写"性格"，请从事迹推断，不要回答"未提及"。
3. 若资料含古文，请附带白话文翻译。
4. 如果【参考资料】与问题无关，请礼貌说明知识库中暂无相关内容，并建议补充对应章节文本。

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
                    temperature=0.7,
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
