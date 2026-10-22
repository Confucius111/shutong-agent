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

            # ⭐ 核心修改：按导师要求升级的“大师级”系统提示词
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
4. 如果用户问到原著中根本没有的人，温和指出即可，绝不瞎编。

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
                    temperature=0.5,  # ⭐ 核心修改：调低温度，减少胡编乱造，让赏析更严谨深刻
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
