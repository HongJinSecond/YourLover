"""
Descripttion: RAG知识库上传工具，基于 Streamlit 的简单前端页面
version: 1.0.0
Author: HongJin
email: 756547077@qq.com
"""

import streamlit as st

from agent.models.embeddings import (
    EMBEDDING_PROVIDER_NONE,
    get_embedding_provider,
)
from agent.tools.RAGTool import RAG
from utils.config_manager import config as settings


@st.cache_resource
def get_rag():
    """缓存 RAG 实例，避免每次交互都重新初始化（重建 BM25 索引耗时）"""
    return RAG(settings)


def main():
    st.set_page_config(page_title="RAG 知识库管理", page_icon="📚", layout="centered")
    st.title("📚 RAG 知识库上传")

    provider = get_embedding_provider(settings)
    if provider == EMBEDDING_PROVIDER_NONE:
        st.error("没有设置Embedding，RAG功能已被禁用。请在应用设置的「Embedding 服务」中选择本地 Ollama 或 OpenAI 兼容接口。")
        st.stop()

    rag = get_rag()

    # 侧边栏：知识库统计
    with st.sidebar:
        st.header("📊 知识库统计")
        doc_count = len(rag.bm25_retriever.documents)
        st.metric("文档段数", doc_count)
        if provider == "openai":
            st.caption(f"向量化模型: `{settings.EMBEDDING_OPENAI_MODEL_ID}`（OpenAI 兼容接口）")
            st.caption(f"Embedding API: `{settings.EMBEDDING_OPENAI_BASE_URL}`")
        else:
            st.caption(f"向量化模型: `{settings.EMBEDDING_MODEL_NAME}`（本地 Ollama）")
            st.caption(
                f"Embedding 地址: `{settings.EMBEDDING_API_BASE or 'http://localhost:11434'}`"
            )
        st.caption(f"向量库路径: `{settings.CHROMA_PATH}`")

    st.caption("支持文件上传或直接粘贴文本，写入后会同时更新向量库与 BM25 索引")

    tab_file, tab_text = st.tabs(["📁 上传文件", "✍️ 粘贴文本"])

    # ---------- 方式1：文件上传 ----------
    with tab_file:
        uploaded_files = st.file_uploader(
            "选择文件上传到知识库",
            type=["txt", "md"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.info(f"已选择 {len(uploaded_files)} 个文件")
            if st.button("开始上传文件", type="primary"):
                for f in uploaded_files:
                    with st.spinner(f"处理中: {f.name}"):
                        content = f.read().decode("utf-8", errors="ignore")
                        result = rag.upload_by_str(content, f.name)
                    st.success(f"{f.name}: {result}")

    # ---------- 方式2：粘贴文本 ----------
    with tab_text:
        filename = st.text_input("文档名称（作为来源标记）", value="manual_input.md")
        text = st.text_area("粘贴文本内容", height=300)
        if st.button("上传文本", type="primary"):
            if not text.strip():
                st.warning("文本内容不能为空")
            else:
                with st.spinner("处理中..."):
                    result = rag.upload_by_str(text, filename)
                st.success(result)


if __name__ == "__main__":
    main()
