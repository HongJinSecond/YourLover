from datetime import datetime

"""
Descripttion: RAG查询工具，用于检索和查询信息
version: 1.0.0
Author: HongJin
email: 756547077@qq.com
"""
import hashlib
import os

import jieba
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from agent.models.embeddings import (
    EMBEDDING_PROVIDER_NONE,
    create_embeddings,
    get_embedding_provider,
)
from agent.tools.BaseTool import BaseTool
from utils.config_manager import ConfigManager
from utils.config_manager import config as app_config

RAG_DISABLED_MESSAGE = "没有设置Embedding，RAG功能已被禁用"


class RAGTool(BaseTool):
    """
    RAG查询工具，用于检索和查询信息, 包含额外的数据库内容，用于补充模型中不存在的信息或过时的信息，以解决模型幻觉问题。

    EMBEDDING_PROVIDER=none 时知识库整体禁用：所有调用返回固定提示，
    不初始化 Chroma，也不发起任何 embedding 请求。
    """

    name = "rag"
    description = """
    RAG查询工具，用于检索和查询信息, 包含额外的数据库内容，用于补充模型中不存在的信息或过时的信息，以解决模型幻觉问题。

    Search the internal knowledge base to retrieve relevant documents, policies, product specifications, or historical data. 

    USE this tool when:
    - The user asks about company-specific information, internal processes, or proprietary data
    - The query requires up-to-date facts, recent events, or real-time data beyond your training cutoff
    - You need to cite specific sources, document IDs, or exact figures
    - You are uncertain about domain-specific details and need to verify before answering

    DO NOT use this tool for:
    - General knowledge, common sense, or widely known public facts
    - Pure reasoning, math, code generation, or language translation tasks
    - Casual conversation or chitchat
    - Questions already answered in the current conversation context
    
    """
    param_descriptions = {
        "query": "查询关键词，必填，用于检索信息",
    }

    def __init__(self):
        # embedding 禁用时跳过 Chroma/BM25 初始化，工具调用统一返回禁用提示。
        self.enabled = get_embedding_provider(app_config) != EMBEDDING_PROVIDER_NONE
        self.RAG = RAG(app_config) if self.enabled else None

    def _tool_call(self, query: str) -> str:
        """调用RAG工具，返回查询结果"""
        if not self.enabled:
            return f"[禁用] {RAG_DISABLED_MESSAGE}"
        return self.RAG.query(query, n_results=self.RAG.config.RAG_N_RESULTS)


class RAG:
    """
    RAG实体类。
    """

    def __init__(self, config: ConfigManager):
        self.config = config
        # embedding 提供方由 EMBEDDING_PROVIDER 决定（ollama/openai）；
        # none 时 RAGTool 不会构造本类。
        self.chroma_db = Chroma(
            collection_name="RAG_DB",
            embedding_function=create_embeddings(config),
            persist_directory=config.CHROMA_PATH,
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,  # 分割后的文本段最大长度
            chunk_overlap=config.CHUNK_OVERLAP,  # 连续文本段之间的字符串重叠数量
            separators=config.SEPARATORS,  # 自然段落划分的符号 分隔符
            length_function=len,
        )  # 文本分割器

        # 新增：BM25 检索器，启动时从 Chroma 重建索引
        self.bm25_retriever = BM25Retriever()
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self):
        """从 ChromaDB 拉取全部文档，构建 BM25 索引"""
        all_data = self.chroma_db.get()  # 返回 {'ids','documents','metadatas'}
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_data["documents"], all_data["metadatas"])
        ]
        self.bm25_retriever.build_from_documents(documents)

    def upload_by_str(self, data, filename):
        """将传入的字符串向量化，并存入到数据库中"""
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已经存储在知识库中"

        if len(data) > self.config.MAX_SPLIT_CHAR_NUMBER:
            knowledge_chunks = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": self.config.RAG_FILE_UPLOADER,
        }

        # 加载到向量数据库中
        self.chroma_db.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        # 增量更新 BM25 索引（基于已缓存文档 + 新文档重建，无需从 Chroma 全量拉取）
        new_docs = [Document(page_content=chunk, metadata=metadata) for chunk in knowledge_chunks]
        self.bm25_retriever.add_documents(new_docs)

        # 记录数据到md5记录中
        save_md5(md5_hex)

        return "[成功]内容已经上传到了数据库"

    def query(self, query: str, n_results: int = 10) -> str:
        """查询数据库，返回查询结果（向量 + BM25 多路召回，RRF 融合）
        Args:
            query (str): 查询关键词
            n_results (int, optional): 最终返回的结果数量，默认10
        Returns:
            str: 格式化后的查询结果字符串，供LLM/Agent直接使用
        """
        # 多路召回时各路多取一些候选，便于融合后仍有足够 top-k
        k_candidates = n_results * 2

        # 向量召回需要 embedding 服务：EMBEDDING_PROVIDER=none 时 RAGTool
        # 已整体禁用本工具；openai 提供方走远程 embedding 接口。
        use_vector = get_embedding_provider(self.config) != EMBEDDING_PROVIDER_NONE
        vec_results = (
            []
            if not use_vector
            else self.chroma_db.similarity_search_with_relevance_scores(
                query=query,
                k=k_candidates,
            )
        )

        # 2. BM25 检索（稀疏检索，擅长关键词精确匹配）
        bm25_results = self.bm25_retriever.search(query, k=k_candidates)

        # 3. 空结果保护
        if not vec_results and not bm25_results:
            return "[无结果] 知识库中未找到与当前查询相关的内容。"

        # 4. RRF 融合两路结果
        fused = self._rrf_fuse(vec_results, bm25_results, k=60)

        # 5. 取 top-n_results
        top_results = fused[:n_results]

        # 6. 格式化输出（包含融合分数 + 来源元数据 + 文本内容）
        formatted_chunks = []
        for idx, (doc, rrf_score) in enumerate(top_results, 1):
            source = doc.metadata.get("source", "未知来源")
            create_time = doc.metadata.get("create_time", "未知时间")
            chunk_text = (
                f"[{idx}] (融合相关度: {rrf_score:.4f} | 来源: {source} | 时间: {create_time})\n"
                f"{doc.page_content.strip()}"
            )
            formatted_chunks.append(chunk_text)

        method = "向量+BM25多路召回" if use_vector else "BM25 关键词召回"
        header = f"[检索成功] {method}，共返回 {len(formatted_chunks)} 条结果:\n{'=' * 50}\n"
        return header + "\n\n".join(formatted_chunks)

    @staticmethod
    def _rrf_fuse(vec_results, bm25_results, k=60):
        """
        Reciprocal Rank Fusion 融合两路检索结果
        只看每路的排名，不看绝对分数，天然解决向量与 BM25 分数量纲不一致的问题。
        Args:
            vec_results: [(Document, score), ...] 已按相关度降序
            bm25_results: [(Document, score), ...] 已按相关度降序
            k: RRF 常数，默认 60
        Returns:
            [(Document, rrf_score), ...] 按 RRF 分数降序
        """
        rrf_scores = {}  # key: doc.page_content, value: [doc, score]

        # 向量路：rank 从 1 开始
        for rank, (doc, _) in enumerate(vec_results, 1):
            key = doc.page_content
            if key not in rrf_scores:
                rrf_scores[key] = [doc, 0.0]
            rrf_scores[key][1] += 1.0 / (k + rank)

        # BM25 路
        for rank, (doc, _) in enumerate(bm25_results, 1):
            key = doc.page_content
            if key not in rrf_scores:
                rrf_scores[key] = [doc, 0.0]
            rrf_scores[key][1] += 1.0 / (k + rank)

        # 按融合分数降序
        return [
            (doc, score)
            for doc, score in sorted(rrf_scores.values(), key=lambda x: x[1], reverse=True)
        ]


class BM25Retriever:
    """基于 rank_bm25 的中文 BM25 检索器"""

    def __init__(self):
        self.bm25: BM25Okapi = None
        self.documents: list[Document] = []  # 与 bm25 索引一一对应

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """中文分词：jieba 切词后过滤空白和标点"""
        tokens = jieba.lcut(text)
        return [t.strip() for t in tokens if t.strip()]

    def build_from_documents(self, documents: list[Document]) -> None:
        """从 LangChain Document 列表构建 BM25 索引"""
        self.documents = documents
        if not documents:  # 空集合保护，避免 BM25Okapi 初始化时除零
            self.bm25 = None
            return
        corpus_tokens = [self.tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(corpus_tokens)

    def add_documents(self, new_documents: list[Document]) -> None:
        """追加新文档后重建索引（基于已缓存文档 + 新文档，无需从 Chroma 全量拉取）"""
        self.documents.extend(new_documents)
        if not self.documents:  # 空集合保护
            self.bm25 = None
            return
        corpus_tokens = [self.tokenize(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, k: int = 20) -> list[tuple[Document, float]]:
        """查询返回 (Document, bm25_score) 列表，按分数降序"""
        if self.bm25 is None or not self.documents:
            return []

        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # 取 top-k（scores 是与 documents 一一对应的数组）
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.documents[i], float(scores[i])) for i in top_indices]


def check_md5(md5_str: str):
    """检查传入的md5字符串是否已经被处理过了"""
    if not os.path.exists(app_config.MD5_PATH):
        # 表示md5不存在---还没处理过
        # 创造新的文件
        open(app_config.MD5_PATH, "w", encoding="utf-8").close()
        return False

    else:
        for line in open(app_config.MD5_PATH, "r", encoding="utf-8").readlines():
            line = line.strip()  # 删除回车和空格
            if line == md5_str:  # 查找到了对应的内容
                return True

        return False


def save_md5(md5_str: str):
    """将传入的md5字符串保存到文件内"""
    with open(app_config.MD5_PATH, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str: str, encoding="utf-8"):
    """将传入的字符串变成md5格式"""

    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)

    # 创建md5对象
    md5_obj = hashlib.md5()  # md5对象
    md5_obj.update(str_bytes)  # 更新内容
    md5_hex = md5_obj.hexdigest()  # 转换为16进制

    return md5_hex
