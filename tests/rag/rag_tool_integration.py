"""
Descripttion: RAGTool tool_call 功能测试
              先上传若干测试文档到知识库，再通过 tool_call 查询验证检索是否命中正确内容。
              覆盖语义查询（向量检索优势）与关键词/精确实体查询（BM25 优势）两类场景。
version: 1.0.0
Author: HongJin
email: 756547077@qq.com
"""

import sys
from pathlib import Path

# 把项目根目录加入 sys.path，支持直接运行此集成脚本。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.tools.RAGTool import RAGTool
from utils.config_manager import config as settings

# ---------- 测试数据：先上传到知识库 ----------
TEST_DOCUMENTS = [
    {
        "filename": "产品功能介绍.md",
        "content": (
            "校园情感陪伴助手 功能介绍\n\n"
            "校园情感陪伴助手是一款面向高校学生的心理健康陪伴应用，主要功能包括：\n\n"
            "1. 情感对话：基于大语言模型提供温暖、共情的对话体验，能够识别用户的情绪状态"
            "（如焦虑、抑郁、孤独），并给予针对性的心理支持。\n\n"
            "2. 情绪记忆：长期记忆用户的关键信息与情感历史，建立持续性的陪伴关系，"
            "随着交互积累逐步提升好感度。\n\n"
            "3. 危机干预：当检测到用户存在自伤、自杀等高风险信号时，立即触发危机干预流程，"
            "联系学校心理咨询中心。\n\n"
            "4. 工具调用：集成数学计算、网页搜索、知识库检索等工具，辅助完成学习与生活咨询。"
        ),
    },
    {
        "filename": "数据隐私政策.md",
        "content": (
            "用户数据隐私与安全政策\n\n"
            "本应用严格保护用户隐私，所有数据处理遵循以下原则：\n\n"
            "1. 数据加密：用户对话内容在传输与存储过程中均采用 AES-256 加密算法。\n\n"
            "2. 本地化部署：向量化模型与聊天模型均部署在学校内网服务器，数据不出校。\n\n"
            "3. 最小化保留：情感记忆仅保留与心理支持相关的关键信息，用户可随时申请删除全部数据。\n\n"
            "4. 访问控制：仅用户本人与授权的心理咨询师可查看对话记录，其他人员无权访问。"
        ),
    },
    {
        "filename": "版本更新日志_v2.0.md",
        "content": (
            "版本更新日志 v2.0.0\n\n"
            "发布日期：2026-07-01\n\n"
            "【新增功能】\n"
            "- 多路召回检索：集成向量检索与 BM25 字符匹配检索，通过 RRF 融合提升知识库查询准确率。\n"
            "- 情绪分析模块：新增基于 LangGraph 的情绪状态机，支持情绪转移与好感度演化。\n\n"
            "【优化改进】\n"
            "- 向量化模型升级为 qwen3-embedding:4b，中文语义理解能力提升 15%。\n"
            "- 记忆系统重构，采用 MemoryItem 结构化存储，支持 LangGraph checkpoint 持久化。\n\n"
            "【修复问题】\n"
            "- 修复 ToolMessage 缺失 tool_call_id 导致的运行时错误。\n"
            "- 修复 LangGraph 反序列化时 MemoryItem 类型未注册的警告。"
        ),
    },
]


# ---------- 测试查询 ----------
# expect_keywords: 返回结果中应至少命中其中一个，否则判定失败
TEST_QUERIES = [
    {
        "query": "助手有哪些功能",
        "expect_keywords": ["情感对话", "情绪记忆", "危机干预", "工具调用"],
        "desc": "语义查询（向量检索优势）——应命中功能介绍文档",
    },
    {
        "query": "v2.0.0 版本更新了什么",
        "expect_keywords": ["多路召回", "情绪分析模块", "qwen3-embedding", "MemoryItem"],
        "desc": "精确版本号查询（BM25 优势）——应命中更新日志",
    },
    {
        "query": "用户数据怎么加密",
        "expect_keywords": ["AES-256", "本地化部署", "加密"],
        "desc": "关键词混合查询——应命中隐私政策",
    },
    {
        "query": "校园情感陪伴助手",
        "expect_keywords": ["校园情感陪伴助手"],
        "desc": "精确实体查询（BM25 优势）——验证关键词匹配",
    },
    {
        "query": "好感度是怎么变化的",
        "expect_keywords": ["好感度", "情绪记忆", "情绪分析模块"],
        "desc": "语义近义查询——验证跨文档语义召回",
    },
]


def upload_test_docs(rag_tool: RAGTool):
    """上传测试文档到知识库（已存在的内容会按 md5 自动跳过）"""
    print("=" * 60)
    print("📥 上传测试文档")
    print("=" * 60)
    for doc in TEST_DOCUMENTS:
        result = rag_tool.RAG.upload_by_str(doc["content"], doc["filename"])
        print(f"[{doc['filename']}] {result}")


def run_queries(rag_tool: RAGTool):
    """执行查询测试，验证 tool_call 返回结果"""
    print("\n" + "=" * 60)
    print("🔍 查询测试（tool_call）")
    print("=" * 60)

    passed = 0
    total = len(TEST_QUERIES)

    for i, t in enumerate(TEST_QUERIES, 1):
        print(f"\n--- 测试 {i}/{total}: {t['desc']} ---")
        print(f"查询: {t['query']}")

        result = rag_tool.tool_call(query=t["query"])
        print(f"返回:\n{result}")

        # 断言1：不能是无结果
        if "[无结果]" in result:
            print("❌ 失败：未检索到任何内容")
            continue

        # 断言2：期望关键词至少命中一个
        hit = [kw for kw in t["expect_keywords"] if kw in result]
        if hit:
            print(f"✅ 通过：命中关键词 {hit}")
            passed += 1
        else:
            print(f"❌ 失败：未命中任何期望关键词 {t['expect_keywords']}")

    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    print("=" * 60)
    return passed, total


def main():
    print("🧪 RAGTool tool_call 功能测试")
    print(f"向量化模型: {settings.EMBEDDING_MODEL_NAME}")
    print(f"向量库路径: {settings.CHROMA_PATH}")
    print("BM25 文档数: (初始化后可知)")
    print()

    rag_tool = RAGTool()

    upload_test_docs(rag_tool)

    # 上传后打印索引规模，确认向量库与 BM25 同步更新
    doc_count = len(rag_tool.RAG.bm25_retriever.documents)
    print(f"\n知识库文档段数: {doc_count}（向量库与 BM25 同步）")

    passed, total = run_queries(rag_tool)

    # 退出码：全部通过返回 0，否则返回 1，方便 CI 判断
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
