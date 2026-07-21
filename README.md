# LegalRAG

面向法律领域的检索增强生成（RAG）问答系统。针对法规、合规制度等文本，提供可溯源、可治理、
可评估的问答能力。项目采用**渐进式、可插拔**的架构，每个组件（分块器 / 检索器 / 重排器 / LLM）
都能独立替换，便于对比实验与逐步演进。

## 快速开始

需要 Python 3.11+。推荐使用 [uv](https://github.com/astral-sh/uv) 管理环境。

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

复制环境变量模板并填入 Embedding 与 LLM 的 API 端点（均为 OpenAI 兼容协议）：

```bash
cp .env.example .env
# 编辑 .env，填写 EMBEDDING_API_BASE / EMBEDDING_API_KEY / LLM_API_BASE / LLM_API_KEY 等
```

摄取文档并提问：

```bash
uv run python -m legalrag.cli ingest path/to/document.pdf
uv run python -m legalrag.cli query "劳动合同试用期最长多久？"
```

## 架构

四层分层，依赖只能自上而下：

```
接口层    CLI（当前） / REST API（规划）        —— 组装、编排调用
编排层    pipeline 编排 / Agent（规划）          —— 路由、反思、拒答
能力层    ingest · embedding · retrieval ·       —— 各可插拔组件的具体实现
          rerank · generation · llm · store
契约层    core（models · interfaces · registry） —— 稳定的领域模型与抽象接口
```

- **契约层是稳定基石**：领域模型与抽象接口一次定稿，后续只增加实现、不改签名。
- **能力层组件互不直接依赖**，一律通过组件注册表（registry）按配置注入。
  换一个分块器 / 检索器 / LLM，只需修改 `config/*.yaml` 中的实现名。

## License

本项目基于 MIT License 开源。
