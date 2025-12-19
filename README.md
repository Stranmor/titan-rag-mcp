# 🚀 Titan RAG: High-Performance MCP Code Search

A heavily optimized, GPU-accelerated Model Context Protocol (MCP) server for local code indexing and semantic search.

Based on the original work by [Luoto Company](https://github.com/LuotoCompany/cursor-local-indexing), **Titan RAG** is designed for speed, stability, and superior retrieval quality using Ollama and high-dimensional embedding models.

## 🌟 Key Improvements over Original

- **GPU Acceleration:** Fully offloads embedding computation to GPU via Ollama (optimized for AMD ROCm/Phoenix and NVIDIA).
- **Superior Embeddings:** Uses `bge-m3` (1024d) by default, providing much higher retrieval accuracy than the standard `all-MiniLM-L6-v2`.
- **Instant Responsiveness:** Non-blocking startup with background indexing. Your AI agent (Gemini/Claude) connects instantly without waiting for the index to build.
- **Dynamic Monitoring:** Includes a `get_index_status` tool to track indexing progress in real-time.
- **Stability Fixes:** Resolved critical dimension mismatch bugs and suppressed noisy library warnings that cause MCP connection drops.
- **Efficient Splitting:** Hybrid strategy combining high-quality semantic splitting (tree-sitter) with fast line-based fallbacks.

## 🛠 Prerequisites

1. **Ollama:** Installed and running.
   ```bash
   ollama pull bge-m3
   ```
2. **Python 3.10+**
3. **GPU Drivers:** Properly configured (ROCm for AMD or CUDA for NVIDIA).

## 🚀 Quick Start

### 1. Configure Environment
Create a `.env` file in the project root:
```env
PROJECTS_ROOT=/path/to/your/projects
FOLDERS_TO_INDEX=project-a,project-b
CHROMA_DB_PATH=./chroma_db
```

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Usage with Gemini CLI / Claude Code
Add the server to your MCP configuration:

**Path:** `~/.config/gemini-cli/config.json` (or similar)
```json
{
  "mcpServers": {
    "titan-rag": {
      "command": "/path/to/project/.venv/bin/python",
      "args": ["/path/to/project/ollama_indexer.py"]
    }
  }
}
```

## 🔍 Tools Provided

- `rag_search_code`: Search your codebase using natural language.
- `get_index_status`: Check if indexing is completed or still running in the background.
- `get_system_health`: Monitor GPU temperature, VRAM usage, and system load.

## 🧪 Testing

To verify your installation and connection to Ollama:
```bash
source .venv/bin/activate
python test_rag.py
```

## 🛡 License
This project is licensed under the terms of the original repository (AGPLv3/MIT). See the LICENSE file for details.

---
*Part of the Titan Hive Mind ecosystem.*
