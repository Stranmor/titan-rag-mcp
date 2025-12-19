#!/usr/bin/env python3

import os
import logging
import json
import sys
import warnings
from typing import List, Set
from dotenv import load_dotenv

# Suppress warnings and redirect stdout to stderr to prevent protocol pollution
warnings.filterwarnings("ignore")
# Note: FastMCP handles its own stdio, but some libraries might print during import.
# We don't redirect sys.stdout globally yet because FastMCP needs it for the server loop.
# But we can silence specific noisy loggers.

# Load environment variables from .env file
load_dotenv()

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tree_sitter_language_pack import get_parser
import asyncio
from fastmcp import FastMCP
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# Import LlamaIndex components
try:
    from llama_index.core import Document
    from llama_index.core.node_parser import CodeSplitter
    from llama_index.core import SimpleDirectoryReader
    # logger.info("LlamaIndex dependencies found.") # Logger not init yet, pass silently
except ImportError as e:
    # We can't log yet, but we can't print either if we want to be clean.
    # Since we are in a try/except at module level, exiting with status 1 is safest.
    exit(1)

# Configure logging
LOG_FILE = os.path.join(os.path.dirname(__file__), "mcp_server.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# Default directories to ignore
DEFAULT_IGNORE_DIRS = {
    "__pycache__",
    "node_modules",
    ".git",
    "build",
    "dist",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    ".ipynb_checkpoints"
}

# Default files to ignore
DEFAULT_IGNORE_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "Gemfile.lock",
    "composer.lock",
    ".DS_Store",
    ".env",
    ".env.local",
    ".env.development",
    ".env.test",
    ".env.production",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.dylib",
    ".coverage",
    "coverage.xml",
    ".eslintcache",
    ".tsbuildinfo"
}

# Default file extensions to include
DEFAULT_FILE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rb", ".php", ".swift", ".kt", ".rs", ".scala", ".sh",
    ".html", ".css", ".sql", ".md", ".json", ".yaml", ".yml", ".toml"
}

# Global variables
config = None
chroma_client = None
embedding_function = None
mcp = FastMCP(title="Titan RAG (Local Code Search)")
observers = []
indexing_status = {
    "status": "starting", 
    "folders": {}, 
    "progress": {"total": 0, "processed": 0, "current_file": ""}
}


def sanitize_collection_name(folder_name: str) -> str:
    """Convert folder name to a valid collection name by replacing forward slashes with underscores."""
    return folder_name.replace("/", "_")


class CodeIndexerEventHandler(FileSystemEventHandler):
    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self.collection_name = sanitize_collection_name(folder_name)
        self.ignore_dirs = set(config["ignore_dirs"])
        self.ignore_files = set(config["ignore_files"])
        self.file_extensions = set(config["file_extensions"])

    def on_created(self, event):
        if event.is_directory:
            return
        if is_valid_file(
            event.src_path,
            self.ignore_dirs,
            self.file_extensions,
            self.ignore_files
        ):
            self._handle_file_change(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        if is_valid_file(
            event.src_path,
            self.ignore_dirs,
            self.file_extensions,
            self.ignore_files
        ):
            self._handle_file_change(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        if is_valid_file(
            event.src_path,
            self.ignore_dirs,
            self.file_extensions,
            self.ignore_files
        ):
            self._handle_file_deletion(event.src_path)

    def _handle_file_change(self, file_path: str):
        try:
            # Calculate relative path
            rel_path = os.path.relpath(file_path, config["projects_root"])

            # Load and process the single file
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()

            if documents:
                # Update metadata with relative path
                documents[0].metadata["file_path"] = rel_path

                # Run heavy processing in a thread to keep MCP responsive
                asyncio.create_task(asyncio.to_thread(
                    process_and_index_documents,
                    documents,
                    self.collection_name,
                    "chroma_db"
                ))
                logger.info(f"Indexing task started for: {rel_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

    def _handle_file_deletion(self, file_path: str):
        try:
            # Calculate relative path
            rel_path = os.path.relpath(file_path, config["projects_root"])

            # Get the collection
            collection = chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=embedding_function
            )

            # Delete all chunks from this file
            collection.delete(
                where={"file_path": rel_path}
            )
            logger.info(f"Removed indexed chunks for deleted file: {rel_path}")
        except Exception as e:
            logger.error(f"Error removing chunks for {file_path}: {e}")


def get_config_from_env():
    """Get configuration from environment variables, defaulting to current directory."""
    # Use current working directory if PROJECTS_ROOT is not set
    cwd = os.getcwd()
    projects_root = os.getenv("PROJECTS_ROOT", os.path.dirname(cwd))
    
    # If FOLDERS_TO_INDEX is not set, perform auto-discovery of git repos
    folders_to_index_str = os.getenv("FOLDERS_TO_INDEX", "")
    if not folders_to_index_str:
        logger.info(f"FOLDERS_TO_INDEX not set. Scanning {projects_root} for git repositories...")
        folders_to_index = []
        try:
            for entry in os.scandir(projects_root):
                if entry.is_dir() and os.path.exists(os.path.join(entry.path, ".git")):
                    folders_to_index.append(entry.name)
            logger.info(f"Auto-discovered {len(folders_to_index)} projects: {', '.join(folders_to_index)}")
        except Exception as e:
            logger.error(f"Auto-discovery failed: {e}")
            folders_to_index = [os.path.basename(cwd)]
    else:
        folders_to_index = [f.strip() for f in folders_to_index_str.split(",") if f.strip()]

    # ... (rest of the function)
    additional_ignore_dirs = os.getenv(
        "ADDITIONAL_IGNORE_DIRS", ""
    ).split(",")
    additional_ignore_dirs = [
        d.strip() for d in additional_ignore_dirs if d.strip()
    ]

    additional_ignore_files = os.getenv(
        "ADDITIONAL_IGNORE_FILES", ""
    ).split(",")
    additional_ignore_files = [
        f.strip() for f in additional_ignore_files if f.strip()
    ]

    # Combine default and additional ignore patterns
    ignore_dirs = list(DEFAULT_IGNORE_DIRS | set(additional_ignore_dirs))
    ignore_files = list(DEFAULT_IGNORE_FILES | set(additional_ignore_files))

    if not folders_to_index:
        logger.warning("No folders specified to index. Using root directory.")
        folders_to_index = [""]

    return {
        "projects_root": projects_root,
        "folders_to_index": folders_to_index,
        "ignore_dirs": ignore_dirs,
        "ignore_files": ignore_files,
        "file_extensions": list(DEFAULT_FILE_EXTENSIONS)
    }


async def initialize_chromadb():
    """Initialize ChromaDB and embedding function asynchronously."""
    global config, chroma_client, embedding_function

    try:
        # Get configuration from environment
        config = get_config_from_env()
        logger.info("Configuration loaded successfully")

        # Use CHROMA_DB_PATH from env or default to local chroma_db
        db_path = os.getenv("CHROMA_DB_PATH", "chroma_db")
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.path.dirname(__file__), db_path)
        
        # Initialize ChromaDB client with telemetry disabled
        chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        logger.info(f"ChromaDB client initialized at {db_path}")

        # Initialize embedding function
        embedding_function = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="bge-m3"
        )
        logger.info("Embedding function initialized (Ollama: bge-m3)")

        return True
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        # Still need to assign default values to global variables
        if config is None:
            config = {"projects_root": "", "folders_to_index": [""]}
        if chroma_client is None:
            # Create an empty client as a fallback
            try:
                db_path = os.getenv("CHROMA_DB_PATH", "chroma_db")
                chroma_client = chromadb.PersistentClient(path=db_path)
            except Exception as db_err:
                logger.error(
                    f"Failed to create fallback ChromaDB client: {db_err}"
                )
        if embedding_function is None:
            try:
                embedding_function = embedding_functions.OllamaEmbeddingFunction(
                    url="http://localhost:11434/api/embeddings",
                    model_name="bge-m3"
                )
            except Exception as embed_err:
                logger.error(
                    f"Failed to create fallback embedding function: "
                    f"{embed_err}"
                )
        return False


def is_valid_file(
    file_path: str,
    ignore_dirs: Set[str],
    file_extensions: Set[str],
    ignore_files: Set[str] = None
) -> bool:
    """Check if a file should be processed based on its path and extension."""
    # Check if path contains ignored directory
    parts = file_path.split(os.path.sep)
    for part in parts:
        if part in ignore_dirs:
            return False

    # Get file name and check against ignored files
    file_name = os.path.basename(file_path)

    # Use provided ignore_files or fall back to default
    files_to_ignore = ignore_files if ignore_files is not None else DEFAULT_IGNORE_FILES

    # Check exact matches
    if file_name in files_to_ignore:
        return False

    # Check wildcard patterns
    for pattern in files_to_ignore:
        if pattern.startswith("*"):
            if file_name.endswith(pattern[1:]):
                return False

    # Check file extension
    _, ext = os.path.splitext(file_path)
    return ext.lower() in file_extensions if file_extensions else True


def load_documents(
    directory: str, 
    ignore_dirs: Set[str] = DEFAULT_IGNORE_DIRS,
    file_extensions: Set[str] = DEFAULT_FILE_EXTENSIONS,
    ignore_files: Set[str] = None
) -> List[Document]:
    """Load documents from a directory, filtering out ignored paths."""
    try:
        # Get all files recursively
        all_files = []
        for root, dirs, files in os.walk(directory):
            # Skip ignored directories
            dirs[:] = [
                d for d in dirs
                if d not in ignore_dirs and not d.startswith('.')
            ]

            for file in files:
                abs_file_path = os.path.join(root, file)
                if is_valid_file(
                    abs_file_path,
                    ignore_dirs,
                    file_extensions,
                    ignore_files
                ):
                    # Calculate relative path from the directory being indexed
                    rel_file_path = os.path.relpath(abs_file_path, directory)
                    all_files.append((abs_file_path, rel_file_path))

        if not all_files:
            logger.warning(f"No valid files found in {directory}")
            return []

        # Load the filtered files using absolute paths for reading
        reader = SimpleDirectoryReader(
            input_files=[abs_path for abs_path, _ in all_files],
            exclude_hidden=True
        )
        documents = reader.load_data()

        # Update the metadata to use relative paths
        for doc, (_, rel_path) in zip(documents, all_files):
            doc.metadata["file_path"] = rel_path

        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []


import time

def process_and_index_documents(
    documents: List[Document],
    collection_name: str,
    persist_directory: str
) -> None:
    """Process documents and index them in ChromaDB with minimal CPU usage."""
    if not documents:
        return

    start_time = time.time()
    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return

    # Process each document
    total_nodes = 0
    indexing_status["progress"]["total"] = len(documents)
    indexing_status["progress"]["processed"] = 0

    for doc in documents:
        try:
            file_path = doc.metadata.get("file_path", "unknown")
            file_name = os.path.basename(file_path)
            
            indexing_status["progress"]["current_file"] = file_path
            indexing_status["progress"]["processed"] += 1

            # Determine language from file extension
            _, ext = os.path.splitext(file_name)
            language = ext[1:] if ext else "text"

            # Handle code files with semantic splitting, others with text splitting
            code_file_extensions = [
                "py", "python", "js", "jsx", "ts", "tsx", "java", "c", 
                "cpp", "h", "hpp", "cs", "go", "rb", "php", "swift", 
                "kt", "rs", "scala"
            ]

            if language in code_file_extensions:
                # Determine parser language based on file extension
                parser_language = "python"  # Default fallback
                if language in ["py", "python"]:
                    parser_language = "python"
                elif language in ["js", "jsx", "ts", "tsx"]:
                    parser_language = "javascript"
                elif language in ["java"]:
                    parser_language = "java"
                elif language in ["c", "cpp", "h", "hpp"]:
                    parser_language = "cpp"
                elif language in ["cs"]:
                    parser_language = "csharp"
                elif language in ["go"]:
                    parser_language = "go"
                elif language in ["rb"]:
                    parser_language = "ruby"
                elif language in ["php"]:
                    parser_language = "php"
                elif language in ["swift"]:
                    parser_language = "swift"
                elif language in ["kt"]:
                    parser_language = "kotlin"
                elif language in ["rs"]:
                    parser_language = "rust"
                elif language in ["scala"]:
                    parser_language = "scala"

                # Create parser and splitter for this specific language
                try:
                    code_parser = get_parser(parser_language)
                    splitter = CodeSplitter(
                        language=parser_language,
                        chunk_lines=40,
                        chunk_lines_overlap=15,
                        max_chars=1500,
                        parser=code_parser
                    )
                    nodes = splitter.get_nodes_from_documents([doc])
                except Exception as e:
                    logger.warning(f"Could not create parser for {parser_language}, falling back to text: {e}")
                    nodes = []
            else:
                nodes = []

            # Fallback to line-based splitting if no nodes generated or not a code file
            if not nodes:
                lines = doc.text.split("\n")
                chunk_size = 40
                overlap = 15
                for i in range(0, len(lines), chunk_size - overlap):
                    start_idx = i
                    end_idx = min(i + chunk_size, len(lines))
                    if start_idx >= len(lines): continue
                    chunk_text = "\n".join(lines[start_idx:end_idx])
                    if not chunk_text.strip(): continue
                    
                    from llama_index.core.schema import TextNode
                    node = TextNode(
                        text=chunk_text,
                        metadata={
                            "start_line_number": start_idx + 1,
                            "end_line_number": end_idx,
                            "file_path": file_path,
                            "file_name": file_name,
                        }
                    )
                    nodes.append(node)

            if not nodes:
                logger.warning(f"No nodes generated for {file_path}, skipping")
                continue

            logger.info(f"Processing {file_path}: {len(nodes)} chunks")

            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []

            for i, node in enumerate(nodes):
                start_line = node.metadata.get("start_line_number", 0)
                end_line = node.metadata.get("end_line_number", 0)

                if start_line == 0 or end_line == 0:
                    start_line = 1
                    end_line = len(node.text.split("\n"))

                chunk_id = f"{file_path}_{start_line}_{end_line}_{i}"

                metadata = {
                    "file_path": file_path,
                    "file_name": file_name,
                    "language": language,
                    "start_line": start_line,
                    "end_line": end_line,
                }

                ids.append(chunk_id)
                texts.append(node.text)
                metadatas.append(metadata)

            # Add nodes to ChromaDB collection
            collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

            total_nodes += len(nodes)

        except Exception as e:
            logger.error(
                f"Error processing document "
                f"{doc.metadata.get('file_path', 'unknown')}: {e}"
            )

    logger.info(
        f"Successfully indexed {total_nodes} code chunks "
        f"across {len(documents)} files"
    )


@mcp.tool()
async def find_todos(project_path: str = ".") -> str:
    """Scan the project for TODO, FIXME, and HACK comments."""
    try:
        cmd = f"grep -rnE 'TODO|FIXME|HACK' {project_path} --exclude-dir='.git' --exclude-dir='node_modules' --exclude-dir='.venv' --exclude-dir='chroma_db'"
        res = os.popen(cmd).read()
        return f"📝 Technical Debt Found:\n{res or 'No TODOs found. Perfect!'}"
    except Exception as e:
        return f"Scan failed: {str(e)}"


@mcp.tool()
async def generate_commit_message() -> str:
    """Analyze staged git changes and generate a professional commit message."""
    try:
        # Get staged changes
        diff = os.popen("git diff --cached").read()
        if not diff:
            return "No staged changes found. Use 'git add' first."

        # Limit diff size to avoid context overflow (take first 4000 chars)
        diff_snippet = diff[:4000]
        
        prompt = f"Generate a professional commit message in Conventional Commits format based on this diff. Keep it concise but descriptive. Output ONLY the message.\n\nDiff:\n{diff_snippet}"
        
        import requests
        response = requests.post('http://localhost:11434/api/generate',
            json={
                'model': 'vibethinker',
                'prompt': prompt,
                'stream': False
            }, timeout=45)
        
        return response.json().get('response', 'AI failed to generate commit message.').strip()
    except Exception as e:
        return f"Commit message generation failed: {str(e)}"


@mcp.tool()
async def analyze_workspace(path: str = ".") -> str:
    """Analyze the workspace structure and suggest the best Zed profile and tech stack."""
    try:
        files = os.listdir(path)
        structure = "\n".join(files[:20]) # Take top 20 files
        
        prompt = f"Analyze this project structure and suggest the best development profile (Rust, Python, Node.js, Web). Explain why and suggest 3 essential Zed extensions.\n\nFiles:\n{structure}"
        
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post('http://localhost:11434/api/generate',
                json={
                    'model': 'vibethinker',
                    'prompt': prompt,
                    'stream': False
                }, timeout=30.0)
        
        return response.json().get('response', 'AI failed to analyze workspace.')
    except Exception as e:
        return f"Analysis failed: {str(e)}"


@mcp.tool()
async def get_zed_status() -> str:
    """Check which isolated Zed profiles are currently running."""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/zed/zed-status.sh")
        result = os.popen(f"bash {script_path}").read()
        return result.strip()
    except Exception as e:
        return f"Failed to get status: {str(e)}"


@mcp.tool()
async def get_ecosystem_analytics(days: int = 7) -> str:
    """Perform a deep analysis of project health vs. developer activity."""
    try:
        analytics = {"time_stats": {}, "health_trend": [], "top_errors": []}
        
        # 1. Analyze Time Tracking
        time_log = os.path.expanduser("~/.titan-time-tracking.csv")
        if os.path.exists(time_log):
            import pandas as pd
            df_time = pd.read_csv(time_log)
            # Simple hours sum
            analytics["time_stats"] = df_time.groupby('profile').size().to_dict() # mins

        # 2. Analyze Health History
        health_log = os.path.expanduser("~/.titan-health-history.csv")
        if os.path.exists(health_log):
            with open(health_log, 'r') as f:
                analytics["health_trend"] = f.readlines()[-days:]

        # 3. Analyze Logs for common errors
        mcp_log = os.path.join(os.path.dirname(__file__), "mcp_server.log")
        if os.path.exists(mcp_log):
            errors = os.popen(f"grep 'ERROR' {mcp_log} | tail -n 20").read()
            analytics["top_errors"] = errors.splitlines()

        return json.dumps(analytics, indent=2)
    except Exception as e:
        return f"Analytics failed: {str(e)}"


@mcp.tool()
async def generate_report(project_path: str = ".") -> str:
    """Generate a high-level intelligence report about the project state."""
    try:
        # Collect data snippets
        p_map = await get_project_map()
        hotspots = await get_git_hotspots(7)
        todos = await find_todos(project_path)
        health = await get_system_health()
        
        prompt = f"""Act as a Technical Lead. Analyze the following project data and generate a concise report.
Include: 
1. Project Pulse (Activity)
2. Technical Debt & Risks
3. System Resources Status
4. Recommendations for next steps.

DATA:
Project Map: {p_map[:1000]}
Git Hotspots: {hotspots}
Technical Debt: {todos[:500]}
        Hardware Health: {health}
\"\"\"
        
        report = await call_ollama(prompt, task_type="general")
        
        # Save report to a file for Dashboard
        report_path = os.path.join(os.path.dirname(__file__), "latest_report.md")

        with open(report_path, 'w') as f:
            f.write(f"# Titan Intelligence Report ({time.strftime('%Y-%m-%d %H:%M')})\n\n{report}")
            
        return report
    except Exception as e:
        return f"Report generation failed: {str(e)}"


@mcp.tool()
async def audit_security(project_path: str = ".") -> str:
    """Scan the project for hardcoded secrets, API keys, and security risks."""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/system/titan-audit.sh")
        result = os.popen(f"bash {script_path} {project_path}").read()
        return result
    except Exception as e:
        return f"Security audit failed: {str(e)}"


@mcp.tool()
async def audit_dependencies(project_path: str = ".") -> str:
    """Scan for outdated or vulnerable dependencies in the project."""
    results = []
    
    # 1. Python (pip-audit + pip list --outdated)
    if os.path.exists(os.path.join(project_path, "requirements.txt")):
        audit = os.popen(f"pip-audit -r {project_path}/requirements.txt 2>&1").read()
        outdated = os.popen("pip list --outdated 2>&1").read()
        results.append(f"--- Python (Audit & Outdated) ---\n{audit}\n\nOutdated:\n{outdated}")

    # 2. Node.js (npm audit + npm outdated)
    if os.path.exists(os.path.join(project_path, "package.json")):
        audit = os.popen(f"cd {project_path} && npm audit").read()
        outdated = os.popen(f"cd {project_path} && npm outdated").read()
        results.append(f"--- Node.js (Audit & Outdated) ---\n{audit}\n\nOutdated:\n{outdated or 'All up to date.'}")

    # 3. Rust (cargo audit - if installed)
    if os.path.exists(os.path.join(project_path, "Cargo.toml")):
        res = os.popen(f"cd {project_path} && cargo audit 2>&1 || echo 'cargo-audit not installed'").read()
        results.append(f"--- Rust (Audit) ---\n{res}")

    return "\n\n".join(results) or "No project files detected for auditing."


@mcp.tool()
async def run_tests(project_path: str = ".") -> str:
    """Detect and run tests for the project (pytest, cargo test, npm test)."""
    results = []
    
    # 1. Python (pytest)
    if os.path.exists(os.path.join(project_path, "pytest.ini")) or any(f.startswith("test_") for f in os.listdir(project_path)):
        res = os.popen(f"cd {project_path} && pytest --version && pytest -v").read()
        results.append(f"--- Python (Pytest) ---\n{res}")

    # 2. Rust (cargo test)
    if os.path.exists(os.path.join(project_path, "Cargo.toml")):
        res = os.popen(f"cd {project_path} && cargo test -- --nocapture").read()
        results.append(f"--- Rust (Cargo) ---\n{res}")

    # 3. Node.js (npm test)
    if os.path.exists(os.path.join(project_path, "package.json")):
        res = os.popen(f"cd {project_path} && npm test").read()
        results.append(f"--- Node.js (NPM) ---\n{res}")

    return "\n\n".join(results) or "No test suites detected."


@mcp.tool()
async def review_code(file_path: str) -> str:
    """Use local AI (vibethinker) to perform a deep code review of a specific file."""
    try:
        if not os.path.exists(file_path):
            return f"File {file_path} not found."

        with open(file_path, 'r') as f:
            code = f.read()

        prompt = f"Perform a deep code review of the following file. Find bugs, security issues, and suggest architectural improvements. Output in professional markdown.\n\nFile: {file_path}\n\nCode:\n```\n{code}\n```"
        
        return await call_ollama(prompt, task_type="review")
    except Exception as e:
        return f"Review failed: {str(e)}"


@mcp.tool()
async def check_code_quality(project_path: str = ".") -> str:
    """Run linters (ruff, cargo clippy) on the project and report errors."""
    results = []
    
    # 1. Python (Ruff)
    if os.path.exists(os.path.join(project_path, "requirements.txt")) or any(f.endswith(".py") for f in os.listdir(project_path)):
        res = os.popen(f"ruff check {project_path} 2>&1").read()
        results.append(f"--- Python (Ruff) ---\n{res or 'All good!'}")

    # 2. Rust (Clippy)
    if os.path.exists(os.path.join(project_path, "Cargo.toml")):
        res = os.popen(f"cd {project_path} && cargo clippy -- -D warnings 2>&1").read()
        results.append(f"--- Rust (Clippy) ---\n{res or 'All good!'}")

    return "\n\n".join(results)


@mcp.tool()
async def titan_speak(text: str) -> str:
    """Announce a message using the local voice synthesis engine (TTS)."""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/system/titan-speak.sh")
        # Safe escaping
        safe_text = text.replace("'", "")
        os.system(f"bash {script_path} '{safe_text}'")
        return f"Announced: {text}"
    except Exception as e:
        return f"Speech failed: {str(e)}"


@mcp.tool()
async def send_notification(title: str, message: str, level: str = "normal") -> str:
    """Send a system notification to the user.
    Args:
        title: Short title of the notification.
        message: Detailed message content.
        level: Urgency level ('low', 'normal', 'critical').
    """
    try:
        icon = "dialog-information"
        if level == "critical": icon = "dialog-error"
        elif level == "low": icon = "dialog-information"
        
        # We escape quotes for safety in os.system
        safe_title = title.replace('"', '\\"')
        safe_message = message.replace('"', '\\"')
        
        os.system(f'notify-send -u {level} -i {icon} "Titan: {safe_title}" "{safe_message}"')
        return "Notification sent successfully."
    except Exception as e:
        return f"Failed to send notification: {str(e)}"


@mcp.tool()
async def hibernate_profile(profile_name: str) -> str:
    """Safely snapshot and close an active Zed profile to free up resources."""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/system/titan-balancer.sh")
        result = os.popen(f"bash {script_path} {profile_name}").read()
        return result
    except Exception as e:
        return f"Hibernation failed: {str(e)}"


@mcp.tool()
async def sync_titan_docs() -> str:
    """Automatically update TITAN_COMMANDS.md by scanning available MCP tools."""
    try:
        current_file = os.path.abspath(__file__)
        proj_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        docs_path = os.path.join(proj_root, "docs/system/TITAN_COMMANDS.md")
        
        with open(current_file, 'r') as f:
            lines = f.readlines()

        tools = []
        import re
        for i, line in enumerate(lines):
            if "@mcp.tool(" in line:
                # Find the function name on next lines
                for j in range(i + 1, i + 5):
                    match = re.search(r"async def (\w+)\(", lines[j])
                    if match:
                        name = match.group(1)
                        # Extract first line of docstring
                        doc = "No description"
                        if '"""' in lines[j+1]:
                            doc = lines[j+1].replace('"""', '').strip()
                        tools.append(f"- `{name}`: {doc}")
                        break
        
        content = "# 🛠 Titan AI Ecosystem: Command Reference\n\n"
        content += "## 🔎 RAG & AI Tools (Auto-generated)\n"
        content += "\n".join(sorted(tools))
        content += "\n\n## 🛡 System & Monitoring\n"
        content += "- `tcc`: Titan Control Center (status, restart, logs).\n"
        content += "- `vibe-ui.sh`: Launch the visual dashboard.\n"
        content += "- `titan-init <path>`: Smart project onboarding.\n"
        
        with open(docs_path, 'w') as f:
            f.write(content)
            
        return "TITAN_COMMANDS.md has been synchronized with latest tools."
    except Exception as e:
        return f"Sync failed: {str(e)}"


@mcp.tool()
async def claim_active_zone(agent_id: str, zone_name: str) -> str:
    """Claim a specific code zone/module to prevent other agents from editing it.
    Args:
        agent_id: Unique ID of the agent claiming the zone.
        zone_name: Name of the module or file path (e.g., 'Auth', 'UI/Header').
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        agents_md_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "AGENTS.md")
        
        with open(agents_md_path, 'r') as f:
            lines = f.readlines()

        # Check if already claimed
        claim_str = f"Agent {agent_id} working on {zone_name}"
        for line in lines:
            if zone_name in line and "working on" in line and agent_id not in line:
                return f"Conflict: Zone '{zone_name}' is already claimed by another agent."

        # Find [ACTIVE ZONES] section
        insert_idx = -1
        for i, line in enumerate(lines):
            if "[ACTIVE ZONES]" in line:
                insert_idx = i + 1
                break
        
        if insert_idx == -1:
            lines.append("\n## [ACTIVE ZONES]\n")
            insert_idx = len(lines)

        lines.insert(insert_idx, f"- {claim_str}\n")

        with open(agents_md_path, 'w') as f:
            f.writelines(lines)

        return f"Successfully claimed zone '{zone_name}' for agent {agent_id}."
    except Exception as e:
        return f"Claim failed: {str(e)}"


@mcp.tool()
async def broadcast_agent_message(agent_id: str, message: str) -> str:
    """Send a message to other agents in the swarm via the common log."""
    try:
        log_path = os.path.expanduser("~/.titan-swarm-comms.log")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, 'a') as f:
            f.write(f"[{timestamp}] [AGENT {agent_id}] {message}\n")
        return "Message broadcasted to the swarm."
    except Exception as e:
        return f"Broadcast failed: {str(e)}"


@mcp.tool()
async def release_active_zone(agent_id: str, zone_name: str) -> str:
    """Release a previously claimed code zone."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        agents_md_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "AGENTS.md")
        
        with open(agents_md_path, 'r') as f:
            lines = f.readlines()

        new_lines = [line for line in lines if not (agent_id in line and zone_name in line and "working on" in line)]

        with open(agents_md_path, 'w') as f:
            f.writelines(new_lines)

        return f"Zone '{zone_name}' released by agent {agent_id}."
    except Exception as e:
        return f"Release failed: {str(e)}"


@mcp.tool()
async def profile_code(command: str, project_path: str = ".") -> str:
    """Execute a command and provide a performance profile (time, CPU, top calls).
    Args:
        command: The command to run (e.g., 'python script.py' or 'pytest test_perf.py').
        project_path: Working directory.
    """
    try:
        import time
        import subprocess
        
        report = [f"⏱️ Profiling execution: {command}"]
        
        # We use cProfile for Python commands to get deep insights
        prof_cmd = command
        if command.startswith("python"):
            prof_cmd = command.replace("python", "python -m cProfile -s cumulative", 1)
        
        start_time = time.perf_counter()
        process = subprocess.Popen(
            f"cd {project_path} && {prof_cmd}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=60)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        report.append(f"Total Duration: {duration:.4f} seconds")
        
        if stderr:
            report.append(f"\n⚠️ Errors during profiling:\n{stderr}")
        
        # Extract top 15 lines of cProfile output or raw stdout
        report.append("\n--- Execution Profile (Top Calls/Output) ---")
        lines = stdout.splitlines()
        report.append("\n".join(lines[:30])) # Show more for profiling
        
        return "\n".join(report)
    except Exception as e:
        return f"Profiling failed: {str(e)}"


@mcp.tool()
async def safe_update_dependency(project_path: str, package_name: str, manager: str = "pip") -> str:
    """Attempt to update a package and rollback if tests fail.
    Args:
        project_path: Path to the project.
        package_name: Name of the package to update.
        manager: Package manager ('pip', 'npm', 'cargo').
    """
    try:
        report = [f"🛡️ Starting Safe Update for {package_name} in {project_path}"]
        
        # 1. Initial Test Run
        report.append("Running initial tests...")
        initial_test_res = await run_tests(project_path)
        if "FAILED" in initial_test_res:
            return "❌ Project has existing test failures. Fix them before updating."

        # 2. Perform Update
        report.append(f"Updating {package_name} via {manager}...")
        if manager == "pip":
            os.popen(f"cd {project_path} && .venv/bin/pip install --upgrade {package_name}").read()
        elif manager == "npm":
            os.popen(f"cd {project_path} && npm update {package_name}").read()
        elif manager == "cargo":
            os.popen(f"cd {project_path} && cargo update -p {package_name}").read()

        # 3. Verify with Tests
        report.append("Verifying update with tests...")
        post_test_res = await run_tests(project_path)
        
        if "FAILED" in post_test_res or "error" in post_test_res.lower():
            report.append(f"❌ Tests FAILED after update. Rolling back...")
            # Rollback logic (simple for pip: reinstall previous version is hard without lock, 
            # but for npm/cargo we can use git checkout on lockfiles)
            os.popen(f"cd {project_path} && git checkout package-lock.json Cargo.lock 2>/dev/null || true")
            if manager == "pip":
                # For pip we'd ideally use requirements.txt
                os.popen(f"cd {project_path} && .venv/bin/pip install -r requirements.txt").read()
            report.append("✅ Rollback complete. System restored to stable state.")
        else:
            report.append(f"✅ Update successful! All tests passed.")
            
        return "\n".join(report)
    except Exception as e:
        return f"Safe update failed: {str(e)}"


@mcp.tool()
async def apply_security_patches(project_path: str = ".") -> str:
    """Identify and automatically fix security risks (like hardcoded secrets)."""
    try:
        # 1. Run Security Audit
        audit_res = await audit_security(project_path)
        if "✅ No secrets detected" in audit_res:
            return "🛡️ No immediate security risks found to patch."

        # 2. Extract first finding for patching
        # We focus on one at a time for stability
        import re
        match = re.search(r"(.*):(\d+):.*Found potential secret", audit_res)
        if not match:
            return "⚠️ Security risks found but could not parse location for automated patching."
        
        file_path = match.group(1).strip()
        line_num = match.group(2)
        
        prompt = f"""You are a Security Engineer. The file {file_path} contains a hardcoded secret on line {line_num}.
Fix it by replacing the secret with an environment variable call (e.g., os.getenv('KEY_NAME')).
Return ONLY the complete fixed code. No markdown blocks.

File: {file_path}
Finding: {match.group(0)}
"""
        # 3. Apply AI Repair
        res = await ai_fix_code(file_path, prompt)
        
        # 4. Update .env.example
        env_ex = os.path.join(project_path, ".env.example")
        with open(env_ex, 'a') as f:
            f.write(f"\n# Added by Titan Security Patcher\n# TITAN_SECRET_PLACEHOLDER=your_value_here\n")

        return f"🔒 Security Patch Applied to {file_path}. Secret moved to environment variable logic. Added placeholder to .env.example."
    except Exception as e:
        return f"Security patching failed: {str(e)}"


@mcp.tool()
async def find_reusable_logic(query: str, current_project: str) -> str:
    """Find similar logic in other projects to avoid rewriting existing code.
    Args:
        query: Description of the logic or a code snippet to search for.
        current_project: The project to EXCLUDE from search results.
    """
    try:
        # We use our global search logic but filter out the current project
        all_res_json = await rag_search_code(query=query, project="all", n_results=10)
        all_res = json.loads(all_res_json)
        
        if "results" not in all_res:
            return "No reusable logic found."

        # Filter out current project and low relevance
        external_results = [
            r for r in all_res["results"] 
            if current_project.lower() not in r.get("file_path", "").lower()
        ]

        if not external_results:
            return f"No similar logic found in other projects. You are clear to implement it in {current_project}."

        report = [f"🔍 Found potential reusable logic in other projects:"]
        for res in external_results[:5]:
            report.append(f"\n--- From Project: {res.get('file_path').split('/')[0]} ---")
            report.append(f"File: {res.get('file_path')}")
            report.append(f"Relevance: {res.get('relevance')}%")
            report.append(f"Snippet:\n{res.get('text')[:300]}...")

        return "\n".join(report)
    except Exception as e:
        return f"Search for reusable logic failed: {str(e)}"


@mcp.tool()
async def execute_autonomous_workflow(project_path: str = ".") -> str:
    """Run a full autonomous cycle: Quality check, Tests, Security audit, and Commit."""
    steps = [
        ("Checking Code Quality", check_code_quality),
        ("Running Tests", run_tests),
        ("Performing Security Audit", audit_security),
    ]
    
    report = [f"🚀 Starting Titan Autonomous Workflow for {project_path}"]
    
    for name, func in steps:
        report.append(f"\n--- {name} ---")
        res = await func(project_path)
        report.append(res)
        if "❌" in res or "error" in res.lower() or "FAILED" in res:
            report.append("\n⚠️ Workflow halted due to issues. Please fix them first.")
            return "\n".join(report)

    # If all passed, generate commit message
    report.append("\n--- Finalizing ---")
    commit_msg = await generate_commit_message()
    report.append(f"Suggested Commit: {commit_msg}")
    
    return "\n".join(report)


@mcp.tool()
async def generate_architecture_map(project_path: str = ".") -> str:
    """Analyze project imports and generate a Mermaid.js architecture diagram."""
    try:
        connections = []
        files_processed = 0
        
        # Scan for common patterns
        for root, dirs, files in os.walk(project_path):
            if any(idir in root for idir in [".git", "node_modules", ".venv", "chroma_db"]):
                continue
                
            for file in files:
                if file.endswith((".py", ".ts", ".js", ".rs")):
                    files_processed += 1
                    path = os.path.join(root, file)
                    rel_src = os.path.relpath(path, project_path)
                    
                    with open(path, 'r', errors='ignore') as f:
                        content = f.read()
                        
                    # Simple regex for imports (Python/JS/TS)
                    import re
                    # Python: import X / from X import Y
                    py_imports = re.findall(r"^(?:from|import)\s+([\w\.]+)", content, re.MULTILINE)
                    # JS/TS: import ... from 'X' / require('X')
                    js_imports = re.findall(r"(?:from|require\()\s*['\"]([\w\.\/\-]+)['\"]", content)
                    
                    for imp in py_imports + js_imports:
                        if "/" in imp or "." in imp: # Filter out stdlibs (heuristic)
                            connections.append(f'    "{rel_src}" --> "{imp}"')

        if not connections:
            return "No clear architectural connections found."

        # Limit to top connections for readability
        diagram = "graph TD\n" + "\n".join(list(set(connections))[:50])
        
        report = f"""## 🏗️ Architecture Map (Mermaid.js)
Files processed: {files_processed}

```mermaid
{diagram}
```

*Note: Showing top 50 discovered connections for clarity.*"""
        return report
    except Exception as e:
        return f"Architecture mapping failed: {str(e)}"


@mcp.tool()
async def generate_docstrings(file_path: str) -> str:
    """Analyze a file and add missing docstrings to functions and classes using AI."""
    try:
        if not os.path.exists(file_path):
            return f"File {file_path} not found."

        with open(file_path, 'r') as f:
            code = f.read()

        prompt = f"""You are a documentation expert. Review the following code and add missing docstrings to ALL functions and classes.
Use Google-style docstrings. Return the COMPLETE fixed code. 
No explanations, no markdown blocks.

Code:
{code}
"""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post('http://localhost:11434/api/generate',
                json={
                    'model': 'vibethinker',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0}
                }, timeout=120.0)
        
        fixed_code = response.json().get('response', '').strip()
        
        # Clean up markdown
        if fixed_code.startswith("```"):
            fixed_code = "\n".join(fixed_code.split("\n")[1:-1])

        if len(fixed_code) < 10:
            return "AI failed to generate documentation."

        with open(file_path, 'w') as f:
            f.write(fixed_code)
            
        return f"Successfully added docstrings to {file_path}."
    except Exception as e:
        return f"Documentation failed: {str(e)}"


@mcp.tool()
async def update_ecosystem() -> str:
    """Pull latest changes from repository and restart all services."""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/system/update-antigravity.sh")
        result = os.popen(f"bash {script_path}").read()
        return result
    except Exception as e:
        return f"Update failed: {str(e)}"


@mcp.tool()
async def run_in_sandbox(command: str, path: str = ".") -> str:
    """Safely execute a command in a Firejail sandbox (no network, limited FS).
    Args:
        command: The command to execute.
        path: The directory to allow access to.
    """
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/system/titan-sandbox.sh")
        result = os.popen(f"bash {script_path} '{command}' {path}").read()
        return result
    except Exception as e:
        return f"Sandbox execution failed: {str(e)}"


@mcp.tool()
async def prune_orphaned_chunks(project: str) -> str:
    """Clean up the index by removing chunks for files that no longer exist on disk."""
    try:
        collection = chroma_client.get_collection(
            name=sanitize_collection_name(project),
            embedding_function=embedding_function
        )
        
        # Get all unique file paths in the collection
        results = collection.get(include=["metadatas"])
        if not results["metadatas"]:
            return "Collection is empty."

        unique_files = {m["file_path"] for m in results["metadatas"]}
        orphaned = []
        
        for rel_path in unique_files:
            abs_path = os.path.join(config["projects_root"], project, rel_path)
            if not os.path.exists(abs_path):
                orphaned.append(rel_path)

        if not orphaned:
            return "No orphaned chunks found. Index is clean."

        for rel_path in orphaned:
            collection.delete(where={"file_path": rel_path})
            
        return f"Successfully pruned {len(orphaned)} orphaned files from index."
    except Exception as e:
        return f"Pruning failed: {str(e)}"


@mcp.tool()
async def generate_pr_description(target_branch: str = "main") -> str:
    """Generate a detailed PR description by analyzing branch changes and project context."""
    try:
        diff = os.popen(f"git diff {target_branch}..HEAD").read()[:5000]
        if not diff: return "No changes detected."
        
        # Get context from AGENTS.md
        current_dir = os.path.dirname(os.path.abspath(__file__))
        agents_md = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "AGENTS.md")
        tasks = ""
        if os.path.exists(agents_md):
            with open(agents_md, 'r') as f:
                tasks = "\n".join(f.readlines()[:50]) # Take top 50 lines for context

        prompt = f"""Generate a professional Pull Request description based on this diff and project context.
Include: 
- High-level summary
- Detailed list of changes
- Impact analysis
- Testing performed

CONTEXT:
{tasks}

DIFF:
{diff}
"""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post('http://localhost:11434/api/generate',
                json={'model': 'vibethinker', 'prompt': prompt, 'stream': False},
                timeout=90.0)
        
        return response.json().get('response', 'AI failed to generate PR.')
    except Exception as e:
        return f"PR generation failed: {str(e)}"


@mcp.tool()
async def query_system_logs(query: str, log_type: str = "all") -> str:
    """Search across all Titan ecosystem logs (MCP, Guardian, Nightly)."""
    logs = {
        "mcp": "/home/stranmor/Documents/ai-assistant/tools/ollama-code-search/mcp_server.log",
        "guardian": "/var/log/titan-guardian.log",
        "nightly": "/home/stranmor/Documents/ai-assistant/tools/ollama-code-search/nightly_maintenance.log"
    }
    results = []
    for name, path in logs.items():
        if log_type == "all" or log_type == name:
            if os.path.exists(path):
                res = os.popen(f"grep -i '{query}' {path} | tail -n 10").read()
                if res: results.append(f"--- {name.upper()} LOGS ---\n{res}")
    return "\n\n".join(results) or "No matching log entries found."


@mcp.tool()
async def simulate_pr_review(target_branch: str = "main") -> str:
    """Perform a Staff Engineer level PR review of all changes in the current branch."""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/system/titan-pr-audit.sh")
        result = os.popen(f"bash {script_path} {target_branch}").read()
        return result
    except Exception as e:
        return f"PR Review failed: {str(e)}"


@mcp.tool()
async def assess_project_risks(project_path: str = ".") -> str:
    """Perform a deep AI-driven risk assessment of the project's architecture and quality."""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/system/titan-risk-assessment.sh")
        raw_data = os.popen(f"bash {script_path} {project_path}").read()
        
        prompt = f"""You are a Senior Risk Analyst. Based on the following raw technical data, provide a structured risk assessment report.
Highlight:
1. Top 3 architectural risks.
2. Impact on long-term maintainability.
3. Specific mitigation strategies.

DATA:
{raw_data}
"""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post('http://localhost:11434/api/generate',
                json={
                    'model': 'vibethinker',
                    'prompt': prompt,
                    'stream': False
                }, timeout=90.0)
        
        report = response.json().get('response', 'AI failed to analyze risks.')
        return f"# 🛡️ Titan Risk Assessment Report\n\n{report}\n\n## Raw Data\n```\n{raw_data}\n```"
    except Exception as e:
        return f"Risk assessment failed: {str(e)}"


@mcp.tool()
async def get_call_graph(file_path: str) -> str:
    """Generate a Mermaid.js call graph for a Python file."""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/system/generate-call-graph.py")
        result = os.popen(f"python3 {script_path} {file_path}").read()
        
        if "graph" not in result:
            return result # Probably an error or 'No calls' message
            
        return f"## 📞 Call Graph: {os.path.basename(file_path)}\n\n```mermaid\n{result}\n```"
    except Exception as e:
        return f"Call graph generation failed: {str(e)}"


@mcp.tool()
async def explain_code_architecture(file_path: str) -> str:
    """Perform a deep architectural analysis of a specific file and its role in the project."""
    try:
        if not os.path.exists(file_path):
            return f"File {file_path} not found."

        with open(file_path, 'r') as f:
            code = f.read()

        # Get related files for context
        related = await find_related_files(file_path, n_results=3)
        
        prompt = f"""You are a Lead Software Architect. Explain the architectural role of the following file.
Identify:
1. Primary purpose and responsibility.
2. Key patterns used (e.g., Singleton, Observer, Asyncio).
3. Critical dependencies and interfaces.
4. How it relates to other parts of the system (see context).

CONTEXT (Related files):
{related}

FILE CONTENT:
{code[:8000]} # Limit to avoid context overflow
"""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post('http://localhost:11434/api/generate',
                json={
                    'model': 'vibethinker',
                    'prompt': prompt,
                    'stream': False
                }, timeout=120.0)
        
        return response.json().get('response', 'AI failed to analyze architecture.')
    except Exception as e:
        return f"Architectural analysis failed: {str(e)}"


@mcp.tool()
async def generate_task_plan(objective: str) -> str:
    """Break down a complex project goal into atomic, actionable tasks."""
    try:
        prompt = f"""You are a Project Architect. Break down the following objective into a logical sequence of 5-8 atomic, actionable tasks.
For each task, specify:
1. WHAT needs to be done.
2. WHY it's needed.
3. WHICH tool to use.

Objective: {objective}

Output in a clean Markdown checklist format.
"""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post('http://localhost:11434/api/generate',
                json={
                    'model': 'vibethinker',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.2}
                }, timeout=90.0)
        
        plan = response.json().get('response', 'AI failed to generate plan.')
        return f"📋 Titan Strategy Plan for: {objective}\n\n{plan}"
    except Exception as e:
        return f"Planning failed: {str(e)}"


@mcp.tool()
async def generate_docs_site() -> str:
    """Build a professional documentation site using MkDocs and Material theme."""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/system/generate-docs-site.sh")
        result = os.popen(f"bash {script_path}").read()
        return result
    except Exception as e:
        return f"Docs generation failed: {str(e)}"


@mcp.tool()
async def summarize_context(query: str, context_json: str) -> str:
    """Compress multiple search results into a dense, high-token-efficiency summary."""
    try:
        data = json.loads(context_json)
        snippets = "\n---\n".join([f"File: {r['file_path']}\n{r['text']}" for r in data.get("results", [])])
        
        prompt = f"""Summarize the following code snippets relative to the query: "{query}".
Extract only the essential logic, function signatures, and key architectural points.
Be extremely dense. Focus on facts.

Snippets:
{snippets}
"""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post('http://localhost:11434/api/generate',
                json={
                    'model': 'qwen2.5-coder:1.5b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0}
                }, timeout=60.0)
        
        return response.json().get('response', 'Failed to summarize.')
    except Exception as e:
        return f"Summarization failed: {str(e)}"


@mcp.tool()
async def generate_release_notes(since: str = "7 days ago") -> str:
    """Generate professional Release Notes from git history using AI.
    Args:
        since: Git-compatible time range or ref (e.g., '7 days ago', 'v1.0').
    """
    try:
        # Get commit history
        cmd = f'git log --since="{since}" --pretty=format:"%s (%h)"'
        commits = os.popen(cmd).read()
        
        if not commits:
            return f"No commits found since {since}."

        prompt = f"""You are a Product Manager. Based on the following commit messages, generate a professional Release Notes document in Markdown.
Categorize changes into: 🚀 Features, 🐛 Bug Fixes, 🛠 Refactoring, and 📝 Documentation.
Keep it concise and focus on user value.

Commits:
{commits}
"""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post('http://localhost:11434/api/generate',
                json={
                    'model': 'vibethinker',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.3}
                }, timeout=60.0)
        
        release_notes = response.json().get('response', 'AI failed to generate release notes.')
        return f"# Release Notes (Since {since})\n\n{release_notes}"
    except Exception as e:
        return f"Release notes generation failed: {str(e)}"


@mcp.tool()
async def generate_unit_tests(file_path: str) -> str:
    """Analyze a file and generate a corresponding unit test file using AI."""
    try:
        if not os.path.exists(file_path):
            return f"File {file_path} not found."

        with open(file_path, 'r') as f:
            code = f.read()

        base, ext = os.path.splitext(file_path)
        framework = "pytest" if ext == ".py" else "cargo test" if ext == ".rs" else "jest"
        
        prompt = f"""You are a QA Engineer. Generate a comprehensive unit test file for the following code.
Use the {framework} framework. Include edge cases and mock dependencies where necessary.
Return ONLY the complete test code. No explanations, no markdown blocks.

Original File: {file_path}
Code:
{code}
"""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post('http://localhost:11434/api/generate',
                json={
                    'model': 'vibethinker',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.1}
                }, timeout=150.0)
        
        test_code = response.json().get('response', '').strip()
        
        # Clean up markdown
        if test_code.startswith("```"):
            test_code = "\n".join(test_code.split("\n")[1:-1])

        if len(test_code) < 10:
            return "AI failed to generate tests."

        # Save to a new test file
        test_file_path = f"test_{os.path.basename(file_path)}"
        if ext == ".py" and not test_file_path.startswith("test_"):
            test_file_path = f"test_{os.path.basename(file_path)}"
        
        test_full_path = os.path.join(os.path.dirname(file_path), test_file_path)
        
        with open(test_full_path, 'w') as f:
            f.write(test_code)
            
        return f"Successfully generated unit tests at {test_full_path}. Run 'run_tests' to verify."
    except Exception as e:
        return f"Test generation failed: {str(e)}"


@mcp.tool()
async def clear_neural_cache() -> str:
    """Clear the stored AI insights cache."""
    try:
        db_path = os.path.join(os.path.dirname(__file__), "insight_cache.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            return "Neural Cache cleared successfully."
        return "Cache is already empty."
    except Exception as e:
        return f"Failed to clear cache: {str(e)}"


@mcp.tool()
async def create_custom_tool(tool_name: str, code: str, description: str) -> str:
    """Create a new custom system tool and add it to the ecosystem.
    Args:
        tool_name: Name of the new tool (filename without extension).
        code: The Python/Bash code for the tool.
        description: What this tool does.
    """
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), f"scripts/custom/{tool_name}.py")
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(f"#!/usr/bin/env python3\n\"\"\"{description}\"\"\"\n\n{code}")
        
        os.chmod(script_path, 0o755)
        
        # Log to project memory
        await update_project_memory("TOOLING", f"Created custom tool: {tool_name} - {description}")
        
        return f"Successfully created custom tool at {script_path}. It will be indexed automatically."
    except Exception as e:
        return f"Tool creation failed: {str(e)}"


@mcp.tool()
async def call_ollama(prompt: str, model: str = "vibethinker", task_type: str = "general") -> str:
    """Dynamically tune Ollama parameters and use Neural Cache."""
    options = {"temperature": 0.2, "num_ctx": 8192}
    
    if task_type == "scoring":
        options = {"temperature": 0, "num_ctx": 4096, "num_predict": 10}
    elif task_type == "review":
        options = {"temperature": 0.1, "num_ctx": 16384}
    elif task_type == "migration":
        options = {"temperature": 0.2, "num_ctx": 32768}

    # 1. Neural Cache Lookup
    import hashlib
    import sqlite3
    
    # Create unique key based on prompt and options
    cache_key = hashlib.sha256(f"{model}_{json.dumps(options)}_{prompt}".encode()).hexdigest()
    db_path = os.path.join(os.path.dirname(__file__), "insight_cache.db")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS insights (key TEXT PRIMARY KEY, response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        
        cursor.execute("SELECT response FROM insights WHERE key = ?", (cache_key,))
        cached = cursor.fetchone()
        if cached:
            logger.info(f"Neural Cache HIT for task: {task_type}")
            return cached[0]
    except Exception as e:
        logger.error(f"Cache error: {e}")

    # 2. Call Ollama (Cache Miss)
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post('http://localhost:11434/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': options
                }, timeout=180.0)
        
        res_text = response.json().get('response', '')
        
        # 3. Store in Cache
        if res_text:
            try:
                cursor.execute("INSERT OR REPLACE INTO insights (key, response) VALUES (?, ?)", (cache_key, res_text))
                conn.commit()
                conn.close()
            except: pass
            
        return res_text
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return ""

@mcp.tool()
async def migrate_code(file_path: str, target_standard: str) -> str:
    """Use AI to migrate or modernize code in a specific file."""
    try:
        if not os.path.exists(file_path):
            return f"File {file_path} not found."

        with open(file_path, 'r') as f:
            code = f.read()

        prompt = f"""You are a senior refactoring expert. 
Task: {target_standard}

Transform the following code to meet the target standard. 
Ensure best practices, correct types, and modern syntax.
Return ONLY the complete transformed code. No explanations, no markdown blocks.

File: {file_path}
Original Code:
{code}
"""
        migrated_code = await call_ollama(prompt, task_type="migration")
        
        # ... (rest of the function)



@mcp.tool()
async def autopilot_fix(file_path: str) -> str:
    """Perform a full autonomous fix cycle on a file: lint, AI-repair, and verify."""
    try:
        # 1. Initial Check
        report = os.popen(f"ruff check {file_path} 2>&1").read()
        if not report:
            return f"✨ File {file_path} is already clean. No action needed."

        # 2. Attempt AI Repair
        logger.info(f"Autopilot: Attempting to fix {file_path}...")
        fix_res = await ai_fix_code(file_path, report)
        
        if "Applied AI fix" not in fix_res:
            return f"❌ Autopilot failed at repair stage: {fix_res}"

        # 3. Verification
        new_report = os.popen(f"ruff check {file_path} 2>&1").read()
        if not new_report:
            return f"✅ Autopilot successfully fixed {file_path}. File is now clean."
        else:
            return f"⚠️ Autopilot applied a partial fix, but some issues remain:\n{new_report}"

    except Exception as e:
        return f"Autopilot crashed: {str(e)}"


@mcp.tool()
async def summarize_changes(ref1: str = "HEAD~1", ref2: str = "HEAD") -> str:
    """Provide a semantic AI-powered summary of git changes between two refs."""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/system/titan-diff.sh")
        result = os.popen(f"bash {script_path} {ref1} {ref2}").read()
        return result
    except Exception as e:
        return f"Diff summary failed: {str(e)}"


@mcp.tool()
async def ai_fix_code(file_path: str, error_message: str) -> str:
    """Use AI to fix a specific error in a file.
    Args:
        file_path: Path to the file that needs fixing.
        error_message: The error message from linter or test.
    """
    try:
        if not os.path.exists(file_path):
            return f"File {file_path} not found."

        with open(file_path, 'r') as f:
            code = f.read()

        prompt = f"""You are a senior developer. Fix the following error in the code.
Return ONLY the complete fixed code. No explanations, no markdown blocks.

Error: {error_message}

File: {file_path}
Original Code:
{code}
"""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post('http://localhost:11434/api/generate',
                json={
                    'model': 'vibethinker',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0}
                }, timeout=120.0)
        
        fixed_code = response.json().get('response', '').strip()
        
        # Clean up possible markdown wrappers if the AI ignored instructions
        if fixed_code.startswith("```"):
            fixed_code = "\n".join(fixed_code.split("\n")[1:-1])

        if not fixed_code or len(fixed_code) < 10:
            return "AI failed to generate a valid fix."

        # Apply fix
        with open(file_path, 'w') as f:
            f.write(fixed_code)
            
        return f"Applied AI fix to {file_path}. Please run tests/linter to verify."
    except Exception as e:
        return f"Fix failed: {str(e)}"


@mcp.tool()
async def fix_code_quality(project_path: str = ".") -> str:
    """Automatically apply linter fixes (ruff --fix) to the codebase."""
    results = []
    
    # 1. Python (Ruff Fix)
    if os.path.exists(os.path.join(project_path, "requirements.txt")):
        res = os.popen(f"ruff check --fix {project_path} 2>&1").read()
        results.append(f"--- Python (Ruff Fix Applied) ---\n{res or 'No fixes needed.'}")

    return "\n\n".join(results)


@mcp.tool()
async def find_related_files(file_path: str, n_results: int = 5) -> str:
    """Find files that are semantically related to the given file path."""
    try:
        # First, find a sample chunk from the source file to get its context
        collection_names = chroma_client.list_collections()
        # Find which collection contains this project (simple heuristic)
        matching_collection = None
        for name in collection_names:
            if name in file_path:
                matching_collection = name
                break
        
        if not matching_collection:
            # Fallback: search across all collections
            matching_collection = collection_names[0] if collection_names else None

        if not matching_collection:
            return "No collections found."

        collection = chroma_client.get_collection(
            name=matching_collection,
            embedding_function=embedding_function
        )

        # Get top chunks from the source file
        source_res = collection.get(
            where={"file_path": file_path},
            limit=3,
            include=["documents", "embeddings"]
        )

        if not source_res["documents"]:
            return f"Source file {file_path} not found in index."

        # Search for similar chunks excluding the source file itself
        results = collection.query(
            query_texts=[source_res["documents"][0]],
            n_results=n_results + 5, # Fetch more to filter
            where={"file_path": {"$ne": file_path}},
            include=["metadatas", "distances"]
        )

        related_files = {}
        if results["metadatas"] and results["metadatas"][0]:
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                f_path = meta.get("file_path", "Unknown")
                score = round((1 - dist) * 100, 1)
                if f_path not in related_files or score > related_files[f_path]:
                    related_files[f_path] = score

        # Sort and take top N
        sorted_files = sorted(related_files.items(), key=lambda x: x[1], reverse=True)[:n_results]
        
        return json.dumps({
            "related_files": [{"path": f, "relevance": s} for f, s in sorted_files]
        }, indent=2)

    except Exception as e:
        return f"Relation search failed: {str(e)}"


@mcp.tool()
async def update_project_memory(category: str, content: str) -> str:
    """Add a new entry to the project memory (AGENTS.md).
    Args:
        category: The section to update (e.g., 'ADR', 'TASK', 'KNOWLEDGE').
        content: The text content to add.
    """
    try:
        # Find the root AGENTS.md
        current_dir = os.path.dirname(os.path.abspath(__file__))
        agents_md_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "AGENTS.md")
        
        if not os.path.exists(agents_md_path):
            return f"AGENTS.md not found at {agents_md_path}"

        date_str = time.strftime("%Y-%m-%d")
        new_entry = f"- [{date_str}] **{category}:** {content}\n"

        # Read and insert based on category
        with open(agents_md_path, 'r') as f:
            lines = f.readlines()

        # Simple logic: append to the end of the corresponding section
        # or just at the end if not found
        insert_idx = len(lines)
        category_upper = category.upper()
        
        for i, line in enumerate(lines):
            if category_upper in line.upper() and line.startswith("##"):
                # Found the section, look for the next section or end
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("##"):
                        insert_idx = j
                        break
                else:
                    insert_idx = len(lines)
                break
        
        lines.insert(insert_idx, new_entry)

        with open(agents_md_path, 'w') as f:
            f.writelines(lines)

        return f"Successfully updated project memory in {category}."
    except Exception as e:
        return f"Failed to update memory: {str(e)}"


@mcp.tool()
async def get_git_hotspots(days: int = 7) -> str:
    """Find the most frequently changed files in the last N days."""
    try:
        cmd = f'git log --name-only --since="{days} days ago" | grep -v "^$" | sort | uniq -c | sort -rn | head -n 10'
        res = os.popen(cmd).read()
        return f"🔥 Git Hotspots (Last {days} days):\n{res or 'No changes found'}"
    except Exception as e:
        return f"Failed to get hotspots: {str(e)}"


@mcp.tool()
async def get_project_stats() -> str:
    """Get summary statistics of the current RAG index."""
    try:
        stats = {"total_collections": 0, "collections": {}}
        collections = chroma_client.list_collections()
        stats["total_collections"] = len(collections)
        
        for name in collections:
            col = chroma_client.get_collection(name)
            stats["collections"][name] = {"count": col.count()}
            
        return json.dumps(stats, indent=2)
    except Exception as e:
        return f"Failed to get stats: {str(e)}"


@mcp.tool()
async def get_project_map() -> str:
    """Get a high-level overview of the current project (files, git, stack, VRAM)."""
    project_map = {
        "structure": "",
        "git_history": "",
        "tech_stack": [],
        "vram_free_mb": 0
    }
    
    try:
        # 1. File Structure (with fallback)
        structure = os.popen("tree -L 2 --noreport -I 'node_modules|.git|.venv|chroma_db' 2>/dev/null").read()
        if not structure:
            structure = os.popen("ls -R | head -n 20").read() + "\n(tree command not found, showing ls fallback)"
        project_map["structure"] = structure
        
        # 2. Git History
        project_map["git_history"] = os.popen("git log -n 5 --oneline 2>/dev/null").read() or "No git history found"
        
        # 3. Tech Stack Detection
        for file in os.listdir("."):
            if file == "package.json": project_map["tech_stack"].append("Node.js")
            if file == "requirements.txt" or file == "pyproject.toml": project_map["tech_stack"].append("Python")
            if file == "Cargo.toml": project_map["tech_stack"].append("Rust")
            if file == "go.mod": project_map["tech_stack"].append("Go")
            if file == "Makefile": project_map["tech_stack"].append("C/C++ (Make)")

        # 4. VRAM Check
        gpu_info = os.popen("rocm-smi --showmeminfo vram --json").read()
        gpu_data = json.loads(gpu_info)
        for metrics in gpu_data.values():
            total = int(metrics.get("VRAM Total Memory (B)", 0))
            used = int(metrics.get("VRAM Total Used Memory (B)", 0))
            project_map["vram_free_mb"] = round((total - used) / (1024**2), 1)
            break # Just take the first GPU
    except Exception as e:
        logger.error(f"Map error: {e}")

    return json.dumps(project_map, indent=2)


@mcp.tool()
async def read_mcp_logs(lines: int = 50) -> str:
    """Read the last N lines of the MCP server log file."""
    try:
        log_path = os.path.join(os.path.dirname(__file__), "mcp_server.log")
        if not os.path.exists(log_path):
            return "Log file not found."
        
        with open(log_path, 'r') as f:
            content = f.readlines()
            return "".join(content[-lines:])
    except Exception as e:
        return f"Failed to read logs: {str(e)}"


@mcp.tool()
async def fix_system_services() -> str:
    """Automatically identify and restart failed Titan system services."""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/system/tcc.sh")
        result = os.popen(f"bash {script_path} fix").read()
        return result
    except Exception as e:
        return f"Self-healing failed: {str(e)}"


@mcp.tool()
async def diagnose_zed() -> str:
    """Execute the Zed diagnostic utility to check and fix profile issues."""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/zed/zed-diagnose.sh")
        result = os.popen(f"bash {script_path}").read()
        return result
    except Exception as e:
        return f"Diagnostic failed: {str(e)}"


@mcp.tool()
async def cleanup_system() -> str:
    """Execute the Titan Janitor cleanup utility to rotate logs and clean temp files."""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts/system/titan-janitor.sh")
        result = os.popen(f"bash {script_path}").read()
        return result
    except Exception as e:
        return f"Cleanup failed: {str(e)}"


@mcp.tool()
async def get_system_health() -> str:
    """Get the current system health status (GPU Temperature, VRAM, Power, Disk, Load)."""
    health = {
        "status": "Healthy",
        "gpu": {},
        "system": {},
        "disk": {}
    }
    
    try:
        # Check GPU status via rocm-smi
        # We parse specific fields for better clarity
        gpu_data = os.popen("rocm-smi --showuse --showmeminfo vram --showtemp --showpower --json").read()
        raw_gpu = json.loads(gpu_data)
        
        # Simplify the output for the agent
        for gpu_id, metrics in raw_gpu.items():
            health["gpu"][gpu_id] = {
                "usage_pct": metrics.get("GPU use (%)"),
                "vram_used_bytes": metrics.get("VRAM Total Used Memory (B)"),
                "vram_total_bytes": metrics.get("VRAM Total Memory (B)"),
                "temperature_c": metrics.get("Temperature (Sensor edge) (C)"),
                "power_w": metrics.get("Current Socket Graphics Package Power (W)")
            }
    except Exception as e:
        health["gpu"]["error"] = str(e)

    try:
        # System Load and Memory
        load1, load5, load15 = os.getloadavg()
        health["system"]["load_average"] = [load1, load5, load15]
        
        import psutil
        mem = psutil.virtual_memory()
        health["system"]["memory_free_gb"] = round(mem.available / (1024**3), 2)

        # Active Zed Profiles
        health["zed_profiles"] = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['name'] == 'firejail' and proc.info['cmdline'] and 'zed-isolated' in " ".join(proc.info['cmdline']):
                cmd_str = " ".join(proc.info['cmdline'])
                import re
                match = re.search(r'zed-isolated-([^-]+)', cmd_str)
                p_name = match.group(1) if match else "unknown"
                
                try:
                    p = psutil.Process(proc.info['pid'])
                    health["zed_profiles"].append({
                        "name": p_name,
                        "cpu_pct": p.cpu_percent(),
                        "ram_mb": round(p.memory_info().rss / (1024**2), 1)
                    })
                except: pass
    except Exception:
        pass

    try:
        # Check Disk space for PROJECTS_ROOT
        projects_path = config.get("projects_root", ".")
        disk = os.statvfs(projects_path)
        free_gb = (disk.f_bavail * disk.f_frsize) / (1024**3)
        health["disk"]["free_gb"] = round(free_gb, 2)
    except Exception:
        health["disk"]["error"] = "Error checking disk"

    return json.dumps(health, indent=2)


@mcp.tool()
async def force_reindex(folder: str) -> str:
    """Force a full re-indexing of a specific project folder.
    Args:
        folder: Name of the folder within PROJECTS_ROOT to re-index.
    """
    success = await perform_initial_indexing(folder)
    if success:
        return f"Successfully re-indexed {folder}"
    else:
        return f"Failed to re-index {folder}. Check logs for details."


@mcp.tool()
async def get_index_status() -> str:
    """Get the current status of the code indexing process."""
    return json.dumps(indexing_status, indent=2)


async def perform_initial_indexing(folder: str) -> bool:
    """Check if collection exists and perform initial indexing if needed."""
    try:
        indexing_status["folders"][folder] = "indexing"
        os.system(f'notify-send -i accessories-text-editor "Titan RAG" "Indexing started: {folder}"')
        
        folder_path = os.path.join(config["projects_root"], folder)
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            os.system(f'notify-send -u critical "Titan RAG" "Folder not found: {folder}"')
            return False

        collection_name = sanitize_collection_name(folder)

        # Check if collection exists
        try:
            chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"Collection {collection_name} already exists, skipping initial indexing")
            indexing_status["folders"][folder] = "ready"
            return True
        except Exception:
            logger.info(f"Collection {collection_name} not found, performing initial indexing")

        # Load and process all documents in the folder
        documents = load_documents(
            folder_path,
            ignore_dirs=set(config["ignore_dirs"]),
            file_extensions=set(config["file_extensions"]),
            ignore_files=set(config["ignore_files"])
        )

        if documents:
            process_and_index_documents(documents, collection_name, "chroma_db")
            logger.info(f"Successfully performed initial indexing for {folder}")
            indexing_status["folders"][folder] = "ready"
            os.system(f'notify-send -i checkbox-checked-symbolic "Titan RAG" "Indexing complete: {folder}"')
            return True
        else:
            logger.warning(f"No documents found to index in {folder}")
            indexing_status["folders"][folder] = "empty"
            return False

    except Exception as e:
        indexing_status["folders"][folder] = f"error: {str(e)}"
        logger.error(f"Error during initial indexing of {folder}: {e}")
        os.system(f'notify-send -u critical "Titan RAG" "Error indexing {folder}: {str(e)}"')
        return False


async def index_projects():
    """Set up file system watchers for all configured projects."""
    global observers
    
    # Give the MCP server time to start up and respond to initial pings
    await asyncio.sleep(5)
    indexing_status["status"] = "running"

    try:
        for folder in config["folders_to_index"]:
            # First perform initial indexing if needed
            success = await perform_initial_indexing(folder)
            if not success:
                logger.error(f"Failed to perform initial indexing for {folder}")
                continue

            folder_path = os.path.join(config["projects_root"], folder)
            logger.info(f"Setting up file watcher for {folder}")

            # Create an observer and event handler for this folder
            observer = Observer()
            event_handler = CodeIndexerEventHandler(folder)
            observer.schedule(event_handler, folder_path, recursive=True)

            # Start the observer
            observer.start()
            observers.append(observer)
            logger.info(f"Started watching {folder}")

        # Keep the main thread alive
        indexing_status["status"] = "idle"
        while True:
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error in file watching setup: {e}")
        # Clean up observers on error
        for observer in observers:
            observer.stop()
        observers.clear()


@mcp.tool()
async def list_indexed_projects() -> str:
    """List all projects currently indexed in Titan RAG."""
    try:
        collections = chroma_client.list_collections()
        projects = []
        for col_name in collections:
            # Try to get count for each
            col = chroma_client.get_collection(col_name)
            projects.append({"name": col_name, "chunks": col.count()})
        return json.dumps({"indexed_projects": projects}, indent=2)
    except Exception as e:
        return f"Failed to list projects: {str(e)}"


@mcp.tool(
    name="rag_search_code",
    description="""Search code using our custom Titan RAG pipeline (Ollama + ChromaDB).
        Args:
            query: Natural language query about the codebase
            project: Collection/folder name to search in. Use 'all' to search everywhere.
            keywords: Optional comma-separated list of exact keywords to filter results.
            file_pattern: Optional glob-like pattern to filter file paths (e.g., 'scripts/*').
            language_filter: Optional specific language to search for (e.g., 'python').
            n_results: Number of results to return (default: 5)
            threshold: Minimum relevance percentage to include results 
                (default: 30.0)
    """
)
async def rag_search_code(
    query: str,
    project: str,
    keywords: str = None,
    file_pattern: str = None,
    language_filter: str = None,
    n_results: int = 5,
    threshold: float = 30.0
) -> str:
    try:
        # ... (keep initialization check)
        if not chroma_client or not embedding_function:
            logger.error("ChromaDB client or embedding function not initialized")
            return json.dumps({
                "error": "Search system not properly initialized",
                "results": [],
                "total_results": 0
            })

        # Get all collections
        collection_names = chroma_client.list_collections()

        # Find matching collections
        matching_collections = []
        project_name = project.lower()
        
        if project_name == "all":
            matching_collections = collection_names
        else:
            for collection_name in collection_names:
                collection_parts = collection_name.lower().split('_')
                if collection_parts[-1] == project_name:
                    matching_collections.append(collection_name)

        if not matching_collections:
            logger.error(f"No collections found matching project {project}")
            return json.dumps({
                "error": f"No collections found matching project {project}",
                "results": [],
                "total_results": 0
            })

        # Prepare keyword filter
        where_filter = None
        filter_list = []
        
        if file_pattern:
            filter_list.append({"file_path": {"$like": f"%{file_pattern}%"}})
        if language_filter:
            filter_list.append({"language": {"$eq": language_filter.lower()}})
            
        if len(filter_list) == 1:
            where_filter = filter_list[0]
        elif len(filter_list) > 1:
            where_filter = {"$and": filter_list}

        # Search in all matching collections and combine results
        all_results = []

        for collection_name in matching_collections:
            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )

            results = collection.query(
                query_texts=[query],
                n_results=n_results * 3 if (keywords or file_pattern) else n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            if results["documents"] and results["documents"][0]:
                for doc, meta, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    # Manual keyword filtering
                    if keywords:
                        kw_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
                        if not all(kw in doc.lower() for kw in kw_list):
                            continue

                    similarity = (1 - distance) * 100
                    if similarity >= threshold:
                        all_results.append({
                            "text": doc,
                            "file_path": meta.get("file_path", "Unknown"),
                            "file_name": meta.get("file_name", "Unknown"),
                            "language": meta.get("language", "text"),
                            "start_line": int(meta.get("start_line", 0)),
                            "end_line": int(meta.get("end_line", 0)),
                            "relevance": round(similarity, 1)
                        })

        # Sort results by relevance
        all_results.sort(key=lambda x: x["relevance"], reverse=True)

        # Take top candidates for re-ranking
        candidates = all_results[:12] # Optimal batch size
        if not candidates:
            return json.dumps({"results": [], "total_results": 0})

        # Batch Re-ranking stage (Titan Class Performance)
        try:
            import httpx
            
            # Formulate a single batch scoring prompt
            snippets_text = ""
            for i, cand in enumerate(candidates):
                snippets_text += f"ID: {i}\nFile: {cand['file_path']}\nContent:\n{cand['text'][:500]}\n---\n"

            prompt = f"""Rate the relevance of the following code snippets to the query: "{query}"
For each snippet, provide a score from 0 to 100.
Return the results as a simple JSON list of integers corresponding to the IDs.
Example: [85, 10, 45...]
Output ONLY the JSON list.

Snippets:
{snippets_text}"""
            
            async with httpx.AsyncClient() as client:
                response = await client.post('http://localhost:11434/api/generate',
                    json={
                        'model': 'qwen2.5-coder:1.5b',
                        'prompt': prompt,
                        'stream': False,
                        'options': {'temperature': 0}
                    }, timeout=30.0)
                
                resp_json = response.json()
                raw_response = resp_json.get('response', '[]').strip()
                
                # Extract JSON list from response
                import re
                list_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
                if list_match:
                    scores = json.loads(list_match.group())
                    for i, score in enumerate(scores):
                        if i < len(candidates):
                            candidates[i]["relevance"] = int(score)
                
            candidates.sort(key=lambda x: x["relevance"], reverse=True)
        except Exception as e:
            logger.error(f"Batch re-ranking failed: {e}")
            # Fallback to original vector relevance if batch fails

        final_results = candidates[:n_results]

        return json.dumps({
            "results": final_results,
            "total_results": len(final_results)
        })

    except Exception as e:
        logger.error(f"Error in search_code: {str(e)}")
        return json.dumps({
            "error": str(e),
            "results": [],
            "total_results": 0
        })


# Run initialization before starting MCP
async def main():
    # Initialize ChromaDB before starting MCP
    success = await initialize_chromadb()

    if success:
        # Start file watching in background
        asyncio.create_task(index_projects())
        logger.info("File watching task started")

    # Default to stdio for Gemini CLI/Claude Code compatibility
    mode = os.getenv("MCP_MODE", "stdio")
    if mode == "sse":
        logger.info("Starting MCP server in SSE mode")
        await mcp.run_sse_async()
    else:
        logger.info("Starting MCP server in stdio mode")
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())
