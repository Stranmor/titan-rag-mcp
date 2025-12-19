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
indexing_status = {"status": "starting", "folders": {}}


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
    for doc in documents:
        try:
            file_path = doc.metadata.get("file_path", "unknown")
            file_name = os.path.basename(file_path)

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
        
        import requests
        response = requests.post('http://localhost:11434/api/generate',
            json={
                'model': 'vibethinker',
                'prompt': prompt,
                'stream': False
            }, timeout=30)
        
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
"""
        
        import requests
        response = requests.post('http://localhost:11434/api/generate',
            json={
                'model': 'vibethinker',
                'prompt': prompt,
                'stream': False
            }, timeout=90)
        
        report = response.json().get('response', 'AI failed to generate report.')
        
        # Save report to a file for Dashboard
        report_path = os.path.join(os.path.dirname(__file__), "latest_report.md")
        with open(report_path, 'w') as f:
            f.write(f"# Titan Intelligence Report ({time.strftime('%Y-%m-%d %H:%M')})\n\n{report}")
            
        return report
    except Exception as e:
        return f"Report generation failed: {str(e)}"


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
        
        import requests
        response = requests.post('http://localhost:11434/api/generate',
            json={
                'model': 'vibethinker',
                'prompt': prompt,
                'stream': False
            }, timeout=60)
        
        return response.json().get('response', 'AI failed to generate review.')
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
        candidates = all_results[:15]
        if not candidates:
            return json.dumps({"results": [], "total_results": 0})

        # Re-ranking stage (Titan Class Innovation)
        # We ask the LLM to score the candidates
        reranked_results = []
        for cand in candidates:
            try:
                # Prepare a scoring prompt
                prompt = f"Query: {query}\n\nCode Snippet from {cand['file_path']}:\n{cand['text']}\n\nTask: Rate the relevance of this snippet to the query from 0 to 100. Output ONLY the number."
                
                response = requests.post('http://localhost:11434/api/generate',
                    json={
                        'model': 'qwen2.5-coder:1.5b',
                        'prompt': prompt,
                        'stream': False,
                        'options': {'num_predict': 5, 'temperature': 0}
                    }, timeout=10)
                
                score_str = response.json().get('response', '0').strip()
                # Extract digits only
                import re
                score_match = re.search(r'\d+', score_str)
                score = int(score_match.group()) if score_match else 0
                
                cand["relevance"] = score # Override with LLM score
                reranked_results.append(cand)
            except:
                reranked_results.append(cand) # Fallback to original score

        # Sort by new scores and take top n_results
        reranked_results.sort(key=lambda x: x["relevance"], reverse=True)
        final_results = reranked_results[:n_results]

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
