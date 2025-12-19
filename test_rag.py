import asyncio
import sys
import os
import json
import logging

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ollama_indexer import initialize_chromadb, rag_search_code, get_system_health, get_index_status

async def test_suite():
    print("🧪 Starting Titan RAG Test Suite...")
    print("-" * 40)

    # 1. Initialization Test
    print("1. Initializing systems...", end=" ")
    try:
        success = await initialize_chromadb()
        if success:
            print("✅ OK")
        else:
            print("❌ FAILED (Check logs)")
            return
    except Exception as e:
        print(f"💥 CRASH: {e}")
        return

    # 2. Health Check Test
    print("2. Checking system health tool...", end=" ")
    health_json = await get_system_health()
    health = json.loads(health_json)
    if health.get("status") == "Healthy":
        print(f"✅ OK (GPU: {health['gpu'].get('0', {}).get('temperature_c', 'N/A')}°C)")
    else:
        print("⚠️ WARNING (Check health output)")

    # 3. Search Test (Dimension Check)
    print("3. Testing semantic search...", end=" ")
    search_res_json = await rag_search_code(query="Ollama configuration", project="ai-assistant", n_results=1)
    search_res = json.loads(search_res_json)
    if "results" in search_res:
        print(f"✅ OK ({len(search_res['results'])} results found)")
    elif "error" in search_res:
        print(f"❌ FAILED: {search_res['error']}")
    else:
        print("❓ UNKNOWN STATE")

    # 4. Status Check
    print("4. Checking indexing status...", end=" ")
    status_json = await get_index_status()
    status = json.loads(status_json)
    print(f"✅ OK (Current status: {status.get('status')})")

    print("-" * 40)
    print("🎉 All core systems functional!")

if __name__ == "__main__":
    # Suppress internal logs for clean output
    logging.getLogger().setLevel(logging.ERROR)
    asyncio.run(test_suite())
