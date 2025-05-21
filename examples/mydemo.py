import os
import asyncio
import inspect
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status
import numpy as np
from datetime import datetime

from dotenv import load_dotenv
# from database import DailyIncContent, get_date_from_url, session

WORKING_DIR = "./dickens"

load_dotenv()
ROOT_DIR = os.environ.get("ROOT_DIR")
WORKING_DIR = f"{WORKING_DIR}/dickens-pg"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

# AGE
os.environ["AGE_GRAPH_NAME"] = "11_daily"

os.environ["POSTGRES_HOST"] = "15.1.162.105"
os.environ["POSTGRES_PORT"] = "5432"
os.environ["POSTGRES_USER"] = "rag"
os.environ["POSTGRES_PASSWORD"] = "rag"
os.environ["POSTGRES_DATABASE"] = "rag"


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_compatible_demo.log"))

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "ERROR",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    return await openai_complete_if_cache(
        # "qwen-plus",
        "Qwen/Qwen3-32B-FP8",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="token-abc123",
        base_url="http://LLM_URL:8001/v1",
        # api_key="TOKEN",
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await qwen_embedding_func(texts)


async def qwen_embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts=texts,
        model="Alibaba-NLP/gte-Qwen2-7B-instruct",
        base_url="http://EMBEDDING_URL:18001",
        api_key="TOKEN",
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await qwen_embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result.shape)


asyncio.run(test_funcs())


async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)


async def initialize_rag():
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
        max_parallel_insert=4,
        # namespace_prefix="11daily2023",
        addon_params={
            "language": "english",
            "entity_types": [
                "Person",
                "Category",
                "Location",
                "Company",
                "Organization",
                "date",
                "Award",
                "Benefit",
                "Career",
                "Event",
                "Community",
                "Product",
                "Tips",
                "Technology",
                "platform",
            ],
            "example_number": 2,
            "disable_qwen_thinking": True,
        },
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        graph_storage="PGGraphStorage",
        vector_storage="PGVectorStorage",
        auto_manage_storages_states=False,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_rag()

        # with open("./book.txt", "r", encoding="utf-8") as f:
        #     await rag.ainsert(f.read())
        # query = session.query(DailyIncContent).filter(DailyIncContent.status == 1).all()
        # for index, item in enumerate(query):
        #     date = get_date_from_url(str(item.source_url))
        #     page_content = f"""
        #     "publish_time": {date} \n
        #     "keywords": {item.meta_data.get("keywords", "")} \n
        #     "description": {item.meta_data.get("description", "")} \n
        #     {item.markdown_content}
        #     """
        #     if date is not None and date > "2025-05-01" and page_content is not None:
        #         with open("./hashs.txt", "a", encoding="utf-8") as f:
        #             f.write(f"{item.markdown_hash}\n")
        #         print(item.markdown_hash)
                # await rag.ainsert(
                #     page_content,
                #     ids=item.markdown_hash,
                #     # file_paths=item.source_url,
                #     create_time=date,
                # )
                # item.created_at = datetime.strptime(date, "%Y-%m-%d").isoformat()
                # session.commit()
                # print(f"Inserted {index + 1} items url: {item.source_url} len: {len(page_content)}")

        # Perform naive search
        question = "What is the Integrity at 11? now: 2025 year, 4th month, 15st day"
        # question = "Introduce the ceo of 11"

        # print("\n=====================")
        # print("Query mode: naive")
        # print("=====================")
        # resp = await rag.aquery(
        #     question,
        #     param=QueryParam(mode="naive", stream=True, only_need_context=False),
        # )
        # if inspect.isasyncgen(resp):
        #     await print_stream(resp)
        # else:
        #     print(resp)

        # Perform local search
        print("\n=====================")
        print("Query mode: local")
        print("=====================")
        resp = await rag.aquery(
            question,
            param=QueryParam(
                mode="local",
                stream=True,
                only_need_prompt=False,
                top_k=10,
                max_token_for_local_context=1000
            ),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform global search
        print("\n=====================")
        print("Query mode: global")
        print("=====================")
        resp = await rag.aquery(
            question,
            param=QueryParam(mode="global", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        resp = await rag.aquery(
            question,
            param=QueryParam(mode="hybrid", stream=True, only_need_context=False),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
