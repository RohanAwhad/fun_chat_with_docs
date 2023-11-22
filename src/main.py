import os
import elasticsearch
import openai
import redis
import requests
import time
import traceback

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel

LOGGING_DIR = os.getenv("LOGGING_DIR", "logs")
logger.add(
    f"{LOGGING_DIR}/{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
)

# TODO (rohan): add healthchecks
CRAWLER_URL = os.getenv("CRAWLER_URL", "http://localhost:8001")
READABLE_URL = os.getenv("READABLE_URL", "http://localhost:8002")
EMBEDDER_URL = os.getenv("EMBEDDER_URL", "http://localhost:8003")
openai.api_key = os.getenv("OPENAI_API_KEY")


# LLM Setup

# llm prompt
prompt = """Answer the question provided at the end, using the following chunks:
---
### Chunks:

{retrieved_chunks}
---
Question: {q}"""


def gpt_completion(prompt, temperature=0.4):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"]


# ES Setup
es_client = elasticsearch.Elasticsearch(
    hosts=[
        {
            "host": os.getenv("ES_HOST", "localhost"),
            "port": os.getenv("ES_PORT", 9200),
        }
    ]
)
ES_INDEX = os.getenv("ES_INDEX", "test")
res = requests.post(EMBEDDER_URL, json=["test"]).json()["embeddings"][0]
print(len(res))

mapping = {
    "mappings": {
        "properties": {
            "link": {"type": "keyword"},
            "chunk": {"type": "text"},
            "embeddings": {"type": "dense_vector", "dims": len(res)},
        }
    }
}

if not es_client.indices.exists(index=ES_INDEX):
    es_client.indices.create(index=ES_INDEX, body=mapping)


def add_to_es(link: str, chunk: str, embeddings: list):
    doc = {"link": link, "chunk": chunk, "embeddings": embeddings}
    es_client.index(index=ES_INDEX, id=link, body=doc)


# REDIS Setup
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=os.getenv("REDIS_PORT", 6379),
    db=os.getenv("REDIS_DB", 0),
)

# FastAPI Setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexInput(BaseModel):
    url: str
    path: str
    max_depth: int = 3


class IndexOutput(BaseModel):
    status: str


@app.post("/index")
async def index(inp: IndexInput):
    logger.info(f"indexing {inp.url} with path={inp.path} max_depth={inp.max_depth}")
    # call crawler
    logger.debug("calling crawler")
    res = requests.post(CRAWLER_URL, json=inp.model_dump())
    logger.debug(f"crawler done: {res.status_code}")
    crawled_data = res.json()
    logger.debug(f"crawler response: {crawled_data}")
    domain = None
    all_links = set()
    for data in crawled_data["data"]:
        # TODO (rohan): use data[url] instead of domain. And handle relative urls
        # extract domain from url
        if domain is None:
            domain = "/".join(data["url"].split("/")[:3])
        all_links.update(data["links"])

    # call readable
    link_to_text = {}
    for link in all_links:
        final_url = f"{domain}{link}"
        try:
            logger.debug(f"calling readable for '{final_url}'")
            res = requests.post(READABLE_URL, json={"url": final_url})
            logger.debug(f"readable response: {res.status_code}")
            data = res.json()
            logger.debug(f"readable response for '{final_url}': {data}")
            text = data["text"].strip()
            link_to_text[final_url] = text
        except Exception as e:
            logger.error(f"error while processing '{final_url}': {e}")
            logger.error(traceback.print_exc())

    chunk_size = 3
    for link, text in link_to_text.items():
        # extract sentences
        sentences = [x for x in sent_tokenize(text) if x]
        chunks = [
            " ".join(sentences[i : i + chunk_size])
            for i in range(0, len(sentences), chunk_size)
        ]
        if not chunks:
            continue

        logger.debug(f"processing '{link}' with {len(chunks)} chunks")
        logger.debug(f"chunks: {chunks}")

        # call embedder
        logger.debug(f"calling embedder for '{link}'")
        res = requests.post(EMBEDDER_URL, json=chunks)
        logger.debug(f"embedder response: {res.status_code}")
        data = res.json()
        embeddings = data["embeddings"]
        for chunk, embedding in zip(chunks, embeddings):
            add_to_es(link, chunk, embedding)

    logger.info("indexing done")
    return IndexOutput(status="success")


# ES search body
ES_SEARCH_SIZE = os.getenv("ES_SEARCH_SIZE", 25)
body = {
    "size": ES_SEARCH_SIZE,
    "_source": {"includes": ["link", "chunk"]},
    "query": {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                "params": {"query_vector": None},
            },
        }
    },
}


@app.post("/query")
async def query(q: str):
    logger.info(f"querying for '{q}'")
    logger.debug("calling embedder")
    res = requests.post(EMBEDDER_URL, json=[q])
    logger.debug(f"embedder response: {res.status_code}")
    q_embeddings = res.json()["embeddings"][0]
    body["query"]["script_score"]["script"]["params"]["query_vector"] = q_embeddings
    search_results = es_client.search(index=ES_INDEX, body=body)
    body["query"]["script_score"]["script"]["params"]["query_vector"] = None
    logger.debug(f"[/query] search results: {search_results}")
    retrieved_chunks = []
    for hit in search_results["hits"]["hits"]:
        retrieved_chunks.append(hit["_source"]["chunk"])

    # TODO (rohan): add a reranker here

    gpt_prompt = prompt.format(
        retrieved_chunks="\n".join(
            [f"[{i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)]
        ),
        q=q,
    )
    logger.debug(f"[/query] gpt prompt: {gpt_prompt}")
    answer = gpt_completion(gpt_prompt)
    logger.info(f"answer: {answer}")
    return {"answer": answer}
