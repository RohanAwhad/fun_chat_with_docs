import os
import elasticsearch
import openai
import re
import redis
import requests
import time
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from urllib.parse import urlparse

if "LOGGING_DIR" in os.environ:
    LOGGING_DIR = os.getenv("LOGGING_DIR", "logs")
    logger.add(
        f"{LOGGING_DIR}/{time.strftime('%Y-%m-%d')}.log",
        format="{time} | {level} | {message}",
    )

# TODO (rohan): add healthchecks
CRAWLER_URL = os.getenv("CRAWLER_URL", "http://localhost:8001")
READABLE_URL = os.getenv("READABLE_URL", "http://localhost:8002")
EMBEDDER_URL = os.getenv("EMBEDDER_URL", "http://localhost:8003")
RERANKER_URL = os.getenv("RERANKER_URL", "http://localhost:8004")
openai.api_key = os.getenv("OPENAI_API_KEY")
logger.debug("Microservice URLs and API keys loaded")


# LLM Setup

# llm prompt
prompt = """### Chunks:

{retrieved_chunks}
---
Question: {q}"""

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.4))
REFERENCE_SIZE = int(os.getenv("REFERENCE_SIZE", 5))


def gpt_completion(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": "Answer the question provided at the end, using the following chunks as references. Answer in detail.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=LLM_TEMPERATURE,
    )
    return response["choices"][0]["message"]["content"]


logger.debug("LLM setup done")

# ES Setup
logger.debug("connecting to ES")
try:
    if "FOUNDELASTICSEARCH_URL" in os.environ:
        url = os.environ["FOUNDELASTICSEARCH_URL"]
        es_client = elasticsearch.Elasticsearch(url)

    elif "SEARCHBOX_URL" in os.environ:
        url = urlparse(os.environ.get("SEARCHBOX_URL"))

        es_client = elasticsearch.Elasticsearch(
            [url.hostname],
            http_auth=(url.username, url.password),
            scheme=url.scheme,
            port=url.port,
        )

    else:
        if "BONSAI_URL" in os.environ:
            bonsai_url = os.environ["BONSAI_URL"]
            auth = re.search("https\:\/\/(.*)\@", bonsai_url).group(1).split(":")
            host = bonsai_url.replace("https://%s:%s@" % (auth[0], auth[1]), "")

            match = re.search("(:\d+)", host)
            if match:
                p = match.group(0)
                host = host.replace(p, "")
                port = int(p.split(":")[1])
            else:
                port = 443

            auth = (auth[0], auth[1])
        else:
            host = os.getenv("ES_HOST", "localhost")
            port = os.getenv("ES_PORT", 9200)
            auth = None

        use_ssl = port == 443

        es_client = elasticsearch.Elasticsearch(
            [{"host": host, "port": port, "use_ssl": use_ssl, "http_auth": auth}]
        )

    if not es_client.ping():
        raise ValueError("Connection failed")
    logger.debug("connected to ES")
except Exception as e:
    logger.error(f"error while connecting to ES: {e}")
    exit(1)

ES_INDEX = os.getenv("ES_INDEX", "test")
EMBED_DIM = int(os.getenv("EMBED_DIM", 768))
logger.info(f"EMBED_DIM: {EMBED_DIM}")
mapping = {
    "mappings": {
        "properties": {
            "link": {"type": "keyword"},
            "chunk": {"type": "text"},
            "embeddings": {"type": "dense_vector", "dims": EMBED_DIM},
        }
    }
}

if not es_client.indices.exists(index=ES_INDEX):
    es_client.indices.create(index=ES_INDEX, body=mapping)


def add_to_es(link: str, chunk: str, embeddings: list):
    doc = {"link": link, "chunk": chunk, "embeddings": embeddings}
    es_client.index(index=ES_INDEX, id=link, body=doc)


logger.debug("ES setup done")


# REDIS Setup
logger.debug("connecting to redis")
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=os.getenv("REDIS_PORT", 6379),
        db=os.getenv("REDIS_DB", 0),
    )
    redis_client.ping()
    logger.debug("connected to redis")
except Exception as e:
    logger.error(f"error while connecting to redis: {e}")
    redis_client = None

# FastAPI Setup
logger.debug("setting up FastAPI")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.debug("FastAPI setup done")


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
    if res.status_code != 200:
        logger.error(f"error while crawling: {res.text}")
        return HTTPException(
            status_code=500, detail=f"error while crawling. {res.text}"
        )
    crawled_data = res.json()
    logger.debug(f"crawler response: {crawled_data}")
    # check if there is any error
    if crawled_data["error"] is not None:
        logger.error(f"error while crawling: {crawled_data['error']}")
        return IndexOutput(status="error")

    # get all links
    all_links = crawled_data["data"]

    # call readable
    link_to_text = {}
    for final_url in all_links:
        try:
            logger.debug(f"calling readable for '{final_url}'")
            res = requests.post(READABLE_URL, json={"url": final_url})
            logger.debug(f"readable response: {res.status_code}")
            data = res.json()
            logger.debug(f"readable response for '{final_url}': {data}")
            text = data["text"].strip()
            link_to_text[final_url] = text
        except Exception as e:
            logger.error(traceback.print_exc())
            logger.error(f"error while processing '{final_url}': {e}")

    chunk_size = 3
    link_chunks = []
    for link, text in link_to_text.items():
        logger.debug(f"processing '{link}' with {len(chunks)} chunks")
        logger.debug(f"chunks: {chunks}")

        # extract sentences
        sentences = [x for x in sent_tokenize(text) if x]
        chunks = [
            " ".join(sentences[i : i + chunk_size])
            for i in range(0, len(sentences), chunk_size)
        ]
        if not chunks:
            continue

        link_chunks.extend([(link, x) for x in chunks])

    all_chunks = [chunk for _, chunk in link_chunks]

    # call embedder
    logger.debug(f"calling embedder")
    res = requests.post(EMBEDDER_URL, json=all_chunks)
    logger.debug(f"embedder response: {res.status_code}")
    data = res.json()
    embeddings = data["embeddings"]
    for (link, chunk), embedding in zip(link_chunks, embeddings):
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

    logger.debug("calling reranker")
    res = requests.post(RERANKER_URL, json={"query": q, "passages": retrieved_chunks})
    logger.debug(f"reranker response: {res.status_code}")
    logger.debug(f"scores: {res.json()['scores']}")
    scores = res.json()["scores"]
    references = sorted(
        zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True
    )[:REFERENCE_SIZE]

    logger.debug(f"references: {references}")
    gpt_prompt = prompt.format(
        retrieved_chunks="\n".join(
            [f"[{i+1}] {chunk[0]}" for i, chunk in enumerate(references)]
        ),
        q=q,
    )
    logger.debug(f"[/query] gpt prompt: {gpt_prompt}")
    answer = gpt_completion(gpt_prompt)
    logger.info(f"answer: {answer}")
    return {"answer": answer}
