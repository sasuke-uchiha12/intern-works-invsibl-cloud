from custom_demo import *
from documents import *
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    FactualCorrectness,
    NoiseSensitivity,
)
from ragas_eval import *

# doc_embedding(index="rag_doc_cc", documents=documents_climate_change)
# doc_embedding(index="rag_doc_ai", documents=documents_ai)


def agent_a(query):
    rag_agenta = rag(index="rag_doc_cc", model="gemini-pro")
    response = rag_agenta.run(query)
    replies = response["rag"]["replies"][0]
    replies = response["replies"]
    context = response["context"]
    # ragas_eval_openai(query=query, replies=replies, content_list=context)
    # ragas_eval_grokai(query=query, replies=replies, content_list=context)
    return replies


def agent_b(query):
    rag_agentb = rag(index="rag_doc_ai", model="gemini-pro")
    response = rag_agentb.run(query)
    replies = response["replies"]
    context = response["context"]
    ragas_eval_openai(query=query, replies=replies, content_list=context)
    # ragas_eval_grokai(query=query, replies=replies, content_list=context)
    return replies
