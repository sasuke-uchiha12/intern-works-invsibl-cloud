from custom_demo import *
from documents import *
from ragas.metrics import faithfulness, answer_correctness, FactualCorrectness, NoiseSensitivity
from ragas_demo import *
doc_embedding(index="rag_doc_cc", documents=documents_climate_change)
doc_embedding(index="rag_doc_ai", documents=documents_ai)


def agent_a(query):
    query_pipeline = Pipeline()
    rag_agenta = rag(index="rag_doc_cc", model="gemini-pro")
    query_pipeline.add_component("rag", rag_agenta)
    response = query_pipeline.run({"rag": {"query": query}})

    replies = response["rag"]["replies"][0]

    logging.info(replies)
    return replies


def agent_b(query):
    rag_agentb = rag(index="rag_doc_ai", model="gemini-pro")
    response = rag_agentb.run(query)
    replies = response["replies"]
    context = response["context"]
    ragas_eval_openai(query=query, replies=replies, content_list=context)
    ragas_eval_grokai(query=query, replies=replies, content_list=context)
    return replies


# def ragas_eval(query, replies, content_list):
#     """ragas metrics - Faithufulness"""
#     reference = "Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Advances in the field of deep learning have allowed neural networks to surpass many previous approaches in performance."
#     data_samples = {
#             'question': [query],
#             'response': [replies[0]],
#             'retrieved_contexts': [content_list],
#             'reference': [reference]
#         }

#     dataset = Dataset.from_dict(data_samples)
#     noise_sensitivity = NoiseSensitivity()
#     score_faithfulness = evaluate(dataset, metrics=[noise_sensitivity, faithfulness])
#     score_faithfulness.upload()









# from custom_demo import *
# from documents import *
# import asyncio

# # Calling doc_embedding to initialize the documents
# doc_embedding(index="rag_doc_cc", documents=documents_climate_change)
# doc_embedding(index="rag_doc_ai", documents=documents_ai)

# async def agent_a(query):
#     query_pipeline = Pipeline()
#     rag_agenta = rag(index="rag_doc_cc", model="gemini-pro")
#     query_pipeline.add_component("rag", rag_agenta)
    
#     # Await the response from the pipeline
#     response = await query_pipeline.run({"rag": {"query": query}})

#     replies = response["rag"]["replies"][0]
#     # logging.info(replies)
#     return replies

# async def agent_b(query):
#     rag_agentb = rag(index="rag_doc_ai", model="gemini-pro")
    
#     # Await the response from rag.run
#     response = await rag_agentb.run(query)
    
#     replies = response["replies"][0]
#     return replies