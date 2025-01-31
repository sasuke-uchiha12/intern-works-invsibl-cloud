import os
from dotenv import load_dotenv
from haystack import component
from haystack import Pipeline
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchEmbeddingRetriever,
)
from haystack_integrations.components.generators.google_ai import (
    GoogleAIGeminiGenerator,
)
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
import logging
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from ragas import SingleTurnSample, evaluate
from ragas.metrics import faithfulness, answer_correctness, FactualCorrectness
from ragas.metrics import BleuScore
from datasets import Dataset


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-agent")


def doc_embedding(index, documents):
    opensearch_store = OpenSearchDocumentStore(
        hosts="http://localhost:9200",
        index=index,
        use_ssl=False,
        embedding_field="embedding",
        embedding_dim=768,
        similarity="cosine",
    )

    embed_pipeline = Pipeline()
    embed_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
    embed_pipeline.add_component(
        "writer",
        DocumentWriter(document_store=opensearch_store, policy=DuplicatePolicy.SKIP),
    )
    embed_pipeline.connect("embedder", "writer")

    embed_pipeline.run({"documents": documents})


@component
class rag:
    def __init__(self, index: str, model: str, top_k: int = 1):
        self.index = index
        self.top_k = top_k
        self.model = model
        # self.query = query

    def run(self, query: str):

        API_KEY = os.getenv("API_KEY")
        os.environ["GOOGLE_API_KEY"] = API_KEY
        os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")
        os.environ["RAGAS_APP_TOKEN"] = os.getenv("RAGAS_API_KEY")

        opensearch_store = OpenSearchDocumentStore(
            hosts="http://localhost:9200",
            index=self.index,
            use_ssl=False,
            embedding_field="embedding",
            embedding_dim=768,
            similarity="cosine",
        )
        template = """ 
            You are an intelligent assistant providing accurate, context-aware, and user-friendly responses based on retrieved documents. Your answers should balance precision and clarity while being conversational.

            ---

            ### Retrieved Documents:
            {% for doc in documents %}
                Document {{ loop.index }}: {{ doc.content }}
            {% endfor %}

            ### User Query:
            {{ query }}

            ---

            ### Response Generation Guidelines:
            1. Prioritize information from the retrieved documents as the primary source of truth.
            2. Address the user’s query comprehensively but concisely.
            3. If a direct answer isn’t available, summarize the relevant context or provide logical inferences.
            4. Maintain a conversational tone, ensuring the response feels natural and user-centric.
            5. Include references or citations to the documents for transparency when needed.

            ---

            ### Response:
            **Answer:** [Directly address the user’s query with relevant details.]

            **Supporting Context:** [Optional - Summarize or reference key points from the retrieved documents.]

            **Additional Notes:** [Optional - Add extra clarifications, helpful suggestions, or links to explore further.]


        """

        """query embedding - retrieving process - prompt builder - llm"""
        # logging.info(template)
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("embedder_text", SentenceTransformersTextEmbedder())
        rag_pipeline.add_component(
            "retriever",
            OpenSearchEmbeddingRetriever(
                document_store=opensearch_store, top_k=self.top_k
            ),
        )
        llm = GoogleAIGeminiGenerator(model=self.model)
        rag_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
        rag_pipeline.add_component("llm_gemini", llm)

        rag_pipeline.connect("embedder_text.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm_gemini")

        result = rag_pipeline.run({"embedder_text": {"text": query}})
        replies = result["llm_gemini"]["replies"]
        print(f"replies: {replies}")

        embedding_text = SentenceTransformersTextEmbedder()
        embedding_text.warm_up()
        embedding_result = embedding_text.run(text=query)
        query_embedding = embedding_result["embedding"]
        # print(query_embedding)
        retrievals = OpenSearchEmbeddingRetriever(
            document_store=opensearch_store, top_k=self.top_k
        )
        document_retrieve = retrievals.run(query_embedding=query_embedding)

        context_response = document_retrieve["documents"]
        # logging.info(context_response)

        content_list = [doc.content for doc in context_response]

        # dataset = SingleTurnSample(
        #     user_input=query,
        #     response=replies[0],
        #     retrieved_contexts= content_list
        # )

        """ragas metrics - Faithufulness"""
        # data_samples = {
        #     'question': [query],
        #     'response': [replies[0]],
        #     'retrieved_contexts': [content_list]
        # }

        # dataset = Dataset.from_dict(data_samples)
        # score_faithfulness = evaluate(dataset, metrics=[faithfulness])
        # score_faithfulness.upload()
        # df = score_faithfulness.to_pandas()
        # df.to_csv('score.csv', index=False)

        """ragas metrics - bleuScore"""
        # content_string = " ".join(doc.content for doc in context_response)
        # test_data = {
        #     "user_input": query,
        #     "response": replies[0],
        #     "reference": content_string
        # }

        # metric = BleuScore()
        # test_data = SingleTurnSample(**test_data)
        # score = metric.single_turn_score(test_data)
        # print(f"score: {score}")
        return {"replies": replies, "context": content_list}


def generic_llm(query, prompt_template):
    API_KEY = os.getenv("API_KEY")
    os.environ["GOOGLE_API_KEY"] = API_KEY

    pipeline = Pipeline()

    pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    pipeline.add_component("llm", GoogleAIGeminiGenerator(model="gemini-pro"))
    pipeline.connect("prompt_builder", "llm")

    response = pipeline.run({"prompt_builder": {"query": query}})

    replies = response["llm"]["replies"][0]
    return replies


def formatter(text, prompt_template):
    API_KEY = os.getenv("API_KEY")
    os.environ["GOOGLE_API_KEY"] = API_KEY

    pipeline = Pipeline()

    pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    pipeline.add_component("llm", GoogleAIGeminiGenerator(model="gemini-pro"))
    pipeline.connect("prompt_builder", "llm")

    response = pipeline.run({"prompt_builder": {"context": text}})
    # logging.info(response)
    return response["llm"]["replies"][0]


# import os
# from dotenv import load_dotenv
# from haystack import component
# from haystack import Pipeline
# from haystack_integrations.components.retrievers.opensearch import (
#     OpenSearchEmbeddingRetriever,
# )
# from haystack_integrations.components.generators.google_ai import (
#     GoogleAIGeminiGenerator,
# )
# from haystack.document_stores.types import DuplicatePolicy
# from haystack.components.embedders import SentenceTransformersTextEmbedder
# from haystack.components.builders import PromptBuilder
# import logging
# from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
# from haystack.components.embedders import SentenceTransformersDocumentEmbedder
# from haystack.components.writers import DocumentWriter
# from ragas import SingleTurnSample
# from ragas.metrics import ResponseRelevancy
# import asyncio

# load_dotenv()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("multi-agent")


# def doc_embedding(index, documents):
#     opensearch_store = OpenSearchDocumentStore(
#         hosts="http://localhost:9200",
#         index=index,
#         use_ssl=False,
#         embedding_field="embedding",
#         embedding_dim=768,
#         similarity="cosine",
#     )

#     embed_pipeline = Pipeline()
#     embed_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
#     embed_pipeline.add_component(
#         "writer",
#         DocumentWriter(document_store=opensearch_store, policy=DuplicatePolicy.SKIP),
#     )
#     embed_pipeline.connect("embedder", "writer")

#     embed_pipeline.run({"documents": documents})


# @component
# class rag:
#     def __init__(self, index: str, model: str, top_k: int = 2):
#         self.index = index
#         self.top_k = top_k
#         self.model = model

#     async def run(self, query: str):  # Make the run method async

#         API_KEY = os.getenv("API_KEY")
#         os.environ["GOOGLE_API_KEY"] = API_KEY

#         opensearch_store = OpenSearchDocumentStore(
#             hosts="http://localhost:9200",
#             index=self.index,
#             use_ssl=False,
#             embedding_field="embedding",
#             embedding_dim=768,
#             similarity="cosine",
#         )
#         template = """
#             You are an intelligent assistant providing accurate, context-aware, and user-friendly responses based on retrieved documents. Your answers should balance precision and clarity while being conversational.

#             ---

#             ### Retrieved Documents:
#             {% for doc in documents %}
#                 Document {{ loop.index }}: {{ doc.content }}
#             {% endfor %}

#             ### User Query:
#             {{ query }}

#             ---

#             ### Response Generation Guidelines:
#             1. Prioritize information from the retrieved documents as the primary source of truth.
#             2. Address the user’s query comprehensively but concisely.
#             3. If a direct answer isn’t available, summarize the relevant context or provide logical inferences.
#             4. Maintain a conversational tone, ensuring the response feels natural and user-centric.
#             5. Include references or citations to the documents for transparency when needed.

#             ---

#             ### Response:
#             **Answer:** [Directly address the user’s query with relevant details.]

#             **Supporting Context:** [Optional - Summarize or reference key points from the retrieved documents.]

#             **Additional Notes:** [Optional - Add extra clarifications, helpful suggestions, or links to explore further.]
#         """

#         rag_pipeline = Pipeline()
#         rag_pipeline.add_component("embedder_text", SentenceTransformersTextEmbedder())
#         rag_pipeline.add_component(
#             "retriever",
#             OpenSearchEmbeddingRetriever(
#                 document_store=opensearch_store, top_k=self.top_k
#             ),
#         )
#         llm = GoogleAIGeminiGenerator(model=self.model)
#         rag_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
#         rag_pipeline.add_component("llm_gemini", llm)

#         rag_pipeline.connect("embedder_text.embedding", "retriever.query_embedding")
#         rag_pipeline.connect("retriever", "prompt_builder.documents")
#         rag_pipeline.connect("prompt_builder", "llm_gemini")

#         result = rag_pipeline.run({"embedder_text": {"text": query}})
#         replies = result["llm_gemini"]["replies"]
#         print(f"replies: {replies}")

#         embedding_text = SentenceTransformersTextEmbedder()
#         embedding_text.warm_up()
#         embedding_result = embedding_text.run(text=query)
#         query_embedding = embedding_result["embedding"]

#         retrievals = OpenSearchEmbeddingRetriever(document_store=opensearch_store, top_k=self.top_k)
#         document_retrieve = retrievals.run(query_embedding=query_embedding)

#         context_response = document_retrieve["documents"]

#         content_list = [doc.content for doc in context_response]

#         dataset = SingleTurnSample(
#             user_input=query,
#             response=replies[0],
#             retrieved_contexts=content_list
#         )

#         scorer = ResponseRelevancy(llm=llm, embeddings=SentenceTransformersTextEmbedder())
#         score = await scorer.single_turn_ascore(dataset)  # Await the async method
#         print(score)

#         return {"replies": replies}


# async def main():
#     # Example usage of the rag class
#     agent = rag(index="my_index", model="gemini-pro")
#     response = await agent.run(query="What is the weather like today?")
#     print(response)

# if __name__ == "__main__":
#     asyncio.run(main())

# async def generic_llm(query, prompt_template):
#     API_KEY = os.getenv("API_KEY")
#     os.environ["GOOGLE_API_KEY"] = API_KEY

#     pipeline = Pipeline()

#     pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
#     pipeline.add_component("llm", GoogleAIGeminiGenerator(model="gemini-pro"))
#     pipeline.connect("prompt_builder", "llm")

#     response = pipeline.run({"prompt_builder": {"query": query}})

#     replies = response["llm"]["replies"][0]
#     return replies


# # Formatter Function
# async def formatter(text, prompt_template):
#     API_KEY = os.getenv("API_KEY")
#     os.environ["GOOGLE_API_KEY"] = API_KEY

#     pipeline = Pipeline()

#     pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
#     pipeline.add_component("llm", GoogleAIGeminiGenerator(model="gemini-pro"))
#     pipeline.connect("prompt_builder", "llm")

#     response = pipeline.run({"prompt_builder": {"context": text}})
#     return response["llm"]["replies"][0]
