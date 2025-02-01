from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import EvaluationDataset
from ragas.metrics import (
    faithfulness,
    FactualCorrectness,
    NoiseSensitivity,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ContextEntityRecall,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas.llms import LangchainLLMWrapper
from custom_demo import *
from langchain_groq import ChatGroq


def ragas_eval_openai(query, replies, content_list):
    llm = ChatOpenAI(model="gpt-4o")

    """ragas metrics - Faithufulness"""
    # reference = "Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Advances in the field of deep learning have allowed neural networks to surpass many previous approaches in performance."
    # reference = "Artificial intelligence (AI) is a set of technologies that allow computers to perform tasks that usually require human intelligence. AI can analyze data, understand language, and make recommendations."
    # reference = "Natural language processing (NLP) is a subfield of computer science and artificial intelligence (AI) that uses machine learning to enable computers to understand and communicate with human language. Applications include chatbots, sentiment analysis, and translation services."
    reference = "Reinforcement Learning is a branch of AI that focuses on training agents to make decisions through interaction with an environment. The agent takes actions and receives feedback in the form of rewards or penalties, which it uses to refine its strategies over time. Unlike supervised learning, it does not rely on labeled data but learns through trial and error to maximize long-term rewards. This approach is widely applied in various domains, such as robotics, game AI, autonomous vehicles, resource management, and recommendation systems, where dynamic decision-making is critical."
    data_samples = []
    data_samples.append(
        {
            "user_input": query,
            "response": replies[0],
            "retrieved_contexts": content_list,
            "reference": reference,
        }
    )

    evaluation_dataset = EvaluationDataset.from_list(data_samples)
    evaluator_llm = LangchainLLMWrapper(llm)
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            faithfulness,
            FactualCorrectness(),
            NoiseSensitivity(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            ContextEntityRecall(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
        ],
        llm=evaluator_llm,
    )
    result.upload()


def ragas_eval_grokai(query, replies, content_list):

    chat = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="mixtral-8x7b-32768",
    )

    """ragas metrics - Faithufulness"""
    reference = "Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Advances in the field of deep learning have allowed neural networks to surpass many previous approaches in performance."
    # reference = "Artificial intelligence (AI) is a set of technologies that allow computers to perform tasks that usually require human intelligence. AI can analyze data, understand language, and make recommendations."
    # reference = "Natural language processing (NLP) is a subfield of computer science and artificial intelligence (AI) that uses machine learning to enable computers to understand and communicate with human language. Applications include chatbots, sentiment analysis, and translation services."
    data_samples = []
    data_samples.append(
        {
            "user_input": query,
            "response": replies[0],
            "retrieved_contexts": content_list,
            "reference": reference,
        }
    )

    evaluation_dataset = EvaluationDataset.from_list(data_samples)
    evaluator_llm = LangchainLLMWrapper(chat)
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[faithfulness, FactualCorrectness(), NoiseSensitivity()],
        llm=evaluator_llm,
    )
    result.upload()
