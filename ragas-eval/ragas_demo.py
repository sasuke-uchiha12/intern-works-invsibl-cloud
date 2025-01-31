from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import EvaluationDataset
from ragas.metrics import faithfulness, answer_correctness, FactualCorrectness, NoiseSensitivity
from ragas.llms import LangchainLLMWrapper
from custom_demo import *
from langchain_groq import ChatGroq

def ragas_eval_openai(query, replies, content_list):
    llm = ChatOpenAI(model="gpt-4o")

    """ragas metrics - Faithufulness"""
    reference = "Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Advances in the field of deep learning have allowed neural networks to surpass many previous approaches in performance."
    #reference = "Artificial intelligence (AI) is a set of technologies that allow computers to perform tasks that usually require human intelligence. AI can analyze data, understand language, and make recommendations."
    #reference = "Natural language processing (NLP) is a subfield of computer science and artificial intelligence (AI) that uses machine learning to enable computers to understand and communicate with human language. Applications include chatbots, sentiment analysis, and translation services."
    data_samples = []
    data_samples.append({
            'user_input': query,
            'response': replies[0],
            'retrieved_contexts': content_list,
            'reference': reference
    })
    
    evaluation_dataset = EvaluationDataset.from_list(data_samples)
    evaluator_llm = LangchainLLMWrapper(llm)
    result = evaluate(dataset=evaluation_dataset,metrics=[faithfulness, FactualCorrectness(), NoiseSensitivity() ],llm=evaluator_llm)
    result.upload()


def ragas_eval_grokai(query, replies, content_list):

    chat = ChatGroq(groq_api_key = "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", model_name="mixtral-8x7b-32768")

    """ragas metrics - Faithufulness"""
    reference = "Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Advances in the field of deep learning have allowed neural networks to surpass many previous approaches in performance."
    #reference = "Artificial intelligence (AI) is a set of technologies that allow computers to perform tasks that usually require human intelligence. AI can analyze data, understand language, and make recommendations."
    #reference = "Natural language processing (NLP) is a subfield of computer science and artificial intelligence (AI) that uses machine learning to enable computers to understand and communicate with human language. Applications include chatbots, sentiment analysis, and translation services."
    data_samples = []
    data_samples.append({
            'user_input': query,
            'response': replies[0],
            'retrieved_contexts': content_list,
            'reference': reference
    })
    
    evaluation_dataset = EvaluationDataset.from_list(data_samples)
    evaluator_llm = LangchainLLMWrapper(chat)
    result = evaluate(dataset=evaluation_dataset,metrics=[faithfulness, FactualCorrectness(), NoiseSensitivity()],llm=evaluator_llm)
    result.upload()
