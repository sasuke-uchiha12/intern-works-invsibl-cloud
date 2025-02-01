from haystack import Document

documents_climate_change = [
    Document(
        content="Climate change refers to long-term changes in temperature, precipitation, and weather patterns, primarily caused by human activities like burning fossil fuels."
    ),
    Document(
        content="Greenhouse gases, including carbon dioxide and methane, trap heat in Earth's atmosphere, leading to global warming and its associated effects."
    ),
    Document(
        content="Deforestation contributes to climate change by reducing the planet's ability to absorb carbon dioxide and altering regional weather patterns."
    ),
    Document(
        content="Renewable energy sources, such as wind, solar, and hydropower, are critical for reducing carbon emissions and mitigating climate change."
    ),
    Document(
        content="Ocean acidification, a result of increased CO2 absorption, threatens marine ecosystems, particularly coral reefs and shell-forming organisms."
    ),
    Document(
        content="International agreements like the Paris Agreement aim to limit global warming to 1.5°C above pre-industrial levels through collective action."
    ),
    Document(
        content="Individual actions, including reducing energy use, minimizing waste, and adopting sustainable practices, contribute significantly to combating climate change."
    ),
]

documents_ai = [
    Document(
        content="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn. It encompasses fields like machine learning, natural language processing, and robotics."
    ),
    Document(
        content="Machine Learning (ML) is a subset of AI that focuses on building algorithms that allow computers to learn from and make predictions based on data without being explicitly programmed."
    ),
    Document(
        content="Natural Language Processing (NLP) is a branch of AI that enables machines to understand, interpret, and respond to human language. Applications include chatbots, sentiment analysis, and translation services."
    ),
    Document(
        content="Deep Learning is a type of machine learning that uses neural networks with multiple layers to analyze and learn from large amounts of data. It has driven advances in image and speech recognition."
    ),
    Document(
        content="Reinforcement Learning is a type of AI where agents learn to make decisions by performing actions and receiving rewards or penalties. It's commonly used in robotics and game AI."
    ),
    Document(
        content="Generative AI models, like GPT and DALL·E, are designed to create new content, such as text, images, and music. They work by learning patterns from existing data."
    ),
    Document(
        content="Ethics in AI involves addressing issues like bias, transparency, and accountability in AI systems to ensure they are fair, trustworthy, and beneficial to society."
    ),
]


"""follwing prompt template can be given below"""

prompt_template_decision = """ Classify the query as 'Artificial Intelligence' or 'Climate Change' or 'General Query':
       \nQuery: {{query}}
       """
prompt_template_gquery = """Give a detailed explaination in conversational tone about this: {{query}}
       """

"""formatter - v1"""
# prompt_template_formatter = """
#     Rewrite the following text in a friendly and conversational tone. Ensure it is engaging, easy to understand, and free of grammatical errors.
#     If possible, include relevant examples to enhance clarity. Slightly tailor the response based on the given query to ensure it remains relevant and aligned with the context.
#     **Note:** Do not fabricate information. Use the provided query and input text as references to ensure accuracy.
#     Also return the answer in a string format!

#     Query: {{query}}

#     Input Text: {{text}}
# """

"""formatter - v2"""
prompt_template_formatter = """
    You are an expert assistant who excels at creating user-friendly, conversational responses. Your task is to rewrite the provided structured response into a natural, engaging, and easy-to-read format. Make sure the information flows logically and includes relevant examples or additional context when appropriate.

    ---

    ### Guidelines:
    1. Merge all structured elements into a unified explanation, removing explicit labels such as "Answer," "Supporting Context," or "Additional Notes."
    2. Use a conversational tone that is engaging and relatable, making the information easy to follow.
    3. Include relevant examples, analogies, or context when appropriate to clarify the concepts.
    4. Ensure the response is accurate, concise, and free of unnecessary jargon, simplifying complex terms where needed.
    5. Structure the response logically, ensuring smooth transitions between ideas.

    ---

    ### Input:
    **context:** {{ context }}

    **Structured Response:**
    {{ structured_response }}

    ---

    ### Reformatted Response:
    [Return the rewritten response here in a conversational and engaging format.]

"""
