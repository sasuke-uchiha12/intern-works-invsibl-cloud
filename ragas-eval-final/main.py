from custom_demo import *
from documents import *
from multi_agent import *

query = input("Enter the query: ")

decison_response = generic_llm(query=query, prompt_template=prompt_template_decision)
logging.info(f"decision_response: {decison_response}")

if decison_response.lower() == "climate change":
    response = agent_a(query)
elif decison_response.lower() == "artificial intelligence":
    response = agent_b(query)
elif decison_response.lower() == "general query":
    response = generic_llm(query=query, prompt_template=prompt_template_gquery)
else:
    response = "can't classify the query!"

logging.info(f"actual response: {response}")
formatted_response = formatter(prompt_template=prompt_template_formatter, text=response)

print(f"formatted response: {formatted_response}")