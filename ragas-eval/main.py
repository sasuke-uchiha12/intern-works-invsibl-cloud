from custom_demo import *
from documents import *
from multi_agent_custom import *

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


# import asyncio
# import logging
# from custom_demo import *
# from documents import *
# from multi_agent_custom import *


# async def main():
#     query = input("Enter the query: ")

#     # Await the generic_llm call since it's async
#     decision_response = await generic_llm(query=query, prompt_template=prompt_template_decision)
#     logging.info(f"decision_response: {decision_response}")

#     # Perform decision based on the response
#     if decision_response.lower() == "climate change":
#         response = await agent_a(query)  # Ensure agent_a is async if needed
#     elif decision_response.lower() == "artificial intelligence":
#         response = await agent_b(query)  # Ensure agent_b is async if needed
#     elif decision_response.lower() == "general query":
#         response = await generic_llm(query=query, prompt_template=prompt_template_gquery)
#     else:
#         response = "can't classify the query!"

#     logging.info(f"actual response: {response}")

#     # Format the response
#     formatted_response = await formatter(prompt_template=prompt_template_formatter, text=response)

#     print(f"formatted response: {formatted_response}")

# # Run the asynchronous main function
# if __name__ == "__main__":
#     asyncio.run(main())
