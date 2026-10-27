from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from pathlib import Path
from dotenv import load_dotenv

# load the .env file that sits next to this script (works no matter where you run it from)
# load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()

# create model instance
base_url = os.getenv("BASE_URL")
model = os.getenv("MODEL") 

print("base_url: ", base_url)
print("model: ", model)
print("=========================================")

# define the llm 
llm = ChatNVIDIA(base_url=base_url, model=model, temperature=0)

# llms jump to conclusions 
# by their nature of generate what is most likely to come next 
# is not surprising to see a llm to jump to conclusions when complex reasoning is needed
print("Multiplication with python: 345*888: ", 345*888)

# perform multiplication with llm 
print("Multiplication with llm: ")
print(llm.invoke('What is 345 * 888?').content)

print("=========================================")
print("=====CHAIN OF THOUGH MULTIPLICACION =====")
print("=========================================")

# problem to solve
example_problem = 'What is 678 * 789?'

# provide a example of step by step reasoning - chain of thoughts example
example_cot = '''\
Let me break this down into steps. First I'll break down 789 into hundreds, tens, and ones:

789 -> 700 + 80 + 9

Next I'll multiply 678 by each of these values, storing the intermediate results:

678 * 700 -> 678 * 7 * 100 -> 4746 * 100 -> 474600

My first intermediate result is 474600.

678 * 80 -> 678 * 8 * 10 -> 5424 * 10 -> 54240

My second intermediate result is 54240.

678 * 9 -> 6102

My third intermediate result is 6102.

My three intermediate results are 474600, 54240, and 6102.

Adding the first two intermediate results I get 474600 + 54240 -> 528840.

Adding 528840 to the last intermediate result I get 528840 + 6102 -> 534942

The final result is 534942.
'''

# build a one shot prompt template with the problem and the CoT example
multiplication_template = ChatPromptTemplate.from_messages([
    ('human', example_problem), 
    ('ai', example_cot),
    ('human', '{long_multiplication_prompt}')
])

# define the chain 
multiplication_chain = multiplication_template | llm | StrOutputParser()

# multiplication problem with CoT 
# goes thought a step by step process to generate the response
print(multiplication_chain.invoke("What is 345 * 888?"))

# Zero-shot Chain of thought prompting 
# It is an approach in which we do not send a verbose prompt that indicates how the reasoning will be do it 
# in this case we only include in the prompt "Think step by step" without supplying CoT examples 

print("=========================================")
print("===== zero shot CoT =====")
print("=========================================")

zero_shot_cot_prompt = ChatPromptTemplate([
    ("human", "{long_multplicatioin_prompt}  Let's think step by step.")
])

zero_shot_multiplication_chain = zero_shot_cot_prompt | llm | StrOutputParser() 

print(zero_shot_multiplication_chain.invoke('What is 345 * 888?'))


