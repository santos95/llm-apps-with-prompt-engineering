from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# define parameters
api_key = os.getenv("API_KEY")
base_url = "https://integrate.api.nvidia.com/v1"
model = 'meta/llama-3.1-8b-instruct'

# instance of the llm 
llm = ChatNVIDIA(
    base_url=base_url, 
    model=model, 
    api_key=api_key, 
    temperature=0
)

# experiment the the rola base interaction with the model
# instruct models are chat models - means that interact in a role-based conversation 
# the not chat models just try to predict whatever ought to come next

# human message
prompt_template = ChatPromptTemplate.from_template("{prompt}")

prompt = prompt_template.invoke("hello there!")

# this is ChatPromptValue - by default is intended to use the structure 
# the value is HumanMessage - that the message is from a humen in the role-based interaction
print(prompt)

# creates a simple chain to verify the type of the response 
chain = prompt_template | llm
response = chain.invoke("Hello there!")

print(response)

# langchain provides an explicit way to manage roles from_message - receives a list of messages which each one is a 2-tuple ("role", "message")
# this code replicates the same message that by default langchain produce using from_templates - human message type
template = ChatPromptTemplate.from_messages([
    ("human", "{prompt}")
])

prompt = template.invoke({"prompt": "Obi-Wan: - Hello There!"})
print(prompt)

chain = template | llm 

response = chain.invoke({"prompt": "Obi-Wan: - Hello There!"})

print(response)