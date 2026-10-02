from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

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

print(llm)
