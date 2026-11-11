import os 
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv() 

# create llm instance 
base_url = os.getenv("BASE_URL")
model = os.getenv("MODEL") 

llm = ChatNVIDIA(base_url=base_url, model=model, temperature=0)

class Chatbot:

    def __init__(self, llm):
        
        # define the chat_conversation_template
        chat_conversation_template = ChatPromptTemplate.from_messages([
            ('placeholder', '{chat_conversation}')
        ])

        # define the chain 
        self.chat_chain = chat_conversation_template | llm | StrOutputParser

        # define an empty list properties for the messages history 
        self.chat_conversation = []

    # chat - expects a simple string prompt
    def chat(self, prompt):

        # append the prompt as user message to chat conversation
        self.chat_conversation.append(('user', prompt))

        response = self.chat_chain.invoke({'chat_conversation': self.chat_conversation})

        return response 

