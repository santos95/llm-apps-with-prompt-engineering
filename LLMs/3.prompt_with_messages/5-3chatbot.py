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
        self.chat_chain = chat_conversation_template | llm | StrOutputParser()

        # define an empty list properties for the messages history 
        self.chat_conversation = []

    # chat - expects a simple string prompt
    def chat(self, prompt):

        # append the prompt as an user message 
        self.chat_conversation.append(('user', prompt))

        # get the response from the model 
        response = self.chat_chain.invoke({'chat_conversation': self.chat_conversation})

        # append the response to the chat history as ai message 
        self.chat_conversation.append(('ai', response))

        # return the response to the user 
        return response

    def clear_history(self):
        self.chat_conversation = []  


# test 
chatbot = Chatbot(llm)

print(chatbot.chat("Hello there! My name is John Connor!"))

print(chatbot.chat("Do you remember my name?"))

print(chatbot.chat("can we be friends?"))

print(chatbot.chat("will you know me in the futere and how you call me?"))


