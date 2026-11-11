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



# build a rudimentary conversation history mechanism
# 1 - create a placeholder template then pass past messages 
chat_conversation_template = ChatPromptTemplate.from_messages([
    ('placeholder', '{chat_conversation}')
])

chat_chain = chat_conversation_template | llm | StrOutputParser()

# 2 - create a list to store the conversation - and fill it with the messages  
chat_conversation = []

chat_conversation.append(('user', 'Hello!, my name is John Connor!'))

# 3 - invoke the chat_chain with the current conversation list 
response = chat_chain.invoke({'chat_conversation': chat_conversation})
# add the response message to the history - to keep track of the history
chat_conversation.append(('ai', response))

# 4 - try to refers to previous information or early messages information to work with the llm 
chat_conversation.append(('user', "Do you remember who am i?"))

response =  chat_chain.invoke({'chat_conversation': chat_conversation})

chat_conversation.append(response)

print(chat_conversation)


