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

# place holder messages - hold a place for a list of messages (history)
template_with_placeholder = ChatPromptTemplate.from_messages([
    ('placeholder', '{messages}'),
    ('human', '{prompt}')
])

# list of history messages
messages = [
    ('human', 'The sun came up today.'),
    ('ai', 'That is wonderful!'),
    ('human', 'The sun went down today.'),
    ('ai', 'That is also wonderful!.')
]

prompt = 'What happened today?'

# similar to other templates this spect a list of messages 
# we provide the values according to the template - in this case a list of messags for the placeholder message type 
# we pass messages to fullfit the messages template and a string to fulfit the templates prompt parameter 
template_with_placeholder.invoke({'messages': messages, 'prompt': prompt})

# create the chain 
chain = template_with_placeholder | llm | StrOutputParser()

# invoque que chain
result = chain.invoke({
    'messages': messages, 'prompt': prompt
})

#  the process when from passing messages like a history, then the prompt, response - based on the history 
# use the placeholder template to keep a history of the conversation (context)
print(result)

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

print(chat_conversation)

