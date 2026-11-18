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

class ChatbotWithRole: 

    def __init__(self, llm, system_message=''):

        # define the template 
        chat_with_role_conversation_template = ChatPromptTemplate.from_messages([
            ('system', system_message),
            ('placeholder', '{chat_conversation}')
        ])

        # define the chain
        self.chat_chain = chat_with_role_conversation_template | llm | StrOutputParser()

        # define the history 
        self.chat_conversation = []

    def chat(self, prompt):

        self.chat_conversation.append(('user', prompt))

        response = self.chat_chain.invoke({'chat_conversation': self.chat_conversation})

        self.chat_conversation.append(('ai', response))

        return response

    def clear_history(self):
        self.chat_conversation = []



brief_chatbot_system_message = "You always answer as briefly and concisely as possible."
curious_chatbot_system_message = """\
You are incredibly curious, and often respond with reflections and followup questions that lean the conversation in the direction of playfully \
understanding more about the subject matters of the conversation."""
increased_vocabulary_system_message = """\
You always respond using challenging and often under-utilized vocabulary words, even when your response could be made more simply."""

# create the objects 
brief_chatbot = ChatbotWithRole(llm, system_message=brief_chatbot_system_message)
curious_chatbot = ChatbotWithRole(llm, system_message=curious_chatbot_system_message)
increased_vocabulary_chatbot = ChatbotWithRole(llm, system_message=increased_vocabulary_system_message)

print(brief_chatbot.chat("Who is John Connor?"))
print(curious_chatbot.chat("Who is John Connor?"))
print(increased_vocabulary_chatbot.chat("Who is John Connor?"))

