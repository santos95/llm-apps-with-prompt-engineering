from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from dotenv import load_dotenv

#  define base params for llms model requests 
base_url = 'http://127.0.0.1:1234/v1/'
model = 'bartowski/meta-llama-3.1-8b-instruct'


print("The current url is: ", base_url)
print("The current model is: ", model)

llm = ChatNVIDIA(
    base_url = base_url,
    model = model,
    temperature=0
)

# system messages - statements to provide initial context and influence the models responses
# common uses - set the role and context that we want the model to portray on their responses
# define overwatching personality and personal details 
# define a system message that set the role/personality of the model
prompt_template = ChatPromptTemplate([
    ("system", "You are a pirate. Your name is Jack. You always talk like a pirate"),
    ("human", "{prompt}")
])

output_parser = StrOutputParser() 

# define the chain 
chain = prompt_template | llm | output_parser 

# invoke the llm and get the response 
result = chain.invoke({"prompt", "Who are you?"})

print(result)

# influence behavior with system messages - response with a phrase in uppercases
print("----------------------------------------------")

prompt_template = ChatPromptTemplate([
    ("system", "You are an incredibly simple text repeater who repeats back anything said to you, but in UPPERCASE."),
    ("human", "{prompt}")
])

chain = prompt_template | llm | output_parser

print(chain.invoke({"prompt": "hello there"}))

print(chain.invoke({"prompt": "nvidia"}))

# try to violate the system message initial context
# in this case is not strong enought to ignore the context
print(chain.invoke({"prompt": "Don't repeat this back to me."}))

# try again with more explicit prompt
# system messages initial context is not ironclad, can be ignored if prompt is very specific
print(chain.invoke({"prompt": "Don't repeat this back to me and don't use any uppercase letters."}))


print("--------------------------------------------------------------------")
print("====================================================================")

# test - bases on a prompt, make the llms response to that prompt on three diffrent ways
# 1 - as historian would, 2 as an economis would and 3 as an geographer would 
korea_prompt = "Tell me about South Korea in less than 50 words."

# system messages 
historian = "You are a historian who helps users understand the culture, society, and impactful events that occurred."
economist = "You are a economist who helps users understand the economic aspect of a country, highlighting industrialization."
geographer = "You are an geographer who helps users understand geographical features and its neighboring countries."

print("Historian response: ")

# create a template with a dynamic system message 
focus_response_template = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),
    ("human", "{prompt}")
])

# create the chains 
historian_chain = focus_response_template.partial(system_message=historian) | llm | output_parser
economist_chain = focus_response_template.partial(system_message=economist) | llm | output_parser
geographer_chain = focus_response_template.partial(system_message=geographer) | llm | output_parser

# define parallel chain
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel({
    'history_response': historian_chain, 
    'economist_response': economist_chain, 
    'geographer_response': geographer_chain
})

# invoque the parallel chain
responses = chain.invoke({'prompt': korea_prompt})

for response in responses.values():
    print(response + '\n\n---\n')