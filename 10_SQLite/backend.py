from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import BaseMessage, HumanMessage
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv; load_dotenv()
from typing import TypedDict, Annotated
import os
import sqlite3

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages] # ~ We can use operator.add as well, but this more optimized for BaseMessages

def chat_node(state: ChatState):

    prompt = f"You are a helpful assistant, answer the following query in as short as possible: {state['messages']}"
    response = model.invoke(prompt)

    return {'messages': [response]}

def ret_all_threads():

    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1',
    task='text-generation',
    huggingfacehub_api_token=os.getenv('HF_TOKEN')
)
model = ChatHuggingFace(llm=llm)

'''
check_same_thread is a boolean parameter used in the sqlite3.connect() function that 
controls whether the connection can be shared across multiple threads.

Key Behavior:- 

    True (Default): The module raises a ProgrammingError if you try to use a database 
    connection in a thread other than the one that created it.

    False: You can share the connection across multiple threads. However, you 
    must manage write serialization yourself (e.g., using a threading lock) to 
    prevent data corruption or transaction errors.
'''
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

workflow = graph.compile(checkpointer=checkpointer)

# CONFIG = {'configurable': {'thread_id': '1'}}
# response = workflow.invoke({'messages': [HumanMessage(content='What is my chat hostory so far...')]}, config=CONFIG)
# print(response)

'''
Input - Pasta Types and origins...

Output - {'messages': [HumanMessage(content='Pasta Types and origins...', additional_kwargs={}, response_metadata={}, id='018c0782-eb0c-4739-92dd-93de061f0a7b'), AIMessage(content='\nHere\'s a concise overview of key pasta types and their origins:\n\n1. **Spaghetti**: Long, thin strands. Originated in **Naples, Italy**.\n2. **Penne**: Tube-shaped with angled cuts. From **Campania/Liguria, Italy**.\n3. **Fettuccine**: Flat ribbons. Classic to **Roman cuisine (Lazio, Italy)**.\n4. **Tortellini**: Stuffed ring-shaped pasta. Traditional from **Bologna, Emilia-Romagna, Italy**.\n5. **Farfalle**: Bow-tie/butterfly shape. Originated in **Lombardy/Emilia-Romagna, Italy**.\n6. **Orecchiette**: Small "ear"-shaped discs. Native to **Puglia, Italy**.\n7. **Lasagna**: Layered sheets. Ancient roots across **Italy**, notably Emilia-Romagna.\n8. **Ravioli**: Stuffed squares/pouches. Earliest records in **Central/Northern Italy** (14th century).\n\nMost pasta shapes developed regionally across Italy, with variations tied to local ingredients and traditions.', additional_kwargs={}, response_metadata={'token_usage': {'completion_tokens': 467, 'prompt_tokens': 71, 'total_tokens': 538}, 'model_name': 'deepseek-ai/DeepSeek-R1', 'system_fingerprint': '', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019e630e-18b3-7422-9c60-60dd3a02ff61-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 71, 'output_tokens': 467, 'total_tokens': 538})]}

Input - What is my chat hostory so far...

Output - {'messages': [HumanMessage(content='Pasta Types and origins...', additional_kwargs={}, response_metadata={}, id='018c0782-eb0c-4739-92dd-93de061f0a7b'), AIMessage(content='\nHere\'s a concise overview of key pasta types and their origins:\n\n1. **Spaghetti**: Long, thin strands. Originated in **Naples, Italy**.\n2. **Penne**: Tube-shaped with angled cuts. From **Campania/Liguria, Italy**.\n3. **Fettuccine**: Flat ribbons. Classic to **Roman cuisine (Lazio, Italy)**.\n4. **Tortellini**: Stuffed ring-shaped pasta. Traditional from **Bologna, Emilia-Romagna, Italy**.\n5. **Farfalle**: Bow-tie/butterfly shape. Originated in **Lombardy/Emilia-Romagna, Italy**.\n6. **Orecchiette**: Small "ear"-shaped discs. Native to **Puglia, Italy**.\n7. **Lasagna**: Layered sheets. Ancient roots across **Italy**, notably Emilia-Romagna.\n8. **Ravioli**: Stuffed squares/pouches. Earliest records in **Central/Northern Italy** (14th century).\n\nMost pasta shapes developed regionally across Italy, with variations tied to local ingredients and traditions.', additional_kwargs={}, response_metadata={'token_usage': {'completion_tokens': 467, 'prompt_tokens': 71, 'total_tokens': 538}, 'model_name': 'deepseek-ai/DeepSeek-R1', 'system_fingerprint': '', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019e630e-18b3-7422-9c60-60dd3a02ff61-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 71, 'output_tokens': 467, 'total_tokens': 538}), HumanMessage(content='What is my chat hostory so far...', additional_kwargs={}, response_metadata={}, id='f1de9fed-0d6d-4375-ad70-056ceda93dd1'), AIMessage(content='\nYour chat history so far:\n\n1. **You asked:** "Pasta Types and origins..."  \n2. **I responded** with a concise list of 8 pasta types and their Italian regional origins (e.g., Spaghetti from Naples, Tortellini from Bologna).  \n\nThat’s the full history! 😊', additional_kwargs={}, response_metadata={'token_usage': {'completion_tokens': 247, 'prompt_tokens': 527, 'total_tokens': 774}, 'model_name': 'deepseek-ai/DeepSeek-R1', 'system_fingerprint': '', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019e6310-a122-7eb3-8b50-3238105ec183-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 527, 'output_tokens': 247, 'total_tokens': 774})]}
'''