from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
import os
from dotenv import load_dotenv; load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

# MCP client for local FastMCP server
client = MultiServerMCPClient(
    {
        "test": {
            "transport": "stdio",
            "command": "python3",
            "args": ["/home/pawan/Projects/Langgraph/14_MCPClients/temp_mcp.py"],
        },
    }
)

model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1",
        task="text-generation",
        huggingfacehub_api_token=os.getenv('HF_TOKEN')
    )
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph():

    tools = await client.get_tools()
    print(tools)
    """
    Starting MCP server 'MyTools' with transport 'stdio'        transport.py:209
    [StructuredTool(name='search_web', args_schema={'additionalProperties': False,
     'properties': {'query': {'type': 'string'}}, 'required': ['query'], 'type': 
    'object'}, metadata={'_meta': {'fastmcp': {'tags': []}}}, 
    response_format='content_and_artifact', coroutine=<function 
    convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x788acac34c20>), 
    StructuredTool(name='calculator', args_schema={'additionalProperties': False, 
    'properties': {'first_num': {'type': 'number'}, 'second_num': {'type': 
    'number'}, 'operation': {'type': 'string'}}, 'required': ['first_num', 
    'second_num', 'operation'], 'type': 'object'}, metadata={'_meta': {'fastmcp': 
    {'tags': []}}}, response_format='content_and_artifact', coroutine=<function 
    convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x788acac34860>)]
    """

    model_with_tools = model.bind_tools(tools=tools)

    async def chat_node(state: ChatState):

        messages = state['messages']
        response = await model_with_tools.ainvoke(messages)
        return {'messages': [response]}

    # Internally async
    tool_node = ToolNode(tools=tools)
    
    graph = StateGraph(ChatState)

    graph.add_node('chat_node', chat_node)
    graph.add_node('tools', tool_node)

    graph.add_edge(START, 'chat_node')
    graph.add_conditional_edges('chat_node', tools_condition)
    graph.add_edge('tools', 'chat_node')

    workflow = graph.compile()

    return workflow

async def main():

    chatbot = await build_graph()
    result = await chatbot.ainvoke({'messages': [HumanMessage(content='Latest global news...')]})
    print(result["messages"][-1].content)

if __name__=='__main__':
    asyncio.run(main())