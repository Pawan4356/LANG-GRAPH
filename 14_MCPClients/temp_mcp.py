from fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun

mcp = FastMCP('MyTools')
search = DuckDuckGoSearchRun()

@mcp.tool()
def search_web(query: str):
    return search.run(query)

@mcp.tool()
def calculator(
    first_num: float,
    second_num: float,
    operation: str
):
    if operation == "add":
        return first_num + second_num
    elif operation == "sub":
        return first_num - second_num
    elif operation == "mul":
        return first_num * second_num
    elif operation == "div":
        return first_num / second_num

if __name__ == "__main__":
    mcp.run()