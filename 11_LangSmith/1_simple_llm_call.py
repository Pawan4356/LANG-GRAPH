import os
from dotenv import load_dotenv; load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id='deepseek-ai/DeepSeek-R1',
        task='text-generation',
        huggingfacehub_api_token=os.getenv('HF_TOKEN')
    )
)
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Peru? (Keep the answer as minimal as possible!)"})
print(result)

'''
Lima
'''