import os
from dotenv import load_dotenv; load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
# To create seperate project in langsmith for different workflows
os.environ['LANGSMITH_PROJECT'] = 'temp2'

prompt1 = PromptTemplate(
    template='Generate a 3 points report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a one pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id='deepseek-ai/DeepSeek-R1',
        task='text-generation',
        huggingfacehub_api_token=os.getenv('HF_TOKEN')
    )
)
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

# Custom run_name, tags and metadata
config = {
    'run_name': 'SequentialChain',
    'tags': ['llm app', 'summarization'],
    'metadata': {'model': 'deepseek-ai/DeepSeek-R1'}
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)