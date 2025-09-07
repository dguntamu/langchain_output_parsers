from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

json_parser = JsonOutputParser()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': json_parser.get_format_instructions()}
)

# Way -1 without using chain
# prompt = template1.invoke({'topic':'circket world cup 2023'})
# result = model.invoke(prompt)

# json_output = json_parser.parse(result.content)
# print(json_output)


# Way -2 using chain

chain = template1 | model | json_parser
result = chain.invoke({'topic':'java opening in 2025 in India'})
print(result)