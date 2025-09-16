from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(name='name', description="the name of the person")
    age: int = Field(name='age', gt=18, description="the age of the person")
    city: str = Field(name='city', description="the city where the person lives")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate a {person} with a name, age, and city.\n{format_instructions}",
    input_variables=['person'],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)


chain = template | model | parser
res = chain.invoke({'person': 'MS Dhoni'})
print(res)


# # print(template)
# # print("---------------------------------------------------")

# prompt = template.invoke({"person": "MS Dhoni"})
# # print(prompt)
# # print("---------------------------------------------------")
# response = model.invoke(prompt)
# final_result = parser.parse(response.content)
# print(final_result)