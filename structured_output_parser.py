from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
        ResponseSchema(name="name", description="the name of the person"),
        ResponseSchema(name="age", description="the age of the person"),
        ResponseSchema(name="city", description="the city where the person lives")
    ]


parser = StructuredOutputParser.get_format_instructions(schema)

template = PromptTemplate(
    template="Generate a {person} with a name, age, and city.\n{format_instructions}",
    input_variables=['person'],
    partial_variables={"format_instructions": schema.get_format_instructions()}
)

prompt = template.invoke({"person": "MS Dhoni"})
#print(prompt)
#prompt = template.format_prompt(person="person")

response = model.invoke(prompt)
final_result = parser.parse(response.content)
print(final_result)


