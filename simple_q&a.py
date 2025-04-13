from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the Ollama LLM with the updated class
llm = OllamaLLM(model="qwen2:7b")

# Create a simple prompt template
prompt = PromptTemplate(
    template="Question: {question}\nAnswer:",
    input_variables=["question"]
)

# Build a simple chain
chain = prompt | llm | StrOutputParser()

question=input("enter ur query:")
# Test the chain
response = chain.invoke({"question": question})
print(response)