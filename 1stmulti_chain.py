from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import SequentialChain  # Updated import path
from langchain.chains import LLMChain  # Updated import path
from langchain_community.llms import Ollama

# Initialize the model
llm = Ollama(model="qwen2:7b")

# Create prompt templates for each step
question_analyzer_prompt = ChatPromptTemplate.from_template(
    "Analyze this question and identify the key concepts I need to understand: {question}"
)

research_prompt = ChatPromptTemplate.from_template(
    "Based on the analysis: {analysis}, provide relevant information about these concepts."
)

answer_prompt = ChatPromptTemplate.from_template(
    "Using this research: {research}, provide a comprehensive answer to the original question: {question}"
)

# Create chains for each step
question_analyzer_chain = LLMChain(
    llm=llm,
    prompt=question_analyzer_prompt,
    output_key="analysis"
)

research_chain = LLMChain(
    llm=llm,
    prompt=research_prompt,
    output_key="research"
)

answer_chain = LLMChain(
    llm=llm,
    prompt=answer_prompt,
    output_key="answer"
)

# Combine the chains
sequential_chain = SequentialChain(
    chains=[question_analyzer_chain, research_chain, answer_chain],
    input_variables=["question"],
    output_variables=["analysis", "research", "answer"],
    verbose=True
)
question=input("enter ur query:")
# Test the chain
result = sequential_chain({"question":question})
print("\nFinal Answer:")
print(result["answer"])