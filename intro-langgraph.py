# Install the required package first if not already installed
# pip install langchain-ollama

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# Import the correct Ollama class
from langchain_ollama import OllamaLLM

# Define our state
class AgentState(TypedDict):
    messages: list
    next_step: Literal["research", "answer", "clarify", "end"]

# Initialize the model with a model that's likely available
# Note: Use a model you've already pulled or one of the default models
llm = OllamaLLM(model="llama3:8b")  # Change from llama3 to llama2 or another model you have

# Define our nodes
def determine_next_step(state: AgentState) -> AgentState:
    """Determine what to do next based on the conversation."""
    messages = state["messages"]
    
    # Convert messages to a string representation for the LLM
    message_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    response = llm.invoke(
        f"""Based on the conversation so far, determine the next step to take.
        Conversation: {message_text}
        Choose from: 'research', 'answer', 'clarify', or 'end'."""
    )
    
    # Simple parsing of the response
    if "research" in response.lower():
        state["next_step"] = "research"
    elif "answer" in response.lower():
        state["next_step"] = "answer"
    elif "clarify" in response.lower():
        state["next_step"] = "clarify"
    else:
        state["next_step"] = "end"
    
    return state

def research(state: AgentState) -> AgentState:
    """Research information related to the question."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    research_result = llm.invoke(
        f"Research this topic and provide relevant information: {last_message}"
    )
    
    state["messages"].append(AIMessage(content=f"Research findings: {research_result}"))
    return state

def answer_question(state: AgentState) -> AgentState:
    """Provide a final answer based on the conversation."""
    messages = state["messages"]
    
    # Convert messages to a string representation for the LLM
    message_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    
    answer = llm.invoke(
        f"Based on our conversation and research, provide a final answer: {message_text}"
    )
    
    state["messages"].append(AIMessage(content=f"Final answer: {answer}"))
    state["next_step"] = "end"
    return state

def ask_for_clarification(state: AgentState) -> AgentState:
    """Ask for clarification about the question."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    clarification = llm.invoke(
        f"Ask for clarification about this question: {last_message}"
    )
    
    state["messages"].append(AIMessage(content=clarification))
    return state

# Define a router function to determine the next step
def router(state: AgentState):
    return state["next_step"]

# Build our graph
workflow = StateGraph(AgentState)

# Add our nodes
workflow.add_node("determine_next_step", determine_next_step)
workflow.add_node("research", research)
workflow.add_node("answer", answer_question)
workflow.add_node("clarify", ask_for_clarification)

# Add conditional edges using the router
workflow.add_conditional_edges(
    "determine_next_step",
    router,
    {
        "research": "research",
        "answer": "answer",
        "clarify": "clarify",
        "end": END
    }
)

# Add regular edges
workflow.add_edge("research", "determine_next_step")
workflow.add_edge("clarify", "determine_next_step")
workflow.add_edge("answer", END)

# Set our entry point
workflow.set_entry_point("determine_next_step")

# Compile the graph
graph = workflow.compile()
c=input("enter ur query:")
# Test our graph
result = graph.invoke({
    "messages": [HumanMessage(content=c)],
    "next_step": "determine_next_step"
})

# Print the conversation
for message in result["messages"]:
    if isinstance(message, HumanMessage):
        print(f"Human: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")