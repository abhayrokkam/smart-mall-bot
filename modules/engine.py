from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from modules.tools import find_similar_shops
from modules.prompts import sam_prompt_template

def get_llm_flow_graph(prompt_template: PromptTemplate,
                       llm: ChatOpenAI) -> CompiledStateGraph:
    """
    Constructs and compiles a state graph for an LLM workflow that retrieves context 
    and generates a response based on a user query.

    Args:
        prompt_template (PromptTemplate): A prompt template used to format the input 
            question and retrieved context for the LLM.
        llm (ChatOpenAI): An instance of a ChatOpenAI language model used to generate responses.

    Returns:
        CompiledStateGraph: A compiled state graph representing the flow of retrieving 
        context and generating an answer.

    Notes:
        The flow consists of two main steps:
        - `retrieve`: Uses `find_similar_shops` to fetch context relevant to the query.
        - `generate`: Formats the question and context with the prompt template and 
          invokes the LLM to produce an answer.

        The graph executes in the following order:
        START → retrieve → generate.    
    """
    class State(TypedDict):
        question: str
        context: str
        answer: str

    def retrieve(state: State):
        retrieved_info = find_similar_shops(state['question'])
        context = " \n ".join(retrieved_info)
        
        return {"context": context}

    def generate(state: State):
        messages = prompt_template.invoke({"question": state["question"], "context": state['context']})
        response = llm.invoke(messages)
        
        return {"answer": response.content}

    graph_builder = StateGraph(State)

    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)

    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")

    graph = graph_builder.compile()
    
    return graph

def get_llm_response(user_query: str) -> str:
    """
    Generates a response to a user's query using a prompt-driven LLM workflow.

    Args:
        user_query (str): The input question or query from the user.

    Returns:
        str: The LLM-generated answer based on the query and retrieved context.

    Notes:
        - Uses a predefined prompt template and the GPT-4 Turbo model via `ChatOpenAI`.
        - The workflow is defined as a state graph built by `get_llm_flow_graph`, 
          which first retrieves relevant context and then generates a response.
    """
    prompt_template = PromptTemplate(input_variables=["context", "question"],
                                    template=sam_prompt_template)

    llm = ChatOpenAI(model_name="gpt-4-turbo")
    
    graph = get_llm_flow_graph(prompt_template=prompt_template, llm=llm)
    
    response = graph.invoke({"question": user_query})
    
    return response['answer']