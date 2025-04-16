from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from modules.tools import find_similar_shops
from modules.prompts import sam_prompt_template

class MallAssistant():
    """
    A virtual assistant for answering queries using a language model and a prompt-based flow graph.
    
    This class utilizes a pre-defined prompt template and a language model to process user queries 
    and return intelligent responses. It is designed for use cases such as guiding users in a mall environment.
    """
    def __init__(self,
                 llm_model_name: str = 'gpt-4o-mini'):
        """
        Initializes the MallAssistant with a language model and a prompt-based flow graph.

        Sets up the prompt template with required input variables and loads a lightweight GPT-4 model.
        Constructs the flow graph that will be used to handle user queries.
        """
        prompt_template = PromptTemplate(input_variables=["context", "question"], 
                                         template=sam_prompt_template)
        
        llm = ChatOpenAI(model_name=llm_model_name)
        
        self.graph = self._init_llm_graph(prompt_template=prompt_template,
                                          llm=llm)
    
    def _init_llm_graph(self,
                        prompt_template: PromptTemplate,
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
            prompt = prompt_template.invoke({"question": state["question"], "context": state['context']})
            response = llm.invoke(prompt)
            
            return {"answer": response.content}

        graph_builder = StateGraph(State)

        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("generate", generate)

        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")

        graph = graph_builder.compile()
        
        return graph
    
    def get_response(self, user_query: str):
        """
        Processes a user query and returns a response generated by the language model.

        Args:
            user_query (str): The user's input question or query.

        Returns:
            str: The assistant's generated answer to the query.
        """
        response = self.graph.invoke({"question": user_query})
        return response['answer']