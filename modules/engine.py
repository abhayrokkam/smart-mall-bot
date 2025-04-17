from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from typing_extensions import Annotated, TypedDict, Dict
from operator import add

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
        Initializes and compiles a stateful LLM graph for context retrieval, 
        response generation, and conversation history tracking.

        Args:
            prompt_template (PromptTemplate): A template used to format the prompt 
                with the user's question and retrieved context.
            llm (ChatOpenAI): An instance of a ChatOpenAI model used to generate responses.

        Returns:
            CompiledStateGraph: A compiled graph that handles the flow of retrieving 
            context, generating a response, and maintaining message history.

        Notes:
            - The state includes `question`, `context`, `answer`, and a message history (`messages`).
            - The graph consists of three nodes:
                1. `retrieve`: Finds context relevant to the user's question using `find_similar_shops`.
                2. `generate`: Uses the prompt template and context to query the LLM.
                3. `history`: Stores the user question and generated answer as message objects.
            - The graph is checkpointed using `MemorySaver` to preserve state transitions.
            - Execution flow: START → retrieve → generate → history.
        """
        memory = MemorySaver()
        
        class State(TypedDict):
            question: str
            messages: Annotated[list[AnyMessage], add]
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
        
        def maintain_history(state: State):
            human_message = HumanMessage(state["question"])
            ai_message = AIMessage(state["answer"])
            
            return {"messages": [human_message, ai_message]}

        graph_builder = StateGraph(State)

        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("generate", generate)
        graph_builder.add_node("history", maintain_history)

        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", "history")

        graph = graph_builder.compile(checkpointer = memory)
        
        return graph
    
    def process_user_query(self, 
                           user_query: str,
                           config: Dict):
        """
        Processes a user query using the initialized LLM graph and returns the response 
        along with the updated conversation history.

        Args:
            user_query (str): The input question or query from the user.
            config (Dict): A configuration dictionary that must include a `"thread_id"` key 
                to maintain conversation history across interactions.

        Returns:
            Dict: A dictionary containing:
                - "response" (str): The generated answer from the LLM.
                - "history" (list): The updated list of messages in the conversation thread.

        Raises:
            ValueError: If "thread_id" is not provided in the config dictionary.
        """
        if "thread_id" not in config:
            raise ValueError('"thread_id" is required in the config to track conversation history.')
        
        response = self.graph.invoke({"question": user_query}, {"configurable": config})
        
        output = {"response": response['answer'],
                  "history": response['messages']}
        
        return output