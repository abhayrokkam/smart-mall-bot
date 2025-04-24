import logging
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

# Logger
logger = logging.getLogger(__name__)

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
        logger.info(f"Initializing MallAssistant with model: {llm_model_name}")
        try:
            prompt_template = PromptTemplate(input_variables=["context", "question"], 
                                            template=sam_prompt_template)
            llm = ChatOpenAI(model_name=llm_model_name)
            
            self.graph = self._init_llm_graph(prompt_template=prompt_template, llm=llm)
            logger.info("LLM graph initialized successfully.")
        except Exception as e:
            logger.exception("Error during MallAssistant initialization.")
            raise
    
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
        logger.info("Initializing LLM graph components.")
        memory = MemorySaver()
        
        class State(TypedDict):
            question: str
            messages: Annotated[list[AnyMessage], add]
            context: str
            answer: str

        def retrieve(state: State):
            logger.debug(f"Retrieving context for question: {state['question'][:50]}...")
            try:
                retrieved_info = find_similar_shops(state['question'])
                context = " \n ".join(retrieved_info)
                logger.info(f"Retrieved {len(retrieved_info)} context snippets.")
                return {"context": context}
            except Exception as e:
                logger.error(f"Error during context retrieval: {e}", exc_info=True)
                return {"context": ""}

        def generate(state: State):
            logger.debug("Generating response based on context.")
            try:
                prompt = prompt_template.invoke({"question": state["question"], "context": state['context']})
                response = llm.invoke(prompt)
                logger.info("LLM response generated successfully.")
                return {"answer": response.content}
            except Exception as e:
                logger.error(f"Error during response generation: {e}", exc_info=True)
                return {"answer": "Sorry, I encountered an error while generating the response."}
        
        def maintain_history(state: State):
            logger.debug("Maintaining conversation history.")
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
        logger.info(f"Processing user query for thread_id: {config.get('thread_id', 'N/A')}")
        if "thread_id" not in config:
            logger.error("Missing 'thread_id' in config.")
            raise ValueError('"thread_id" is required in the config to track conversation history.')
        
        try:
            response = self.graph.invoke({"question": user_query}, {"configurable": config})
            
            output = {"response": response['answer'],
                      "history": response['messages']}
            logger.info(f"Successfully processed query for thread_id: {config['thread_id']}")
            return output
        except Exception as e:
            logger.exception(f"Error processing query for thread_id {config.get('thread_id', 'N/A')}: {e}") 
            raise 
