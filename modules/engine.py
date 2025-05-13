import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from sentence_transformers import CrossEncoder

from typing_extensions import Annotated, TypedDict, Dict
from operator import add

from modules.tools import find_similar_shops
from modules.prompts import sam_prompt_template

# Logger
logger = logging.getLogger(__name__)

class MallAssistant():
    """
    A virtual assistant that utilizes a language model and a prompt-based flow graph 
    to answer user queries, particularly for use in mall guidance scenarios.

    Attributes:
        graph (CompiledStateGraph): The compiled flow graph used to process queries 
            through retrieval, response generation, and history maintenance steps.
    """
    # Initialize the models and the graph
    def __init__(self,
                 llm_model_name: str = 'gpt-4o-mini',
                 reranker_model_name: str = 'cross-encoder/ms-marco-MiniLM-L6-v2'):
        """
        Initializes the MallAssistant by setting up the prompt template, language model, 
        reranker, and compiling the state graph used for processing queries.

        Args:
            llm_model_name (str): The name of the language model to use. Defaults to 'gpt-4o-mini'.
            reranker_model_name (str): The name of the reranker model for reordering retrieved 
                documents. Defaults to 'cross-encoder/ms-marco-MiniLM-L6-v2'.

        Raises:
            Exception: If initialization of the graph or any components fails.
        """
        logger.info(f"Initializing MallAssistant with model: {llm_model_name}")
        try:
            prompt_template = PromptTemplate(input_variables=["context", "question"], 
                                            template=sam_prompt_template)
            llm = ChatOpenAI(model_name=llm_model_name)
            reranker = CrossEncoder(reranker_model_name)

            self.graph = self._init_llm_graph(prompt_template=prompt_template, llm=llm, reranker=reranker)
            logger.info("LLM graph initialized successfully.")
        except Exception as e:
            logger.exception("Error during MallAssistant initialization.")
            raise
    
    # Function to initialize the graph
    def _init_llm_graph(self,
                        prompt_template: PromptTemplate,
                        llm: ChatOpenAI,
                        reranker: CrossEncoder) -> CompiledStateGraph:
        """
        Initializes and compiles the LLM graph used for processing user queries.

        The graph follows a three-step pipeline:
        1. Retrieve relevant context using semantic similarity and reranking.
        2. Generate an answer based on the retrieved context using the language model.
        3. Maintain the history of the conversation.

        Args:
            prompt_template (PromptTemplate): The template used to structure prompts for the language model.
            llm (ChatOpenAI): The language model instance used to generate responses.
            reranker (CrossEncoder): The reranker used to reorder retrieved context documents by relevance.

        Returns:
            CompiledStateGraph: The compiled flow graph for handling queries.
        """
        logger.info("Initializing LLM graph components.")
        # Memory saver for conversational history
        memory = MemorySaver()
        
        # State to maintain objects across graph
        class State(TypedDict):
            question: str
            messages: Annotated[list[AnyMessage], add]
            context: str
            answer: str

        # Retreiving relevant stores with reranking
        def retrieve(state: State):
            """
            Retrieves relevant context documents for the user's question using similarity search 
            and reranking.

            Args:
                state (State): Contains the user's question.

            Returns:
                dict: Retrieved context under the key 'context'.
            """
            logger.debug(f"Retrieving context for question: {state['question'][:50]}...")
            try:
                retrieved_docs = find_similar_shops(state['question'])

                # Reranking (filters top 10 from the given 20)
                pairs = [(state['question'], doc) for doc in retrieved_docs]
                scores = reranker.predict(pairs)
                sorted_docs = sorted(zip(scores, retrieved_docs), reverse=True)
                relevant_shops = [doc[1] for doc in sorted_docs[:10]]

                context = " \n ".join(relevant_shops)
                logger.info(f"Retrieved relevant shops and combined to context.")
                return {"context": context}
            except Exception as e:
                logger.error(f"Error during context retrieval: {e}", exc_info=True)
                return {"context": ""}

        # Generates a response using conversation history and context
        def generate(state: State):
            """
            Generates an answer using the language model based on the question and retrieved context.

            Args:
                state (State): Includes the user's question and context.

            Returns:
                dict: Generated answer under the key 'answer'.
            """
            logger.debug("Generating response based on context.")
            try:
                # Filter conversation history (last 3 or less exchanges)
                history_str = state['messages'][-6:] if len(state['messages']) >= 6 else state['messages']

                prompt = prompt_template.invoke({"question": state["question"], "history": history_str, "context": state['context']})
                response = llm.invoke(prompt)
                logger.info("LLM response generated successfully.")
                return {"answer": response.content}
            except Exception as e:
                logger.error(f"Error during response generation: {e}", exc_info=True)
                return {"answer": "Sorry, I encountered an error while generating the response."}
        
        # Maintaining conversational history
        def maintain_history(state: State):
            """
            Updates the conversation history with the latest question and answer.

            Args:
                state (State): Contains the question and generated answer.

            Returns:
                dict: Updated messages under the key 'messages'.
            """
            logger.debug("Maintaining conversation history.")
            human_message = f"visitor: {state["question"]}"
            ai_message = f"ai_assistant: {state["answer"]}"
            
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
    
    # Used to process the user_query given thread_id
        # thread_id is used to maintain chat history
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

        # Throw error if 'thread_id' is missing
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
