import logging
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

def find_similar_shops(query: str):
    """
    Performs a similarity search on the 'shops' ChromaDB collection using the provided query.

    Args:
        query (str): A text query used to find semantically similar shop entries.

    Returns:
        List[str]: A list of 20 shop document strings that are most similar to the query.
    """
    logger.info(f"Performing similarity search for query: {query[:50]}...")
    
    # Using OpenAI Embedding Model
    embed_model = OpenAIEmbeddings(model='text-embedding-3-small')
    vectordb = Chroma(
        collection_name='shops',
        embedding_function=embed_model,
        persist_directory="./chromadb")
    
    # Getting the top 20 results using similarity search
    results = vectordb.similarity_search(query, k=20)
    logger.info(f"Found {len(results)} similar shops for query.")

    return [result.page_content for result in results]