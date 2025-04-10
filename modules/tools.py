from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def find_similar_shops(query: str):
    """
    Performs a similarity search on the 'shops' ChromaDB collection using the provided query.

    Args:
        query (str): A text query used to find semantically similar shop entries.

    Returns:
        List[str]: A list of up to 4 shop document strings that are most similar to the query.
    """    
    embed_model = OpenAIEmbeddings(model='text-embedding-3-small')
    vectordb = Chroma(
        collection_name='shops',
        embedding_function=embed_model,
        persist_directory="./chromadb")
    
    results = vectordb.similarity_search(query, k=4)

    return [result.page_content for result in results]