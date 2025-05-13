import json
import logging

from dotenv import load_dotenv
load_dotenv()

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

def push_to_chroma(data_path,
                   persist_path = './chromadb'):
    """
    Loads shop data from a JSON file, generates embeddings using OpenAI, and adds 
    the data to a ChromaDB collection for persistent storage.

    Args:
        data_path (str): Path to the JSON file containing shop data.
        persist_path (str, optional): Directory path where the ChromaDB instance 
            will persist data. Defaults to './chromadb'.

    Raises:
        Exception: If there is an error creating or retrieving the ChromaDB collection.

    Notes:
        - Each shop entry must include 'title', 'venue', 'categories', 'keywords', 
          and 'description'. 'Subcategories' is optional.
        - Uses OpenAI's 'text-embedding-3-small' model for embedding.
        - Document content is formatted for readability; embeddings use a simplified input.
    """
    logger.info(f"Starting push to ChromaDB from data path: {data_path}, persist path: {persist_path}")
    
    # Collection
    try:
        client = chromadb.PersistentClient(path=persist_path)
        model = OpenAIEmbeddings(model='text-embedding-3-small')
        collection_name = "shops"
        
        collection = client.get_or_create_collection(name=collection_name)
        
        # Load shop data
        logger.debug(f"Loading shop data from: {data_path}")
        with open(data_path, 'r') as f:
            shops = json.load(f)
        logger.info(f"Loaded {len(shops)} shop entries from {data_path}")
        
        # Lists for collection
        ids = []
        documents = []
        documents_for_embeddings = []
        metadatas = []

        for shop in shops:
            # Ids
            ids.append(f'{shop['title']} | {shop['venue']}')
            
            # Documents
            content = f"""
            Title: {shop['title']}
            Venue: {shop['venue']}
            Categories: {', '.join(shop['categories'])}
            Subcategories: {', '.join(shop.get('subcategories', []))}
            Description: {shop['description']}
            """
            documents.append(content)
            
            # Documents uniquely for creating embeddings
            parts = []
            parts.append(shop['title'])
            parts.append(", ".join(shop['categories']))
            parts.append(", ".join(shop['subcategories']))
            parts.append(", ".join(shop['keywords']))
            documents_for_embeddings.append(" | ".join(parts))
            
            # Metadata of documents
            metadata={
                'title': shop['title'],
                'categories': ', '.join(shop['categories']),
                'subcategories': ', '.join(shop['subcategories']),
                'venue': shop['venue'],
            }
            metadatas.append(metadata)

        # Embeddings 
        embeddings = model.embed_documents(documents_for_embeddings)
        logger.info(f"Generated {len(embeddings)} embeddings.")
        
        # Add to collection
        logger.info(f"Adding {len(ids)} documents to ChromaDB collection '{collection_name}'")
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info("Successfully added data to ChromaDB.")
    except Exception as e:
        logger.exception(f"Error during push_to_chroma from {data_path}: {e}") 
        raise

def messages_to_string(messages: list) -> str:
    """
    Converts a list of HumanMessage and AIMessage objects into a formatted string.

    Each HumanMessage is labeled as "visitor:", and each AIMessage is labeled as "assistant:".
    Messages are separated by newlines.

    Args:
        messages (list): A list containing HumanMessage and AIMessage objects.

    Returns:
        str: A formatted string representation of the message list. Returns an empty string if no messages are provided.
    """
    # Empty list
    if not messages: 
        logger.info("Received empty message list, returning empty string.")
        return ""

    logger.info(f"Formatting messages into string")
    lines = []
    
    # Process one-by-one with 'isinstance'
    for message in messages:
        try:
            if isinstance(message, HumanMessage):
                lines.append(f"visitor:\n{message.content}")
            elif isinstance(message, AIMessage):
                lines.append(f"assistant:\n{message.content}")
            else:
                logger.warning(f"Skipped unsupported message type: {type(message).__name__}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            continue
    
    # Join all strings from the list
    result = "\n".join(lines)
    logger.info(f"Messages formatted to string")
    return result