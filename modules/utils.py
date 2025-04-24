import os
import json
import logging

from dotenv import load_dotenv
load_dotenv()

import chromadb
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

def cleaning_json_files(data_folder):
    """
    Reads and processes JSON files from the specified folder, extracting relevant shop data.

    This function iterates through all JSON files in the given directory, parses each file,
    and extracts information about shops including title, categories, subcategories, venue,
    keywords, and description. The extracted data from each shop is stored in a dictionary
    and added to a list, which is returned at the end.

    Args:
        data_folder (str): The path to the folder containing the JSON files.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains the extracted
            information for a shop. Each dictionary has the following structure:
            {
                'title': str,
                'categories': list[str],
                'subcategories': list[str],
                'venue': str,
                'keywords': list[str],
                'description': str
            }

    Raises:
        FileNotFoundError: If the specified folder does not exist.
        json.JSONDecodeError: If any JSON file is not properly formatted.
        KeyError: If expected keys are missing in the JSON structure.
    """
    # List all JSON files in the folder
    try:
        json_files = os.listdir(data_folder)
    except FileNotFoundError:
        logger.error(f"Data folder not found: {data_folder}")
        raise

    # Shops list
    shops = []

    # Read each JSON file
    for file_name in json_files:
        file_path = os.path.join(data_folder, file_name)
        logger.debug(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                for datapoint in data['docs']:
                    # Extracting shop details one-by-one
                    store = {}
                    store['title'] = datapoint['title']
                    
                    # Filtering categories and subcategories
                    categories = []
                    subcategories = []
                    for category in datapoint['categoryTree']:
                        categories.append(category['title'])
                        
                        for sub in category['subs']:
                            subcategories.append(sub['title'])
                        
                    store['categories'] = categories
                    store['subcategories'] = subcategories
                    
                    # Storing venue
                    store['venue'] = datapoint['venue']
                    
                    # Filtering and storing keywords as list
                    keywords = []
                    keywords_str = datapoint['keywords']
                    for keyword in keywords_str.split(','):
                        if(keyword != '' and keyword != '&'):
                            keywords.append(keyword)
                    store['keywords'] = keywords
                    
                    # Storing description of shop
                    store['description'] = datapoint['text']
                    
                    shops.append(store)
        except json.JSONDecodeError as jde:
            logger.error(f"Error decoding JSON from file {file_path}: {jde}")
            continue # Skip corrupted files or handle differently
        except KeyError as ke:
            logger.error(f"Missing expected key in file {file_path}: {ke}")
            continue # Skip files with unexpected structure
        except Exception as e:
            logger.error(f"Unexpected error processing file {file_path}: {e}", exc_info=True)
            continue
    
    logger.info(f"Finished JSON cleaning. Extracted data for {len(shops)} shops.")  
    return shops

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
            Keywords: {', '.join(shop['keywords'])}
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