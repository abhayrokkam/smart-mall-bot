import os
import json

from dotenv import load_dotenv
load_dotenv()

import chromadb
from langchain_openai import OpenAIEmbeddings

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
    json_files = os.listdir(data_folder)

    # Shops list
    shops = []

    # Read each JSON file
    for file_name in json_files:
        file_path = os.path.join(data_folder, file_name)
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
    client = chromadb.PersistentClient(path=persist_path)
    model = OpenAIEmbeddings(model='text-embedding-3-small')
    collection_name = "shops"
    
    # Collection
    try:
        collection = client.get_or_create_collection(name=collection_name)
    except Exception as e:
        print(f"Error creating or getting collection: {e}")
        
    # Load shop data
    with open(data_path, 'r') as f:
        shops = json.load(f)
    
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
    
    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )