import os
import json

def cleaning_json_files(data_folder):
    """
    Process and clean shop data from multiple JSON files in a directory.
    
    Extracts shop entries from each file's 'docs' array, transforms hierarchical categories
    into flat lists, cleans keyword strings, and returns standardized shop records.
    
    Args:
        data_folder (str): Path to directory containing source JSON files
    
    Returns:
        list: Dictionaries with structured shop data containing:
            - title (str)
            - categories (list)
            - subcategories (list) 
            - level (dict)
            - keywords (list)
            - description (str)
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