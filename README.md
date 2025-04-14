# AI Mall Assistant: Smart Navigation & Recommendations

## Introduction 

Welcome to the AI Mall Assistant project, designed to enhance the visitor experience in large shopping malls through an intelligent chatbot interface.

This assistant employs a Large Language Model (LLM) to handle natural language interaction and query processing. The LLM interfaces with a structured knowledge base detailing specific mall information, including:
    - Store directories (names, categories, keywords)
    - Spatial layout data (floor plans, relative locations)

By processing user queries against this mall data via the LLM, the chatbot provides:

- AI-Powered Navigation: Translates natural language requests into navigational guidance within the mall context.
- Intelligent Recommendations: Filters and retrieves relevant shops, restaurants, or services from the knowledge base based on user needs expressed conversationally.

The primary goal is to deliver a robust and efficient system for real-time mall navigation assistance and personalized recommendations, leveraging the integration of an LLM with specific domain data.

- [AI Mall Assistant: Smart Navigation \& Recommendations](#ai-mall-assistant-smart-navigation--recommendations)
  - [Introduction](#introduction)
  - [System Information](#system-information)
  - [Workflow](#workflow)

## System Information

1. **Data Processing & Storage**
   - Mall information (shop details, categories, descriptions, keywords, venue) is expected in JSON format located within the `data/` directory.
   - The `utils.py` module contains functions responsible for parsing these JSON files and extracting relevant shop attributes.
   - Processed shop data is then embedded using OpenAI's `text-embedding-3-small` model.
   - These embeddings, along with associated metadata, are stored persistently in a ChromaDB vector database, with data files located in the `chromadb/` directory.

2. **Retrieval**
    - The `tools.py` module defines functions that perform semantic similarity searches against the ChromaDB 'shops' collection.
    - Given a user query, this component retrieves the most relevant shop information from the vector database to be used as context.

3. **LLM Engine**
    - The `engine.py` module orchestrates the core RAG workflow, likely using a state graph.
    - It takes a user query, triggers the retrieval step, formats the query and retrieved context using templates defined in `prompts.py`, and then generates a response.
    - The primary Large Language Model used for generation is `gpt-4-turbo` accessed via the `ChatOpenAI` integration.

4. **Configuration**
    - Sensitive information, such as API keys, should be managed via environment variables stored in the `.env` file.

## Workflow

1. **Query Input**
   - The user submits a natural language query to the AI Mall Assistant (e.g., "Where can I find a coffee shop near the entrance?", "Suggest a store for buying sportswear").

2. **Context Retrieval**

   - The query is used to perform a semantic search against the ChromaDB vector store.
   - The most relevant shop information based on the query's meaning is retrieved from ChromaDB.

3. **Prompt Construction**

   - A predefined prompt template is populated with the original user query and the context retrieved in the previous step.
   - This structured prompt prepares the information for the Language Model, instructing it on how to generate the desired response using the provided context.

4. **LLM Generation**

   - The complete, formatted prompt is sent to the configured Large Language Model.
   - The LLM processes the input, synthesizing the retrieved context to generate a coherent and helpful answer tailored to the user's original query.

5. **Response Output**

   - The final generated text response from the LLM is returned to the user, providing the requested navigation guidance or shop recommendation.