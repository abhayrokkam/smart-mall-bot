
# SMART MALL ASSISTANT

A production-grade, AI-powered digital concierge designed to serve real-time, context-aware assistance to visitors of Sunway Pyramid Mall. This project demonstrates advanced natural language processing, semantic search, and modular backend orchestration — tailored for large-scale commercial deployments.

Built using **FastAPI**, **OpenAI embeddings**, and **Chroma vector search**, the system supports intelligent retrieval and dialogue management, enabling users to find shops and services through natural language queries in a kiosk or digital assistant environment.

- [SMART MALL ASSISTANT](#smart-mall-assistant)
  - [Project Objectives](#project-objectives)
  - [Core Features](#core-features)
    - [Conversational Intelligence with Domain Context](#conversational-intelligence-with-domain-context)
    - [Semantic Search Engine with Embedding Matching](#semantic-search-engine-with-embedding-matching)
    - [Modular Data Ingestion Pipeline](#modular-data-ingestion-pipeline)
    - [Scalable Backend Architecture](#scalable-backend-architecture)
    - [Logging and Observability](#logging-and-observability)
  - [Architecture Overview](#architecture-overview)
    - [1. Data Layer](#1-data-layer)
    - [2. AI Orchestration Layer](#2-ai-orchestration-layer)
    - [3. API Layer](#3-api-layer)
  - [Project Structure](#project-structure)
  - [Real-World Use Case](#real-world-use-case)
  - [Technology and Skills Demonstrated](#technology-and-skills-demonstrated)
    - [Artificial Intelligence](#artificial-intelligence)
    - [Software Engineering](#software-engineering)
    - [Backend Infrastructure](#backend-infrastructure)
  - [Requirements](#requirements)

## Project Objectives

The core goal of this assistant is to provide a **highly responsive, intelligent, and scalable interface** that allows mall visitors to interact with a kiosk or app as if speaking with a human concierge. It addresses common real-world challenges such as:

- Efficiently locating shops based on vague or open-ended descriptions.
- Maintaining dialogue context across multiple user interactions.
- Providing domain-specific responses grounded in an up-to-date knowledge base.

This implementation is optimized for real-world usage in environments with heavy foot traffic, diverse visitor intents, and a broad range of retail services.

## Core Features

### Conversational Intelligence with Domain Context

- Designed around a persona-driven prompt to simulate a knowledgeable mall assistant named *Sam*.
- Responses are generated with **LLM prompting strategies** that encourage helpfulness, completeness, and consistency.
- Multi-turn dialogue support via **threaded conversation memory** powered by LangChain’s message abstraction.

### Semantic Search Engine with Embedding Matching

- Embeds incoming user queries and shop data using **OpenAI’s `text-embedding-3-small` model**.
- Executes **semantic similarity search** against a local **ChromaDB** vector index.
- Returns results that are meaningfully related to user intent, rather than relying on exact keyword matches.

### Modular Data Ingestion Pipeline

- Accepts structured `.json` files representing store data and ingests them into the system dynamically.
- Prepares multiple textual variations of metadata fields for robust embedding accuracy.
- Automatically generates document IDs and stores metadata alongside embeddings for future reference.

### Scalable Backend Architecture

- Built with **FastAPI**, enabling clean API exposure and efficient routing.
- Fully asynchronous request handling for high concurrency under user load.
- Structured logging with `RotatingFileHandler` to capture critical application events and diagnostics.

### Logging and Observability

- Centralized logging architecture implemented via `logging_config.py`.
- Logs all operations from API calls to vector search results, aiding in debugging and monitoring in production environments.

## Architecture Overview

The system consists of three main functional layers:

### 1. Data Layer
Handles ingestion and persistent storage of domain knowledge:

- Converts raw shop metadata into embedded vector format.
- Stores vectors in **ChromaDB** for low-latency, high-relevance similarity search.
- Enables dynamic updates by re-ingesting modified `.json` files without full re-deployment.

### 2. AI Orchestration Layer
Implements logic for processing and responding to user queries:

- Routes messages through the `MallAssistant` engine (`engine.py`).
- Maintains session state and historical memory using `thread_id` tracking.
- Conducts vector retrieval via `tools.py` and composes final responses with context-aware prompt injection.

### 3. API Layer
Exposes endpoints through FastAPI for:

- Uploading new shop data
- Interacting with the assistant
- Retrieving historical dialogue sessions

All endpoints are documented in the source code and follow RESTful principles.

## Project Structure

```
.
├── main.py                         # FastAPI app and API routing
├── chromadb/                       # Local vector DB storage (persisted embeddings)
└── modules/
    ├── engine.py                   # MallAssistant orchestration logic
    ├── tools.py                    # Vector similarity search using ChromaDB
    ├── utils.py                    # Embedding and document processing pipeline
    ├── prompts.py                  # Persona-driven assistant prompt
    ├── logging_config.py           # Logging setup (file and console handlers)
```

## Real-World Use Case

This assistant is designed to simulate a **smart kiosk system** in a large shopping mall setting, capable of:

- Interpreting user requests in everyday language.
- Recommending five or more relevant shops or services per query.
- Maintaining a coherent and helpful personality across a full conversation.

Potential deployment contexts include:

- **Retail malls**: Shop locators, promotions, event info.
- **Airports**: Lounge guidance, terminal services.
- **Hospitals**: Department navigation and patient services.
- **Expos and conventions**: Booth directories, session schedules.

This project is highly adaptable and can be extended with voice interfaces, frontend UIs, or multi-language support.

## Technology and Skills Demonstrated

This assistant is an end-to-end demonstration of professional AI system design and deployment readiness. The following technologies and skills are demonstrated:

### Artificial Intelligence

- OpenAI Embedding APIs for document encoding.
- Prompt engineering and persona construction.
- Vector-based information retrieval using semantic similarity.

### Software Engineering

- Modular Python codebase with clear separation of logic.
- RESTful API development with FastAPI and Pydantic models.
- Exception handling and system logging for production stability.

### Backend Infrastructure

- Local vector database management with **ChromaDB**.
- Asynchronous file and request handling for scalability.
- Threaded conversation state using **LangChain**'s memory model.

## Requirements

- Python 3.12+

For setup and installation, install dependencies as listed in `requirements.txt`.