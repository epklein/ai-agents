# AI Agents

This project is a personal AI-powered agent experiment, designed for my custom needs. I'm developing it for learning purposes.

This agent is build on top of LangChain ReAct agent. The first feature (Langchain Tool) I developed is designed to fetch and search for articles from my readwise account.

## Features

- **Readwise API Integration**: This agent utilizes a custom tool to search articles in the user's Readwise account.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/epklein/ai-agents.git
   cd ai-agents

2. Install dependencies:
   `pip install -r requirements.txt`

3. Set up environment variables: Create a .env file in the root directory and add your API keys and configurations. For example:

   ```bash
   READWISE_API_KEY=your_readwise_api_key
   OPENAI_API_KEY=your_openai_api_key

## Usage

Run the main script: `python3 main.py`

The agent will execute the template prompt and return all articles from eduklein.com.br saved in your Readwise acocunt.