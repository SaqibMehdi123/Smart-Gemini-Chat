# Smart-Gemini-Chat
A conversational AI chatbot using Google's Gemini API with semantic search capabilities powered by sentence transformers for intelligent conversation history analysis.

## Features

Conversational AI powered by Google's Gemini 2.0 Flash
Semantic Search using sentence transformers
Cosine Similarity analysis with previous conversations
Persistent Chat History with automatic save/load
Real-time Similarity Reports for each message

## Installation

### 1. Clone and setup

bashgit clone https://github.com/SaqibMehdi123/Smart-Gemini-Chat.git
cd Smart-Gemini-Chat
pip install requests python-dotenv sentence-transformers scikit-learn numpy

### 2. Create .env file

envGEMINI_API_KEY=your_gemini_api_key_here
Get your API key from Google AI Studio.

## Usage
python gemini_chat.py

### Commands:

Chat normally with Gemini
clear - Clear conversation history
exit - Exit application

## Similarity Scoring

0.8-1.0: Very similar messages
0.6-0.8: Moderately similar
0.4-0.6: Some similarity
0.0-0.4: Low similarity

## How It Works

Messages are converted to embeddings using sentence transformers
Cosine similarity is calculated against conversation history
Real-time similarity reports show related previous conversations
Chat history is automatically saved and loaded

## Author
Saqib Mehdi - @SaqibMehdi123
