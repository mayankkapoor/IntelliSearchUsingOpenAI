# IntelliSearchUsingOpenAI

A powerful Retrieval-Augmented Generation (RAG) search application built with Streamlit and OpenAI's Responses API. This application allows users to search through documents stored in an OpenAI vector store and receive AI-generated responses with proper citations.

## 🌟 Features

- Interactive search interface built with Streamlit
- Utilizes OpenAI's Responses API for intelligent search
- Supports multiple OpenAI models (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- Configurable search parameters
- Displays search results with proper citations
- Maintains search history for easy reference
- Clean, responsive UI with custom styling

## 📋 Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Vector store ID from OpenAI

## 🚀 Installation

1. Clone the repository:
   ```
   git clone https://github.com/mayankkapoor/IntelliSearchUsingOpenAI.git
   cd IntelliSearchUsingOpenAI
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -e .
   ```
   
   Or using the project's dependencies from pyproject.toml:
   ```
   pip install openai>=1.66.2 streamlit>=1.43.2 python-dotenv
   ```

4. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 💻 Usage

1. Start the Streamlit application:
   ```
   streamlit run main.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Enter your search query in the search box

4. Optionally, adjust the advanced options:
   - Select the OpenAI model to use
   - Adjust the maximum number of results
   - Toggle whether to include search results

5. Click the "Search" button to perform the search

6. View the AI-generated response with citations and the original search results

## 🔧 Configuration

The application uses a vector store ID that is currently hardcoded in the `main.py` file. Modify this value to your own vector store ID.

## 🧩 Project Structure

- `main.py`: The main Streamlit application
- `utils.py`: Contains the RAGSearchClient class and utility functions
- `static/styles.css`: Custom CSS styling for the application
- `.env`: Environment variables file (contains your OpenAI API key)

## 📝 License

MIT License
