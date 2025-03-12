import streamlit as st
import os
import re
from utils import RAGSearchClient, extract_citations
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now you can access the API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify the API key was loaded (optional)
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Set page configuration
st.set_page_config(
    page_title="RAG Search by OpenAI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    with open("static/styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize the session state for storing search history
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

if 'current_result' not in st.session_state:
    st.session_state.current_result = None

def main():
    # Load CSS
    load_css()
    
    # Application header
    st.markdown('<h1 class="title">RAG Search</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Retrieve information from documents using OpenAI\'s Responses API</p>', unsafe_allow_html=True)
    
    # Vector store ID - hardcoded from requirements, but could be made configurable
    vector_store_id = "vs_67d1a6aeb41c8191a29b940182a6e272"
    
    # Initialize the RAG search client
    try:
        rag_client = RAGSearchClient(vector_store_id=vector_store_id, api_key=openai_api_key)
    except Exception as e:
        st.error(f"Error initializing RAG search client: {str(e)}")
        st.stop()
    
    # Create search form
    with st.form(key="search_form", clear_on_submit=False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your search query",
                placeholder="e.g., What is Deep Research by OpenAI?",
                key="query_input"
            )
        
        with col2:
            # Advanced options
            with st.expander("Advanced Options"):
                model = st.selectbox(
                    "Model",
                    options=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                    index=0
                )
                
                max_results = st.slider(
                    "Max Results",
                    min_value=1,
                    max_value=10,
                    value=5
                )
                
                include_search_results = st.checkbox(
                    "Include Search Results",
                    value=True
                )
        
        # Submit button
        submit_button = st.form_submit_button(
            label="Search",
            use_container_width=True,
            type="primary"
        )
    
    # Handle search submission
    if submit_button and query:
        with st.spinner("Searching..."):
            try:
                # Perform the search
                search_result = rag_client.search(
                    query=query,
                    model=model,
                    max_results=max_results,
                    include_search_results=include_search_results
                )
                
                # Store the result in session state
                st.session_state.current_result = search_result
                
                # Add to search history
                st.session_state.search_history.append({
                    "query": query,
                    "timestamp": import_time_module().strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                st.error(f"Error performing search: {str(e)}")
    
    # Display search results
    if st.session_state.current_result:
        display_search_results(st.session_state.current_result)
    
    # Display search history
    if st.session_state.search_history:
        with st.expander("Search History"):
            for idx, item in enumerate(reversed(st.session_state.search_history)):
                st.text(f"{item['timestamp']} - {item['query']}")

def display_search_results(search_result):
    """Display the search results in a formatted way."""
    if search_result.get("error"):
        st.error(f"Search error: {search_result.get('message', 'Unknown error')}")
        return
    
    # Extract the answer and related information
    answer = search_result.get("answer", "No answer found.")
    files_used = search_result.get("files_used", [])
    annotations = search_result.get("annotations", [])
    
    # Main answer card
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='card-title'>Answer</h3>", unsafe_allow_html=True)
    
    # Format the answer text with citation markers
    formatted_answer = answer
    citation_map = {}
    
    # If there are annotations, add citation markers
    if annotations:
        # Create a map of filenames to citation numbers
        for i, filename in enumerate(files_used):
            citation_map[filename] = i + 1
        
        # Sort annotations by index in reverse order to avoid messing up indices
        sorted_annotations = sorted(annotations, key=lambda x: x.get("index", 0), reverse=True)
        
        # Insert citation references
        for annotation in sorted_annotations:
            index = annotation.get("index")
            filename = annotation.get("filename", "Unknown Source")
            citation_num = citation_map.get(filename, "?")
            
            if index and index < len(formatted_answer):
                citation_ref = f"<sup class='citation-badge'>[{citation_num}]</sup>"
                formatted_answer = formatted_answer[:index] + citation_ref + formatted_answer[index:]
    
    # Display the formatted answer
    st.markdown(f"<div class='card-content'>{formatted_answer}</div>", unsafe_allow_html=True)
    
    # Display metadata about the sources
    if files_used:
        st.markdown("<div class='card-meta'>", unsafe_allow_html=True)
        st.markdown(f"Sources: {', '.join(files_used)}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display citations and snippets
    if files_used:
        st.markdown("<h3>Sources & Citations</h3>", unsafe_allow_html=True)
        citations = extract_citations(search_result)
        
        for i, citation in enumerate(citations):
            filename = citation.get("filename", "Unknown Source")
            snippets = citation.get("snippets", [])
            score = citation.get("score")
            
            st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"<h4 class='card-title'><sup class='citation-badge'>[{i+1}]</sup> {filename}</h4>", unsafe_allow_html=True)
            
            if score is not None:
                st.markdown(f"<span class='relevance-score'>Relevance Score: {score:.4f}</span>", unsafe_allow_html=True)
            
            if snippets:
                for snippet in snippets:
                    st.markdown(f"<div class='citation'>{snippet}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<p>No snippets available</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Display raw search results (optional)
    if search_result.get("search_results"):
        with st.expander("Raw Search Results"):
            st.json(search_result.get("search_results"))

def import_time_module():
    """Import time module for timestamps."""
    import time
    from datetime import datetime
    return datetime.now()

if __name__ == "__main__":
    main()
