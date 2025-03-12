import os
import re
from openai import OpenAI
from typing import Dict, List, Optional, Any

class RAGSearchClient:
    """Client for RAG search using OpenAI's Responses API."""
    
    def __init__(self, vector_store_id: str, api_key: Optional[str] = None):
        """
        Initialize the RAG search client.
        
        Args:
            vector_store_id: The ID of the vector store to search.
            api_key: Optional OpenAI API key. If None, reads from environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.vector_store_id = vector_store_id
    
    def search(self, 
               query: str, 
               model: str = "gpt-4o", 
               max_results: int = 5,
               include_search_results: bool = True) -> Dict[str, Any]:
        """
        Perform RAG search using OpenAI's Responses API.
        
        Args:
            query: The search query.
            model: The OpenAI model to use.
            max_results: Maximum number of search results to return.
            include_search_results: Whether to include raw search results.
            
        Returns:
            Dictionary containing the search response.
        """
        try:
            # Parameters for the include option
            include_params = []
            if include_search_results:
                include_params.append("output[*].file_search_call.search_results")
            
            # Create the request
            response = self.client.responses.create(
                model=model,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                input=query,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store_id],
                    "max_num_results": max_results
                }],
                include=include_params if include_params else None
            )
            
            # Process the response
            return self._process_response(response)
        except Exception as e:
            # Handle errors
            return {
                "error": True,
                "message": str(e),
                "query": query
            }
    
    def _process_response(self, response) -> Dict[str, Any]:
        """
        Process the RAG search response.
        
        Args:
            response: The response from the API.
            
        Returns:
            Processed response with extracted information.
        """
        output = response.output
        
        # Initialize result structure
        result = {
            "error": False,
            "message": None,
            "file_search_call": None,
            "answer": None,
            "annotations": [],
            "files_used": set(),
            "search_results": None
        }
        
        # Extract file_search_call
        file_search_call = next((item for item in output if item.type == "file_search_call"), None)
        if file_search_call:
            result["file_search_call"] = {
                "id": file_search_call.id,
                "status": file_search_call.status,
                "queries": file_search_call.queries
            }
            
            # Include search results if available
            if hasattr(file_search_call, "search_results") and file_search_call.search_results:
                result["search_results"] = []
                for sr in file_search_call.search_results:
                    search_result = {
                        "filename": sr.filename,
                        "file_id": sr.file_id,
                        "score": sr.score,
                        "content": [c.text for c in sr.content if hasattr(c, "text")]
                    }
                    result["search_results"].append(search_result)
        
        # Extract the message content
        message = next((item for item in output if item.type == "message"), None)
        if message and message.content:
            text_content = next((content for content in message.content if content.type == "output_text"), None)
            if text_content:
                result["answer"] = text_content.text
                
                # Extract annotations
                if hasattr(text_content, "annotations") and text_content.annotations:
                    for annotation in text_content.annotations:
                        if annotation.type == "file_citation":
                            annotation_info = {
                                "type": annotation.type,
                                "index": annotation.index,
                                "file_id": annotation.file_id,
                                "filename": annotation.filename
                            }
                            result["annotations"].append(annotation_info)
                            result["files_used"].add(annotation.filename)
        
        # Convert files_used from set to list
        result["files_used"] = list(result["files_used"])
        
        return result
    
    def format_answer_with_citations(self, search_result: Dict[str, Any]) -> str:
        """
        Format the answer with proper citations.
        
        Args:
            search_result: The processed search result.
            
        Returns:
            Formatted answer with citations.
        """
        if not search_result or search_result.get("error"):
            return "Error retrieving answer."
        
        answer = search_result.get("answer", "")
        annotations = search_result.get("annotations", [])
        
        # Sort annotations by index in reverse order to avoid messing up indices
        sorted_annotations = sorted(annotations, key=lambda x: x.get("index", 0), reverse=True)
        
        # Insert citation references
        for i, annotation in enumerate(sorted_annotations):
            index = annotation.get("index")
            filename = annotation.get("filename", "Unknown Source")
            
            if index and index < len(answer):
                citation_ref = f" [{i+1}]"
                answer = answer[:index] + citation_ref + answer[index:]
        
        return answer

def extract_citations(search_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract citations from search result.
    
    Args:
        search_result: The processed search result.
        
    Returns:
        List of citations with filename and snippets.
    """
    citations = []
    files_used = search_result.get("files_used", [])
    search_results = search_result.get("search_results", [])
    
    if not search_results:
        # If search_results not available, just return filenames
        return [{"filename": filename, "snippets": []} for filename in files_used]
    
    # Group search results by filename
    for filename in files_used:
        snippets = []
        for result in search_results:
            if result.get("filename") == filename:
                content = result.get("content", [])
                # Limit content to avoid very long outputs
                snippets.extend(content[:2])
                # Add score
                score = result.get("score", 0)
                
        citations.append({
            "filename": filename,
            "snippets": snippets[:3],  # Limit to top 3 snippets
            "score": score if 'score' in locals() else None
        })
    
    return citations
