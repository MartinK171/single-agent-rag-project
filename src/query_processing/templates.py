from typing import Dict, Optional
from .analyzer import QueryAnalysis

class ResponseTemplate:
    """Manages response templates for different query types."""

    TEMPLATES = {
        "advanced": """
You are an assistant that provides detailed answers.

Question: {query}

Context:
{context}

Please provide a detailed answer in valid JSON format, **without any code fences or extra formatting**, ensuring all control characters (like newlines and tabs) are properly escaped, and the JSON is on a single line without any line breaks:

{{"answer": "<Your detailed answer here>"}}
""",
        "entity_focused": """
You are an assistant that focuses on entities.

Question: {query}

Entities Detected: {entities}

Please provide an answer in valid JSON format, **without any code fences or extra formatting**, ensuring all control characters (like newlines and tabs) are properly escaped, and the JSON is on a single line without any line breaks:

{{"answer": "<Your answer here>"}}
""",
        "question": """
You are an assistant that answers questions.

Question: {query}

Please provide an answer in valid JSON format, **without any code fences or extra formatting**, ensuring all control characters (like newlines and tabs) are properly escaped, and the JSON is on a single line without any line breaks:

{{"answer": "<Your answer here>"}}
""",
        "standard": """
You are an assistant.

{query}

Please provide an answer in valid JSON format, **without any code fences or extra formatting**, ensuring all control characters (like newlines and tabs) are properly escaped, and the JSON is on a single line without any line breaks:

{{"answer": "<Your answer here>"}}
"""
    }


    @classmethod
    def get_template(cls, path: str, analysis: QueryAnalysis) -> str:
        """Get appropriate template based on path and analysis."""
        template = cls.TEMPLATES.get(path, cls.TEMPLATES["standard"])
        
        # Customize template based on analysis
        if analysis.complexity > 0.7:
            template = cls._add_complexity_section(template)
        
        if analysis.entities:
            template = cls._add_entity_section(template)
            
        return template
    
    @classmethod
    def _add_complexity_section(cls, template: str) -> str:
        """Add section for complex queries."""
        return template + "\nComplexity Analysis:\n{complexity_analysis}"
    
    @classmethod
    def _add_entity_section(cls, template: str) -> str:
        """Add section for entity information."""
        return template + "\nEntity Details:\n{entity_details}"