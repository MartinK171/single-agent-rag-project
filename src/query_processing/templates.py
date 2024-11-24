from typing import Dict, Optional
from .analyzer import QueryAnalysis

class ResponseTemplate:
    """Manages response templates for different query types."""
    
    # Template definitions
    TEMPLATES = {
        "advanced": """
        Detailed Response:
        
        Analysis: {analysis}
        
        Explanation:
        {explanation}
        
        Additional Information:
        {additional_info}
        """,
        
        "entity_focused": """
        Entity Information:
        
        Entities Found: {entities}
        
        Details:
        {details}
        """,
        
        "question": """
        Answer:
        {answer}
        
        Reasoning:
        {reasoning}
        """,
        
        "standard": """
        Response:
        {response}
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