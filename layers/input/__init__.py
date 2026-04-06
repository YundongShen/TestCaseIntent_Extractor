"""Input layer - Data preprocessing."""

from layers.input.preprocessor import Preprocessor

class InputLayer:
    """Input layer for data preprocessing."""
    
    def __init__(self):
        print("[InputLayer] Initializing")
        self.preprocessor = Preprocessor()
    
    def process(self, raw_data):
        """Process raw data. Returns: processed data dict."""
        print("[InputLayer] Processing")
        content = raw_data.get("content", "")
        cleaned_content = self.preprocessor.process(content)
        
        # Return processed content along with other metadata
        return {
            **raw_data,
            "content": cleaned_content
        }

__all__ = ['InputLayer']
