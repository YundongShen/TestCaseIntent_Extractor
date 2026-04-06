"""
Intent layer - Validate and adjust extracted intent elements.
"""

from layers.intent.validator import Validator
from layers.intent.adjuster import Adjuster

class IntentLayer:
    """Intent layer for validation and adjustment."""
    
    def __init__(self):
        print("[IntentLayer] Initializing")
        self.validator = Validator()
        self.adjuster = Adjuster()
    
    def process(self, extracted_data):
        """Validate and adjust intent elements. Returns: processed intent data."""
        print("[IntentLayer] Processing")
        
        objects = extracted_data.get("objects", [])
        goals = extracted_data.get("goals", [])
        activities = extracted_data.get("activities", [])
        
        # Validate
        validation_result = self.validator.validate(objects, goals, activities)
        is_matched = validation_result.get("matched", False)
        
        if not is_matched:
            print("[IntentLayer] Validation failed, adjusting")
        else:
            print("[IntentLayer] Validation passed")
        
        # Adjust
        adjusted_result = self.adjuster.adjust(objects, goals, activities)
        
        # Combine results
        intent_data = {
            **extracted_data,
            "objects": adjusted_result.get("objects", objects),
            "goals": adjusted_result.get("goals", goals),
            "activities": adjusted_result.get("activities", activities),
            "validation_result": validation_result,
            "specificity": adjusted_result.get("specificity", 0.5)
        }
        
        print("[IntentLayer] Processing done")
        return intent_data

__all__ = ['IntentLayer']
