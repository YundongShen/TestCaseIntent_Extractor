"""
Extract layer - Extract objects, activities, and goals from content.
"""

from layers.extract.object_extractor import ObjectExtractor
from layers.extract.activity_extractor import ActivityExtractor
from layers.extract.goal_extractor import GoalExtractor

class ExtractLayer:
    """Extract layer for key information extraction."""
    
    def __init__(self):
        print("[ExtractLayer] Initializing")
        self.object_extractor = ObjectExtractor()
        self.activity_extractor = ActivityExtractor()
        self.goal_extractor = GoalExtractor()
    
    def process(self, preprocessed_data):
        """Extract objects, activities, goals. Returns: dict with results."""
        print("[ExtractLayer] Extracting")
        
        content = preprocessed_data.get("content", "")
        
        objects_result = self.object_extractor.extract(content)
        activities_result = self.activity_extractor.extract(content)
        goals_result = self.goal_extractor.extract(content)
        
        objects = objects_result.get("objects", []) if isinstance(objects_result, dict) else objects_result
        activities = activities_result.get("activities", []) if isinstance(activities_result, dict) else activities_result
        goals = goals_result.get("goals", []) if isinstance(goals_result, dict) else goals_result
        
        extracted_data = {
            **preprocessed_data,
            "objects": objects,
            "activities": activities,
            "goals": goals,
        }
        
        print(f"[ExtractLayer] Done | objects={len(objects)} activities={len(activities)} goals={len(goals)}")
        return extracted_data

__all__ = ['ExtractLayer']
