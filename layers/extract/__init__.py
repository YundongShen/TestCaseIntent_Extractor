"""
Extract layer - Extract objects, activities, and goals from content.
Supports three modes: 
  - independent (separate LLM calls, no dependencies)
  - combined (single LLM call for all three)
  - chain (sequential extraction with context passing: objects → goals → activities)
"""

from layers.extract.object_extractor import ObjectExtractor
from layers.extract.activity_extractor import ActivityExtractor
from layers.extract.goal_extractor import GoalExtractor

class ExtractLayer:
    """Extract layer for key information extraction."""
    
    def __init__(self, extract_mode="independent"):
        """
        Initialize Extract Layer.
        
        Args:
            extract_mode: "independent" (default) - separate LLM calls for each item type
                         "combined" - single LLM call for all three item types
                         "chain" - sequential extraction with context passing
        """
        print(f"[ExtractLayer] Initializing with mode: {extract_mode}")
        self.extract_mode = extract_mode
        
        if extract_mode == "combined":
            from layers.extract.combined_extractor import CombinedExtractor
            self.combined_extractor = CombinedExtractor()
        elif extract_mode == "chain":
            # Chain mode: sequential extraction with context passing
            from layers.extract.chain import ChainObjectExtractor, ChainGoalExtractor, ChainActivityExtractor
            self.chain_object_extractor = ChainObjectExtractor()
            self.chain_goal_extractor = ChainGoalExtractor()
            self.chain_activity_extractor = ChainActivityExtractor()
        else:  # "independent" mode
            self.object_extractor = ObjectExtractor()
            self.activity_extractor = ActivityExtractor()
            self.goal_extractor = GoalExtractor()
    
    def process(self, preprocessed_data):
        """Extract objects, activities, goals. Returns: dict with results."""
        print(f"[ExtractLayer] Extracting (mode: {self.extract_mode})")
        
        content = preprocessed_data.get("content", "")
        
        if self.extract_mode == "combined":
            # Single LLM call for all three item types
            result = self.combined_extractor.extract(content)
            objects = result.get("objects", [])
            activities = result.get("activities", [])
            goals = result.get("goals", [])
        elif self.extract_mode == "chain":
            # Chain mode: sequential extraction with context passing
            # Step 1: Extract objects (no context needed)
            objects_result = self.chain_object_extractor.extract(content)
            objects = objects_result.get("objects", []) if isinstance(objects_result, dict) else objects_result
            
            # Step 2: Extract goals (with objects context)
            goals_result = self.chain_goal_extractor.extract(content, objects=objects)
            goals = goals_result.get("goals", []) if isinstance(goals_result, dict) else goals_result
            
            # Step 3: Extract activities (with objects and goals context)
            activities_result = self.chain_activity_extractor.extract(content, objects=objects, goals=goals)
            activities = activities_result.get("activities", []) if isinstance(activities_result, dict) else activities_result
        else:  # independent mode
            # Separate LLM calls for each item type
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
