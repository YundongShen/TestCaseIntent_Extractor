"""
Onboarding Generator - business layer/output component
Uses LLM to generate Onboarding documents based on provided prompts and test intents
"""

import json
import re
from datetime import datetime
import os


class OnboardingGenerator:
    """Uses LLM to generate formatted Onboarding documents"""
    
    def __init__(self):
        print("[ONBOARDING GENERATOR] Initializing generator")
    
    def generate(self, prompt_template, objects, goals, activities):
        """
        Generate Onboarding document using LLM.
        
        Args:
            prompt_template: PROMPT template (format guidance)
            objects: Adjusted test object list
            goals: Adjusted test goal list
            activities: Adjusted test activity list
            
        Returns:
            Dict with {"status": "success/error", "document": str, "error": str}
        """
        # Format guidance + core content = complete prompt
        prompt = self._build_prompt(prompt_template, objects, goals, activities)
        
        print("[ONBOARDING GENERATOR] Calling LLM to generate")
        return self._generate_with_model(prompt)
    
    def _generate_with_model(self, prompt):
        """Generate document using LLM inference service"""
        from model.inference_service import get_inference_service
        
        # Use inference service instead of direct model calls
        service = get_inference_service()
        document = service.infer(prompt, max_tokens=800)
        
        print("[ONBOARDING GENERATOR] Generation completed")
        return {"status": "success", "document": document, "error": None}
    
    def _build_prompt(self, format_guide, objects, goals, activities):
        """
        Build generation prompt: format guidance + test intent content.
        """
        objects_str = "\n".join([f"- {obj}" for obj in objects])
        goals_str = "\n".join([f"- {goal}" for goal in goals])
        activities_str = "\n".join([f"- {activity}" for activity in activities])
        
        return f"""{format_guide}

【Test Intent Content to Generate】
Objects:
{objects_str}

Goals:
{goals_str}

Activities:
{activities_str}

Based on the above content and format requirements, generate Onboarding document:"""
    
    def save_document(self, document):
        """
        Save generated document to onboarding_result folder (auto-generate timestamped filename).
        
        Args:
            document: Document content
            
        Returns:
            Dict with {"status": "success/error", "path": str, "error": str}
        """
        try:
            # Create onboarding_result folder (if not exists)
            result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "onboarding_result")
            os.makedirs(result_dir, exist_ok=True)
            
            # Generate timestamped filename: onboarding_20260326_141530.md
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"onboarding_{timestamp}.md"
            output_path = os.path.join(result_dir, filename)
            
            # Save document
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(document)
            
            print(f"[ONBOARDING GENERATOR] Document saved to: {output_path}")
            
            return {
                "status": "success",
                "path": output_path,
                "error": None
            }
        except Exception as e:
            print(f"[ONBOARDING GENERATOR] Save failed: {e}")
            return {
                "status": "error",
                "path": "",
                "error": str(e)
            }
