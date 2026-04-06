"""
Output layer initialization file
Responsible for document generation and persistence
"""

from layers.output.document_writer import DocumentWriter
from layers.business.onboarding_generator import OnboardingGenerator


class OutputLayer:
    """Output layer - document generation and persistence"""
    
    def __init__(self):
        print("[OUTPUT LAYER] Initializing output layer")
        self.generator = OnboardingGenerator()
        self.document_writer = DocumentWriter(output_dir="Result/onboarding_result")
    
    def process(self, business_data):
        """
        Generate and save the final Onboarding document.
        
        Args:
            business_data: Output from business layer (contains prompt, objects, goals, activities, etc.)
            
        Returns:
            Result dict with status and document info
        """
        print("[OUTPUT LAYER] Starting Onboarding document generation")
        
        # Extract necessary information
        prompt = business_data.get("prompt", "")
        objects = business_data.get("objects", [])
        goals = business_data.get("goals", [])
        activities = business_data.get("activities", [])
        test_case_id = business_data.get("test_case_id", "tc_unknown")
        
        # Call Generator to generate document
        gen_result = self.generator.generate(prompt, objects, goals, activities)
        
        if gen_result["status"] != "success":
            print(f"[OUTPUT LAYER] Document generation failed: {gen_result.get('error')}")
            return {
                "success": False,
                "error": gen_result.get('error')
            }
        
        # Save generated document
        document = gen_result.get("document", "")
        save_result = self.document_writer.save_document(document)
        
        if save_result["status"] == "success":
            print("[OUTPUT LAYER] Document generation and persistence completed")
            return {
                "success": True,
                "filepath": save_result.get("path"),
                "test_case_id": test_case_id
            }
        else:
            print(f"[OUTPUT LAYER] Document persistence failed: {save_result.get('error')}")
            return {
                "success": False,
                "error": save_result.get('error')
            }


__all__ = ['OutputLayer']
