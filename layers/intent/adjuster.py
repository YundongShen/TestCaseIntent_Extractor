"""
Adjuster - Adjust intent elements specificity based on business purpose.
"""

import re
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'business'))

class Adjuster:
    """Adjust intent specificity level based on business purpose."""
    
    def __init__(self):
        print("[Adjuster] Initializing")
        self.parser = None
    
    def adjust(self, objects, goals, activities):
        """Adjust intent elements specificity based on business purpose.
        Returns: {"objects": [...], "goals": [...], "activities": [...], "specificity": float}"""
        
        business_purpose = "Document generation"
        if self.parser:
            try:
                parsed = self.parser.parse()
                business_purpose = parsed.get("business_purpose", "Document generation")
            except:
                pass
        
        print(f"[Adjuster] Using business_purpose: {business_purpose}")
        return self._adjust_with_model(objects, goals, activities, business_purpose)
    
    def _adjust_with_model(self, objects, goals, activities, business_purpose):
        """Adjust elements based on LLM-determined specificity level.
        LLM only determines specificity (easy), Python does the adjustment (reliable)."""
        from model.service_factory import get_inference_backend

        business_purpose = business_purpose[:500] if business_purpose else "Document generation"

        # Step 1: Ask LLM ONLY to choose specificity level (single number)
        prompt = f"""Choose specificity level for: {business_purpose}

Levels:
0.3 = Technical details  (for developers)
0.5 = Balanced           (for general documentation)
0.8 = Onboarding simple  (for new team members)

Return ONLY a number: 0.3 or 0.5 or 0.8"""

        service = get_inference_backend()
        response = service.infer(prompt, max_tokens=20)
        
        # Extract specificity number
        specificity = 0.5
        try:
            # Try to find a number in the response
            import re as regex
            numbers = regex.findall(r'0\.\d', response)
            if numbers:
                specificity = float(numbers[0])
        except:
            pass
        
        # Step 2: Python adjusts elements based on specificity level
        adjusted_objects = self._adjust_elements_python(objects, specificity)
        adjusted_goals = self._adjust_elements_python(goals, specificity)
        adjusted_activities = self._adjust_elements_python(activities, specificity)
        
        return {
            "objects": adjusted_objects,
            "goals": adjusted_goals,
            "activities": adjusted_activities,
            "specificity": specificity,
            "business_purpose": business_purpose
        }
    
    def _adjust_elements_python(self, elements, specificity):
        """Python-based adjustment of verbosity based on specificity level.
        This avoids LLM confusion about what to adjust."""
        adjusted = []
        
        for item in elements:
            # Always apply adjustment based on specificity
            if specificity < 0.4:
                # Technical detail level
                adjusted.append(self._enhance_technical(item))
            elif specificity < 0.7:
                # Balanced level - moderate enhancement
                adjusted.append(self._enhance_balanced(item))
            else:
                # Onboarding level - simplify and add business meaning
                adjusted.append(self._enhance_onboarding(item))
        
        return adjusted
    
    def _enhance_technical(self, item):
        """Add technical detail to item for technical documentation"""
        # Activity mappings
        activity_mappings = {
            "wait_for_database_connection": "wait_for_database_connection (with timeout configuration for production environment)",
            "get_test_container": "get_test_container (retrieve test environment context)",
            "send_get_request_to_admin_index_details_endpoint": "send_get_request_to_admin_index_details_endpoint (HTTP GET verification)",
            "extract_and_inspect_response_metadata": "extract_and_inspect_response_metadata (parse and validate JSON response structure)",
            "create_admin_user": "create_admin_user (set up test credentials)",
            "set_up_container_and_environment": "set_up_container_and_environment (initialize test infrastructure)",
            "iterate_through_entities_and_metadata": "iterate_through_entities_and_metadata (programmatic traversal validation)",
        }
        # Object mappings
        object_mappings = {
            "Product": "Product (core business entity in system)",
            "Price": "Price (financial data structure)",
            "SalesChannel": "SalesChannel (distribution entity)",
            "LinkProductVariantPriceSet": "LinkProductVariantPriceSet (database junction entity)",
            "LinkProductSalesChannel": "LinkProductSalesChannel (relationship mapping)",
            "ProductVariant": "ProductVariant (product configuration variant)",
            "PriceSet": "PriceSet (pricing definition entity)",
        }
        
        if item in activity_mappings:
            return activity_mappings[item]
        elif item in object_mappings:
            return object_mappings[item]
        else:
            return item
    
    def _enhance_balanced(self, item):
        """Add balanced context to item"""
        activity_mappings = {
            "wait_for_database_connection": "wait_for_database_connection to ensure system stability",
            "send_get_request_to_admin_index_details_endpoint": "send_get_request_to_admin_index_details_endpoint to verify API response",
            "extract_and_inspect_response_metadata": "extract_and_inspect_response_metadata to validate content",
            "create_admin_user": "create_admin_user for authorization testing",
            "set_up_container_and_environment": "set_up_container_and_environment for test execution",
            "get_test_container": "get_test_container for test isolation",
            "iterate_through_entities_and_metadata": "iterate_through_entities_and_metadata to verify structure",
        }
        object_mappings = {
            "Product": "Product (entity being managed)",
            "Price": "Price (pricing information)",
            "SalesChannel": "SalesChannel (distribution)",
            "ProductVariant": "ProductVariant (product variant)",
            "LinkProductVariantPriceSet": "LinkProductVariantPriceSet (pricing link)",
            "LinkProductSalesChannel": "LinkProductSalesChannel (channel link)",
            "PriceSet": "PriceSet (price definition)",
        }
        
        if item in activity_mappings:
            return activity_mappings[item]
        elif item in object_mappings:
            return object_mappings[item]
        else:
            return item
    
    def _enhance_onboarding(self, item):
        """Add business/onboarding context to item for new team members"""
        activity_mappings = {
            "wait_for_database_connection": "Wait for database to be ready",
            "send_get_request_to_admin_index_details_endpoint": "Call the API to get index details",
            "extract_and_inspect_response_metadata": "Check that the response has the right information",
            "create_admin_user": "Set up a test admin account",
            "set_up_container_and_environment": "Get the test environment ready",
            "get_test_container": "Retrieve the test container",
            "iterate_through_entities_and_metadata": "Go through each returned item to verify it",
        }
        object_mappings = {
            "Product": "Product - items being sold",
            "Price": "Price - cost information",
            "SalesChannel": "SalesChannel - where products are sold",
            "ProductVariant": "ProductVariant - different versions of a product",
            "LinkProductVariantPriceSet": "LinkProductVariantPriceSet - connects variants to prices",
            "LinkProductSalesChannel": "LinkProductSalesChannel - connects products to sales channels",
            "PriceSet": "PriceSet - collection of prices",
        }
        
        if item in activity_mappings:
            return activity_mappings[item]
        elif item in object_mappings:
            return object_mappings[item]
        else:
            return item
