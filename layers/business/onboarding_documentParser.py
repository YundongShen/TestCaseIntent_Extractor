"""
Onboarding Document Parser - business layer component
Extracts document format and prompt template from onboarding example documents
"""

import re
import json


class OnboardingDocumentParser:
    """Parses onboarding document examples and extracts format/prompt template"""
    
    def __init__(self):
        print("[ONBOARDING DOCUMENT PARSER] Initializing parser")
        self.example_path = "/Users/yundong/test-intent-extraction/layers/business/onboarding_templates/onboarding_example1.md"
    
    def parse(self):
        """
        Parse onboarding example document and extract format/prompt template.
        Save generated prompt to onboarding_template.py
        
        Returns:
            Dict with {"format_template": {...}, "prompt_template": str, "business_purpose": str}
        """
        try:
            with open(self.example_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print("[ONBOARDING DOCUMENT PARSER] Starting document format parsing")
            
            # Extract document structure
            document_structure = self._extract_structure(content)
            
            # Generate prompt template
            prompt_template = self._generate_prompt_template(document_structure)
            
            # Extract business_purpose
            business_purpose = self._extract_business_purpose(content)
            
            # Save prompt to onboarding_template.py
            self._save_prompt_to_template(prompt_template)
            
            result = {
                "format_template": document_structure,
                "prompt_template": prompt_template,
                "business_purpose": business_purpose,
                "status": "success"
            }
            
            print("[ONBOARDING DOCUMENT PARSER] Document format parsing completed")
            print("[ONBOARDING DOCUMENT PARSER] Prompt saved to onboarding_template.py")
            return result
            
        except Exception as e:
            print(f"[ONBOARDING DOCUMENT PARSER] Parsing failed: {e}")
            return {
                "format_template": {},
                "prompt_template": "",
                "business_purpose": "",
                "status": "error",
                "error": str(e)
            }
    
    def _extract_structure(self, content):
        """
        Extract document structure (sections and characteristics of each section).
        
        Args:
            content: Markdown content
            
        Returns:
            Dict with {"sections": [...], "total_sections": int}
        """
        # Extract all level-1 headers (starting with ##)
        sections = re.findall(r'^##\s+(.+)$', content, re.MULTILINE)
        
        # Extract features of each section
        structure = {
            "sections": [],
            "total_sections": len(sections),
            "document_type": "onboarding",
            "has_code_examples": "```" in content,
            "has_bullet_lists": "-" in content or "•" in content,
            "has_emoji_headers": any(emoji in content for emoji in ["👤", "🎯", "📋", "📝", "🔑"])
        }
        
        # Detailed analysis of each section
        for i, section in enumerate(sections):
            section_pattern = rf'^##\s+{re.escape(section)}$.*?(?=^##|\Z)'
            section_match = re.search(section_pattern, content, re.MULTILINE | re.DOTALL)
            
            if section_match:
                section_content = section_match.group()
                structure["sections"].append({
                    "index": i + 1,
                    "title": section,
                    "has_list": bool(re.search(r'^\s*[-•]\s', section_content, re.MULTILINE)),
                    "lines_count": len(section_content.strip().split('\n')),
                    "has_code": "```" in section_content,
                    "list_items": len(re.findall(r'^\s*[-•]\s', section_content, re.MULTILINE))
                })
        
        return structure
    
    def _generate_prompt_template(self, structure):
        """
        Generate usable prompt template (for generating new documents in same format).
        
        Args:
            structure: Document structure
            
        Returns:
            Prompt template string
        """
        sections_desc = "\n".join([
            f"- {s['title']}" for s in structure.get('sections', [])
        ])
        
        # Build code examples line conditionally
        code_examples_line = "- Include code examples section demonstrating key code snippets\n" if structure.get('has_code_examples') else ""
        
        prompt_template = f"""You are an Onboarding document expert. Generate Onboarding documents according to the following format:

【Document Format Structure】
Total {structure.get('total_sections', 0)} main sections, in order:
{sections_desc}

【Format Characteristics】
- Use markdown format
- Use emoji symbols with titles to enhance visual guidance (e.g.: 👤 Test Subject, 🎯 Test Objectives)
- Include bullet point lists explaining key points
{code_examples_line}- Include final documentation section explaining document purpose and applicable scenarios

【Generation Requirements】
1. Follow the section order and format above
2. Each section content should be practical and easy to understand
3. Code examples should be concise and clear
4. Overall length should be maintained in similar range

Please generate Onboarding documents with the same format based on given test content (subjects, objectives, activities)."""
        
        return prompt_template
    
    def _extract_business_purpose(self, content):
        """
        Extract business_purpose from document (last explanation portion).
        
        Args:
            content: Markdown content
            
        Returns:
            business_purpose string
        """
        # Extract last line or last paragraph as business_purpose
        lines = content.strip().split('\n')
        
        # Find last meaningful line
        for line in reversed(lines):
            if line.strip() and not line.strip().startswith('---'):
                if 'document purpose' in line or 'applicable' in line or 'scenario' in line:
                    return line.strip()
        
        # If not found, extract last markdown paragraph
        last_paragraph = re.findall(r'.*[。，！？]', content, re.DOTALL)
        if last_paragraph:
            return last_paragraph[-1].strip()[-100:]  # Take last 100 chars
        
        return "Generate Onboarding documents for new team members to quickly understand test design"
    
    def _save_prompt_to_template(self, prompt):
        """
        Save generated prompt to onboarding_template.py.
        
        Args:
            prompt: Generated prompt string
        """
        template_path = "/Users/yundong/test-intent-extraction/layers/business/onboarding_template.py"
        
        # Escape quotes in prompt
        escaped_prompt = prompt.replace('"""', r'\"\"\"').replace("'", r"\'")
        
        # Generate template file content
        template_content = f"""\"\"\"
Onboarding Document Prompt Template - Pure Data File
Auto-generated by onboarding_documentParser, should not be modified manually
This prompt is used by Output Layer to have LLM generate Onboarding documents
\"\"\"

PROMPT = \"\"\"{prompt}\"\"\"
"""
        
        try:
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
            print(f"[ONBOARDING DOCUMENT PARSER] Prompt saved to: {template_path}")
        except Exception as e:
            print(f"[ONBOARDING DOCUMENT PARSER] Saving prompt failed: {e}")
