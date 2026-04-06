"""
Preprocessor - input layer component
Performs basic cleaning and formatting of test case files

Preprocessing Rules (Four Steps):
1. Remove leading/trailing whitespace (trim)
2. Merge consecutive blank lines into 1 line
3. Delete pure single-line comments (e.g., //, #, not inline comments)
4. Normalize line indentation/newlines (format only, no code change)

Purpose: Ensure input testcases receive better preprocessing before intent extraction by LLM
"""

import re
from datetime import datetime
from pathlib import Path


class Preprocessor:
    """Preprocessor for basic processing of test case files"""
    
    def __init__(self):
        print("[PREPROCESSOR] Initializing preprocessor")
    
    def process(self, raw_code):
        """
        Process raw code (four-step preprocessing).
        
        Args:
            raw_code: Raw code string
            
        Returns:
            Cleaned code string
        """
        print("[PREPROCESSOR] Starting code preprocessing")
        
        # Four-step preprocessing rules
        cleaned_code = self._trim_whitespace(raw_code)
        cleaned_code = self._merge_blank_lines(cleaned_code)
        cleaned_code = self._remove_comment_lines(cleaned_code)
        cleaned_code = self._normalize_indentation(cleaned_code)
        
        original_lines = len(raw_code.strip().split('\n')) if raw_code.strip() else 0
        cleaned_lines = len(cleaned_code.strip().split('\n')) if cleaned_code.strip() else 0
        
        print(f"[PREPROCESSOR] Preprocessing completed: {original_lines} -> {cleaned_lines} lines")
        
        return cleaned_code
    
    def process_file(self, file_path):
        """
        Process test case file.
        
        Args:
            file_path: Test case file path
            
        Returns:
            Cleaned test case and metadata
        """
        filepath = Path(file_path)
        print(f"[PREPROCESSOR] Reading file: {filepath.name}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_code = f.read()
        
        result = self.process(raw_code)
        result['file_name'] = filepath.name
        result['file_path'] = str(filepath)
        
        return result
    
    @staticmethod
    def _trim_whitespace(code):
        """Rule 1: Remove leading/trailing whitespace"""
        lines = code.split('\n')
        trimmed_lines = [line.rstrip() for line in lines]
        return '\n'.join(trimmed_lines)
    
    @staticmethod
    def _merge_blank_lines(code):
        """Rule 2: Merge consecutive blank lines into 1 line"""
        # Replace consecutive blank lines with single blank line
        code = re.sub(r'\n\s*\n+', '\n\n', code)
        return code
    
    @staticmethod
    def _remove_comment_lines(code):
        """Rule 3: Delete pure single-line comments (//, # at start), not inline comments"""
        lines = code.split('\n')
        result_lines = []
        
        for line in lines:
            stripped = line.lstrip()
            
            # If line starts with comment symbol (ignoring leading whitespace), skip
            if stripped.startswith('//') or stripped.startswith('#'):
                continue
            
            # Otherwise keep (including inline comments)
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @staticmethod
    def _normalize_indentation(code):
        """Rule 4: Normalize line indentation/newlines (format only, no code change)"""
        lines = code.split('\n')
        result_lines = []
        
        for line in lines:
            # Keep each line's indentation structure, standardize to multiples of 2 spaces
            # Get line's leading whitespace
            stripped = line.lstrip()
            if not stripped:
                # Empty lines stay empty
                result_lines.append('')
            else:
                # Calculate indentation level (original indent length, keep relative relationship)
                original_indent = len(line) - len(stripped)
                # Convert to multiples of 2 spaces
                normalized_indent = (original_indent // 2) * 2
                # Rebuild line
                normalized_line = ' ' * normalized_indent + stripped
                result_lines.append(normalized_line)
        
        return '\n'.join(result_lines)
    
    @staticmethod
    def validate(cleaned_data):
        """Validate validity of cleaned data"""
        print("[PREPROCESSOR] Validating cleaned data")
        
        if not cleaned_data or not cleaned_data.get("content"):
            print("[PREPROCESSOR] Error: Data empty or content missing")
            return False
        
        print("[PREPROCESSOR] Data validation passed")
        return True
