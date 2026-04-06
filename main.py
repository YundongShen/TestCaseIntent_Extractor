#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intent Extraction & Onboarding Document Generation - Main Pipeline
Five-layer architecture: Input → Extract → Intent → Business → Output
"""

import sys
import os
import json
from datetime import datetime

from layers import InputLayer, ExtractLayer, IntentLayer, BusinessLayer, OutputLayer


def save_intermediate_result(result_type, data):
    """Save intermediate processing results to JSON files."""
    output_dir = f"Result/{result_type}_result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result_type}_result_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {result_type.upper()}: {filepath}")
    return filepath


def run_pipeline(raw_data):
    """Execute the complete processing pipeline. Returns: {"success": bool, ...}"""
    print("\n" + "="*80)
    print("[Intent Extraction & Document Generation System]")
    print("="*80)
    print(f"Start time: {datetime.now().isoformat()}\n")
    
    try:
        # Layer 1: Input - Preprocess data
        print("="*80)
        print("Layer 1: INPUT LAYER (Data Preprocessing)")
        print("="*80)
        data1 = InputLayer().process(raw_data)
        print(f"✓ Input layer done: {len(data1.get('content', ''))} chars\n")
        
        # Layer 2: Extract - Key information extraction
        print("="*80)
        print("Layer 2: EXTRACT LAYER (Objects/Goals/Activities)")
        print("="*80)
        data2 = ExtractLayer().process(data1)
        print(f"✓ Extraction done: objects={len(data2.get('objects', []))} goals={len(data2.get('goals', []))} activities={len(data2.get('activities', []))}\n")
        
        # Save Extract result
        save_intermediate_result('extract', {
            "timestamp": datetime.now().isoformat(),
            "objects": data2.get('objects', []),
            "goals": data2.get('goals', []),
            "activities": data2.get('activities', []),
            "test_case_id": data2.get('test_case_id', 'unknown'),
            "user_id": data2.get('user_id', 'unknown')
        })
        
        # Layer 3: Intent - Validate and adjust
        print("="*80)
        print("Layer 3: INTENT LAYER (Validation & Adjustment)")
        print("="*80)
        data3 = IntentLayer().process(data2)
        print(f"✓ Intent processing done\n")
        
        # Save Validate result
        save_intermediate_result('validate', {
            "timestamp": datetime.now().isoformat(),
            "objects": data2.get('objects', []),
            "goals": data2.get('goals', []),
            "activities": data2.get('activities', []),
            "validate_matched": data3.get('validation_result', {}).get('matched', False),
            "validate_reasoning": data3.get('validation_result', {}).get('reasoning', ''),
            "specificity": data3.get('specificity', 0.5),
            "test_case_id": data3.get('test_case_id', 'unknown'),
            "user_id": data3.get('user_id', 'unknown')
        })
        
        # Save Adjust result
        save_intermediate_result('adjust', {
            "timestamp": datetime.now().isoformat(),
            "before_adjustment": {
                "objects": data2.get('objects', []),
                "goals": data2.get('goals', []),
                "activities": data2.get('activities', [])
            },
            "after_adjustment": {
                "objects": data3.get('objects', []),
                "goals": data3.get('goals', []),
                "activities": data3.get('activities', [])
            },
            "specificity": data3.get('specificity', 0.5),
            "test_case_id": data3.get('test_case_id', 'unknown'),
            "user_id": data3.get('user_id', 'unknown')
        })
        
        # Layer 4: Business - Assemble prompt template
        print("="*80)
        print("Layer 4: BUSINESS LAYER (Prompt Template)")
        print("="*80)
        data4 = BusinessLayer().process(data3)
        print(f"✓ Prompt loaded: {len(data4.get('prompt', ''))} chars\n")
        
        # Layer 5: Output - Generate and save document
        print("="*80)
        print("Layer 5: OUTPUT LAYER (Document Generation)")
        print("="*80)
        output_result = OutputLayer().process(data4)
        
        if not output_result.get("success"):
            print(f"✗ Document generation failed: {output_result.get('error')}\n")
            return {
                "success": False,
                "error": output_result.get('error')
            }
        
        print(f"✓ Document saved: {output_result.get('filepath')}\n")
        
        # Complete
        print("="*80)
        print("[Pipeline completed successfully!]")
        print("="*80 + "\n")
        
        return {
            "success": True,
            "output": output_result,
            "data": data4
        }
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def create_sample_data():
    """Load test data from file."""
    test_file_path = "testcases/ContactUs_StepDefs.java"
    
    with open(test_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"✓ Loaded from: {test_file_path}")
    
    return {
        "content": content,
        "user_id": "test_user_bug",
        "test_case_id": "tc_bug_ant_solid"
    }


def main():
    """Main entry point."""
    
    print("\n[Loading test data]")
    sample_data = create_sample_data()
    print(f"✓ Data loaded")
    print(f"  User ID: {sample_data['user_id']}")
    print(f"  Test ID: {sample_data['test_case_id']}")
    print(f"  Content: {len(sample_data['content'])} chars\n")
    
    result = run_pipeline(sample_data)
    
    if result["success"]:
        print("[Final Result]")
        print(f"✓ Document: {result['output'].get('filepath', 'N/A')}")
        return 0
    else:
        print(f"✗ Failed: {result.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
