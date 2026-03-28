#!/usr/bin/env python3
"""
Quick fix for the Extended Window FULCCA JSON serialization error.
"""

import json
import numpy as np

def fix_json_serialization():
    """Fix the JSON serialization issue in Extended Window FULCCA."""
    
    # Read the Extended_Window_FULCCA.py file
    with open('Extended_Window_FULCCA.py', 'r') as f:
        content = f.read()
    
    # Replace the problematic JSON dump line
    old_line = '        json.dump(best_config, f, indent=2)'
    new_line = '''        # Convert numpy arrays to lists for JSON serialization
        config_to_save = best_config.copy()
        if 'detailed_results' in config_to_save:
            detailed = config_to_save['detailed_results']
            if 'predictions' in detailed:
                detailed['predictions'] = detailed['predictions'].tolist()
            if 'targets' in detailed:
                detailed['targets'] = detailed['targets'].tolist()
        json.dump(config_to_save, f, indent=2)'''
    
    # Replace the line
    content = content.replace(old_line, new_line)
    
    # Write the fixed file
    with open('Extended_Window_FULCCA_fixed.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed JSON serialization issue!")
    print("📁 Fixed file saved as: Extended_Window_FULCCA_fixed.py")

if __name__ == "__main__":
    fix_json_serialization()
