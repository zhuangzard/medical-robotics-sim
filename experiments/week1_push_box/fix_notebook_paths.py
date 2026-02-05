#!/usr/bin/env python3
"""
Fix Notebook - Ensure all cells use correct working directory
"""

import json
from pathlib import Path

def fix_notebook(input_path, output_path):
    """Add directory checks to all cells that run Python scripts"""
    
    with open(input_path) as f:
        nb = json.load(f)
    
    # Directory check code to prepend
    dir_check = """# ✅ Ensure we're in the project directory
import os
if not os.path.exists('baselines'):
    if os.path.exists('medical-robotics-sim'):
        os.chdir('medical-robotics-sim')
        print('✅ Changed to project directory')
    else:
        print('❌ Error: medical-robotics-sim directory not found!')
        print(f'Current directory: {os.getcwd()}')
        raise FileNotFoundError('Project directory not found')

print(f'📂 Working directory: {os.getcwd()}')
print()

"""
    
    # Process each cell
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
            
        source = ''.join(cell['source'])
        
        # Cells that need directory check
        needs_fix = (
            ('!python3 baselines/' in source or
             '!python3 training/' in source or
             '!python3 experiments/' in source) and
            'Ensure we\'re in the project directory' not in source and
            '!git clone' not in source
        )
        
        if needs_fix:
            # Prepend directory check
            if isinstance(cell['source'], list):
                cell['source'] = dir_check.split('\n') + ['\n'] + cell['source']
            else:
                cell['source'] = dir_check + cell['source']
            
            print(f"✅ Fixed Cell {i}")
    
    # Save fixed notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Fixed notebook saved to: {output_path}")

if __name__ == '__main__':
    input_nb = Path('notebooks/educational_training.ipynb')
    output_nb = Path('notebooks/educational_training.ipynb')  # Overwrite
    
    print("🔧 Fixing notebook paths...")
    fix_notebook(input_nb, output_nb)
