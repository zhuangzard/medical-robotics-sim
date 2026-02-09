#!/usr/bin/env python3
"""
Inject enrichment HTML fragments into GeometricDL V2 chapter files.

Usage: python3 inject_enrichments.py <chapter_num>

Reads enrichment fragments from enrichments/ch{N}_sec_{section_id}.html
and injects them after the corresponding section in the chapter's index.html.

Also injects the enrichment CSS into <head>.
"""

import re
import sys
from pathlib import Path

BASE = Path("/Users/taisen/.openclaw/workspace/medical-robotics-sim/docs/单项学习/GeometricDL_V2")

def inject_css(html_content):
    """Inject enrichment CSS if not already present."""
    if 'enrichment-block' in html_content and '.enrichment-block' in html_content:
        return html_content  # CSS already present
    
    css_file = BASE / "enrichment_css.html"
    if not css_file.exists():
        print("WARNING: enrichment_css.html not found!")
        return html_content
    
    css_content = css_file.read_text(encoding='utf-8')
    
    # Inject before </head>
    if '</head>' in html_content:
        html_content = html_content.replace('</head>', css_content + '\n</head>', 1)
    
    return html_content

def find_section_end(lines, section_id, start_search=0):
    """Find the line index where a section's content ends.
    
    Strategy: find the section by its id, then find the next section start
    or the next major structural element.
    """
    # Find the section start
    section_start = -1
    pattern = re.compile(rf'id="{re.escape(section_id)}"')
    
    for i in range(start_search, len(lines)):
        if pattern.search(lines[i]):
            section_start = i
            break
    
    if section_start == -1:
        return -1
    
    # Find the next section start (any h2 or h3 with id, or section tag)
    next_section_patterns = [
        re.compile(r'<section\s+id="'),
        re.compile(r'<h[23][^>]*\s+id="(?!' + re.escape(section_id) + r')'),
    ]
    
    for i in range(section_start + 1, len(lines)):
        for pat in next_section_patterns:
            if pat.search(lines[i]):
                # Insert before this next section
                # Go back to find a good insertion point (before blank lines)
                insert_at = i
                while insert_at > section_start + 1 and lines[insert_at - 1].strip() == '':
                    insert_at -= 1
                return insert_at
    
    return -1

def inject_enrichments(chapter_num):
    """Inject all enrichment fragments for a chapter."""
    html_path = BASE / f"chapter{chapter_num}" / "index.html"
    enrichment_dir = BASE / "enrichments"
    
    if not html_path.exists():
        print(f"ERROR: {html_path} not found!")
        return
    
    # Read original HTML
    content = html_path.read_text(encoding='utf-8')
    original_size = len(content)
    
    # Inject CSS
    content = inject_css(content)
    
    # Find all enrichment fragments for this chapter
    fragments = sorted(enrichment_dir.glob(f"ch{chapter_num}_sec_*.html"))
    
    if not fragments:
        print(f"No enrichment fragments found for chapter {chapter_num}")
        return
    
    print(f"Found {len(fragments)} enrichment fragments for chapter {chapter_num}")
    
    lines = content.split('\n')
    injected = 0
    offset = 0  # Track line offset from previous injections
    
    for frag_path in fragments:
        # Extract section_id from filename: ch1_sec_overview.html -> overview
        section_id = frag_path.stem.replace(f"ch{chapter_num}_sec_", "")
        
        frag_content = frag_path.read_text(encoding='utf-8').strip()
        if not frag_content:
            continue
        
        # Find where to inject
        insert_line = find_section_end(lines, section_id)
        
        if insert_line == -1:
            print(f"  WARNING: Could not find section '{section_id}' in chapter {chapter_num}")
            # Try fuzzy match
            for i, line in enumerate(lines):
                if section_id.replace('-', '') in line.replace('-', '').lower():
                    insert_line = i + 5  # Insert a few lines after the match
                    print(f"  Fuzzy match found at line {insert_line}")
                    break
        
        if insert_line == -1:
            print(f"  SKIPPED: section '{section_id}'")
            continue
        
        # Insert the fragment
        frag_lines = frag_content.split('\n')
        lines = lines[:insert_line] + ['', '<!-- === ENRICHMENT START: ' + section_id + ' === -->'] + frag_lines + ['<!-- === ENRICHMENT END: ' + section_id + ' === -->', ''] + lines[insert_line:]
        
        injected += 1
        print(f"  ✅ Injected '{section_id}' at line {insert_line} ({len(frag_lines)} lines)")
    
    # Write back
    new_content = '\n'.join(lines)
    html_path.write_text(new_content, encoding='utf-8')
    new_size = len(new_content)
    
    print(f"\nDone! Injected {injected}/{len(fragments)} fragments")
    print(f"Size: {original_size} -> {new_size} bytes ({new_size/original_size:.1f}x)")

if __name__ == '__main__':
    chapter = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    inject_enrichments(chapter)
