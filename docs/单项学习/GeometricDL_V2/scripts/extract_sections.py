#!/usr/bin/env python3
"""Extract section boundaries from GeometricDL V2 HTML files.
Outputs a JSON with section IDs, titles, line ranges, and content snippets."""

import re
import json
import sys
from pathlib import Path

BASE = Path("/Users/taisen/.openclaw/workspace/medical-robotics-sim/docs/单项学习/GeometricDL_V2")

def extract_sections(chapter_num):
    html_path = BASE / f"chapter{chapter_num}" / "index.html"
    if not html_path.exists():
        print(f"File not found: {html_path}")
        return
    
    content = html_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    
    # Find section boundaries
    sections = []
    section_pattern = re.compile(r'<section\s+id="([^"]*)"', re.IGNORECASE)
    section_end_pattern = re.compile(r'</section>', re.IGNORECASE)
    
    current_section = None
    for i, line in enumerate(lines, 1):
        match = section_pattern.search(line)
        if match:
            if current_section:
                current_section['end_line'] = i - 1
                sections.append(current_section)
            current_section = {
                'id': match.group(1),
                'start_line': i,
                'content_preview': ''
            }
        
        end_match = section_end_pattern.search(line)
        if end_match and current_section:
            # Only close if this section started before
            pass  # sections may be nested, handle carefully
    
    if current_section:
        current_section['end_line'] = len(lines)
        sections.append(current_section)
    
    # Also find by div with class="section" or id patterns
    if not sections:
        # Fallback: find by heading tags
        heading_pattern = re.compile(r'<h[23][^>]*id="([^"]*)"[^>]*>(.+?)</h[23]>', re.IGNORECASE | re.DOTALL)
        for i, line in enumerate(lines, 1):
            match = heading_pattern.search(line)
            if match:
                # Clean HTML tags from title
                title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
                sections.append({
                    'id': match.group(1),
                    'title': title,
                    'line': i
                })
    
    # Extract titles from nearby content
    for sec in sections:
        start = sec.get('start_line', sec.get('line', 1))
        end = min(start + 10, len(lines))
        snippet = '\n'.join(lines[start-1:end])
        # Extract title from h2 or h3
        title_match = re.search(r'<h[23][^>]*>(.+?)</h[23]>', snippet, re.DOTALL)
        if title_match and 'title' not in sec:
            sec['title'] = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
    
    result = {
        'chapter': chapter_num,
        'file': str(html_path),
        'total_lines': len(lines),
        'total_bytes': len(content),
        'sections': sections
    }
    
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    chapter = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    extract_sections(chapter)
