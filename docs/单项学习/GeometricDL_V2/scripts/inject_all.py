#!/usr/bin/env python3
"""
Inject ALL enrichment fragments into GeometricDL V2 chapter HTML files.

Handles three types of fragments:
1. Section enrichments: ch{N}_sec_{section_id}.html → inject after matching section
2. Formula annotations: ch{N}_formulas.html → inject at end of main content
3. Glossary/concepts: ch{N}_glossary.html → inject at beginning of main content

Usage: python3 inject_all.py [chapter_num]  (no arg = all chapters)
"""

import re
import sys
from pathlib import Path

BASE = Path("/Users/taisen/.openclaw/workspace/medical-robotics-sim/docs/单项学习/GeometricDL_V2")
ENRICHMENTS = BASE / "enrichments"

def read_css():
    """Read enrichment CSS."""
    css_file = BASE / "enrichment_css.html"
    if css_file.exists():
        return css_file.read_text(encoding='utf-8')
    return ""

def inject_css(content):
    """Inject enrichment CSS into <head> if not present."""
    if '.enrichment-block' in content:
        return content
    
    css = read_css()
    if not css:
        return content
    
    if '</head>' in content:
        content = content.replace('</head>', css + '\n</head>', 1)
    return content

def find_insertion_point(lines, section_id, start_from=0):
    """Find where to insert enrichment for a given section.
    
    Looks for the section by ID, then finds the start of the NEXT section.
    Inserts just before the next section.
    """
    # Find the section
    section_line = -1
    id_pattern = re.compile(rf'id="{re.escape(section_id)}"')
    
    for i in range(start_from, len(lines)):
        if id_pattern.search(lines[i]):
            section_line = i
            break
    
    if section_line == -1:
        # Try partial match
        for i in range(start_from, len(lines)):
            if section_id in lines[i]:
                section_line = i
                break
    
    if section_line == -1:
        return -1
    
    # Find the next section/heading with an id (that's not this one)
    next_patterns = [
        re.compile(r'<section\s[^>]*id="(?!' + re.escape(section_id) + r')'),
        re.compile(r'<h[23][^>]*\s+id="(?!' + re.escape(section_id) + r')'),
    ]
    
    for i in range(section_line + 3, len(lines)):
        for pat in next_patterns:
            if pat.search(lines[i]):
                # Go back past blank lines
                insert = i
                while insert > section_line + 1 and lines[insert - 1].strip() == '':
                    insert -= 1
                return insert
    
    # No next section found, insert before footer or end
    for i in range(section_line + 3, len(lines)):
        if '<footer' in lines[i].lower() or '</main>' in lines[i].lower():
            return i
    
    return len(lines) - 10  # Near the end

def find_main_content_start(lines):
    """Find the start of main content (after first section heading)."""
    for i, line in enumerate(lines):
        if re.search(r'<section\s+id="|<h2[^>]*id="', line):
            return i
    return 50  # Default fallback

def find_main_content_end(lines):
    """Find the end of main content (before footer)."""
    for i in range(len(lines) - 1, 0, -1):
        if '<footer' in lines[i].lower() or '</main>' in lines[i].lower():
            return i
    return len(lines) - 5

def inject_chapter(chapter_num):
    """Inject all enrichments for a single chapter."""
    html_path = BASE / f"chapter{chapter_num}" / "index.html"
    
    if not html_path.exists():
        print(f"  ❌ Chapter {chapter_num} HTML not found")
        return
    
    content = html_path.read_text(encoding='utf-8')
    original_size = len(content)
    
    # Inject CSS
    content = inject_css(content)
    lines = content.split('\n')
    
    # Collect fragments
    sec_fragments = sorted(ENRICHMENTS.glob(f"ch{chapter_num}_sec_*.html"))
    formula_frag = ENRICHMENTS / f"ch{chapter_num}_formulas.html"
    glossary_frag = ENRICHMENTS / f"ch{chapter_num}_glossary.html"
    
    total_frags = len(sec_fragments) + (1 if formula_frag.exists() else 0) + (1 if glossary_frag.exists() else 0)
    
    if total_frags == 0:
        print(f"  ℹ️  No fragments for chapter {chapter_num}")
        return
    
    print(f"  📦 Found {total_frags} fragments ({len(sec_fragments)} sections, "
          f"{'✅' if formula_frag.exists() else '❌'} formulas, "
          f"{'✅' if glossary_frag.exists() else '❌'} glossary)")
    
    injected = 0
    
    # 1. Inject glossary at the beginning of main content
    if glossary_frag.exists():
        frag = glossary_frag.read_text(encoding='utf-8').strip()
        insert_at = find_main_content_start(lines)
        if insert_at > 0:
            frag_lines = ['', '<!-- === GLOSSARY START === -->'] + frag.split('\n') + ['<!-- === GLOSSARY END === -->', '']
            lines = lines[:insert_at] + frag_lines + lines[insert_at:]
            injected += 1
            print(f"  ✅ Glossary injected at line {insert_at}")
    
    # 2. Inject section enrichments
    for frag_path in sec_fragments:
        section_id = frag_path.stem.replace(f"ch{chapter_num}_sec_", "")
        frag = frag_path.read_text(encoding='utf-8').strip()
        
        if not frag:
            continue
        
        insert_at = find_insertion_point(lines, section_id)
        
        if insert_at == -1:
            print(f"  ⚠️  Section '{section_id}' not found, appending to end of content")
            insert_at = find_main_content_end(lines)
        
        frag_lines = ['', f'<!-- === ENRICHMENT: {section_id} === -->'] + frag.split('\n') + [f'<!-- === END ENRICHMENT: {section_id} === -->', '']
        lines = lines[:insert_at] + frag_lines + lines[insert_at:]
        injected += 1
        print(f"  ✅ Section '{section_id}' injected at line {insert_at}")
    
    # 3. Inject formulas at end of main content
    if formula_frag.exists():
        frag = formula_frag.read_text(encoding='utf-8').strip()
        insert_at = find_main_content_end(lines)
        frag_lines = ['', '<!-- === FORMULAS START === -->'] + frag.split('\n') + ['<!-- === FORMULAS END === -->', '']
        lines = lines[:insert_at] + frag_lines + lines[insert_at:]
        injected += 1
        print(f"  ✅ Formulas injected at line {insert_at}")
    
    # Write back
    new_content = '\n'.join(lines)
    new_size = len(new_content)
    html_path.write_text(new_content, encoding='utf-8')
    
    ratio = new_size / original_size
    print(f"  📊 {original_size:,} → {new_size:,} bytes ({ratio:.1f}x) | {injected} fragments injected")

def main():
    print("🚀 GeometricDL V2 Enrichment Injector")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        chapters = [int(sys.argv[1])]
    else:
        chapters = [1, 2, 3, 4, 5, 6, 7]
    
    for ch in chapters:
        print(f"\n📖 Chapter {ch}")
        inject_chapter(ch)
    
    print("\n✅ Done!")
    
    # Summary
    print("\n📊 Final sizes:")
    for ch in chapters:
        html_path = BASE / f"chapter{ch}" / "index.html"
        if html_path.exists():
            size = html_path.stat().st_size
            orig_path = Path(str(html_path).replace("GeometricDL_V2", "GeometricDL"))
            orig_size = orig_path.stat().st_size if orig_path.exists() else size
            ratio = size / orig_size if orig_size > 0 else 1
            print(f"  Ch{ch}: {size:,} bytes ({ratio:.1f}x original)")

if __name__ == '__main__':
    main()
