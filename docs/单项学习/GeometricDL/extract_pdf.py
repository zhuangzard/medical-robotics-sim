#!/usr/bin/env python3
"""
Extract and split the Geometric Deep Learning PDF into chapters.
- Extracts all text per page
- Extracts all images 
- Splits PDF into chapter PDFs
- Generates raw markdown per chapter
"""

import fitz  # PyMuPDF
import os
import json
import sys

PDF_PATH = "/Users/taisen/Library/Mobile Documents/com~apple~CloudDocs/Documents/PPT/BIT/Books/PhysicalModel/Reference_Books_Papers/2104.13478v2.pdf"
OUT_DIR = "/Users/taisen/.openclaw/workspace/medical-robotics-sim/docs/单项学习/GeometricDL"
ASSETS_DIR = os.path.join(OUT_DIR, "assets")
RAW_MD_DIR = os.path.join(OUT_DIR, "raw_markdown")

# Chapter definitions: (name, start_page_0indexed, end_page_0indexed_exclusive)
# Pages in PDF are 0-indexed; the TOC says p5=preface, p8=ch1, etc.
# We need to figure out exact 0-indexed pages. The PDF page numbers might differ from printed pages.
# Let's first detect the actual page mapping.

CHAPTERS = [
    ("chapter1_introduction", "1 Introduction", 7, 8),        # p8 (0-idx: 7) to p8
    ("chapter2_learning_high_dimensions", "2 Learning in High Dimensions", 8, 12),  # p9-12 (0-idx: 8-11)
    ("chapter3_geometric_priors", "3 Geometric Priors", 12, 33),      # p13-33 (0-idx: 12-32)
    ("chapter4_geometric_domains", "4 Geometric Domains: the 5 Gs", 33, 71),  # p34-71 (0-idx: 33-70)
    ("chapter5_geometric_dl_models", "5 Geometric Deep Learning Models", 71, 105),  # p72-105 (0-idx: 71-104)
    ("chapter6_applications", "6 Problems and Applications", 105, 117),  # p106-117 (0-idx: 105-116)
    ("chapter7_historic_perspective", "7 Historic Perspective", 117, None),  # p118+ (0-idx: 117+)
]

def extract_images(doc):
    """Extract all images from the PDF."""
    print("=== Extracting images ===")
    img_count = 0
    seen_xrefs = set()
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        
        for img_idx, img in enumerate(images):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            
            try:
                base_image = doc.extract_image(xref)
                if base_image:
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    
                    # Skip very small images (likely icons/bullets)
                    if width < 50 or height < 50:
                        continue
                    
                    # Determine which chapter this image belongs to
                    ch_label = "misc"
                    for ch_name, ch_title, start, end in CHAPTERS:
                        actual_end = end if end else len(doc)
                        if start <= page_num < actual_end:
                            ch_num = ch_name.split("_")[0].replace("chapter", "ch")
                            ch_label = ch_num
                            break
                    
                    img_filename = f"{ch_label}_p{page_num+1}_img{img_idx}.{ext}"
                    img_path = os.path.join(ASSETS_DIR, img_filename)
                    
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    
                    img_count += 1
                    if img_count % 20 == 0:
                        print(f"  Extracted {img_count} images so far...")
            except Exception as e:
                print(f"  Warning: Could not extract image xref={xref} on page {page_num+1}: {e}")
    
    print(f"  Total images extracted: {img_count}")
    return img_count


def extract_chapter_text(doc, start, end):
    """Extract text from a range of pages."""
    actual_end = end if end else len(doc)
    text_parts = []
    for p in range(start, actual_end):
        page = doc[p]
        text = page.get_text("text")
        text_parts.append(f"--- Page {p+1} ---\n{text}")
    return "\n\n".join(text_parts)


def split_chapters(doc):
    """Split the PDF into per-chapter PDFs and extract markdown."""
    print("=== Splitting into chapter PDFs ===")
    
    for ch_name, ch_title, start, end in CHAPTERS:
        actual_end = end if end else len(doc)
        page_range = list(range(start, actual_end))
        
        # Create chapter PDF
        ch_doc = fitz.open()
        ch_doc.insert_pdf(doc, from_page=start, to_page=actual_end-1)
        ch_pdf_path = os.path.join(OUT_DIR, f"{ch_name}.pdf")
        ch_doc.save(ch_pdf_path)
        ch_doc.close()
        print(f"  {ch_name}.pdf: pages {start+1}-{actual_end} ({actual_end-start} pages)")
        
        # Extract raw markdown text
        raw_text = extract_chapter_text(doc, start, actual_end)
        md_path = os.path.join(RAW_MD_DIR, f"{ch_name}_raw.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {ch_title}\n\n")
            f.write(raw_text)
        print(f"  {ch_name}_raw.md: {len(raw_text)} chars")


def extract_full_text(doc):
    """Extract full text for reference."""
    print("=== Extracting full text ===")
    full_text = []
    for p in range(len(doc)):
        page = doc[p]
        text = page.get_text("text")
        full_text.append(f"=== PAGE {p+1} ===\n{text}")
    
    full_path = os.path.join(RAW_MD_DIR, "full_text.md")
    with open(full_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_text))
    print(f"  Full text: {len(full_text)} pages, saved to full_text.md")


def main():
    print(f"Opening PDF: {PDF_PATH}")
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")
    
    # First, let's check what's on key pages to validate our page mapping
    print("\n=== Validating page mapping ===")
    for check_page in [0, 4, 6, 7, 8, 12, 33, 71, 105, 117]:
        if check_page < total_pages:
            text = doc[check_page].get_text("text")[:200]
            print(f"  Page {check_page+1} (0-idx {check_page}): {text[:100].strip()}...")
    
    # Extract everything
    print()
    extract_full_text(doc)
    print()
    img_count = extract_images(doc)
    print()
    split_chapters(doc)
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Total pages: {total_pages}")
    print(f"Total images extracted: {img_count}")
    print(f"Chapter PDFs: {len(CHAPTERS)}")
    for ch_name, ch_title, start, end in CHAPTERS:
        actual_end = end if end else total_pages
        print(f"  {ch_title}: pages {start+1}-{actual_end} ({actual_end-start} pages)")
    
    doc.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
