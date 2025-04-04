import os
import json
from src.preprocess import read_docx, clean_text
from src.chapter_recognition import get_toc_from_text, split_text_into_chapters
from src.template_extraction import process_chapters_for_templates

def main():
    file_path = "data/tim_ucln_sach_giao_khoa.docx"
    
    # Read/cleaning input
    raw_text = read_docx(file_path)
    cleaned_text = clean_text(raw_text)
    print("Cleaned text loaded.")
    
    # Extract TOC using LLM (apply fallback if needed)
    toc = get_toc_from_text(cleaned_text)
    print("Extracted TOC:", toc)
    
    # Split input into chapters using TOC
    chapters = split_text_into_chapters(cleaned_text, toc)
    print("Extracted Chapters:")
    for title, info in chapters.items():
        print(f"Title: {title}")
        print(f"Char Range: {info['char_range']}")
        print("Content Preview:", info["content"][:200])
        print("----")
    
    # Chapter Recognition -> save output
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    chapters_file = os.path.join(output_folder, "chapters.json")
    with open(chapters_file, "w", encoding="utf-8") as f:
        json.dump(chapters, f, indent=4, ensure_ascii=False)
    print(f"Chapters JSON saved to: {chapters_file}")
    
    # Template Extraction
    templates = process_chapters_for_templates(chapters)
    print("Extracted Templates:")
    print(json.dumps(templates, indent=4, ensure_ascii=False))
    
    # Template Extraction -> save output
    templates_file = os.path.join(output_folder, "templates.json")
    with open(templates_file, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=4, ensure_ascii=False)
    print(f"Templates JSON saved to: {templates_file}")

if __name__ == '__main__':
    main()
