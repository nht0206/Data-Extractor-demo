import os
import re
import json
import openai
from rapidfuzz import fuzz
from src.preprocess import clean_text
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_toc_from_text(text: str) -> dict:
    """
    Sử dụng LLM (OpenAI GPT) để trích xuất block TOC từ văn bản.
    Trả về kết quả dạng JSON với key "toc" chứa danh sách tiêu đề chương.
    Nếu LLM không trả về kết quả hợp lệ, sử dụng fallback bằng regex.
    """
    prompt = f"""Given the following text from a document that contains a Table of Contents (TOC), 
please extract the TOC section as a JSON object. Return only the TOC in the following format:
{{"toc": ["I. Chapter One Title", "II. Chapter Two Title", ...]}}
Text:
{text[:2000]}...
"""
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        toc_text = response.choices[0].message.content
        toc_json = json.loads(toc_text)
    except Exception as e:
        print("Error extracting TOC with LLM:", e)
        toc_json = {"toc": []}

    # Fallback with regex if GPT doesn't give answer
    if not toc_json.get("toc"):
        toc_json = extract_toc_with_regex(text)
        print("Fallback TOC extracted using regex.")
    return toc_json

def extract_toc_with_regex(text: str) -> dict:
    """
    Fallback: Sử dụng regex để trích xuất các dòng tiêu đề chương.
    Giả định tiêu đề chương có dạng "I. <text>" hoặc "II. <text>"...
    """
    pattern = r'^(?:[IVXLCDM]+\.)\s+.*$'
    toc_list = re.findall(pattern, text, re.MULTILINE)
    toc_list = [line.strip() for line in toc_list if len(line.strip()) > 5]
    return {"toc": toc_list}

def fuzzy_find(text: str, target: str, threshold: int = 80) -> int:
    """
    Sử dụng fuzzy matching để tìm vị trí xấp xỉ của target trong text.
    Trả về chỉ số nếu tìm thấy (nếu score vượt threshold), ngược lại trả về -1.
    """
    best_index = -1
    best_score = 0
    for i in range(0, len(text) - len(target)):
        segment = text[i:i+len(target)]
        score = fuzz.ratio(segment, target)
        if score > best_score:
            best_score = score
            best_index = i
        if best_score >= threshold:
            return best_index
    return best_index if best_score >= threshold else -1

def split_text_into_chapters(text: str, toc: dict) -> dict:
    """
    Dựa vào danh sách tiêu đề trong TOC, tách toàn bộ văn bản thành các chương riêng biệt.
    Trả về dict với mỗi key là tiêu đề chương, và giá trị là dict chứa:
      - 'char_range': [start, end]
      - 'content': toàn bộ nội dung của chương đó.
    """
    chapters = {}
    positions = []
    for chapter_title in toc.get("toc", []):
        pos = text.find(chapter_title)
        if pos == -1:
            pos = fuzzy_find(text, chapter_title)
        if pos != -1:
            positions.append((chapter_title, pos))
    
    # Nếu không tìm thấy bất kỳ tiêu đề nào, trả về toàn bộ văn bản dưới dạng một chương
    if not positions:
        chapters["Full Document"] = {"char_range": [0, len(text)], "content": text}
        return chapters

    # Sắp xếp các chương theo vị trí xuất hiện
    positions.sort(key=lambda x: x[1])
    
    # Tách nội dung dựa vào khoảng cách giữa các tiêu đề
    for idx, (title, start_pos) in enumerate(positions):
        end_pos = positions[idx+1][1] if idx < len(positions) - 1 else len(text)
        chapters[title] = {
            "char_range": [start_pos, end_pos],
            "content": text[start_pos:end_pos]
        }
    return chapters

if __name__ == '__main__':
    # Test module
    from src.preprocess import read_docx, clean_text
    sample_file = "data/tim_ucln_sach_giao_khoa.docx"
    sample_text = clean_text(read_docx(sample_file))
    
    toc = get_toc_from_text(sample_text)
    print("Extracted TOC:", toc)
    
    chapters = split_text_into_chapters(sample_text, toc)
    print("Extracted Chapters:")
    for title, info in chapters.items():
        print(f"Chapter: {title}")
        print("Char Range:", info["char_range"])
        print("Content Preview:", info["content"][:200])
        print("----")
