import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def classify_template(text: str) -> bool:
    """
    Sử dụng LLM để phân loại xem văn bản (chương) có chứa
    template giải bài tập (có lời giải chi tiết) hay không.
    Yêu cầu: không được tự chỉnh sửa nội dung, chỉ xác định TRUE/FALSE.
    
    Trả về True nếu text chứa template giải bài tập, False nếu không.
    """
    prompt = f"""Given the following chapter content from an educational document, 
determine whether it contains a complete solution template for an exercise. 
A solution template is defined as a segment that provides a detailed, step-by-step explanation 
to solve a given problem, including the problem description and the corresponding solution steps.
Note: Do not modify the text; only determine if the text is a valid solution template.
Respond in JSON format as: {{"is_template": true}} or {{"is_template": false}}.
Text:
{text[:1500]}...
"""
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        answer_text = response.choices[0].message.content.strip()
        result = json.loads(answer_text)
        return result.get("is_template", False)
    except Exception as e:
        print("Error in classify_template:", e)
        return False

def extract_template(text: str) -> dict:
    """
    Sử dụng LLM để trích xuất template giải bài tập từ văn bản.
    Yêu cầu: Không chỉnh sửa nội dung so với bản gốc.
    Trả về JSON có định dạng:
    Option A:
    {
        "problem": "<exact text of the problem statement>",
        "solution": "<exact text of the solution explanation>"
    }
    Option B:
    {
        "template": "<exact full text of the solution template>"
    }
    Nếu không tìm thấy template rõ ràng, trả về toàn bộ text dưới key "template".
    """
    prompt = f"""You are given a chapter of an educational document that contains a solution template for an exercise. 
The solution template is defined as a contiguous segment of text that includes a complete, step-by-step explanation of how to solve a given problem, including the problem description and the solution process.
The template should be extracted exactly as it appears in the original text, without any modifications.

Important:
1. Do not alter the text.
2. If the text includes additional parts such as examples or practice exercises that are not part of the core solution template, include only the segment that provides the solution explanation.
3. If the text does not clearly separate the problem and the solution, return the entire segment as the solution template under a single key "template".

Please return your answer as a JSON object in one of the following formats:

Option A:
{{
    "problem": "<exact text of the problem statement>",
    "solution": "<exact text of the solution explanation>"
}}

Option B:
{{
    "template": "<exact full text of the solution template>"
}}

Only extract content that is clearly part of the solution template for the exercise, and ignore other parts.

Text:
{text[:2000]}...
"""
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        extracted_text = response.choices[0].message.content.strip()
        result = json.loads(extracted_text)
        return result
    except Exception as e:
        print("Error in extract_template:", e)
        # Return all the text in template if false
        return {"template": text}

def process_chapters_for_templates(chapters: dict) -> dict:
    """
    Duyệt qua kết quả của Chapter Recognition (chương đã tách) và sử dụng LLM để trích xuất template giải bài tập từ từng chương.
    Chỉ giữ lại những chương mà LLM xác định chứa template giải bài tập.
    
    Trả về dict với mỗi key là tiêu đề chương và value là kết quả template (dạng JSON như trên).
    """
    templates = {}
    for title, info in chapters.items():
        content = info.get("content", "")
        # Classify template
        is_template = classify_template(content)
        print(f"Chapter '{title}': is_template = {is_template}")
        if is_template:
            template_data = extract_template(content)
            templates[title] = template_data
        else:
            print(f"Chapter '{title}' does not contain a valid solution template.")
    return templates

if __name__ == '__main__':
    # Test module
    from src.preprocess import read_docx, clean_text
    from src.chapter_recognition import split_text_into_chapters, get_toc_from_text
    sample_file = "data/tim_ucln_sach_giao_khoa.docx"
    sample_text = clean_text(read_docx(sample_file))
    
    toc = get_toc_from_text(sample_text)
    chapters = split_text_into_chapters(sample_text, toc)
    
    templates = process_chapters_for_templates(chapters)
    print("Extracted Templates:")
    print(json.dumps(templates, indent=4, ensure_ascii=False))
