import docx

def read_docx(file_path: str) -> str:
    """
    Đọc file .docx và trả về văn bản thuần.
    """
    doc = docx.Document(file_path)
    text = "\n\n".join([para.text for para in doc.paragraphs])
    return text

def clean_text(text: str) -> str:
    """
    Làm sạch văn bản: loại bỏ khoảng trắng thừa, chuẩn hóa dòng.
    """
    return " ".join(text.split())

if __name__ == '__main__':
    file_path = "data/tim_ucln_sach_giao_khoa.docx"
    raw_text = read_docx(file_path)
    cleaned_text = clean_text(raw_text)
    print("Cleaned Text:")
    print(cleaned_text)
