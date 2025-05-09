# 특정 폴더 논문파일들을 순차적으로 불러와서 자동으로 분류 및 분석
import openai
import json
import fitz  # PyMuPDF
import os
import re
import shutil  # 파일 복사용

# OpenAI API 키 설정 (본인의 API 키를 입력하세요)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)  # 최신 방식

# 📂 논문이 저장된 폴더 경로 (여기 수정)
PAPER_FOLDER_PATH = "D:\\2025\\Readables\\Total_test"  
OUTPUT_FOLDER = "Classified_Papers"  # 정리된 논문이 저장될 폴더
RESULT_JSON_PATH = "analysis_results.json"  # JSON 저장 경로
RESULT_TXT_PATH = "analysis_results.txt"  # TXT 저장 경로

# 📌 논문 분류 기준 (10종)
classification_criteria = [
    "1) Data-Driven Turbulence Modeling : Improvement of turbulence models using machine learning, RANS, LES, DNS, Subgrid modeling, Closure model learning",
    "2) Shock & Boundary Layer Interaction : Shock-boundary layer interaction, SBLI, Shock detection, Supersonic flow, Shock control",
    "3) Hypersonic Flow & High-Speed Aerodynamics : Hypersonic flow, High-temperature gas dynamics, Fluid-structure interaction, FSI, Scramjet, Hypersonic vehicle",
    "4) Reduced-Order Modeling : CFD acceleration, Low-dimensional models, Aerodynamic analysis, Data-driven surrogate models, Surrogate modeling",
    "5) Aerodynamic Shape Optimization : Shape optimization, Automated fluid dynamics design, Genetic algorithm, Reinforcement learning-based optimization",
    "6) Compressible Flow Physics : Compressible flow, Supersonic, Unsteady flow, Aerodynamic performance prediction of vehicles",
    "7) Machine Learning for Flow Control : Flow control, Drag reduction, Efficiency improvement, Jet, Vortex, Flap",
    "8) Multi-Fidelity Modeling & Uncertainty Quantification : High-fidelity vs. low-fidelity, Physical constraints, Uncertainty quantification, Reliability assessment",
    "9) Scientific Machine Learning : Physics-Informed Neural Networks, PINNs, Fluid dynamics theory, Fusion of physical models",
    "10) Experimental Data Fusion & Surrogate Modeling : Wind tunnel experimental data, AI integration, Fusion of CFD and experimental data, Generative AI, Data augmentation",
    "11) Review Papers : Overall trends with a specific view point",
    "12) Etc : None of the aboves"
]

# 📄 PDF에서 Abstract 부분만 추출하는 함수
def extract_abstract_from_pdf(paper_file_path):
    """PDF 파일에서 Abstract(초록) 부분만 추출"""
    doc = fitz.open(paper_file_path)
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"

    # Abstract 찾기 (띄어쓰기 포함)
    abstract_match = re.search(r"(A\s*B\s*S\s*T\s*R\s*A\s*C\s*T|ABSTRACT|Abstract)\s*([\s\S]+?)(?=\n(Introduction|INTRODUCTION|Background|BACKGROUND|Nomenclature|1\.)|$)", text)

    if abstract_match:
        return abstract_match.group(2).strip()
    else:
        return "Abstract not found"

# 📄 PDF에서 초기 300단어 추출하는 함수
def extract_first_300_words(paper_file_path):
    """PDF 파일에서 처음 300단어 추출"""
    doc = fitz.open(paper_file_path)
    text = ""

    # PDF의 모든 페이지에서 텍스트 추출
    for page in doc:
        text += page.get_text("text") + "\n"

    #print(text)

    # 공백 기준으로 단어 단위로 분할
    words = re.findall(r'\S+', text)  # 공백이 아닌 문자들(단어) 추출

    # 300단어까지만 가져와서 반환
    first_300_words = " ".join(words[:300])

    return first_300_words if first_300_words else "No text found"

# 📝 ChatGPT에 보낼 프롬프트 생성
def generate_prompt(extract_text):
    return f"""
    논문의 내용을 분석하여 다음 정보를 한글로 제공해줘:

    1. **이 논문은 아래 12종 분류법 중 어느 카테고리에 속하는지? (복수 가능)**  
       {", ".join(classification_criteria)}
    2. **이 논문에서의 새로운 발견이나 성과는 무엇인가?**  

    논문의 내용:
    {extract_text}

    결과 형식 예시:
    1. 분류번호: 1), 3)
    2. 발견 혹은 성과: 
       - (...) 
       - (...)
    """

# 📡 ChatGPT API 요청 함수
def ask_chatgpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are PaperBot, an AI assistant for academic paper analysis."},
                  {"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content

# 안전한 폴더명 생성 함수 (한글 및 특수문자 제거)
def sanitize_folder_name(folder_name):
    """폴더명에서 한글 및 특수문자 제거"""
    folder_name = re.sub(r'[가-힣]', '', folder_name)  # 한글 제거
    folder_name = re.sub(r'[<>:"/\\|?*]', '', folder_name)  # Windows에서 문제되는 문자 제거
    folder_name = folder_name.strip()  # 앞뒤 공백 제거
    return folder_name[:100]  # 너무 길면 100자로 제한

# 안전한 파일명 생성 함수 (한글 및 특수문자 제거)
def sanitize_filename(filename):
    """파일명에서 한글 및 특수문자 제거 및 길이 제한"""
    filename = re.sub(r'[가-힣]', '', filename)  # 한글 제거
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)  # Windows 문제 문자 제거
    filename = filename.strip()  # 앞뒤 공백 제거
    return filename[:200]  # 최대 길이 제한

def copy_to_classification_folders(paper_file_path, categories):
    """논문을 해당하는 모든 분류 폴더로 복사"""
    base_filename = sanitize_filename(os.path.basename(paper_file_path))  # 안전한 파일명으로 변경

    for category in categories:
        raw_folder_name = classification_criteria[int(category)-1].split(') ')[1].split(':')[0]
        folder_name = sanitize_folder_name(f"{category}) {raw_folder_name}")  # 폴더명 정리
        category_folder = os.path.join(OUTPUT_FOLDER, folder_name)
        os.makedirs(category_folder, exist_ok=True)  # 폴더 생성 (이미 있으면 생략)

        dest_file_path = os.path.join(category_folder, base_filename)  # 새로운 파일 경로
        shutil.copy(paper_file_path, dest_file_path)
        print(f"📂 {paper_file_path} → {category_folder}에 복사 완료!")

# 🧐 ChatGPT 응답에서 분류번호 추출
def extract_categories_from_result(result_text):
    match = re.search(r"분류번호:\s*([\d, )]+)", result_text)
    if match:
        categories = re.findall(r"\d+", match.group(1))  
        return categories
    return []

# 🚀 폴더 내 모든 논문을 분석 및 분류
def analyze_all_papers():
    results_json = []
    results_txt = []

    print(f"📂 폴더 내 모든 논문을 분석합니다: {PAPER_FOLDER_PATH}\n")

    for filename in os.listdir(PAPER_FOLDER_PATH):
        if filename.endswith(".pdf"):
            file_path = os.path.join(PAPER_FOLDER_PATH, filename)
            print(f"\n📄 분석 중: {filename}")

            #extract_text = extract_abstract_from_pdf(file_path)
            extract_text = extract_first_300_words(file_path)
            prompt = generate_prompt(extract_text)
            result = ask_chatgpt(prompt)

            print("\n📌 논문 분석 결과:")
            print(result)

            # JSON 결과 저장
            results_json.append({"filename": filename, "analysis": result})

            # TXT 결과 저장
            results_txt.append(f"\n📄 논문: {filename}\n{result}\n{'-'*80}\n")

            # 📂 논문 분류 폴더로 복사
            categories = extract_categories_from_result(result)
            if categories:
                copy_to_classification_folders(file_path, categories)
            else:
                print("⚠ 분류 정보를 찾을 수 없어 복사하지 않음.")

    # JSON 파일 저장
    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=4)

    # TXT 파일 저장
    with open(RESULT_TXT_PATH, "w", encoding="utf-8") as f:
        f.writelines(results_txt)

    print("\n✅ 모든 논문 분석 완료!")
    print(f"📜 분석 결과 JSON 저장: {RESULT_JSON_PATH}")
    print(f"📄 분석 결과 TXT 저장: {RESULT_TXT_PATH}")

# 실행 (폴더 내 모든 논문 분석)
if __name__ == "__main__":
    analyze_all_papers()