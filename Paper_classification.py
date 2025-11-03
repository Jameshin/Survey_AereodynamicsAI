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

#  PDF에서 Abstract 부분만 추출하는 함수
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

#  PDF에서 초기 300단어 추출하는 함수
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

    # 500단어까지만 가져와서 반환
    first_300_words = " ".join(words[:500])  # 500단어일 때 결과가 괜찮았음

    return first_300_words if first_300_words else "No text found"

#  ChatGPT에 보낼 프롬프트 생성
def generate_prompt(extract_text):
    return f"""
    논문의 내용을 분석하여 다음 정보를 한글로 제공해줘:

    1. **이 논문은 아래 60여종 분류 중 어느 카테고리에 속하는지 분류해줘(복수 가능). 분류번호를 2개 수준으로 부여했어. 예를 들어, 1-1)부터 1-6)까지는 상위 수준 분류번호인 "1) Compressible flow physics"의 하위 수준 분류번호들이야.**  
       **분류번호는 하위 수준 분류번호만 가지게 분류해줘. 예외적으로, 분류번호 13)은 하위 수준이 없으니 그냥 상위 수준 분류번호로 써줘. 즉, 상위 수준은 하위 수준 분류번호를 보면 아니까 상위 수준 분류번호는 "13) Review or survey papers"을 제외하고 꼭 빼야 된다.**
       **그리고, 분류할 때 되도록 Title, Abstract, Keywords라는 단어에 가까운 단어들이 해당 논문의 성격을 가장 많이 규정하기 때문에, 그 단어들을 더 중점적으로 감안해서 분류해줘. **
       **Introduction이라는 단어 이후에 나오는 단어들은 전반적인 동향과 관련되기 때문에 실제 논문과 관련없는 키워들이 많이 등장하니 무시하는게 나아.**
       **그래서, 논문의 파일명(혹은 title)과 Abstract, Keywords의 단어들과 내용들로 판단을 해주면 좋겠어." **
       {", ".join(classification_criteria)}
    2. **이 논문에서의 새로운 발견이나 성과는 무엇인가?**  

       **같은 논문은 항상 동일한 분류를 유지해야 하니 여러 번 분류해보고 가장 높은 확률의 분류번호들을 적어줘. 너의 판단이 매번 달라지는 경우가 많았어.**
       **각 논문별로 분류가 된 다음 상위 수준 분류번호가 여전히 들어가는 경우가 있는데, 13)을 제외하고 상위 수준 분류번호들이 안 나오게 다시 한 번 확인해줘.**
       **결과를 analysis_results.txt에 쓸 때 이모티콘은 안들어가게 해주면 좋겠어.**

    논문의 내용:
    {extract_text}

    결과 형식 예시:
    1. 분류번호: 1-2), 7-3), 11-1)
    2. 발견 혹은 성과: 
       - (...) 
       - (...)
    """

#  ChatGPT API 요청 함수
def ask_chatgpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": "You are PaperBot, an AI assistant for academic paper analysis."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
    
# ChatGPT 응답에서 분류번호 추출
def extract_categories_from_result(result_text):
    match = re.search(r"분류번호:\s*([\d, )]+)", result_text)
    if match:
        categories = re.findall(r"\d+", match.group(1))  
        return categories
    return []

#  폴더 내 모든 논문을 분석 및 분류
def extract_year_from_filename(filename):
    """파일 이름에서 (YYYY) 형식의 연도를 추출"""
    match = re.match(r"\(\s*([^\)]+)\)", filename)  # 괄호 안 내용 추출
    if match:
        content = match.group(1)
        digits = re.findall(r"\d{4}", content)       # 4자리 숫자만 찾기
        if digits:
            return digits[0]                         # 첫 번째 연도만 사용
    return "Unknown"

def extract_classification_and_analysis(result_text):
    """ChatGPT 응답에서 분류번호들과 분석 내용 추출"""


    # 1. 복합 분류번호 (예: 1-2), 3-2), 13)) 모두 인식
    classification_matches = re.findall(r"\b(\d{1,2}(?:-\d+)?\))", result_text)
    classification = classification_matches if classification_matches else ["Not found"]

    # 2. 분석 내용 추출
    analysis_match = re.search(r"발견 혹은 성과:\s*([\s\S]+?)(?:\n\d{1,2}(?:-\d+)?\)|\Z)", result_text)
    analysis = analysis_match.group(1).strip() if analysis_match else "Not found"

    return classification, analysis

def clean_title_from_filename(filename):
    """파일명에서 한글 및 특수문자 제거하고 제목 추정"""
    name = filename.replace(".pdf", "")
    # 괄호 안 숫자 제거 (연도)
    name = re.sub(r"\(\d{4}\)", "", name)
    # 한글 제거
    name = re.sub(r"[가-힣]", "", name)
    # 특수기호 제거
    name = re.sub(r"[^\w\s\-]", "", name)
    return name.strip()

def get_citation_count(title):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "fields": "title,citationCount",
        "limit": 1
    }
    headers = {"User-Agent": "PaperBot"}
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        results = response.json()
        if results.get("data"):
            return results["data"][0].get("citationCount", 0)
    except Exception as e:
        print(f"❌ 인용 검색 오류: {e}")
    return 0

def analyze_all_papers():
    results_json = []
    results_txt = []

    print(f"📂 폴더 내 모든 논문을 분석합니다: {PAPER_FOLDER_PATH}\n")

    for filename in os.listdir(PAPER_FOLDER_PATH):
        if filename.endswith(".pdf"):
            file_path = os.path.join(PAPER_FOLDER_PATH, filename)
            print(f"\n📄 분석 중: {filename}")

            extract_text = extract_first_300_words(file_path)
            prompt = generate_prompt(extract_text)
            result = ask_chatgpt(prompt)

            print("\n📌 논문 분석 결과:")
            print(result)

            # 분류번호와 분석 내용 추출
            classification, analysis = extract_classification_and_analysis(result)

            # 파일명에서 연도 추출
            year = extract_year_from_filename(filename)

            # 인용횟수 추출
            #clean_title = clean_title_from_filename(filename)
            #citation_count = get_citation_count(clean_title)
            
            results_json.append({
                "filename": filename,
                "year": year,
                "classification": classification,
                "analysis": analysis,
                #"citations": citation_count    
            })

            # TXT 결과 저장
            results_txt.append(f"\n📄 논문: {filename} ({year})\n분류: {classification}\n{analysis}\n{'-'*80}\n")

    # JSON 파일 저장
    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=4)

    # TXT 파일 저장
    with open(RESULT_TXT_PATH, "w", encoding="utf-8") as f:
        f.writelines(results_txt)

    print("\n✅ 모든 논문 분석 완료!")
    print(f"📜 분석 결과 JSON 저장: {RESULT_JSON_PATH}")
    print(f"📄 분석 결과 TXT 저장: {RESULT_TXT_PATH}")

    print("\n✅ 모든 논문 분석 완료!")
    print(f"📜 분석 결과 JSON 저장: {RESULT_JSON_PATH}")
    print(f"📄 분석 결과 TXT 저장: {RESULT_TXT_PATH}")

# 실행 (폴더 내 모든 논문 분석)
if __name__ == "__main__":
    analyze_all_papers()
