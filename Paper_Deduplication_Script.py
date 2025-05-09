import re

def normalize_title(title):
    # 번호 제거 및 논문명 정규화: '숫자)'으로 시작하는 번호 제거, 띄어쓰기와 기호 제거
    title_without_number = re.sub(r'^\d+\)\s*', '', title)  # 번호 제거
    return re.sub(r'[\s_-]+', '', title_without_number).lower()  # 띄어쓰기와 기호 제거 후 소문자로 변환

def find_duplicates(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    normalized_titles = {}
    duplicates = []

    for line in lines:
        title = line.strip()  # 줄바꿈 제거
        normalized = normalize_title(title)

        if normalized in normalized_titles:
            duplicates.append(title)  # 원래의 논문명을 중복 리스트에 추가
        else:
            normalized_titles[normalized] = title

    return duplicates

# 실행
file_path = 'Paper_list.txt'  # 논문 리스트가 저장된 파일 경로
duplicates = find_duplicates(file_path)

if duplicates:
    print("중복된 파일명:")
    for duplicate in duplicates:
        print(duplicate)
else:
    print("중복된 파일명이 없습니다.")