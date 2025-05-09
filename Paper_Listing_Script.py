import os

def list_files(directory, output_file):
    # 디렉터리 내 파일 리스트 얻기
    files = os.listdir(directory)
    
    # 파일명에서 확장자 제거
    file_names_without_extension = [os.path.splitext(file_name)[0] for file_name in files]
    
    # 파일명과 번호를 매겨서 리스트
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, file_name in enumerate(file_names_without_extension, 1):
            f.write(f"{i}) {file_name}\n")

# 디렉터리 경로와 출력 파일명 지정
directory_path = './Total'  # 논문 폴더
output_file_name = './Paper_list.txt'  # 출력 파일

# 파일 리스트 출력
list_files(directory_path, output_file_name)
