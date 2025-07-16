
# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
if ! command -v conda &> /dev/null; then
    echo "[INFO] miniconda 설치 시작"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
else
    echo "[INFO] miniconda 이미 설치되어 있음"
fi


# Conda 환경 생성 및 활성화
# Conda 환경 이름: myenv
if [ ! -d "$HOME/miniconda/envs/myenv" ]; then
    echo "[INFO] 가상환경 myenv 생성 중..."
    source "$HOME/miniconda/etc/profile.d/conda.sh"
    conda create -y -n myenv python=3.10
else
    echo "[INFO] 가상환경 myenv 이미 존재함"
fi

# 가상환경 활성화
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    filename=$(basename "$file" .py)
    input_file="../input/${filename}_input"
    output_file="../output/${filename}_output"
    
    echo "[INFO] 실행 중: $file"
    python "$file" < "$input_file" > "$output_file"

done

# mypy 테스트 실행 및 mypy_log.txt 저장
mypy *.py > ../mypy_log.txt

# conda.yml 파일 생성
conda env export --name myenv > ../conda.yml

# 가상환경 비활성화
conda deactivate