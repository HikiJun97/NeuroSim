# NeuroSim 실행 가이드

호스트에서 직접(uv) 돌리는 방법과 Docker 컨테이너로 돌리는 방법 두 가지가 있습니다.
어느 쪽이든 **NVIDIA GPU + 드라이버** 가 필요합니다.

---

## 사전 요구사항

| 요구사항 | 로컬 실행 | Docker 실행 |
|---|---|---|
| NVIDIA GPU | ✅ | ✅ |
| NVIDIA Driver (호스트) | ✅ | ✅ |
| CUDA Toolkit (호스트, nvcc) | ❌ | ❌ |
| `uv` (Python/의존성 관리) | ✅ | ❌ (컨테이너 내부) |
| Docker | ❌ | ✅ |
| NVIDIA Container Toolkit | ❌ | ✅ |

CUDA SDK 는 어느 모드에서도 호스트에 깔 필요가 없습니다. PyTorch wheel 이 자체 CUDA 런타임을 번들합니다.

---

## A. 로컬 실행 (uv)

### 1) 첫 셋업 (한 번만)
```bash
# 의존성 설치 (.venv 생성됨)
uv sync

# TensorRT fork (CIM 모듈) 설치
uv pip install -e ./pytorch-quantization

# C++ NeuroSIM 백엔드 컴파일
cd NeuroSIM && make && cd ..
```

### 2) 모델 가중치 준비
| 데이터셋 | 모델 | 가중치 파일 | 학습 방법 |
|---|---|---|---|
| CIFAR-10 | vgg8 | `models/vgg8_cifar10.pth` | `uv run python train_cifar10_vgg8.py` |
| MNIST | lenet | `models/lenet_mnist.pth` | `uv run python train_mnist_lenet.py` |
| ImageNet | resnet18/50/.../swin_t | (불필요) | PyTorch Hub 자동 다운로드 |

### 3) Gradio UI 실행
```bash
# 일반 실행
uv run python gradio_app.py

# 코드 수정하면서 hot reload
uv run gradio gradio_app.py
```
브라우저: `http://localhost:7860` (또는 콘솔에 찍힌 `https://*.gradio.live`)

### 4) CLI 로 직접 실행
```bash
uv run python inference.py \
    --dataset cifar10 --model vgg8 \
    --data_path ./datasets/ --hardware 1 --bitcell 1 \
    --sub_array "[128,128]" --mem_type resistive \
    --off_state 1e-15 --on_state 1e-14 --adc_precision 7
```

---

## B. Docker 실행

### 1) 호스트 준비 (한 번만)

NVIDIA Container Toolkit 설치:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

연결 확인:
```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 nvidia-smi
```
호스트 `nvidia-smi` 와 같은 GPU 정보가 출력되면 OK.

> WSL2 환경에서는 Windows 의 NVIDIA Driver 만 설치돼 있으면 별도 작업 없이 그대로 됩니다. WSL2 내부에 추가 드라이버 설치는 **금지**.

### 2) 이미지 빌드
```bash
docker build -t neurosim:latest .
```
- 첫 빌드 약 10분 (CUDA + PyTorch 다운로드)
- 최종 이미지 약 6~7 GB
- `.dockerignore` 가 데이터셋/가중치/결과를 제외 → 컨텍스트 전송이 가벼움

### 3) 가중치 미리 준비
호스트의 `models/` 디렉토리에 `.pth` 파일이 있어야 합니다 (볼륨 마운트 됨). 위 A-2 단계와 동일하게 학습.

### 4) 컨테이너 실행
```bash
docker run --rm -it \
    --gpus all \
    -p 7860:7860 \
    -v "$(pwd)/datasets:/app/datasets" \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/gradio_runs:/app/gradio_runs" \
    neurosim:latest
```
브라우저: `http://localhost:7860`

| 옵션 | 의미 |
|---|---|
| `--gpus all` | 호스트 GPU 를 컨테이너에 연결 (Container Toolkit 가 처리) |
| `-p 7860:7860` | Gradio 포트 노출 |
| `-v ./datasets:/app/datasets` | CIFAR/MNIST 캐시 호스트 보존 |
| `-v ./models:/app/models` | 학습 가중치 호스트 보존 |
| `-v ./gradio_runs:/app/gradio_runs` | 시뮬레이션 결과 호스트로 노출 |

### 5) 동작 확인 (선택)
```bash
docker run --rm --gpus all neurosim:latest \
    uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.version.cuda)"
```

---

## 결과물 위치

`--run_dir` 가 자동 설정되어 한 번 실행마다 한 디렉토리에 모입니다.

```
gradio_runs/run_YYYYMMDD_HHMMSS/
├─ stdout.log               # Gradio 가 캡처한 전체 출력
├─ test_log_<timestamp>     # inference.py 내부 logger
├─ accuracy.json            # 최종 정확도
├─ ppa_output.log           # C++ PPA 출력 (ppa=1일 때)
├─ mem_states.csv           # 업로드한 경우 사본
└─ results.zip              # 위 전체 + layer_record_<model>/ + NetWork CSV
```

Gradio UI 의 "결과 zip 다운로드" 버튼으로 `results.zip` 을 받을 수 있습니다.

---

## 자주 사용하는 변형

### CIFAR-10 + VGG8 (기본)
```bash
... --dataset cifar10 --model vgg8 --bitcell 1 \
    --sub_array "[128,128]" --parallel_read 128
```

### MNIST + LeNet
LeNet 의 첫 Conv 가 작아서 sub_array 도 줄여야 PPA C++ 가 segfault 안 남.
```bash
... --dataset mnist --model lenet --bitcell 1 \
    --sub_array "[32,32]" --parallel_read 32
```

### 4-bit MLC 셀
```bash
... --bitcell 4 --mem_states_file mem_states_4bit.csv
```
CSV 행 수 = `2^bitcell` 이어야 함.

### 소프트웨어 베이스라인만 빠르게 측정
```bash
... --hardware 0 --ppa 0 --num_batches 10
```

---

## 중지 / 정리

### Gradio UI 실행 중
- 화면의 **Stop** 버튼: 현재 추론 서브프로세스만 종료 (GPU 즉시 회수)
- 터미널 Ctrl+C 또는 `kill <pid>`: gradio + 자식 모두 정리 (PDEATHSIG + atexit 안전망)

### Docker 컨테이너
```bash
# foreground (Ctrl+C) 면 자동 정리
# 다른 터미널에서 강제 중지:
docker stop <container_id>
```

### 디스크 정리
```bash
# 누적된 실행 결과 제거
rm -rf gradio_runs/run_*

# 이미지 제거
docker rmi neurosim:latest
```

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `The states in the external file do not match the bitcell` | `mem_states.csv` 행 수와 `--bitcell` 불일치. `행 수 = 2^bitcell` 로 맞추기 |
| `ERROR: SubArray Size is too large` + segfault (PPA 단계) | 모델 레이어보다 `--sub_array` 가 큼. LeNet 같은 작은 모델은 `[32,32]` 로 |
| `cuda available: False` (Docker) | Container Toolkit 미설치 또는 `--gpus all` 누락 |
| Gradio 공개 URL 안 만들어짐 | `gradio_app.py` 마지막 줄 `share=True` 확인. 또는 `gradio` CLI 모드는 share 무시함 |
| 로딩 시 `pretrained model ... not found` | `models/<model>_<dataset>.pth` 존재 확인. 직접 학습 필요 (위 A-2) |
