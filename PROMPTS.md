# PROMPTS Log — AI 코딩 툴을 활용한 FoundationPose 재현 (클라우드 GPU 환경 구축 · 데모 추론 분석)

> 본 문서는 AI Coding Tool(Anthropic Claude)과의 대화를 바탕으로, NVIDIA의 공개 6D Pose
> Estimation/Tracking 모델 **FoundationPose**를 클라우드 GPU 인스턴스에서 구축하고
> 공식 model-based 데모(`run_demo.py`, mustard 객체)를 실행·분석한 과정의 재현 로그이다.
> **제3자가 본 로그만 보고 동일 과정을 재현할 수 있도록** 입력 프롬프트, 발생한 에러,
> AI가 제시한 진단·해결 명령, 그리고 검증 결과를 단계별로 기록한다.


## 메타 정보

| 항목 | 내용 |
|---|---|
| 과목 / 팀 | 스마트팩토리캡스톤디자인 (1조) |
| 발표자 | 송영집 (YeongJip Song), thdudwlq2di@skku.edu |
| 대상 모델 | FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects (Wen, Yang, Kautz, Birchfield, CVPR 2024 Highlight, arXiv:2312.08344) |
| 원본 코드 | https://github.com/NVlabs/FoundationPose |
| 사용 AI 도구 | Anthropic Claude (대화형 assistant) |
| 재현 환경 | Elice AICE 클라우드 플랫폼, GPU 인스턴스 G-NAHP-80 (NVIDIA A100 80GB PCIe, 79GiB 가용, sm_80), CPU 16 vCore, Memory 192GiB, 실행 환경 VSCode(Remote) / CUDA 12.4 |

## 문서 구성

- **Part 1 — 클라우드 GPU 환경 구축**: 빈 클라우드 인스턴스에서 시작해 Miniconda, gcc,
  CUDA 12.4, PyTorch 2.6, FoundationPose 의존성, PyTorch3D를 설치하기까지. 호환성 이슈
  2건의 에러→해결 사이클이 핵심.
- **Part 2 — `run_demo.py` 실행 및 추론 파이프라인 분석**: 헤드리스 서버 환경 이슈 2건을
  해결하고, mustard 객체에 대한 포즈 가설 생성 → 트랜스포머 정제 → 계층적 포즈 선정까지
  실제 로그를 추적·분석한다.

각 단계는 **상황·입력 → AI 진단/응답 → 실행 명령 → 결과·검증** 순으로 기술한다.

---

# Part 1 — 클라우드 GPU 환경 구축

## 1.0 재현 전제

- Elice AICE 클라우드 플랫폼에서 GPU 인스턴스 신규 생성: 유형 `G-NAHP-80`
  (NVIDIA A100 80GB PCIe), CPU 16 vCore, Memory 192GiB. 인스턴스 이름은 `FoundationPose`로
  지정, 실행 환경은 VSCode(Remote) 기반 CUDA 12.4 이미지를 선택.
- 목표: NVlabs/FoundationPose 원본 repo를 해당 인스턴스에 설치 → 공식 model-based 데모
  데이터(`mustard0`)에 대해 `run_demo.py`를 실행 → 포즈 추정 파이프라인(객체 표현 →
  가설 생성 → 트랜스포머 정제 → 계층적 선정)이 끝까지 정상 동작하는지 확인.

**재현자 환경 가정**: 본 로그는 재현자가 받는 인스턴스가 OS와 NVIDIA 드라이버만 설치된
**빈 상태**라고 가정한다. Miniconda, CUDA toolkit(conda/pip 경유), PyTorch, FoundationPose
자체 의존성(PyTorch3D 등)은 모두 본 로그의 절차를 따라 처음부터 설치해야 한다.

**전체 재현 순서** (이 순서를 지켜야 각 단계가 성립):

```
[1] GPU 인스턴스 유형 + CUDA 버전 결정           (§1.1)
[2] Miniconda 설치                              (§1.2)
[3] gcc-13 고정 (CUDA 12.4 호환)                 (§1.3) ← [이슈 1]
[4] conda env 생성 + PyTorch 2.6 설치            (§1.4)
[5] FoundationPose 클론 + requirements 설치      (§1.5)
[6] PyTorch3D 소스 빌드                          (§1.6) ← [이슈 2]
[7] 데모 데이터(mustard0) + weight 다운로드       (§1.7)   ← 검증자가 직접 받아야 함
[8] 최초 실행 → libGL.so.1 해결                  (Part 2 §2.1) ← [이슈 3]
[9] run_demo.py 정상 실행 + 추론 파이프라인 분석  (Part 2 §2.2 ~ §2.4)
[10] Qt xcb 디스플레이 경고 — 비치명적임 확인     (Part 2 §2.5) ← [이슈 4]
```

> **소요 시간 주의**: PyTorch3D 소스 빌드(§1.6)는 다수의 `.cu` 파일을 컴파일하므로 A100
> 환경에서도 체감상 적지 않은 시간이 소요된다. 백그라운드 작업 없이 단독으로 빌드를
> 진행하는 것을 권장한다.

## 1.1 클라우드 GPU 인스턴스 선택과 CUDA 버전 결정

**입력 프롬프트** (재구성):
> Elice AICE에서 GPU 인스턴스를 새로 받았는데, A100 80GB짜리야. FoundationPose 돌리려면
> CUDA 버전을 몇으로 잡아야 하는지 모르겠어. PyTorch 버전이랑 같이 정해야 할 것 같은데.

**AI 응답 요지**: FoundationPose의 핵심 의존성인 PyTorch3D, open3d, NVIDIA Warp(`warp-lang`)
등은 모두 CUDA 빌드/런타임에 민감한 패키지다. 따라서 **PyTorch 공식 wheel이 제공하는
최신 CUDA 빌드를 기준으로 나머지 스택을 맞추는 것**이 가장 안전한 순서다. 현재 시점
PyTorch 2.6은 `cu124`(CUDA 12.4) 공식 wheel을 제공하며, A100(`sm_80`)은 CUDA 12.4 +
최신 드라이버 조합에서 완전히 지원된다. FoundationPose가 요구하는 `open3d`, `warp-lang`
역시 CUDA 12.x 계열과 호환되므로, **CUDA 12.4 + PyTorch 2.6**으로 스택을 통일하는 것을
권장.

**실행 명령** (인스턴스 콘솔에서 직접 선택 — CLI 명령 없음):
```
AICE 콘솔 → GPU 인스턴스 생성
  - 인스턴스 유형: G-NAHP-80 (NVIDIA A100 80GB PCIe)
  - 실행 환경: VSCode (CUDA 12.4)
  - 인스턴스 이름: FoundationPose
```

**검증** (인스턴스 부팅 후):
```bash
nvidia-smi
# NVIDIA A100 80GB PCIe, Driver Version: ..., CUDA Version: 12.4
```

## 1.2 Miniconda 설치

**입력 프롬프트** (재구성):
> 인스턴스가 막 받은 상태라 conda가 없어. Miniconda부터 깔아야 할 것 같은데 Linux x86_64
> 버전 맞지?

**AI 응답**: 그렇다. AICE 인스턴스는 Linux x86_64 컨테이너이므로 공식 Miniconda
Linux-x86_64 installer를 받아 사용자 홈 아래에 설치하면 된다.

**실행 명령**:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
source ~/.bashrc
```

**검증**: `which conda` 및 `conda --version`으로 정상 설치 확인.

## 1.3 [이슈 1] CUDA 12.4 빌드 시 gcc 버전 비호환

**입력 프롬프트** (재구성 — FoundationPose 의존성 빌드 중 발생한 에러를 붙여넣으며):
> ```
> nvcc fatal   : Unsupported gnu version! gcc versions later than 13 are not supported!
> Use --allow-unsupported-compiler flag to override this version check, but using an
> unsupported host compiler may cause compilation failure or incorrect run time execution.
> Use at your own risk.
> ```
> 인스턴스 기본 gcc로 그냥 빌드했는데 이 에러가 난다. gcc 버전을 낮춰야 하는 건가?

**AI 진단**: NVIDIA CUDA 12.4 toolkit이 `nvcc`의 host compiler로 공식 지원하는 GCC의
상한은 **13.2**다. AICE 인스턴스 기본 이미지에 설치된 gcc가 이보다 신버전이라 nvcc가
컴파일을 거부하는 것. `--allow-unsupported-compiler`로 강제 우회하는 방법도 있지만,
경고문에 명시된 대로 컴파일 실패나 잘못된 런타임 동작으로 이어질 수 있어 권장하지
않는다. 가장 안전한 해법은 **CUDA 12.4가 공식 지원하는 gcc-13/g++-13을 별도로 설치하고,
`CC`/`CXX` 환경변수로 명시적으로 고정**하는 것이다. 이렇게 하면 이후 PyTorch3D 등의
CUDA 확장 빌드에서도 동일한 컴파일러가 일관되게 사용된다.

**실행 명령**:
```bash
sudo apt-get update
sudo apt-get install -y gcc-13 g++-13

export CC=gcc-13
export CXX=g++-13
# (이후 모든 셸 세션에서 재사용하려면 ~/.bashrc에도 추가)
echo 'export CC=gcc-13'  >> ~/.bashrc
echo 'export CXX=g++-13' >> ~/.bashrc
```

**검증**:
```bash
gcc-13 --version    # gcc-13.x.x 확인
$CC --version        # CC가 gcc-13을 가리키는지 확인
```

**재현 교훈**: CUDA toolkit 버전마다 nvcc가 공식 지지하는 host compiler 상한이 명시되어
있다(CUDA 12.4 → GCC 13.2). 클라우드 이미지의 "기본" 컴파일러가 이 상한보다 신버전인
경우가 실제로 발생하므로, CUDA 확장을 빌드하기 전에는 반드시 컴파일러 버전을 먼저
확인·고정해야 한다.

## 1.4 conda 환경 생성 + PyTorch 2.6 설치

**입력 프롬프트** (재구성):
> gcc는 고정했고, 이제 conda 환경 만들고 PyTorch 깔 차례야. CUDA 12.4에 맞는 PyTorch
> 설치 명령이 정확히 어떻게 되는지？

**AI 응답**: `conda create`로 독립된 Python 환경을 먼저 만들고, PyTorch 공식
`download.pytorch.org` 인덱스에서 `cu124` 빌드를 명시적으로 지정해 설치한다.
`torch`만 설치하면 추후 `torchvision`/`torchaudio`가 별도 인덱스에서 받아져 버전이
어긋날 수 있으므로 세 패키지를 함께 설치하는 것이 안전하다.

**실행 명령**:
```bash
conda create -n foundationpose python=3.11 -y
conda activate foundationpose

pip install torch==2.6.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
```

**검증**:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.6.0+cu124 True
python -c "import torch; print(torch.cuda.get_device_name(0))"
# NVIDIA A100 80GB PCIe
```

## 1.5 FoundationPose 클론 + 의존성 설치

**입력 프롬프트** (재구성):
> 이제 FoundationPose 원본 repo를 받아서 requirements 설치해야 할 것 같은데, README
> Option 2(conda 로컬 설치) 순서를 그대로 따라가면 되는 거지?

**AI 응답**: README의 "Env setup option 2: conda (local)" 절차가 정확히 이 상황을
가정하고 있다. `requirements.txt` 설치 후, PyTorch3D 빌드에 필요한 `fvcore`/`iopath`
(PyTorch3D `setup.py`가 import 시점에 요구하는 헬퍼 패키지)와 `setuptools`/`wheel`
(소스 빌드용 패키징 도구)을 미리 설치해 두면 다음 단계의 빌드 실패를 줄일 수 있다.

**실행 명령**:
```bash
git clone https://github.com/NVlabs/FoundationPose.git
cd FoundationPose

pip install -r requirements.txt
pip install fvcore iopath setuptools wheel
```

**검증**: `pip show fvcore iopath` 등으로 설치 확인. `requirements.txt`에 포함된
`open3d`, `warp-lang`(NVIDIA Warp) 등 핵심 패키지가 에러 없이 설치되었는지 로그로 확인.

## 1.6 [이슈 2] PyTorch3D — CUDA 12.4용 사전 빌드 wheel 부재

**입력 프롬프트** (재구성):
> `pip install pytorch3d`를 그냥 해봤더니 맞는 버전을 못 찾는다고 나온다. cu124 조합이
> 아직 정식으로 안 올라온 건가?

**AI 진단**: PyTorch3D는 PyPI/conda 채널에 일부 (PyTorch 버전, CUDA 버전) 조합에 대해서만
사전 컴파일된 wheel을 배포한다. **PyTorch 2.6 + CUDA 12.4 조합은 현재 사전 빌드 wheel이
제공되지 않으므로 소스에서 직접 빌드해야 한다.** 빌드 시 `--no-build-isolation` 옵션이
필수인데, 이 옵션이 없으면 pip이 격리된 빌드 환경을 새로 만들면서 이미 설치한 `torch`를
다시 설치하려 시도해 버전이 어긋날 수 있다. `--no-build-isolation`을 주면 현재 conda
환경에 이미 설치된 `torch 2.6.0+cu124`를 그대로 헤더/라이브러리로 사용해 빌드한다.
앞 단계(§1.3)에서 `CC=gcc-13`, `CXX=g++-13`로 고정해 두었기 때문에 nvcc 호환성 문제 없이
빌드가 진행된다.

**실행 명령**:
```bash
pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git"
```

**검증**:
```bash
python -c "import pytorch3d; print(pytorch3d.__version__)"
python -c "from pytorch3d import _C; print(_C)"   # CUDA 확장(.so) 정상 로드 확인
```

**재현 교훈**: 바이너리 배포가 따라가지 못하는 최신 (PyTorch, CUDA) 조합을 쓸 경우,
소스 빌드가 기본 경로가 된다는 점을 처음부터 가정하고 빌드 도구(`fvcore`/`iopath`/
`ninja`/컴파일러)를 먼저 갖춰두는 편이 전체 소요 시간을 줄인다.

## 1.7 데모 데이터 + 가중치 준비

**입력 프롬프트** (재구성 — README의 *Data prepare* 섹션을 붙여넣으며):
> ```
> 1. Download all network weights from [here] and put them under weights/.
>    For the refiner, you will need 2023-10-28-18-33-37.
>    For scorer, you will need 2024-01-11-20-02-45.
> 2. Download demo data and extract them under demo_data/
> ```
> mustard 데모만 일단 돌려보고 싶은데, 위 두 개만 받으면 되는 거 맞지?

**AI 응답 + 본인 확인**: 맞다. `run_demo.py`는 기본 argparse 경로가 `mustard0` 데이터와
refiner 가중치(`2023-10-28-18-33-37`)를 가리키도록 되어 있다. scorer 가중치
(`2024-01-11-20-02-45`)는 252개 포즈 후보를 랭킹할 때 필요하므로 함께 받아둔다.

**실행 명령**:
```bash
mkdir -p weights demo_data

# (1) Google Drive에서 weights/ 아래 두 체크포인트 디렉토리를 받아 배치)
ls weights/
#   2023-10-28-18-33-37/   ← refiner
#   2024-01-11-20-02-45/   ← scorer

# (2) mustard0 데모 데이터 (약 345MB .zip) 다운로드 후 압축 해제
unzip mustard0.zip -d demo_data/
ls demo_data/mustard0/    # rgb/, depth/, masks/, mesh/, cam_K.txt 등 확인
```

**배치 검증**:
```bash
ls weights/2023-10-28-18-33-37   # refiner 가중치 존재
ls weights/2024-01-11-20-02-45   # scorer 가중치 존재
ls demo_data/mustard0             # 데모 씬 데이터 존재
```

두 경로가 모두 확인되면 §2(`run_demo.py` 실행)로 진행 가능.

---

## Part 1 재현 체크포인트 요약

빈 클라우드 인스턴스에서 실행 직전까지 도달하기 위한 핵심 순서와 함정:

1. GPU 인스턴스(G-NAHP-80, A100 80GB) + CUDA 12.4 결정 — PyTorch 2.6 wheel 호환 기준
2. Miniconda 설치
3. **[이슈 1]** gcc 비호환 → `gcc-13`/`g++-13` 설치 + `CC`/`CXX` 고정 (CUDA 12.4 상한)
4. conda env(`foundationpose`, Python 3.11) + `torch==2.6.0+cu124` 설치
5. FoundationPose 클론 + `requirements.txt` + `fvcore`/`iopath`/`setuptools`/`wheel`
6. **[이슈 2]** PyTorch3D cu124 사전 wheel 없음 → `--no-build-isolation` 소스 빌드
7. weight(refiner `2023-10-28-18-33-37`, scorer `2024-01-11-20-02-45`) + `mustard0` 데모 데이터 배치

두 이슈는 모두 **"최신 (PyTorch, CUDA) 조합을 클라우드 GPU 인스턴스의 기본 이미지/
배포판 패키지 매니저가 아직 따라오지 못한 데서 발생하는 충돌"**이라는 공통 패턴이다.

---

# Part 2 — `run_demo.py` 실행 및 추론 파이프라인 분석

> Part 1에서 구축한 환경 위에서 공식 model-based 데모(`python run_demo.py`)를 실행하고,
> 헤드리스(headless) 클라우드 서버 특유의 이슈 2건을 해결한 뒤, 실제 로그를 근거로
> FoundationPose의 포즈 가설 생성 → 트랜스포머 정제 → 계층적 선정 파이프라인이 의도대로
> 동작하는지 단계별로 분석한다.

## 2.0 1차 실행 — 목표와 기대 동작

**입력 프롬프트** (재구성):
> 환경은 다 갖춘 것 같으니 이제 `python run_demo.py` 돌려본다. README 보면 mustard
> bottle 객체에 대해서 6D pose를 잡아주는 데모라고 하던데, 별다른 인자 없이 그냥 실행하면
> 되는 거지?

**AI 응답**: 그렇다. `run_demo.py`의 argparse 기본값이 이미 `demo_data/mustard0`와
`weights/2023-10-28-18-33-37`을 가리키도록 설정되어 있어 별도 인자 없이 실행 가능하다.
다만 첫 실행 시 일부 모듈이 온라인으로 JIT 컴파일/캐시를 생성하므로 첫 실행은 다소
느릴 수 있다는 점을 README가 명시하고 있다.

**실행 명령**:
```bash
cd FoundationPose
python run_demo.py
```

## 2.1 [이슈 3] `libGL.so.1` 누락 — 헤드리스 서버에서 `open3d` import 실패

**입력 프롬프트** (원문 취지 재구성 — 최초 실행 시 발생한 에러를 붙여넣으며):
> ```
> ImportError: libGL.so.1: cannot open shared object file: No such file or directory
> ```
> open3d를 import하는 부분에서 막힌다. GPU 인스턴스인데 왜 GL 관련 에러가 나는지
> 모르겠다.

**AI 진단**: Elice AICE의 헤드리스(headless) 서버 이미지에는 데스크탑 GUI 패키지가
설치되어 있지 않은 경우가 많아, OpenGL 런타임 공유 라이브러리(`libGL.so.1`)가
누락되어 있을 수 있다. `open3d`는 시각화 기능을 직접 호출하지 않아도 **모듈을 import
하는 시점에 내부적으로 OpenGL 기반 렌더링 컨텍스트 관련 심볼을 로드**하기 때문에,
순수 numpy/포인트클라우드 연산만 쓰더라도 `libGL.so.1`이 없으면 import 자체가 실패한다.
GPU 연산(CUDA)과는 무관한, OS 레벨의 그래픽 런타임 라이브러리 부재 문제이므로 `apt`로
필요한 공유 라이브러리를 설치하면 해결된다.

**실행 명령**:
```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

**검증**:
```bash
python -c "import open3d; print(open3d.__version__)"   # 에러 없이 import 성공
```

**재현 교훈**: "GPU 인스턴스니까 그래픽 관련 라이브러리는 다 있을 것"이라는 가정은
틀리기 쉽다. 헤드리스 클라우드 이미지는 CUDA 런타임은 갖추고 있어도, 데스크탑용 OpenGL
공유 라이브러리는 별도 설치 대상으로 빠져 있는 경우가 흔하다.

## 2.2 정상 초기화 로그 분석 — Warp / CUDA / 가중치 / 하이퍼파라미터

`libGL` 문제를 해결한 뒤 재실행하면, FoundationPose가 사용하는 NVIDIA Warp 기반
렌더링·연산 백엔드가 정상적으로 초기화되는 로그를 확인할 수 있다.

**확인된 로그 요지**:
```
Warp 1.13.0 initialized.
CUDA Toolkit: 12.9, Driver: 12.4
NVIDIA A100 80GB PCIe (79 GiB, sm_80) detected.
Loading refiner weights: weights/2023-10-28-18-33-37/model_best.pth
```

- **Warp 1.13.0**이 정상 초기화되며, A100(sm_80, 79GiB)을 인식했다는 로그를 확인.
- "CUDA Toolkit 12.9 / Driver 12.4"처럼 두 숫자가 다르게 보이는 것은 비정상이 아니다.
  Warp는 내부에 자체 CUDA JIT 컴파일 툴체인을 함께 배포하므로, 시스템에 맞춘
  Driver/PyTorch 빌드(CUDA 12.4)와 Warp가 내부적으로 사용하는 컴파일러 툴체인 버전이
  서로 다르게 보고될 수 있다. 둘 다 A100 드라이버가 지원하는 범위 내이므로 문제 없음.
- 학습된 refiner 가중치(`model_best.pth`, 체크포인트 디렉토리 `2023-10-28-18-33-37`)가
  정상적으로 로드됨.

**모델 설정(`self.cfg`) 주요 하이퍼파라미터** (로그에서 확인):

| 항목 | 값 |
|---|---|
| `lr` | 0.0001 |
| `n_view` | 1 |
| `rot_rep` | axis_angle |
| `batch_size` | 64 |
| `crop_ratio` | 1.2 |
| `trans_rep` | tracknet |
| `loss_type` | l2 |
| `rot_normalizer` | 0.349 |
| `enable_amp` | True |

→ 체크포인트에 저장된 학습 설정이 로드 시점에 그대로 복원되고 있음을 확인. `enable_amp:
True`로 자동 혼합정밀도(AMP) 추론이 활성화되어 있어 A100의 Tensor Core 연산 효율을
그대로 활용하는 구성임을 알 수 있다.

## 2.3 포즈 가설 생성 → 정제 → 선정 파이프라인 추적

**입력 프롬프트** (재구성):
> 초기화는 잘 됐고, 이제 실제로 포즈를 추정하는 부분의 로그를 같이 보면서 슬라이드에서
> 설명한 "504개/다수의 가설 생성 → 트랜스포머 정제 → 계층적 선정" 구조가 코드에서 실제로
> 어떻게 흘러가는지 확인하고 싶다.

**AI 응답 + 본인 검증**: 로그에 찍힌 함수 호출 순서와 텐서 shape를 그대로 추적하면
파이프라인 단계를 1:1로 매칭할 수 있다.

1. **포즈 후보 배치 구성** — `make_crop_data_batch()`가 반복 호출되며,
   `tf_to_crops → render → warp → pose batch` 순으로 각 포즈 후보에 대한 렌더링/크롭
   이미지를 만든다.
2. **반복적 정제(iterative refinement)** — `predict()` forward가 반복 호출되며
   coarse-to-fine 방식으로 회전/이동 업데이트를 누적 적용한다.
3. **후보 텐서 shape 확인**:
   ```
   ob_in_cams.shape          : (252, 4, 4)   # 252개 포즈 후보(SE(3) 4x4)
   pose_data.rgbAs.shape[0]  : 252            # 252장의 RGB 크롭 이미지
   ```
   → 슬라이드에서 설명한 "42 viewpoints × 12 in-plane rotations = 504개 초기 가설"
   설정과 달리, 이번 실행에서는 정제 단계에 들어가기 전 필터링·반복을 거쳐 **252개
   후보**가 최종 정제·랭킹 대상으로 유지된 것을 실측 로그로 확인.
4. **최종 포즈 확정**:
   ```
   register() final, add_errs min: -1.0
   ```
   `add_errs min: -1.0`은 ground-truth 포즈가 없는 실제 추론(라벨 없는 임의 장면)
   상황에서 오차를 계산할 기준이 없음을 나타내는 placeholder 값이며, 평가용이 아닌
   **실제 추론(inference) 모드**로 정상 동작하고 있음을 보여주는 신호로 해석.
5. **포즈 후보 정렬**:
   ```
   sort ids tensor
   top indices: [67, 98, 0, 1, 6, 7, 30, 31, 68, 72, ...]
   ```
   계층적 포즈 선정 네트워크(scorer, 체크포인트 `2024-01-11-20-02-45`)가 252개 후보를
   점수 기반 내림차순으로 정렬한 인덱스. 0/1번처럼 초기 가설 순서상 앞쪽에 있던 후보와
   67/98처럼 뒤쪽에 있던 후보가 함께 상위권에 오른 것을 보면, **정렬이 초기 생성 순서와
   무관하게 실제 정합 품질만으로 이루어지고 있음**을 시사.

**본인 검증** (AI 설명에 의존하지 않고 로그 수치 직접 대조):
```bash
grep -n "ob_in_cams.shape\|register() final\|sort ids" run_demo.py inference 로그
# → 로그상의 shape/값이 코드의 어느 호출 지점에서 나온 출력인지 직접 라인 단위로 대조
```

## 2.4 포즈 점수(scores) 분석

**확인된 로그**:
```
sorted scores (top): 77.8125, 77.7891, 77.7656, ...
sorted scores (bottom of range): 75.84, ...
```

- **최고 점수 77.8125** — 252개 후보 중 최종 선택된 포즈(상위 인덱스 `67`)의 점수.
- **점수 범위 75.84 ~ 77.81** — 252개 후보 전체의 점수 분포.
- 상위 수십 개 후보가 77.x대에 밀집되어 있어, 단일 후보로 우연히 수렴한 것이 아니라
  **다수의 후보가 비슷하게 높은 정합 품질로 모여드는 강건한 수렴 양상**으로 해석.
  슬라이드 §핵심 연구 방법론에서 설명한 "계층적 비교(로컬 정렬 → 글로벌 비교)" 전략이
  실제 로그에서도 후보들을 한 점으로 몰지 않고 **상위 군집** 형태로 평가하고 있다는
  근거가 된다.

## 2.5 [이슈 4] Qt xcb 디스플레이 경고 — 비치명적 헤드리스 이슈

**입력 프롬프트** (재구성 — 실행 말미에 나타난 경고를 붙여넣으며):
> ```
> qt.qpa.xcb: could not connect to display
> qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
> ```
> 결과가 어느 정도 나온 것 같긴 한데 맨 끝에 이 에러가 뜬다. 이것도 고쳐야 하는 건가?

**AI 진단**: 이 에러는 `cv2.imshow()` 계열의 **GUI 윈도우 시각화 호출**이 디스플레이가
없는 헤드리스 서버에서 발생할 때 나오는 전형적인 경고다. 로그 순서를 보면 252개 포즈
후보 생성, 정제, scorer를 통한 랭킹·정렬까지의 **핵심 포즈 추정 연산은 이 경고가
뜨기 전에 이미 모두 완료**되어 있다. 즉 이 에러는 추론 결과의 정확도와는 무관하며,
"추정된 포즈를 화면에 띄워 보여주는" 마지막 시각화 단계에서만 발생한다.

**선택적 해결책** (결과 자체에는 영향이 없어 필수는 아니나, 경고를 없애고 싶을 경우):
```bash
# Qt를 오프스크린 모드로 강제 — GUI 윈도우 없이 디스플레이 호출을 무시
export QT_QPA_PLATFORM=offscreen
python run_demo.py

# 또는 시각화 단계에서 cv2.imshow 대신 cv2.imwrite로 결과 이미지를 저장하도록
# 코드를 수정하면 헤드리스 환경에서도 결과를 파일로 확인 가능
```

**결론**: 포즈 추정 로직 자체는 정상 동작했으며, 시각화(디스플레이 출력) 단계만
헤드리스 환경 특성상 스킵된 것. 클라우드 GPU 인스턴스에서 데모를 재현하는 경우 흔히
마주치는, 결과에 영향이 없는 이슈로 분류.

## 2.6 최종 결과 정리

| 항목 | 값 |
|---|---|
| 대상 객체 | mustard (model-based demo, `demo_data/mustard0`) |
| 생성·정제된 포즈 후보 수 | 252 |
| 최고 포즈 점수 | 77.8125 |
| 포즈 점수 범위 | 75.84 ~ 77.81 |
| `add_errs min` | -1.0 (GT 없는 실추론 모드, placeholder) |
| 상위 인덱스(랭킹 1~3) | 67, 98, 0 |

→ FoundationPose 6D 포즈 추정 파이프라인(객체 표현 → 가설 생성 → 트랜스포머 기반 반복
정제 → 계층적 포즈 선정)이 Elice AICE A100 80GB 클라우드 인스턴스에서 **엔드투엔드로
정상 재현됨**을 확인.

---

## Part 2 재현 체크포인트 요약

1. `python run_demo.py` 1차 실행 시 **[이슈 3]** `open3d` import에서 `libGL.so.1` 누락
   → `apt install libgl1 libglib2.0-0`
2. 재실행 시 Warp 1.13.0 / A100(sm_80) 인식, refiner 가중치(`2023-10-28-18-33-37`) 로드,
   하이퍼파라미터(`lr=0.0001`, `n_view=1`, `rot_rep=axis_angle`, `batch_size=64` 등) 확인
3. `make_crop_data_batch()` → `predict()` 반복 → `register()`로 포즈 가설 252개
   (`ob_in_cams.shape=(252,4,4)`) 생성·정제 확인
4. scorer(`2024-01-11-20-02-45`)가 252개 후보를 점수 기반 정렬 — 최고 점수 77.8125,
   상위권 군집 분포(75.84~77.81)
5. **[이슈 4]** 결과 산출 이후 `qt.qpa.xcb` 디스플레이 경고 발생 — 비치명적, 핵심 연산은
   이미 완료된 상태이므로 결과에 영향 없음 (필요 시 `QT_QPA_PLATFORM=offscreen`으로 억제)

두 이슈는 모두 **"GPU 연산 자체와는 무관한, 헤드리스 클라우드 서버의 OS/그래픽 스택
누락에서 비롯된 충돌"**이라는 공통 패턴이며, Part 1의 두 이슈("최신 (PyTorch, CUDA)
조합을 배포 채널이 아직 못 따라가는 문제")와는 성격이 다른 별도 카테고리다.

---

# 구현 검증 (Verification)

> 재현자가 "구현이 성공적으로 수행되었는지"를 스스로 판정할 수 있도록, 검증 절차와 기대
> 출력을 한곳에 모은다. 아래 수치는 본 환경(Elice AICE, A100 80GB PCIe)에서 측정한
> 실측값이다.

## V.1 환경 인증 — `check_env.py` + 자체 점검 스크립트

FoundationPose 원본 repo에는 환경 적합성을 1차로 점검할 수 있는 `check_env.py`가
포함되어 있다. 재현자는 본격적인 데모 실행 전에 이를 먼저 돌려 GPU/CUDA/주요 패키지
인식 여부를 확인하는 것을 권장한다.

**실행 명령**:
```bash
cd FoundationPose
conda activate foundationpose
python check_env.py
```

이에 더해, 버전 일관성을 한 번에 출력하는 보조 점검 스크립트를 작성해 두면 재현
환경을 본 로그의 기준과 직접 대조할 수 있다.

**`reproduction/verify_env.sh`** (제안 스크립트):
```bash
#!/usr/bin/env bash
echo "=== [1] GPU & Driver ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "=== [2] Conda Environment ==="
echo "Active env: $CONDA_DEFAULT_ENV"   # 기대값: foundationpose

echo "=== [3] Compiler ==="
echo "CC=$CC CXX=$CXX"                  # 기대값: gcc-13 / g++-13
gcc-13 --version | head -n1

echo "=== [4] Key Package Versions ==="
python - <<'PYEOF'
import torch, pytorch3d, warp, open3d
print("torch       :", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("pytorch3d   :", pytorch3d.__version__)
print("warp        :", warp.__version__)
print("open3d      :", open3d.__version__)
PYEOF

echo "=== [5] Weights & Demo Data ==="
ls weights/2023-10-28-18-33-37 >/dev/null 2>&1 && echo "refiner weight  : OK" || echo "refiner weight  : MISSING"
ls weights/2024-01-11-20-02-45 >/dev/null 2>&1 && echo "scorer weight   : OK" || echo "scorer weight   : MISSING"
ls demo_data/mustard0           >/dev/null 2>&1 && echo "mustard0 data   : OK" || echo "mustard0 data   : MISSING"
```

**판정 기준**: [1]에서 A100 80GB가 인식되고, [3]의 `gcc-13`이 활성화되어 있으며, [4]에서
`torch.cuda.is_available()=True`이고 `pytorch3d`/`warp`/`open3d`가 모두 에러 없이
import되며, [5]의 weight/demo data가 모두 `OK`이면 **환경 구축 성공**으로 판정한다.

## V.2 `run_demo.py` 추론 결과 검증

**실행 명령**:
```bash
cd FoundationPose
conda activate foundationpose
python run_demo.py 2>&1 | tee run_demo.log
```

**검증 절차 및 판정 기준**:
1. 로그 초반에 `Warp 1.13.0 initialized`, `NVIDIA A100 80GB PCIe ... sm_80 detected`,
   `Loading refiner weights: .../2023-10-28-18-33-37/model_best.pth`가 출력되면
   **초기화 성공**.
2. `libGL.so.1` 관련 `ImportError`가 더 이상 나타나지 않아야 함 (§2.1 패치 적용 확인).
3. 로그 중반에 `ob_in_cams.shape: (252, 4, 4)`, `pose_data.rgbAs.shape[0]: 252`가
   출력되고, `register() final, add_errs min: -1.0`까지 도달하면 **포즈 가설 생성·정제
   성공**.
4. 정렬된 점수(`sorted scores`)의 최댓값이 **77 이상**(본 환경 실측 77.8125)이고, 상위
   인덱스 목록(`sort ids`)이 출력되면 **계층적 포즈 선정 성공**.
5. 이후 `qt.qpa.xcb` 경고가 나타나더라도 위 4단계가 이미 로그에 모두 찍혀 있다면 이는
   **무시 가능한 비치명적 이슈**로 판정하고 전체 파이프라인을 **성공**으로 간주한다.

```bash
# 빠른 자동 판정 예시 (run_demo.log 기준)
grep -q "Warp.*initialized" run_demo.log && \
grep -q "ob_in_cams.shape" run_demo.log && \
grep -q "register() final" run_demo.log && \
echo "PIPELINE OK" || echo "PIPELINE INCOMPLETE — 로그 확인 필요"
```

## V.3 검증 자산 일람 (repo)

| 자산 | 경로 | 용도 |
|---|---|---|
| 원본 환경 점검 스크립트 | `check_env.py` (NVlabs 원본 제공) | GPU/CUDA/패키지 1차 점검 |
| 자체 인증 스크립트(제안) | `reproduction/verify_env.sh` | 버전 일관성 + weight/data 배치 통합 점검 |
| 실행 로그 | `run_demo.log` | 추론 파이프라인 단계별 산출물(shape, score) 보존 |
| 데모 실행 스크립트 | `run_demo.py` (NVlabs 원본 제공) | mustard0 model-based 데모 추론 |

---

## 종합 — AI 코딩 툴 활용 패턴 (재현 시 유효했던 방식)

- 에러를 그대로 붙여넣고 "왜 이게 뜨는지"부터 묻는 진단 우선 흐름을 반복 — 추측성
  수정(예: gcc 버전을 무작정 낮추거나 `--allow-unsupported-compiler`로 강제 우회)보다
  **근본 원인(CUDA 12.4의 host compiler 지지 상한, open3d의 OpenGL 의존, headless 환경의
  GUI 부재)을 먼저 확정**한 뒤 한 줄짜리 명령으로 해결.
- 로그에 찍힌 텐서 shape(`ob_in_cams.shape=(252,4,4)`)와 점수 수치를 슬라이드에서
  설명한 이론적 파이프라인(가설 생성 → 정제 → 계층적 선정)에 직접 대조해, AI의 코드
  설명이 실제 실행 로그와 일치하는지 본인이 검증.
- 결과에 영향 없는 경고(Qt xcb)와 치명적 에러(libGL.so.1 `ImportError`)를 구분해, 모든
  경고를 무조건 "고쳐야 할 문제"로 취급하지 않고 **파이프라인의 어느 단계 이전/이후에
  발생했는지**를 기준으로 치명도를 판단.

## 향후 작업

1. `run_demo.py` 인자를 바꿔 다른 객체(driller 등)·다른 모델-프리 참조 이미지 세트로
   확장 재현 — README가 명시한 "no need to retrain" 특성을 추가로 검증.
2. LINEMOD / YCB-Video 등 공개 벤치마크(`run_linemod.py`, `run_ycb_video.py`)로 확장해
   정량 지표(ADD, ADD-S AUC)를 본 클라우드 환경에서 직접 재현, 논문 보고치와 대조.
3. 헤드리스 환경에서도 결과를 바로 확인할 수 있도록 시각화 단계를
   `cv2.imshow` → `cv2.imwrite` 기반 저장 방식으로 코드 수정하여 Qt 의존성 자체를 제거.
4. 본 캡스톤 모델(자체 6D pose 모델)과의 비교/통합을 위해, FoundationPose의 입출력
   인터페이스(RGB-D + mask + CAD/참조이미지 → SE(3) pose)를 동일하게 따르는 어댑터 설계.

---

마지막 갱신: 2026년 6월
