# TurboQuant · PolarQuant · QJL 인터랙티브 시각화 데모

이 저장소는 **TurboQuant**, **PolarQuant**, **QJL**, 그리고 비교 기준이 되는 **기존 Cartesian 양자화 baseline**을 한 화면에서 직접 보고 설명할 수 있게 만든 **Streamlit 기반 시각화 데모**입니다.

핵심 목적은 논문 수식을 그대로 복사해 두는 것이 아니라,

- 각 방법이 **무엇을 보존하려는지**
- 양자화 전후에 **좌표, 각도, 반지름, inner product, attention score**가 어떻게 달라지는지
- 어떤 그림은 **논문에 가까운 표현**이고 어떤 그림은 **설명을 위한 단순화**인지

를 발표나 학습 문맥에서 **직관적으로 이해할 수 있게 보여 주는 것**입니다.

---

## 이 레포가 보여주고 싶은 것

이 앱은 아래 질문에 답하기 위해 만들어졌습니다.

### 1) 그냥 좌표를 바로 양자화하면 어떤 왜곡이 생기나?
기존 양자화 baseline 탭에서 **Cartesian uniform quantization**이 만드는 격자형 스냅을 먼저 보여 줍니다.

### 2) TurboQuant는 왜 회전을 먼저 보나?
TurboQuant 탭에서 **회전 후 좌표별 scalar quantization**이 어떤 구조를 만드는지 보여 줍니다.

### 3) PolarQuant는 왜 각도 관점이 중요한가?
PolarQuant 탭에서 **polar 표현, 각도 스냅, 반지름 변화**를 함께 보여 줍니다.

### 4) QJL은 왜 복원보다 inner-product estimation으로 읽어야 하나?
QJL 탭에서 **sign sketch**, **scaled sketch**, **true vs estimated inner product**, **attention score proxy**를 먼저 보이도록 구성했습니다.

즉, 이 저장소의 중심은 “누가 더 예쁘게 복원되나”만 보는 것이 아니라,
**방법마다 보존하려는 정보가 다르다**는 점을 시각적으로 설명하는 데 있습니다.

---

## 포함된 구성

현재 앱은 다음 흐름으로 구성되어 있습니다.

### 1. 기존 양자화
- 원래 좌표계에서 바로 적용하는 **uniform scalar quantization**
- 공통 격자 코드북 위로 점이 어떻게 붙는지 확인
- Turbo / Polar / QJL과 비교하기 위한 기준선

### 2. TurboQuant
- 무작위 회전 후 좌표별 quantization
- 회전 좌표 공간에서의 **격자형 스냅**
- 복원 전후 구조 변화와 내적 변화 비교

### 3. PolarQuant
- polar 표현으로 바꾼 뒤 angle 중심 양자화
- 좌표쌍 단면에서 보이는 **방사형 구조**
- 각도와 반지름이 어떻게 달라지는지 확인

### 4. QJL
- `Sk → sign(Sk) → scaled sketch` 흐름
- **비대칭 inner-product estimator** 관점 설명
- true / estimated inner product와 attention score proxy 비교
- 3D 그림은 보조 설명용 surrogate visualization

### 5. 비교 / 하이브리드
- **복원 / 구조 보존 축**과 **내적 / 추정 축**을 분리해서 비교
- `Turbo + QJL`, `Polar + QJL` 하이브리드 포함
- 어떤 방법을 어떤 관점으로 읽어야 하는지 정리

### 6. 지표 / 투영 해설
- MSE, MAE, cosine, IP bias, IP MAE, IP corr, score TV 등의 의미 설명
- Random projection / PCA / First 3 coordinates 차이 설명

---

## 화면을 읽는 법

### 색상
- **회색 점/회색 벡터**: 원본 기준
- **색이 있는 점**: 양자화 후의 색상 그룹
- 같은 색 점은 같은 양자화 그룹으로 묶여 읽으면 됩니다.

앱에서는 `001`, `010` 같은 비트 문자열보다 **색상 그룹 중심**으로 보이도록 정리했습니다.

### 3D 과정 보기
- 원본 → 중간 단계 → 최종 단계 흐름을 애니메이션으로 봅니다.
- 점을 클릭하면 **단면 예시 벡터 번호**가 같이 바뀝니다.

### 좌표쌍 구름도 / 단면 예시
- `(x[2i], x[2i+1])` 형태의 2D 좌표쌍을 확대해서 보여 줍니다.
- Turbo는 격자형 스냅,
- Polar는 각도형 스냅,
- baseline은 원래 좌표계에서의 직접 스냅,
- QJL은 복원보다 inner product 관련 그림을 중심으로 읽는 것이 좋습니다.

---

## 이 저장소의 성격

이 프로젝트는 아래 성격에 가깝습니다.

- **논문 발표용 visual explainer**
- **paper-aligned demo**
- **개념 비교용 인터랙티브 도구**

반대로 아래와는 조금 다릅니다.

- 논문 전체 실험을 완전히 재현하는 대규모 reproduction repo
- 실제 학습된 코드북과 모든 최적화 루프를 그대로 구현한 production repo

---

## 논문과의 거리에서 꼭 알아둘 점

### TurboQuant
앱은 **회전 후 좌표별 scalar quantization**이라는 핵심 직관을 보여 주는 데 집중합니다.

### PolarQuant
앱의 방사형 그림은 이해를 돕기 위한 시각화입니다. 실제 논문 구현은 단순 균일 각도 분할만을 그대로 쓰는 그림이라기보다,
**preconditioning 뒤 angle distribution 기반 optimized codebook**과
**level-dependent bit allocation** 관점으로 읽는 편이 더 정확합니다.

### QJL
QJL의 본체는 벡터 복원기가 아니라 **sign sketch 기반 inner-product estimator**입니다.
따라서 이 앱의 QJL 3D 그림은 설명 보조용이고,
핵심 평가는 **inner product / attention score proxy** 쪽입니다.

### Hybrid
- **Turbo + QJL**: 논문 친화적인 2단계 설명용 하이브리드
- **Polar + QJL**: 비교/탐색용 하이브리드

---

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

테스트 실행:

```bash
python -m unittest discover -s tests -v
```

---

## 이 앱이 적합한 사용 장면

- 논문 스터디에서 방법별 차이를 설명할 때
- 발표 자료용 그림을 직접 조작해 보며 설명 포인트를 잡을 때
- TurboQuant / PolarQuant / QJL을 한 프로젝트 안에서 비교해 보고 싶을 때
- “복원 품질”과 “inner-product estimation 품질”을 분리해서 설명하고 싶을 때

---

## 한 줄 요약

이 저장소는 **TurboQuant, PolarQuant, QJL이 각각 어떤 정보를 보존하려 하는지**를,
기존 양자화 baseline까지 포함해 **직접 보고 비교할 수 있게 만든 시각화 설명용 데모**입니다.
