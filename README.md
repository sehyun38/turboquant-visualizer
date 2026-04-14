# TurboQuant · PolarQuant · QJL Interactive Visualizer

TurboQuant, PolarQuant, QJL의 핵심 아이디어를 **직접 보고 비교할 수 있게 만든 Streamlit 시각화 데모**입니다.

이 프로젝트는 논문 전체 실험을 완전히 재현하는 저장소가 아니라,
- 각 방법이 무엇을 보존하려는지,
- 양자화 전후에 좌표·각도·반지름·내적이 어떻게 바뀌는지,
- 어떤 부분은 논문 원안에 가깝고 어떤 부분은 발표용 단순화인지
를 **직관적으로 설명하는 것**에 초점을 둡니다.

## 이 레포가 보여 주는 것

### TurboQuant
- 무작위 회전 뒤 좌표별 scalar quantization
- 좌표쌍 단면에서 보이는 **격자형 코드북 스냅**
- 회전 좌표 분포와 코드북 중심 비교

### PolarQuant
- random preconditioning 뒤 recursive polar transform
- 각도(angle) 중심 양자화
- 좌표쌍 단면에서 보이는 **방사형 코드북 구조**
- 각도와 반지름이 양자화 후 어떻게 달라지는지

### QJL
- 복원형 양자화기보다 **asymmetric inner-product estimator**로서의 QJL
- query는 JL transform, key는 sign-bit sketch + norm 저장
- true vs estimated inner product, bias, MAE, correlation 비교

### Hybrid / 비교
- Turbo + QJL: 논문 친화적 2단계 하이브리드
- Polar + QJL: 비교/교육용 탐색 하이브리드
- 복원 / 구조 보존 축과 내적 / 추정 품질 축을 분리해서 비교

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 탭 구성

### 1) TurboQuant
TurboQuant의 회전 → 좌표 스냅 → 복원 흐름을 보여 줍니다.

주로 보면 좋은 것:
- 3D 과정 보기
- 회전 좌표 분포와 Turbo 코드북 히스토그램
- Turbo 단면 예시와 좌표 변화 표
- Turbo 좌표쌍 구름도와 격자 코드북

### 2) PolarQuant
PolarQuant의 각도 스냅 직관을 보여 줍니다.

주로 보면 좋은 것:
- 3D 과정 보기
- Polar 단면 예시
- Polar 좌표 / 반지름 / 각도 변화 표
- 방사형 각도 코드북과 반지름 동심원
- 1단계 / 깊은 단계 각도 오차, 반지름 변화

### 3) QJL
QJL을 **내적 추정기** 관점으로 설명하는 탭입니다.

주로 보면 좋은 것:
- QJL 3D 과정(설명용)
- true vs estimated inner product 산점도
- QJL 오차 히스토그램
- QJL 핵심 구조 / 공식 / 발표용 해석 expander

### 4) 비교 / 하이브리드
비교 축을 둘로 나눠서 보여 줍니다.

- **복원 / 구조 비교**: TurboQuant, PolarQuant
- **내적 / 추정 비교**: QJL, Turbo + QJL, Polar + QJL
- **하이브리드 메모**: 논문 원안 여부와 발표용 설명 정리

### 5) 지표 / 투영 해설
앱 안에 나오는 점수와 3D 투영 방식을 설명합니다.

- MSE, MAE, Mean cosine
- IP bias, IP MAE, IP corr
- Random projection / PCA / First 3 coordinates 차이

## 주요 설정 설명

### 앱 모드
- **Balanced**: 시각 설명과 직관을 우선
- **Paper-faithful**: 논문 관점 설명을 조금 더 강조

### 벡터 수
시연에 사용할 샘플 개수입니다.

### 차원 d
벡터 차원입니다. 실제 내부 계산 차원이 바뀝니다.

### 데이터 분포
현재 앱에 들어 있는 옵션은 다음과 같습니다.
- `Gaussian`
- `Gaussian + outliers`
- `Unit sphere`
- `Sphere shell + outliers`
- `Ball + outliers`

이 분포들은 실제 데이터셋 자체라기보다,
각 방법이 어떤 기하 구조에서 어떻게 보이는지 설명하기 위한 **시연용 입력 분포**입니다.

### 입력 정밀도 시뮬레이션
입력값을 `fp16-like`, `fp8-like`, `int8-like`처럼 거칠게 만드는 옵션입니다.

### 양자화 비트 수
코드북이 얼마나 촘촘한지 보는 기준 비트 수입니다.

### 랜덤 전처리 적용
random rotation / preconditioning 효과를 켜거나 끄는 옵션입니다.

### 3D 공통 투영 방식
- **Random projection**: 논문 분위기에 가장 가까운 설명용 투영
- **PCA**: 발표에서 가장 보기 쉬운 투영
- **First 3 coordinates**: 축 의미를 직접 설명하기 쉬운 투영

### 단면 예시 벡터 번호
한 개 샘플 벡터를 골라 단면 예시에서 자세히 봅니다.
일부 탭에서는 3D 점이나 구름도 점을 눌러 이 번호를 바꿀 수 있습니다.

### 단면 좌표쌍 번호
`(x[2i], x[2i+1])` 형태의 2D 단면을 고릅니다.

## 도표 읽는 법

### 3D 과정 보기
원본 → 중간 단계 → 최종 단계로 이동하는 흐름을 보여 줍니다.

- 연한 점: 기준 상태 / 이전 단계
- 진한 점: 현재 단계 / 최종 상태
- 같은 색: 같은 양자화 bin 또는 비슷한 코드북 영역

### 단면 예시
하나의 벡터를 2D 좌표쌍 기준으로 확대해서 보여 줍니다.

- 파란 벡터: 원본
- 빨간 벡터: 양자화 후
- 보조선 / 원호 / 동심원: 이동량, 각도 변화, 반지름 변화

### 좌표쌍 구름도
전체 샘플을 특정 좌표쌍 평면에서 보여 줍니다.

- Turbo: 격자형 코드북이 잘 보이는지 확인
- Polar: 방사형 각도 코드북과 반지름 구조를 확인

### 내적 비교 산점도
점들이 `y = x`에 가까울수록 내적 추정이 잘 된다는 뜻입니다.

## 논문과의 관계

### TurboQuant
큰 흐름은 논문과 잘 맞습니다.
다만 앱의 격자 그림은 **학습된 클러스터 맵**이 아니라,
random rotation 뒤 **공통 scalar codebook**이 작동하는 모습을 직관적으로 보여 주는 그림입니다.

### PolarQuant
재귀 polar 변환과 angle quantization 흐름은 논문 취지를 따릅니다.
다만 현재 앱의 방사형 각도 bin 그림은 **직관용 단순화 데모**이고,
논문 구현은 preconditioning 뒤 각도 분포를 바탕으로 **optimized codebook**과
**level-dependent bit allocation**을 사용합니다.

### QJL
QJL의 핵심은 3D 복원이 아니라 **비대칭 inner-product estimation**입니다.
따라서 QJL 탭의 핵심 그래프는 복원 오차보다
`IP bias`, `IP MAE`, `IP corr`, `true vs estimated inner product`입니다.

### Hybrid
- **Turbo + QJL**: 논문 친화적 하이브리드
- **Polar + QJL**: 비교/교육용 탐색 하이브리드

## 발표 때 짧게 설명하는 법

- **TurboQuant**: “회전 후 좌표별 스칼라 코드북으로 스냅합니다.”
- **PolarQuant**: “polar 좌표로 바꾼 뒤 각도를 중심으로 양자화합니다.”
- **QJL**: “벡터를 복원하는 방법이라기보다, key를 1-bit sketch로 저장해서 inner product를 추정합니다.”
- **Turbo + QJL**: “Turbo base로 구조를 잡고 residual만 QJL로 보정합니다.”
- **Polar + QJL**: “논문 원안은 아니고 비교용 탐색 하이브리드입니다.”

## 주의

- 이 앱은 **연구/발표/학습용 시각화 도구**입니다.
- 일부 그림은 이해를 돕기 위한 단순화가 포함됩니다.
- 특히 Polar의 방사형 bin과 QJL의 3D 복원 그림은 **설명용 직관 시각화**로 보는 편이 정확합니다.
