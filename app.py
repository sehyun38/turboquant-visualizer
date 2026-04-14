import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(layout="wide", page_title="TurboQuant / PolarQuant / QJL Explorer")

EPS = 1e-12
BLUE = "#2563eb"
RED = "#dc2626"
BLUE_LIGHT = "#eff6ff"
RED_LIGHT = "#fef2f2"
GRAY_LIGHT = "#f8fafc"
DARK = "#0f172a"
ORIGINAL_POINT_COLOR = "#9ca3af"
ORIGINAL_REFERENCE_COLOR = RED
FINAL_REFERENCE_COLOR = BLUE
ORIGINAL_VECTOR_COLOR = "#475569"
QJL_POSITIVE_COLOR = "#14b8a6"
QJL_NEGATIVE_COLOR = "#f97316"


@dataclass
class MethodResult:
    name: str
    reconstructed: np.ndarray
    metrics: Dict[str, float]
    note: str
    details: Dict[str, np.ndarray]

# -----------------------------
# Theme / UI helpers
# -----------------------------

def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background: white;
            color: #111827;
        }}
        [data-testid="stSidebar"] {{
            background: white;
            border-right: 1px solid #e5e7eb;
        }}
        h1, h2, h3, h4, h5, h6, p, li, span, label, div {{
            color: #111827;
        }}
        
        /* 입력창 및 셀렉트박스 기본 스타일 */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        [data-testid="stNumberInputContainer"] > div,
        [data-testid="stTextInputRootElement"] > div {{
            background: white !important;
            border: 1px solid #d1d5db !important;
            color: #111827 !important;
            border-radius: 12px !important;
        }}

        /* [수정] 입력창 내부의 실제 글자색 강제 지정 (중괄호 2개 사용) */
        div[data-baseweb="input"] input {{
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }}

        /* [추가] 드롭다운(Selectbox) 클릭 시 나타나는 목록(popover) 스타일 */
        div[data-baseweb="popover"] ul {{
            background-color: white !important;
            border: 1px solid #d1d5db !important;
        }}
        div[data-baseweb="popover"] li {{
            background-color: white !important;
            color: #111827 !important;
        }}
        div[data-baseweb="popover"] li:hover {{
            background-color: #f3f4f6 !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
        .stTabs [data-baseweb="tab"] {{
            background: #f8fafc;
            border: 1px solid #dbeafe;
            border-radius: 12px 12px 0 0;
            padding: 10px 16px;
        }}
        .stTabs [aria-selected="true"] {{
            background: {BLUE_LIGHT};
            border-color: #93c5fd;
        }}
        .info-card, .look-card, .paper-card {{
            border-radius: 16px;
            padding: 14px 16px;
            margin: 8px 0 12px 0;
            border: 1px solid #dbeafe;
        }}
        .info-card {{ background: {BLUE_LIGHT}; color: #0f172a; }}
        .look-card {{ background: {RED_LIGHT}; border-color: #fecaca; color: #111827; }}
        .paper-card {{ background: {GRAY_LIGHT}; border-color: #e5e7eb; color: #111827; }}
        .info-card p, .info-card li, .info-card strong, .info-card span,
        .look-card p, .look-card li, .look-card strong, .look-card span,
        .paper-card p, .paper-card li, .paper-card strong, .paper-card span {{
            color: #111827 !important;
        }}
        .step-card {{
            background: white;
            border: 1px solid #dbeafe;
            border-radius: 16px;
            padding: 12px 14px;
            min-height: 118px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }}
        .step-num {{
            display: inline-block;
            background: {BLUE};
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 999px;
            text-align: center;
            line-height: 24px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .metric-chip {{
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: {BLUE_LIGHT};
            border: 1px solid #93c5fd;
            color: #1e3a8a;
            font-size: 12px;
            margin-right: 6px;
            margin-bottom: 6px;
        }}
        .small-note {{
            color: #475569;
            font-size: 0.92rem;
        }}
        .stButton > button {{
            background: white;
            color: #111827;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}
        .stButton > button:hover {{
            border-color: #94a3b8;
            background: #f8fafc;
            color: #0f172a;
        }}
        .stButton > button[kind="primary"] {{
            background: #eff6ff;
            border-color: #93c5fd;
            color: #1e3a8a;
            font-weight: 700;
        }}
        .stButton > button[kind="primary"]:hover {{
            background: #dbeafe;
            border-color: #60a5fa;
            color: #1d4ed8;
        }}
        .stButton > button:focus {{
            border-color: #60a5fa;
            box-shadow: 0 0 0 1px #93c5fd;
        }}
        code, pre, .stCodeBlock, [data-testid="stCodeBlock"] {{
            background: white !important;
            color: #111827 !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 14px !important;
        }}
        .metric-card {{
            background: white;
            border: 1px solid #e5e7eb;
            border-top: 4px solid {BLUE};
            border-radius: 16px;
            padding: 12px 14px;
            min-height: 92px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        }}
        .metric-card.red {{ border-top-color: {RED}; }}
        .metric-label {{ font-size: 0.88rem; color: #475569 !important; margin-bottom: 6px; }}
        .metric-value {{ font-size: 1.35rem; font-weight: 700; color: #111827 !important; }}
        details {{
            background: {GRAY_LIGHT};
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 8px 12px;
        }}
        .js-plotly-plot .plotly .slider-container text,
        .js-plotly-plot .plotly g.slider text,
        .js-plotly-plot .plotly g.updatemenu text,
        .js-plotly-plot .plotly .updatemenu-container text {{
            fill: #000000 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def one_line_box(text: str) -> None:
    st.markdown(f'<div class="info-card"><strong>한 줄 설명</strong><br>{text}</div>', unsafe_allow_html=True)


def look_box(items: List[str]) -> None:
    body = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f'<div class="look-card"><strong>Look for this</strong><ul style="margin-top:8px;">{body}</ul></div>',
        unsafe_allow_html=True,
    )


def metric_cards(metrics: Dict[str, float], accents: List[str] | None = None) -> None:
    items = list(metrics.items())
    cols = st.columns(len(items))
    accents = accents or ["blue"] * len(items)
    for col, (accent, (label, value)) in zip(cols, zip(accents, items)):
        cls = "metric-card" if accent != "red" else "metric-card red"
        col.markdown(
            f'<div class="{cls}"><div class="metric-label">{label}</div><div class="metric-value">{value:.4f}</div></div>',
            unsafe_allow_html=True,
        )


def pipeline_box(title: str, steps: List[Tuple[str, str]]) -> None:
    st.markdown(f"### {title}")
    cols = st.columns(len(steps))
    for idx, (col, (head, desc)) in enumerate(zip(cols, steps), start=1):
        col.markdown(
            f"""
            <div class="step-card">
              <div class="step-num">{idx}</div>
              <div><strong>{head}</strong></div>
              <div class="small-note" style="margin-top:6px;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep] + body)


def note_card(title: str, body: str) -> None:
    st.markdown(f'<div class="paper-card"><strong>{title}</strong><br>{body}</div>', unsafe_allow_html=True)


def fmt_num(x: float) -> str:
    return f"{float(x):.4f}"


def pair_radius(pair: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(pair, dtype=float)))


def pair_angle(pair: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(pair[1]), float(pair[0]))))


def pair_summary_rows(original_pair: np.ndarray, quant_pair: np.ndarray) -> List[List[str]]:
    o = np.asarray(original_pair, dtype=float)
    q = np.asarray(quant_pair, dtype=float)
    return [
        ["좌표 1", fmt_num(o[0]), fmt_num(q[0]), fmt_num(q[0] - o[0])],
        ["좌표 2", fmt_num(o[1]), fmt_num(q[1]), fmt_num(q[1] - o[1])],
        ["반지름 r", fmt_num(pair_radius(o)), fmt_num(pair_radius(q)), fmt_num(pair_radius(q) - pair_radius(o))],
        ["각도 θ (deg)", fmt_num(pair_angle(o)), fmt_num(pair_angle(q)), fmt_num(pair_angle(q) - pair_angle(o))],
    ]


def render_pair_summary(title: str, original_pair: np.ndarray, quant_pair: np.ndarray, emphasis: str) -> None:
    with st.expander(title, expanded=False):
        st.markdown(markdown_table(["항목", "원본", "양자화 후", "변화량"], pair_summary_rows(original_pair, quant_pair)))
        st.caption(emphasis)


def render_turbo_slice_explainer() -> None:
    with st.expander("Turbo 단면 그림 읽는 법", expanded=False):
        st.markdown(
            r"""
- **오른쪽 단면 예시**는 선택한 벡터의 한 좌표쌍 `(x[2i], x[2i+1])` 이 양자화 전후에 어디로 이동했는지 보여 줍니다.
- **TurboQuant 핵심**은 각 좌표를 **공통 scalar codebook**에 각각 스냅하는 것이므로, 단면에서는 점이 **격자(grid)** 위로 붙는 모양이 나타납니다.
- 아래 **좌표 변화 표**에서는 좌표 1·2, pair 반지름, pair 각도 변화량을 함께 읽으면 됩니다.

**간단 공식**
1. 입력 방향을 무작위 회전: `z = R(x / ||x||)`
2. 각 좌표를 코드북에 독립적으로 스냅: `\hat z_j = Q_b(z_j)`
3. 역회전 후 길이를 다시 곱해 복원: `\hat x = ||x|| R^\top \hat z`

**발표할 때는 이렇게 말하면 됩니다**
- Turbo는 **학습된 군집 맵**이 아니라, 회전 후 좌표들이 고르게 퍼질 때 **좌표별 코드북**이 잘 맞는 구조입니다.
- 그래서 단면 그림에서는 원본 점이 가장 가까운 **격자 교차점**으로 이동하는 모습으로 이해하면 됩니다.
            """
        )



def render_polar_slice_explainer() -> None:
    with st.expander("Polar 단면 그림 읽는 법", expanded=False):
        st.markdown(
            r"""
- **Polar 단면 예시**는 선택한 벡터의 한 좌표쌍을 `(반지름 r, 각도 θ)` 관점으로 읽도록 만든 그림입니다.
- **방사형 점선**은 첫 레벨 angle codebook의 직관용 그림이고, **동심원**은 선택 벡터의 원본/양자화 후 반지름을 뜻합니다.
- 논문 구현은 단순 균일 bin이 아니라 **preconditioning 뒤 angle distribution 기반 optimized codebook**을 사용한다는 점을 함께 기억하면 좋습니다.

**간단 공식**
1. 입력을 preconditioning: `z = R x`
2. polar 변환: `(r, \Theta) = \mathrm{Polar}(z)`
3. 각도를 코드북에 스냅: `\hat\Theta = Q(\Theta)`
4. `r` 와 `\hat\Theta` 로 Cartesian 복원

**발표할 때는 이렇게 말하면 됩니다**
- Turbo가 **격자형 스냅**이라면, Polar는 **각도형 스냅**입니다.
- 단면에서는 `θ → θ̂` 변화가 핵심이지만, 재귀 결합 때문에 실제 pair 반지름도 조금 달라질 수 있어 **r / r̂** 도 함께 보는 편이 정확합니다.
            """
        )



def render_qjl_core_explainer() -> None:
    with st.expander("QJL 핵심 설명 / 공식 / 발표 포인트", expanded=False):
        st.markdown(
            r"""
**핵심 한 줄**
- QJL의 본질은 **벡터 복원**이 아니라 **asymmetric inner-product estimation** 입니다.

**무엇을 저장하나**
- **query**: JL transform을 거친 실수 projection `Sq`
- **key**: sign-bit sketch `HS(k)=sign(Sk)` 와 key norm
- **value**: 논문 맥락에서는 별도 **standard token-wise quantization** 으로 처리

**직관 공식**
- `q \mapsto Sq`
- `k \mapsto HS(k)=sign(Sk)`
- `ProdQJL(q,k) \propto \|k\|_2 \langle Sq, HS(k) \rangle`
- 즉, 목표는 `\langle q,k \rangle` 자체를 직접 복원하는 것이 아니라 **비대칭 추정기**로 근사하는 것입니다.

**이 탭을 읽는 법**
- 왼쪽 3D는 **JL 투영 → sign sketch** 흐름을 보여 주는 **설명용 보조 시각화**입니다.
- 진짜 평가는 오른쪽의 **실제 vs 추정 내적**, **IP bias / IP MAE / IP corr**, 그리고 아래의 **attention score proxy**로 보는 편이 정확합니다.
- 그래서 QJL은 Turbo/Polar와 같은 복원형 quantizer 비교축보다, **내적 추정기 / attention score 보존 축**으로 분리해서 설명하는 것이 논문 취지에 더 가깝습니다.
            """
        )



def render_projection_reference(x_points: np.ndarray, x_hat_points: np.ndarray, color_ids: np.ndarray, levels: int, seed: int) -> None:
    with st.expander("3D 투영 방식 설명 / 간단 시연", expanded=False):
        overview_tab, demo_tab = st.tabs(["빠른 표", "간단 시연"])
        with overview_tab:
            st.markdown(markdown_table(["투영 방식", "무엇을 보나", "언제 쓰기 좋은가"], [
                ["Random projection", "임의 축으로 전체 구조를 고르게 펼쳐 보는 방식", "발표에서 형태를 직관적으로 보여 줄 때"],
                ["PCA", "분산이 큰 축을 우선으로 잡아 데이터의 주된 변화를 보여 줌", "구조 차이를 강조하고 싶을 때"],
                ["First 3 coordinates", "원본 좌표의 앞 3축만 그대로 사용", "투영 자체의 가공을 최소화하고 싶을 때"],
            ]))
            st.caption("중요: 이 투영 방식은 시각화용 도구이며, 양자화 알고리즘 자체를 바꾸는 설정은 아닙니다.")
        with demo_tab:
            ref = np.vstack([x_points, x_hat_points])
            cols = st.columns(3)
            for col, method, offset in zip(cols, ["Random projection", "PCA", "First 3 coordinates"], [901, 902, 903]):
                mean_p, basis_p = fit_projector(ref, method, int(seed) + offset, 3)
                proj_x = apply_projector(x_points, mean_p, basis_p)
                proj_hat = apply_projector(x_hat_points, mean_p, basis_p)
                with col:
                    st.plotly_chart(
                        scatter_overlay_3d(
                            proj_x,
                            proj_hat,
                            method,
                            color_ids=color_ids,
                            levels=levels,
                            recon_name="양자화 후",
                        ),
                        width="stretch",
                        theme=None,
                    )


def render_method_sequence_panel(title: str, steps: List[Tuple[str, str]], formulas: List[str], talk: str) -> None:
    with st.expander(title, expanded=False):
        pipeline_box("순서", steps)
        if formulas:
            st.markdown("**핵심 수식**")
            for formula in formulas:
                st.latex(formula)
        st.caption(talk)


def sign_chip_box_html(signs: np.ndarray) -> str:
    arr = np.asarray(signs, dtype=float)
    chips = []
    for idx, value in enumerate(arr.tolist()):
        positive = float(value) > 0
        accent = QJL_POSITIVE_COLOR if positive else QJL_NEGATIVE_COLOR
        bg = "#ecfeff" if positive else "#fff7ed"
        text = "+1" if positive else "-1"
        chips.append(
            f'<span style="display:inline-flex;align-items:center;gap:6px;padding:7px 10px;border-radius:10px;border:1px solid {accent};background:{bg};color:#111827;font-weight:700;font-size:0.88rem;">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:999px;background:{accent};"></span>'
            f'{idx}: {text}</span>'
        )
    return '<div style="display:flex;flex-wrap:wrap;gap:8px;padding:10px 12px;background:white;border:1px solid #e5e7eb;border-radius:14px;">' + ''.join(chips) + '</div>'

def discrete_palette(n: int) -> List[str]:
    base = [
        "#e15759",
        "#f28e2b",
        "#edc948",
        "#59a14f",
        "#00a676",
        "#17becf",
        "#4e79a7",
        "#9c6ade",
        "#d37295",
        "#ff5da2",
        "#8c564b",
        "#bcbd22",
        "#1d4ed8",
        "#7c3aed",
        "#0f766e",
        "#c2410c",
    ]
    if n <= len(base):
        return base[: max(1, n)]
    colors = []
    for i in range(n):
        base_color = base[i % len(base)]
        colors.append(base_color)
    return colors


def color_array_from_ids(ids: np.ndarray, levels: int) -> List[str]:
    palette = discrete_palette(levels)
    safe = np.asarray(ids, dtype=int) % max(1, levels)
    return [palette[int(i)] for i in safe]


def hash_ids(arr: np.ndarray, levels: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=int)
    if arr.ndim == 1:
        return arr % max(1, levels)
    use = arr[:, : min(4, arr.shape[1])]
    weights = np.array([1, 3, 5, 7][: use.shape[1]], dtype=int)
    return (use * weights).sum(axis=1) % max(1, levels)


# -----------------------------
# Data and math helpers
# -----------------------------


@st.cache_data(show_spinner=False)
def make_data(n: int, d: int, dist: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dist == "Unit sphere":
        x = rng.normal(size=(n, d))
        x /= np.linalg.norm(x, axis=1, keepdims=True) + EPS
        return x
    if dist == "Gaussian":
        return rng.normal(scale=1.0, size=(n, d))
    if dist == "Sphere shell + outliers":
        x = rng.normal(size=(n, d))
        x /= np.linalg.norm(x, axis=1, keepdims=True) + EPS
        radii = np.clip(rng.normal(loc=1.0, scale=0.045, size=(n, 1)), 0.84, 1.16)
        x = x * radii
        n_out = max(2, n // 24)
        idx = rng.choice(n, size=n_out, replace=False)
        x[idx] *= rng.uniform(1.7, 2.4, size=(n_out, 1))
        return x
    if dist == "Ball + outliers":
        direction = rng.normal(size=(n, d))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + EPS
        radius = rng.random((n, 1)) ** (1.0 / max(1, d))
        x = direction * radius
        n_out = max(2, n // 24)
        idx = rng.choice(n, size=n_out, replace=False)
        x[idx] *= rng.uniform(1.55, 2.2, size=(n_out, 1))
        return x
    x = rng.normal(scale=0.35, size=(n, d))
    n_out = max(1, n // 20)
    idx = rng.choice(n, size=n_out, replace=False)
    x[idx] += rng.normal(scale=1.2, size=(n_out, d))
    return x


@st.cache_data(show_spinner=False)
def make_query(d: int, dist: str, seed: int) -> np.ndarray:
    query_dist = "Unit sphere" if dist in {"Unit sphere", "Sphere shell + outliers"} else "Gaussian"
    q = make_data(1, d, query_dist, seed + 777)[0]
    return q


def apply_precision_rounding(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "fp16-like":
        return x.copy()
    if mode == "fp8-like":
        return np.round(x * 8.0) / 8.0
    return np.round(x * 2.0) / 2.0


@st.cache_data(show_spinner=False)
def random_orthogonal(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(d, d))
    q, _ = np.linalg.qr(h)
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def build_preconditioner(d: int, seed: int) -> Any:
    if d <= 256:
        return random_orthogonal(d, seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(d)
    signs = rng.choice(np.array([-1.0, 1.0]), size=d)
    return {"kind": "signed_permutation", "perm": perm, "signs": signs}


def apply_preconditioner_rows(x: np.ndarray, preconditioner: Any) -> np.ndarray:
    if isinstance(preconditioner, np.ndarray):
        return x @ preconditioner.T
    perm = np.asarray(preconditioner["perm"], dtype=int)
    signs = np.asarray(preconditioner["signs"], dtype=float)
    return x[:, perm] * signs[None, :]


def invert_preconditioner_rows(x: np.ndarray, preconditioner: Any) -> np.ndarray:
    if isinstance(preconditioner, np.ndarray):
        return x @ preconditioner
    perm = np.asarray(preconditioner["perm"], dtype=int)
    signs = np.asarray(preconditioner["signs"], dtype=float)
    restored = np.empty_like(x)
    restored[:, perm] = x * signs[None, :]
    return restored


def effective_sketch_dim(requested_m: int, d: int) -> int:
    if d <= 512:
        return min(requested_m, d)
    if d <= 1024:
        return min(requested_m, 256)
    if d <= 2048:
        return min(requested_m, 128)
    return min(requested_m, 64)


@st.cache_data(show_spinner=False)
def gaussian_sketch(m: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(m, d))


def fit_projector(reference: np.ndarray, method: str, seed: int, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    mean = reference.mean(axis=0, keepdims=True)
    centered = reference - mean
    d = reference.shape[1]
    if method == "PCA":
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:n_components].T
    elif method == "First 3 coordinates":
        basis = np.eye(d, n_components)
    else:
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(d, n_components))
        basis, _ = np.linalg.qr(raw)
        basis = basis[:, :n_components]
    return mean, basis


def apply_projector(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (x - mean) @ basis


def sample_indices(n: int, limit: int, seed: int) -> np.ndarray:
    if limit >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=limit, replace=False)
    idx.sort()
    return idx


def interpolate_points(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    valid = (an > EPS) & (bn > EPS)
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.sum(a[valid] * b[valid], axis=1) / (an[valid] * bn[valid])))


@st.cache_data(show_spinner=False)
def compute_metrics(x: np.ndarray, x_hat: np.ndarray, q: np.ndarray) -> Dict[str, float]:
    diff = x - x_hat
    mse = float(np.mean(np.sum(diff * diff, axis=1)))
    mae = float(np.mean(np.abs(diff)))
    cos = cosine_mean(x, x_hat)
    true_ip = x @ q
    est_ip = x_hat @ q
    ip_bias = float(np.mean(est_ip - true_ip))
    ip_mae = float(np.mean(np.abs(est_ip - true_ip)))
    corr = float(np.corrcoef(true_ip, est_ip)[0, 1]) if np.std(true_ip) > EPS and np.std(est_ip) > EPS else 1.0
    return {
        "MSE": mse,
        "MAE": mae,
        "Mean cosine": cos,
        "IP bias": ip_bias,
        "IP MAE": ip_mae,
        "IP corr": corr,
    }


def softmax_stable(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temp = max(float(temperature), EPS)
    v = np.asarray(values, dtype=float) / temp
    v = v - float(np.max(v))
    ex = np.exp(v)
    return ex / (float(np.sum(ex)) + EPS)


@st.cache_data(show_spinner=False)
def attention_proxy_metrics(true_ip: np.ndarray, est_ip: np.ndarray, temperature: float, top_k: int = 10) -> Dict[str, float]:
    true_scores = softmax_stable(true_ip, temperature)
    est_scores = softmax_stable(est_ip, temperature)
    score_mae = float(np.mean(np.abs(true_scores - est_scores)))
    score_tv = float(0.5 * np.sum(np.abs(true_scores - est_scores)))
    err = est_ip - true_ip
    ip_err_var = float(np.var(err))
    k = max(1, min(int(top_k), len(true_scores)))
    top_true = set(np.argsort(true_scores)[-k:])
    top_est = set(np.argsort(est_scores)[-k:])
    topk_overlap = float(len(top_true & top_est) / k)
    return {
        "score_mae": score_mae,
        "score_tv": score_tv,
        "ip_err_var": ip_err_var,
        "topk_overlap": topk_overlap,
        "true_scores": true_scores,
        "est_scores": est_scores,
    }


# -----------------------------
# TurboQuant helpers
# -----------------------------


@st.cache_data(show_spinner=False)
def sample_unit_coordinate_distribution(d: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_samples, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True) + EPS
    return z[:, 0]



def lloyd_max_1d(samples: np.ndarray, k: int, n_iter: int = 25) -> np.ndarray:
    if k <= 1:
        return np.array([float(np.mean(samples))])
    centers = np.quantile(samples, np.linspace(0.0, 1.0, k + 2)[1:-1]).astype(float)
    for _ in range(n_iter):
        dist = np.abs(samples[:, None] - centers[None, :])
        idx = np.argmin(dist, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            mask = idx == i
            if np.any(mask):
                new_centers[i] = float(np.mean(samples[mask]))
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return np.sort(centers)


@st.cache_data(show_spinner=False)
def turbo_codebook(d: int, bits: int, seed: int) -> np.ndarray:
    if bits <= 0:
        return np.array([0.0])
    samples = sample_unit_coordinate_distribution(d, n_samples=50000, seed=seed + 13 * d + bits)
    return lloyd_max_1d(samples, 2 ** bits, n_iter=30)



def quantize_by_codebook(values: np.ndarray, codebook: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    distances = np.abs(values[..., None] - codebook[None, ...])
    idx = np.argmin(distances, axis=-1)
    return codebook[idx], idx



@st.cache_data(show_spinner=False)
def turbo_quantize_mse(x: np.ndarray, bits: int, precondition: bool, seed: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    d = x.shape[1]
    levels = max(1, 2 ** bits)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + EPS
    x_unit = x / norms
    rot = build_preconditioner(d, seed + 101) if precondition else np.eye(d)
    x_rot = apply_preconditioner_rows(x_unit, rot)
    codebook = turbo_codebook(d, bits, seed + 202)
    xq_rot, idx = quantize_by_codebook(x_rot, codebook)
    xq_unit = invert_preconditioner_rows(xq_rot, rot)
    xq = xq_unit * norms
    details = {
        "rot": x_rot,
        "rot_scaled": x_rot * norms,
        "q_rot": xq_rot,
        "q_rot_scaled": xq_rot * norms,
        "q_unit": xq_unit,
        "reconstructed": xq,
        "norms": norms[:, 0],
        "codebook": codebook,
        "indices": idx,
        "color_ids": hash_ids(idx, levels),
        "rotation": rot,
    }
    return xq, details



@st.cache_data(show_spinner=False)
def turbo_quantize_prod(x: np.ndarray, bits: int, precondition: bool, seed: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    levels = max(1, 2 ** bits)
    base_bits = max(1, bits - 1)
    x_base, details = turbo_quantize_mse(x, base_bits, precondition, seed)
    residual = x - x_base
    d = x.shape[1]
    sketch_dim = effective_sketch_dim(d, d)
    s = gaussian_sketch(sketch_dim, d, seed + 303)
    qjl = np.sign(residual @ s.T)
    qjl[qjl == 0] = 1
    gamma = np.linalg.norm(residual, axis=1, keepdims=True)
    residual_hat = math.sqrt(math.pi / 2.0) / sketch_dim * gamma * (qjl @ s)
    x_hat = x_base + residual_hat
    details.update(
        {
            "base": x_base,
            "residual": residual,
            "qjl_sign": qjl,
            "gamma": gamma[:, 0],
            "sketch": s,
            "residual_hat": residual_hat,
            "residual_mid": x_base + 0.5 * residual_hat,
            "reconstructed": x_hat,
            "base_bits": np.array([base_bits]),
            "color_ids": (details.get("color_ids", np.zeros(len(x), dtype=int)) + hash_ids((qjl > 0).astype(int), levels)) % levels,
        }
    )
    return x_hat, details


# -----------------------------
# Baseline quantization helpers
# -----------------------------


@st.cache_data(show_spinner=False)
def baseline_uniform_quantize(x: np.ndarray, bits: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    levels = max(1, 2 ** bits)
    qmin = float(np.quantile(x, 0.01))
    qmax = float(np.quantile(x, 0.99))
    if qmax - qmin < EPS:
        qmin, qmax = float(np.min(x)) - 1.0, float(np.max(x)) + 1.0
    clipped = np.clip(x, qmin, qmax)
    codebook = np.linspace(qmin, qmax, levels)
    x_hat, idx = quantize_by_codebook(clipped, codebook)
    details = {
        "clipped": clipped,
        "reconstructed": x_hat,
        "codebook": codebook,
        "indices": idx,
        "qmin": np.array([qmin]),
        "qmax": np.array([qmax]),
        "color_ids": hash_ids(idx, levels),
    }
    return x_hat, details



def render_baseline_slice_explainer() -> None:
    with st.expander("기존 양자화 단면 그림 읽는 법", expanded=False):
        st.markdown(
            r"""
- **기존 양자화 baseline**은 회전이나 polar 변환 없이, 원래 Cartesian 좌표에서 바로 **uniform scalar codebook**에 스냅하는 그림입니다.
- 단면 예시에서는 선택한 좌표쌍 `(x[2i], x[2i+1])` 이 **공통 직교 격자** 위로 붙는 모습을 보면 됩니다.
- Turbo와 달리 이 방법은 좌표계를 바꾸지 않으므로, 입력 분포의 방향성 편향이 그대로 남아 있을 수 있습니다.

**간단 공식**
1. 전역 범위를 잡음: `x_{clip} = \mathrm{clip}(x, q_{min}, q_{max})`
2. 균일 코드북 생성: `\mathcal{C} = \mathrm{linspace}(q_{min}, q_{max}, 2^b)`
3. 각 좌표를 가장 가까운 코드북 값으로 스냅: `\hat x_j = Q_b(x_{clip,j})`

**발표할 때는 이렇게 말하면 됩니다**
- 이 탭은 Turbo/Polar/QJL과 비교하기 위한 **기준선 baseline**입니다.
- 즉, 별도 preconditioning 없이 원래 좌표계에서 바로 양자화하면 어떤 격자형 왜곡이 생기는지 먼저 보는 탭입니다.
            """
        )



def baseline_ideal_explainer(bits: int) -> None:
    with st.expander("기존 양자화 설명 / 공식", expanded=False):
        st.markdown(
            f"""
**이 그림이 뜻하는 것**
- 기존 양자화는 입력을 그대로 Cartesian 좌표계에서 다루고, 각 좌표를 **균일 간격 코드북**에 붙입니다.
- 현재 설정에서는 코드북 레벨 수가 **{2 ** max(1, bits)}개**이고, 좌표쌍 평면에서는 `({2 ** max(1, bits)}) × ({2 ** max(1, bits)})` 격자로 보입니다.
- Turbo/Polar처럼 좌표계를 바꾸지 않으므로, 입력 분포가 한쪽으로 치우치면 그 편향이 단면에 그대로 남습니다.
            """
        )
        st.latex(r"x_{clip} = \mathrm{clip}(x, q_{min}, q_{max})")
        st.latex(r"\mathcal{C} = \mathrm{linspace}(q_{min}, q_{max}, 2^b)")
        st.latex(r"\hat x_j = rg\min_{c \in \mathcal{C}} |x_{clip,j} - c|")
        st.markdown(
            """
**발표용 해석 포인트**
- 이 탭은 “아무 구조 변환 없이 그냥 균일 quantization을 하면 어떤 그림이 나오는가?”를 보여 주는 비교 기준입니다.
- 이후 Turbo 탭에서는 **회전 뒤 좌표별 코드북**, Polar 탭에서는 **각도형 코드북**, QJL 탭에서는 **sign sketch 기반 내적 추정**으로 넘어가면 차이가 더 잘 보입니다.
            """
        )


# -----------------------------
# PolarQuant helpers
# -----------------------------



def polar_forward_single(x: np.ndarray) -> Tuple[float, List[np.ndarray]]:
    cur = x.astype(float).copy()
    levels: List[np.ndarray] = []
    first_angles = []
    pair_radii = []
    for j in range(0, len(cur), 2):
        a = cur[j]
        b = cur[j + 1] if j + 1 < len(cur) else 0.0
        first_angles.append(np.mod(np.arctan2(b, a), 2 * np.pi))
        pair_radii.append(math.hypot(a, b))
    levels.append(np.array(first_angles, dtype=float))
    radii = np.array(pair_radii, dtype=float)
    while len(radii) > 1:
        angles = []
        next_radii = []
        for j in range(0, len(radii), 2):
            left = radii[j]
            right = radii[j + 1] if j + 1 < len(radii) else 0.0
            angles.append(math.atan2(right, left))
            next_radii.append(math.hypot(left, right))
        levels.append(np.array(angles, dtype=float))
        radii = np.array(next_radii, dtype=float)
    return float(radii[0]), levels



def polar_inverse_single(radius: float, levels: List[np.ndarray], output_dim: Optional[int] = None) -> np.ndarray:
    cur = np.array([radius], dtype=float)
    for lvl in range(len(levels) - 1, 0, -1):
        angles = levels[lvl]
        expanded = np.empty(angles.size * 2, dtype=float)
        for i, ang in enumerate(angles):
            parent = cur[i]
            expanded[2 * i] = parent * math.cos(float(ang))
            expanded[2 * i + 1] = parent * math.sin(float(ang))
        cur = expanded
    first = levels[0]
    out = np.empty(first.size * 2, dtype=float)
    for i, ang in enumerate(first):
        r = cur[i]
        out[2 * i] = r * math.cos(float(ang))
        out[2 * i + 1] = r * math.sin(float(ang))
    if output_dim is not None:
        return out[:output_dim]
    return out



def uniform_angle_codebook(lo: float, hi: float, bits: int) -> np.ndarray:
    bins = max(2, 2 ** bits)
    edges = np.linspace(lo, hi, bins + 1)
    return (edges[:-1] + edges[1:]) / 2.0



def quantize_angles(values: np.ndarray, codebook: np.ndarray, circular: bool = False, period: float = 2 * np.pi) -> np.ndarray:
    diff = np.abs(values[:, None] - codebook[None, :])
    if circular:
        diff = np.minimum(diff, period - diff)
    idx = np.argmin(diff, axis=1)
    return codebook[idx]



def polar_level_bits(bits: int) -> Tuple[int, int]:
    if bits <= 1:
        return 2, 1
    if bits == 2:
        return 3, 1
    return 4, 2



@st.cache_data(show_spinner=False)
def polar_quantize(x: np.ndarray, bits: int, precondition: bool, seed: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    d = x.shape[1]
    levels = max(1, 2 ** bits)
    rot = build_preconditioner(d, seed + 404) if precondition else np.eye(d)
    x_rot = apply_preconditioner_rows(x, rot)
    first_bits, upper_bits = polar_level_bits(bits)
    first_codebook = uniform_angle_codebook(0.0, 2 * np.pi, first_bits)
    upper_codebook = uniform_angle_codebook(0.0, np.pi / 2.0, upper_bits)

    reconstructed_rot = []
    lvl_before: Dict[int, List[float]] = {}
    lvl_after: Dict[int, List[float]] = {}
    color_ids: List[int] = []
    for row in x_rot:
        radius, levels_data = polar_forward_single(row)
        q_levels: List[np.ndarray] = []
        for lvl_idx, angles in enumerate(levels_data):
            lvl_before.setdefault(lvl_idx, []).extend(angles.tolist())
            if lvl_idx == 0:
                q = quantize_angles(angles, first_codebook, circular=True)
            else:
                q = quantize_angles(angles, upper_codebook, circular=False)
            lvl_after.setdefault(lvl_idx, []).extend(q.tolist())
            q_levels.append(q)
        first_angle = float(q_levels[0][0]) if len(q_levels[0]) else 0.0
        color_ids.append(int(np.floor((first_angle / (2 * np.pi)) * levels)) % levels)
        reconstructed_rot.append(polar_inverse_single(radius, q_levels, output_dim=d))
    x_hat_rot = np.vstack(reconstructed_rot)
    x_hat = invert_preconditioner_rows(x_hat_rot, rot)
    details = {
        "rot": x_rot,
        "recon_rot": x_hat_rot,
        "reconstructed": x_hat,
        "first_codebook": first_codebook,
        "upper_codebook": upper_codebook,
        "lvl0_before": np.array(lvl_before.get(0, [])),
        "lvl0_after": np.array(lvl_after.get(0, [])),
        "lvllast_before": np.array(lvl_before.get(max(lvl_before.keys(), default=0), [])),
        "lvllast_after": np.array(lvl_after.get(max(lvl_after.keys(), default=0), [])),
        "first_bits": np.array([first_bits]),
        "upper_bits": np.array([upper_bits]),
        "color_ids": np.array(color_ids, dtype=int),
        "rotation": rot,
    }
    return x_hat, details



@st.cache_data(show_spinner=False)
def polar_quantize_prod(x: np.ndarray, bits: int, precondition: bool, seed: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    levels = max(1, 2 ** bits)
    base_bits = max(1, bits - 1)
    x_base, details = polar_quantize(x, base_bits, precondition, seed)
    residual = x - x_base
    d = x.shape[1]
    sketch_dim = effective_sketch_dim(d, d)
    s = gaussian_sketch(sketch_dim, d, seed + 909)
    qjl = np.sign(residual @ s.T)
    qjl[qjl == 0] = 1
    gamma = np.linalg.norm(residual, axis=1, keepdims=True)
    residual_hat = math.sqrt(math.pi / 2.0) / sketch_dim * gamma * (qjl @ s)
    x_hat = x_base + residual_hat
    details.update(
        {
            "base": x_base,
            "residual": residual,
            "qjl_sign": qjl,
            "gamma": gamma[:, 0],
            "sketch": s,
            "residual_hat": residual_hat,
            "residual_mid": x_base + 0.5 * residual_hat,
            "reconstructed": x_hat,
            "base_bits": np.array([base_bits]),
            "color_ids": (details.get("color_ids", np.zeros(len(x), dtype=int)) + hash_ids((qjl > 0).astype(int), levels)) % levels,
        }
    )
    return x_hat, details


# -----------------------------
# QJL helpers
# -----------------------------



@st.cache_data(show_spinner=False)
def qjl_quantize(x: np.ndarray, q: np.ndarray, m: int, seed: int, bits: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    d = x.shape[1]
    levels = max(1, 2 ** bits)
    sketch_dim = effective_sketch_dim(m, d)
    s = gaussian_sketch(sketch_dim, d, seed + 505)
    signed = np.sign(x @ s.T)
    signed[signed == 0] = 1
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    sq = s @ q
    ip_est = math.sqrt(math.pi / 2.0) / sketch_dim * norms[:, 0] * (signed @ sq)
    x_hat = math.sqrt(math.pi / 2.0) / sketch_dim * norms * (signed @ s)
    sign_bits = (signed[:, : min(6, signed.shape[1])] > 0).astype(int)
    weights = 2 ** np.arange(sign_bits.shape[1])
    color_ids = (sign_bits * weights).sum(axis=1) % levels
    details = {
        "signs": signed,
        "norms": norms[:, 0],
        "sketch": s,
        "sq": sq,
        "ip_est": ip_est,
        "ip_true": x @ q,
        "color_ids": color_ids.astype(int),
        "m": np.array([sketch_dim]),
        "requested_m": np.array([m]),
    }
    return x_hat, details


# -----------------------------
# Plot helpers
# -----------------------------

def scatter_overlay_3d(
    original_3d: np.ndarray,
    recon_3d: np.ndarray,
    title: str,
    color_ids: np.ndarray | None = None,
    levels: int = 8,
    original_name: str = "원본",
    recon_name: str = "양자화 후",
) -> go.Figure:
    colors = color_array_from_ids(color_ids if color_ids is not None else np.zeros(len(recon_3d), dtype=int), levels)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=original_3d[:, 0], y=original_3d[:, 1], z=original_3d[:, 2],
            mode="markers", name=original_name,
            marker=dict(size=3.6, opacity=0.34, color=ORIGINAL_POINT_COLOR)
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=recon_3d[:, 0], y=recon_3d[:, 1], z=recon_3d[:, 2],
            mode="markers", name=recon_name,
            marker=dict(size=4.4, opacity=0.92, color=colors, line=dict(width=0.4, color="white"))
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color="black")),
        template="plotly_white",
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            aspectmode="cube",
            bgcolor="white",
            xaxis=dict(title="투영축 1", backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
            yaxis=dict(title="투영축 2", backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
            zaxis=dict(title="투영축 3", backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
        ),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(color="black")),
        paper_bgcolor="white",
    )
    return fig


def process_figure_3d(
    stages: List[Dict[str, Any]],
    color_ids: np.ndarray,
    title: str,
    levels: int,
    point_indices: Optional[np.ndarray] = None,
    selected_index: Optional[int] = None,
) -> go.Figure:
    colors_final = color_array_from_ids(color_ids, levels)
    point_indices = np.arange(len(stages[0]["points"]), dtype=int) if point_indices is None else np.asarray(point_indices, dtype=int)

    normalized_stages: List[Dict[str, Any]] = []
    for idx, stage in enumerate(stages):
        pts = np.asarray(stage["points"], dtype=float)
        stage_colors = stage.get("colors")
        if stage_colors is None:
            stage_colors = colors_final if idx == len(stages) - 1 else [ORIGINAL_POINT_COLOR] * len(pts)
        normalized_stages.append({"label": str(stage["label"]), "points": pts, "colors": list(stage_colors)})

    all_pts = np.vstack([stage["points"] for stage in normalized_stages])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    pads = np.maximum((maxs - mins) * 0.15, 0.25)
    xr = [float(mins[0] - pads[0]), float(maxs[0] + pads[0])]
    yr = [float(mins[1] - pads[1]), float(maxs[1] + pads[1])]
    zr = [float(mins[2] - pads[2]), float(maxs[2] + pads[2])]

    selected_mask = np.asarray(point_indices == int(selected_index), dtype=bool) if selected_index is not None else np.zeros(len(point_indices), dtype=bool)

    timeline: List[Tuple[str, np.ndarray, List[str]]] = []
    first_stage = normalized_stages[0]
    timeline.append((first_stage["label"], first_stage["points"], first_stage["colors"]))
    for prev_stage, next_stage in zip(normalized_stages[:-1], normalized_stages[1:]):
        for t in (0.25, 0.5, 0.75):
            cur = interpolate_points(prev_stage["points"], next_stage["points"], float(t))
            interp_colors = next_stage["colors"] if t >= 0.5 else prev_stage["colors"]
            timeline.append((f"{prev_stage['label']} → {next_stage['label']} {int(round(t * 100))}%", cur, interp_colors))
        timeline.append((next_stage["label"], next_stage["points"], next_stage["colors"]))

    final_points = normalized_stages[-1]["points"]

    def frame_payload(cur_label: str, pts: np.ndarray, colors: List[str]) -> List[go.Scatter3d]:
        frame_data = [
            go.Scatter3d(
                x=final_points[:, 0], y=final_points[:, 1], z=final_points[:, 2],
                mode="markers",
                name="최종 상태",
                customdata=point_indices[:, None],
                marker=dict(size=3.8, opacity=0.85, color=FINAL_REFERENCE_COLOR),
                visible="legendonly",
                hovertemplate="벡터 #%{customdata[0]}<extra></extra>",
            ),
            go.Scatter3d(
                x=normalized_stages[0]["points"][:, 0], y=normalized_stages[0]["points"][:, 1], z=normalized_stages[0]["points"][:, 2],
                mode="markers",
                name="원본 기준",
                customdata=point_indices[:, None],
                marker=dict(size=3.8, opacity=0.85, color=ORIGINAL_REFERENCE_COLOR),
                visible="legendonly",
                hovertemplate="벡터 #%{customdata[0]}<extra></extra>",
            ),
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                name="현재 단계",
                customdata=point_indices[:, None],
                marker=dict(size=4.5, opacity=0.94, color=colors, line=dict(width=0.45, color="white")),
                hovertemplate="벡터 #%{customdata[0]}<extra></extra>",
            ),
        ]
        if np.any(selected_mask):
            sel_pts = pts[selected_mask]
            frame_data.append(
                go.Scatter3d(
                    x=sel_pts[:, 0], y=sel_pts[:, 1], z=sel_pts[:, 2],
                    mode="markers+lines",
                    name=f"선택 벡터 #{int(selected_index)}",
                    customdata=point_indices[selected_mask, None],
                    marker=dict(size=8.5, opacity=1.0, color="#111827", line=dict(width=2.0, color="#f59e0b")),
                    line=dict(color="#f59e0b", width=6),
                    hovertemplate="벡터 #%{customdata[0]}<extra></extra>",
                )
            )
        return frame_data

    frames = [go.Frame(name=label, data=frame_payload(label, pts, colors), traces=list(range(len(frame_payload(label, pts, colors))))) for label, pts, colors in timeline]

    initial_label, initial_points, initial_colors = timeline[-1]
    fig = go.Figure(data=frame_payload(initial_label, initial_points, initial_colors), frames=frames)
    fig.update_layout(
        title=dict(text=title, font=dict(color="black")),
        template="plotly_white",
        height=560,
        margin=dict(l=0, r=0, t=40, b=0),
        clickmode="event+select",
        scene=dict(
            aspectmode="cube",
            bgcolor="white",
            xaxis=dict(title="투영축 1", range=xr, backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
            yaxis=dict(title="투영축 2", range=yr, backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
            zaxis=dict(title="투영축 3", range=zr, backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
        ),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0,
            "y": 1.08,
            "font": {"color": "black"},
            "bgcolor": "white",
            "bordercolor": "#cbd5e1",
            "borderwidth": 1,
            "buttons": [
                {"label": "Play", "method": "animate", "args": [[label for label, _, _ in timeline], {"frame": {"duration": 180, "redraw": True}, "fromcurrent": False, "mode": "immediate", "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": len(timeline) - 1,
            "currentvalue": {"prefix": "3D 단계: ", "font": {"color": "black"}},
            "font": {"color": "black"},
            "bgcolor": "white",
            "activebgcolor": "#e2e8f0",
            "bordercolor": "#cbd5e1",
            "tickcolor": "black",
            "pad": {"t": 36},
            "steps": [
                {"label": label, "method": "animate", "args": [[label], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                for label, _, _ in timeline
            ],
        }],
        legend=dict(bgcolor="rgba(255,255,255,0.82)", font=dict(color="black")),
        paper_bgcolor="white",
        selectionrevision=str(selected_index) if selected_index is not None else "none",
    )
    return fig

def histogram_with_codebook(values: np.ndarray, codebook: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=40, name="분포", opacity=0.8, marker=dict(color=BLUE, line=dict(width=0.4, color="white"))))
    for c in codebook:
        fig.add_vline(x=float(c), line_width=2, line_dash="dash", line_color=RED)
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=360, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"))
    return fig


def scatter_true_vs_est(true_ip: np.ndarray, est_ip: np.ndarray, title: str, name: str = "점", point_colors: List[str] | None = None) -> go.Figure:
    lo = float(min(true_ip.min(), est_ip.min()))
    hi = float(max(true_ip.max(), est_ip.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=true_ip, y=est_ip, mode="markers", marker=dict(size=6, opacity=0.78, color=point_colors or RED, line=dict(width=0.3, color="white")), name=name))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="이상적인 y=x", line=dict(color=BLUE, dash="dash", width=2)))
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="실제 내적", yaxis_title="추정 / 복원 내적", plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"))
    return fig


def error_hist(errors: np.ndarray, title: str, color: str = RED) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=errors, nbinsx=40, opacity=0.82, name="오차", marker=dict(color=color, line=dict(width=0.4, color="white"))))
    fig.add_vline(x=0.0, line_width=2, line_color=BLUE)
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="추정값 - 실제값", plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"))
    return fig


def slice_geometry_figure(original_pair: np.ndarray, quant_pair: np.ndarray, title: str) -> go.Figure:
    max_abs = float(max(np.max(np.abs(original_pair)), np.max(np.abs(quant_pair)), 1.0))
    r_arc = 0.32 * max_abs
    theta_o = math.atan2(float(original_pair[1]), float(original_pair[0]))
    theta_q = math.atan2(float(quant_pair[1]), float(quant_pair[0]))
    arc_o_x = np.cos(np.linspace(0, theta_o, 80)) * r_arc
    arc_o_y = np.sin(np.linspace(0, theta_o, 80)) * r_arc
    arc_q_x = np.cos(np.linspace(0, theta_q, 80)) * (r_arc * 0.78)
    arc_q_y = np.sin(np.linspace(0, theta_q, 80)) * (r_arc * 0.78)

    fig = go.Figure()
    fig.add_hline(y=0.0, line_color="#cbd5e1", line_width=1)
    fig.add_vline(x=0.0, line_color="#cbd5e1", line_width=1)
    fig.add_trace(go.Scatter(x=[0, original_pair[0]], y=[0, original_pair[1]], mode="lines+markers", name="원본", line=dict(width=3, color=ORIGINAL_VECTOR_COLOR), marker=dict(size=8, color=ORIGINAL_VECTOR_COLOR)))
    fig.add_trace(go.Scatter(x=[0, quant_pair[0]], y=[0, quant_pair[1]], mode="lines+markers", name="양자화 후", line=dict(width=3, dash="dash", color=RED), marker=dict(size=8, color=RED)))
    fig.add_trace(go.Scatter(x=arc_o_x, y=arc_o_y, mode="lines", line=dict(color=ORIGINAL_VECTOR_COLOR, width=2), name="원본 각도 θ"))
    fig.add_trace(go.Scatter(x=arc_q_x, y=arc_q_y, mode="lines", line=dict(color=RED, width=2, dash="dot"), name="양자화 각도 θ̂"))
    fig.add_annotation(x=float(arc_o_x[-1]), y=float(arc_o_y[-1]), text="θ", showarrow=False, font=dict(color=ORIGINAL_VECTOR_COLOR, size=14))
    fig.add_annotation(x=float(arc_q_x[-1]), y=float(arc_q_y[-1]), text="θ̂", showarrow=False, font=dict(color=RED, size=14))
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=420, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="좌표축 1", yaxis_title="좌표축 2", xaxis=dict(range=[-1.2 * max_abs, 1.2 * max_abs]), yaxis=dict(range=[-1.2 * max_abs, 1.2 * max_abs], scaleanchor="x", scaleratio=1), plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"))
    return fig


def pair_vector_compare_figure(original_pair: np.ndarray, quant_pair: np.ndarray, title: str) -> go.Figure:
    max_abs = float(max(np.max(np.abs(original_pair)), np.max(np.abs(quant_pair)), 1.0))
    fig = go.Figure()
    fig.add_hline(y=0.0, line_color="#cbd5e1", line_width=1)
    fig.add_vline(x=0.0, line_color="#cbd5e1", line_width=1)
    fig.add_trace(go.Scatter(x=[0, original_pair[0]], y=[0, original_pair[1]], mode="lines+markers", name="원본 벡터", line=dict(width=3, color=ORIGINAL_VECTOR_COLOR), marker=dict(size=8, color=ORIGINAL_VECTOR_COLOR)))
    fig.add_trace(go.Scatter(x=[0, quant_pair[0]], y=[0, quant_pair[1]], mode="lines+markers", name="양자화 벡터", line=dict(width=3, dash="dash", color=RED), marker=dict(size=8, color=RED)))
    fig.add_trace(go.Scatter(x=[original_pair[0], quant_pair[0]], y=[original_pair[1], quant_pair[1]], mode="lines", name="이동량", line=dict(width=2, dash="dot", color="#64748b")))
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=420, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="좌표축 1", yaxis_title="좌표축 2", xaxis=dict(range=[-1.2 * max_abs, 1.2 * max_abs]), yaxis=dict(range=[-1.2 * max_abs, 1.2 * max_abs], scaleanchor="x", scaleratio=1), plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"))
    return fig


def pair_cloud_figure(
    original_pairs: np.ndarray,
    quant_pairs: np.ndarray,
    title: str,
    color_ids: np.ndarray | None = None,
    levels: int = 8,
    point_indices: Optional[np.ndarray] = None,
    selected_index: Optional[int] = None,
    x_grid: Optional[np.ndarray] = None,
    y_grid: Optional[np.ndarray] = None,
    radial_angles: Optional[np.ndarray] = None,
    show_unit_circle: bool = False,
    quant_name: str = "양자화 좌표쌍",
    original_name: str = "원본 좌표쌍",
    selected_original_pair: Optional[np.ndarray] = None,
    selected_quant_pair: Optional[np.ndarray] = None,
    radius_rings: Optional[np.ndarray] = None,
) -> go.Figure:
    point_indices = np.arange(len(original_pairs), dtype=int) if point_indices is None else np.asarray(point_indices, dtype=int)
    fig = go.Figure()
    q_colors = color_array_from_ids(color_ids if color_ids is not None else np.zeros(len(quant_pairs), dtype=int), levels)
    if x_grid is not None:
        for gx in np.asarray(x_grid).tolist():
            fig.add_vline(x=float(gx), line_color="#94a3b8", line_width=1, line_dash="dot")
    if y_grid is not None:
        for gy in np.asarray(y_grid).tolist():
            fig.add_hline(y=float(gy), line_color="#94a3b8", line_width=1, line_dash="dot")
    max_abs = float(max(np.max(np.abs(original_pairs)), np.max(np.abs(quant_pairs)), 1.0))
    if radial_angles is not None:
        ray_r = 1.05 * max_abs
        for ang in np.asarray(radial_angles).tolist():
            fig.add_trace(go.Scatter(x=[0.0, ray_r * math.cos(float(ang))], y=[0.0, ray_r * math.sin(float(ang))], mode="lines", line=dict(color="#94a3b8", width=1, dash="dot"), name="각도 코드북", showlegend=False, hoverinfo="skip"))
    if radius_rings is not None:
        theta = np.linspace(0.0, 2 * np.pi, 220)
        for ridx, radius in enumerate(np.asarray(radius_rings, dtype=float).tolist()):
            if radius <= 0:
                continue
            fig.add_trace(go.Scatter(x=radius * np.cos(theta), y=radius * np.sin(theta), mode="lines", line=dict(color="#cbd5e1", width=1.1, dash="dot"), name="반지름 기준", showlegend=(ridx == 0), hoverinfo="skip"))
    if show_unit_circle:
        theta = np.linspace(0.0, 2 * np.pi, 240)
        fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines", line=dict(color="#cbd5e1", width=1.3, dash="dot"), name="단위 원", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=original_pairs[:, 0], y=original_pairs[:, 1], mode="markers", customdata=point_indices[:, None], marker=dict(size=5, opacity=0.28, color=ORIGINAL_POINT_COLOR), name=original_name, hovertemplate="벡터 #%{customdata[0]}<extra></extra>"))
    fig.add_trace(go.Scatter(x=quant_pairs[:, 0], y=quant_pairs[:, 1], mode="markers", customdata=point_indices[:, None], marker=dict(size=5.5, opacity=0.9, color=q_colors, line=dict(width=0.3, color="white")), name=quant_name, hovertemplate="벡터 #%{customdata[0]}<extra></extra>"))
    if selected_index is not None:
        selected_mask = point_indices == int(selected_index)
        if np.any(selected_mask):
            fig.add_trace(go.Scatter(x=quant_pairs[selected_mask, 0], y=quant_pairs[selected_mask, 1], mode="markers", customdata=point_indices[selected_mask, None], marker=dict(size=10.5, color="#111827", line=dict(width=2.0, color="#f59e0b")), name=f"선택 벡터 #{int(selected_index)}", hovertemplate="벡터 #%{customdata[0]}<extra></extra>"))
    if selected_original_pair is not None:
        op = np.asarray(selected_original_pair, dtype=float)
        fig.add_trace(go.Scatter(x=[0.0, op[0]], y=[0.0, op[1]], mode="lines", line=dict(width=2, color=ORIGINAL_VECTOR_COLOR), name="선택 원본 반지름", hoverinfo="skip"))
    if selected_quant_pair is not None:
        qp = np.asarray(selected_quant_pair, dtype=float)
        fig.add_trace(go.Scatter(x=[0.0, qp[0]], y=[0.0, qp[1]], mode="lines", line=dict(width=2, color=RED, dash="dash"), name="선택 양자화 반지름", hoverinfo="skip"))
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=420, margin=dict(l=10, r=10, t=40, b=10), clickmode="event+select", xaxis_title="좌표축 1", yaxis_title="좌표축 2", yaxis=dict(scaleanchor="x", scaleratio=1), plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"), selectionrevision=str(selected_index) if selected_index is not None else "none")
    return fig


def bit_pattern_label(bit_id: int, bits: int) -> str:
    return f"색상 그룹 {int(bit_id) + 1}"


def color_group_preview_html(group_ids: List[int], levels: int, selected_ids: Optional[set[int]] = None, max_items: int = 24) -> str:
    palette = discrete_palette(levels)
    chips = []
    visible = [int(idx) for idx in group_ids[:max_items]]
    selected_lookup = set(visible if selected_ids is None else [int(v) for v in selected_ids])
    for idx in visible:
        color = palette[idx % len(palette)]
        active = idx in selected_lookup
        border = "#93c5fd" if active else "#e5e7eb"
        bg = "white" if active else "#f8fafc"
        text_color = "#111827" if active else "#94a3b8"
        opacity = "1.0" if active else "0.52"
        chips.append(
            f'<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border:1px solid {border};border-radius:999px;background:{bg};margin:3px 6px 3px 0;font-size:0.84rem;color:{text_color};opacity:{opacity};">'
            f'<span style="display:inline-block;width:11px;height:11px;border-radius:999px;background:{color};border:1px solid rgba(15,23,42,0.18);"></span>{bit_pattern_label(idx, 0)}</span>'
        )
    extra = '' if len(group_ids) <= max_items else f'<span style="color:#64748b;font-size:0.84rem;">외 {len(group_ids) - max_items}개</span>'
    return '<div style="display:flex;flex-wrap:wrap;align-items:center;margin:6px 0 10px 0;">' + ''.join(chips) + extra + '</div>'


def _set_color_group_selection(state_key: str, values: List[int]) -> None:
    st.session_state[state_key] = sorted(int(v) for v in values)


def _toggle_color_group(state_key: str, idx: int) -> None:
    current = {int(v) for v in st.session_state.get(state_key, [])}
    idx = int(idx)
    if idx in current:
        current.remove(idx)
    else:
        current.add(idx)
    st.session_state[state_key] = sorted(current)


def bit_pattern_multiselect(label: str, bits: int, key: str) -> List[int]:
    options = list(range(max(1, 2 ** bits)))
    state_key = f"{key}_selected"
    if state_key not in st.session_state:
        st.session_state[state_key] = options.copy()

    selected = [int(v) for v in st.session_state.get(state_key, options) if int(v) in options]
    selected_set = set(selected)
    st.markdown(f"**{label}**")
    st.markdown(color_group_preview_html(options, len(options), selected_set), unsafe_allow_html=True)

    action_left, action_right = st.columns(2)
    action_left.button("모두 켜기", key=f"{key}_all_on", use_container_width=True, on_click=_set_color_group_selection, args=(state_key, options))
    action_right.button("모두 끄기", key=f"{key}_all_off", use_container_width=True, on_click=_set_color_group_selection, args=(state_key, []))

    row_size = min(4, max(1, len(options)))
    for start in range(0, len(options), row_size):
        cols = st.columns(row_size)
        for col, idx in zip(cols, options[start:start + row_size]):
            active = idx in selected_set
            col.button(
                bit_pattern_label(int(idx), bits),
                key=f"{key}_btn_{idx}",
                use_container_width=True,
                type="primary" if active else "secondary",
                on_click=_toggle_color_group,
                args=(state_key, int(idx)),
            )

    selected = sorted(int(v) for v in st.session_state.get(state_key, options) if int(v) in options)
    if selected:
        st.caption(f"현재 표시 중: {len(selected)} / {len(options)} 색상 그룹")
    else:
        st.markdown('<div style="padding:8px 10px;border:1px dashed #cbd5e1;border-radius:12px;color:#64748b;background:#f8fafc;margin:6px 0 10px 0;">현재 선택된 색상 그룹이 없습니다.</div>', unsafe_allow_html=True)
        st.caption("`모두 켜기` 또는 개별 버튼으로 다시 켤 수 있습니다.")
    return selected

def build_plot_point_index_map(fig: go.Figure) -> Dict[Tuple[int, int], int]:
    point_index_map: Dict[Tuple[int, int], int] = {}
    for curve_number, trace in enumerate(fig.data):
        customdata = getattr(trace, "customdata", None)
        if customdata is None:
            continue
        arr = np.asarray(customdata, dtype=object)
        if arr.ndim == 0:
            continue
        if arr.ndim > 1:
            arr = arr[:, 0]
        point_count = len(arr)
        for point_number in range(point_count):
            try:
                point_index_map[(int(curve_number), int(point_number))] = int(arr[point_number])
            except Exception:
                continue
    return point_index_map


def extract_selected_point_index(event: Any, point_index_map: Optional[Dict[Tuple[int, int], int]] = None) -> Optional[int]:
    if event is None:
        return None

    candidate_points = []
    if isinstance(event, dict):
        if isinstance(event.get("points"), list):
            candidate_points.extend(event.get("points") or [])
        selection = event.get("selection")
        if isinstance(selection, dict) and isinstance(selection.get("points"), list):
            candidate_points.extend(selection.get("points") or [])
    if hasattr(event, "points") and getattr(event, "points"):
        candidate_points.extend(getattr(event, "points"))
    if hasattr(event, "selection") and hasattr(event.selection, "points") and event.selection.points:
        candidate_points.extend(event.selection.points)

    for point in candidate_points:
        customdata = None
        if hasattr(point, "customdata"):
            customdata = point.customdata
        elif isinstance(point, dict):
            customdata = point.get("customdata")
        if isinstance(customdata, (list, tuple, np.ndarray)):
            customdata = customdata[0] if len(customdata) else None
        if customdata is not None:
            try:
                return int(customdata)
            except Exception:
                pass

        curve_number = None
        point_number = None
        if hasattr(point, "curve_number"):
            curve_number = point.curve_number
        elif isinstance(point, dict):
            curve_number = point.get("curve_number", point.get("curveNumber"))
        if hasattr(point, "point_number"):
            point_number = point.point_number
        elif isinstance(point, dict):
            point_number = point.get("point_number", point.get("pointNumber", point.get("pointIndex")))

        if point_index_map is not None and curve_number is not None and point_number is not None:
            mapped = point_index_map.get((int(curve_number), int(point_number)))
            if mapped is not None:
                return int(mapped)
    return None


def plotly_chart_pick(fig: go.Figure, key: str) -> Optional[int]:
    point_index_map = build_plot_point_index_map(fig)
    event = st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        key=key,
        on_select="rerun",
        selection_mode=("points",),
    )
    return extract_selected_point_index(event, point_index_map=point_index_map)


def sync_inspect_vector_from_widget() -> None:
    st.session_state.inspect_vector_value = int(st.session_state.get("inspect_vector_widget", 0))
    st.session_state.inspect_vector_pending = None



def apply_pending_inspect_vector(max_index: int) -> None:
    pending = st.session_state.get("inspect_vector_pending")
    if pending is None:
        current = int(st.session_state.get("inspect_vector_value", st.session_state.get("inspect_vector_widget", 0)))
    else:
        current = int(pending)
        st.session_state.inspect_vector_pending = None
    current = int(max(0, min(max_index, current)))
    st.session_state.inspect_vector_value = current
    st.session_state.inspect_vector_widget = current



def update_inspect_vector(selected_index: Optional[int], max_index: int) -> None:
    if selected_index is None:
        return
    selected_index = int(max(0, min(max_index, selected_index)))
    current = int(st.session_state.get("inspect_vector_value", st.session_state.get("inspect_vector_widget", 0)))
    if current != selected_index or st.session_state.get("inspect_vector_pending") != selected_index:
        st.session_state.inspect_vector_pending = selected_index
        st.rerun()


def ensure_nonempty_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask if np.any(mask) else np.ones_like(mask, dtype=bool)


def filter_mask_from_bins(color_ids: np.ndarray, selected_bins: List[int]) -> np.ndarray:
    if len(selected_bins) == 0:
        return np.ones(len(color_ids), dtype=bool)
    return np.isin(np.asarray(color_ids, dtype=int), np.asarray(selected_bins, dtype=int))


@st.cache_data(show_spinner=False)
def make_ideal_geometry(n: int, d: int, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "구 표면":
        return make_data(n, d, "Unit sphere", seed + 1)
    if kind == "구 표면 + 이상치":
        return make_data(n, d, "Sphere shell + outliers", seed + 2)
    direction = rng.normal(size=(n, d))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + EPS
    radius = rng.random((n, 1)) ** (1.0 / max(1, d))
    points = direction * radius
    if kind == "구 내부 + 이상치":
        n_out = max(2, n // 24)
        idx = rng.choice(n, size=n_out, replace=False)
        points[idx] *= rng.uniform(1.55, 2.2, size=(n_out, 1))
    return points


def colored_map_3d(points_3d: np.ndarray, title: str, color_ids: np.ndarray, levels: int, name: str = "양자화 색상 그룹 맵") -> go.Figure:
    colors = color_array_from_ids(color_ids, levels)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points_3d[:, 0], y=points_3d[:, 1], z=points_3d[:, 2],
            mode="markers", name=name,
            marker=dict(size=4.2, opacity=0.9, color=colors, line=dict(width=0.35, color="white")),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color="black")),
        template="plotly_white",
        height=430,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            aspectmode="cube",
            bgcolor="white",
            xaxis=dict(title="투영축 1", backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
            yaxis=dict(title="투영축 2", backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
            zaxis=dict(title="투영축 3", backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
        ),
        legend=dict(bgcolor="rgba(255,255,255,0.82)", font=dict(color="black")),
        paper_bgcolor="white",
        font=dict(color="black"),
    )
    return fig


def turbo_ideal_explainer(bits: int) -> None:
    with st.expander("Turbo 이상 분포 설명 / 공식", expanded=False):
        st.markdown(
            f"""
**이 그림이 뜻하는 것**
- TurboQuant는 데이터셋 전용 **learned cluster map**을 보여 주는 그림이 아니라, **무작위 회전 뒤 공통 scalar codebook이 잘 작동하는 상황**을 직관적으로 보여 주는 이상화된 입력 분포입니다.
- 구 표면 계열은 **방향이 고르게 퍼진 경우**, 구 내부 계열은 **길이까지 다양하게 섞인 경우**를 뜻합니다.
- 현재 설정에서는 좌표마다 **{2 ** max(1, bits)}개 레벨**로 스냅되므로, 선택한 좌표쌍 평면에서는 `({2 ** max(1, bits)}) × ({2 ** max(1, bits)})` Turbo 격자가 나타납니다.

**핵심 식**
1. 입력을 길이와 방향으로 나눔
2. 방향 벡터를 무작위 회전
3. 회전 좌표를 1차원 코드북으로 각각 스냅
4. 다시 원래 좌표계로 되돌리고 길이를 곱함
            """
        )
        st.latex(r"u = x / \|x\|_2,\quad z = R u")
        st.latex(r"\hat z_j = Q_b(z_j)\quad (j=1,\dots,d)")
        st.latex(r"\hat x = \|x\|_2\, R^	op \hat z")
        st.markdown(
            """
**발표용 해석 포인트**
- 3D 이상 분포는 **회전 후 좌표가 고르게 퍼져 좌표별 코드북이 잘 작동하는 상황**을 보여줍니다.
- 오른쪽 히스토그램의 빨간 점선은 **Turbo 코드북 중심**이고, 아래 Turbo 좌표쌍 구름도에서는 같은 코드북이 **가로/세로 격자**로 보입니다.
            """
        )


def polar_ideal_explainer(bits: int, first_bits: int, upper_bits: int) -> None:
    with st.expander("Polar 이상 분포 설명 / 공식", expanded=False):
        st.markdown(
            f"""
**이 그림이 뜻하는 것**
- PolarQuant는 점을 polar 좌표계로 바꿔 **반지름은 유지하고 각도만 코드북에 스냅**하는 방법입니다.
- 이 이상 분포 그림은 **preconditioning 뒤 각도 분포가 잘 정리된 상태**를 단순화해서 보여줍니다.
- 현재 설정에서 첫 레벨 각도는 **{2 ** max(1, first_bits)}개**, 상위 레벨 각도는 **{2 ** max(1, upper_bits)}개** 코드북으로 스냅됩니다.
- **중요:** 이 앱의 방사형 bin 그림은 angle snap 직관을 위한 단순화이며, 논문 구현은 preconditioning 뒤 angle distribution을 바탕으로 **optimized codebook**과 **level-dependent bit allocation**을 사용합니다.

**핵심 식**
1. 입력을 무작위 회전
2. 회전 좌표를 재귀 polar 변환
3. 첫 레벨 각도는 `[0, 2\\pi)`, 상위 레벨 각도는 `[0, \\pi/2]` 코드북으로 양자화
4. 양자화된 각도로 다시 Cartesian 좌표 복원
            """
        )
        st.latex(r"z = R x")
        st.latex(r"(r, \Theta) = \mathrm{Polar}(z),\quad \hat\Theta = Q(\Theta)")
        st.latex(r"\hat x = R^\top\, \mathrm{Polar}^{-1}(r, \hat\Theta)")
        st.markdown(
            """
**발표용 해석 포인트**
- Polar 좌표쌍 구름도에 보이는 **방사선 묶음**은 첫 레벨 각도 코드북의 단순화된 직관 그림입니다.
- 즉, Turbo가 **격자형 스냅**이라면 Polar는 **각도형 스냅**으로 보면 이해가 쉽습니다.
- 논문 구현 쪽으로 말할 때는 “균일 각도 bin을 그대로 쓰는 방법”이라기보다, **분포 기반 codebook**을 실제 구현에 반영한다는 점을 함께 말하는 편이 안전합니다.
            """
        )


def render_polar_paper_panel(first_bits: int, upper_bits: int, bit_width: int) -> None:
    with st.expander("Polar demo simplification vs paper note", expanded=False):
        st.markdown(markdown_table(["구분", "이 데모가 주로 보여 주는 것", "논문이 추가하는 것"], [
            ["Demo", "recursive polar intuition / angle·radius 시각화 / 단순화된 angle codebook", "-"],
            ["Paper", "-", "optimized codebook / level-dependent bit allocation / 분포 기반 설계"],
        ]))
        demo_tab, paper_tab, bits_tab = st.tabs(["Demo codebook", "Paper note", "Bit allocation"])
        with demo_tab:
            st.markdown(
                f"""
- 현재 앱의 방사형 angle 그림은 **직관용 uniform angle snap 데모**입니다.
- 첫 레벨은 `[0, 2π)` 전체를, 상위 레벨은 `[0, π/2]` 범위를 나눠 보여 줍니다.
- 지금 설정에서는 첫 레벨 각도 bin이 **{2 ** max(1, first_bits)}개**, 상위 레벨 각도 bin이 **{2 ** max(1, upper_bits)}개**입니다.
- 발표에서는 “Turbo가 격자형 스냅이라면 Polar는 각도형 스냅”이라고 소개하면 이해가 빠릅니다.
                """
            )
        with paper_tab:
            st.markdown(
                """
- **중요:** 논문 구현은 단순 균일 각도 bin 그림을 그대로 쓰는 것이 아니라, preconditioning 뒤 얻어지는 **angle distribution**을 바탕으로 **optimized codebook**을 설계합니다.
- 논문 설명 흐름에서는 angle samples를 모은 뒤 **1-D k-means++ / Lloyd-style update**로 codebook을 구성할 수 있다는 관점으로 읽는 편이 맞습니다.
- 따라서 앱의 방사형 그림은 **논문 구현 그 자체**가 아니라, 각도 양자화가 어떤 방식으로 보이는지 설명하는 시각적 단순화입니다.
                """
            )
            st.latex(r"\Theta \xrightarrow{\text{sample}} \{\theta_i\} \xrightarrow{\text{1-D k-means++ / Lloyd}} \mathcal{C}_{\Theta}")
        with bits_tab:
            rows = [
                ["첫 레벨 angle", "[0, 2π)", str(first_bits), str(2 ** max(1, first_bits)), "가장 큰 방향 구분을 먼저 잡음"],
                ["상위 레벨 angle", "[0, π/2]", str(upper_bits), str(2 ** max(1, upper_bits)), "재귀 polar 구조에서 세부 각도 보정"],
                ["현재 앱 해석", "bit_width 기반 단순화", str(bit_width), str(2 ** max(1, bit_width)), "시연용 기준 비트 수"],
            ]
            st.markdown(markdown_table(["레벨", "범위", "비트 수", "bin 수", "의미"], rows))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["첫 레벨", "상위 레벨"], y=[first_bits, upper_bits], name="비트 수", text=[str(first_bits), str(upper_bits)], textposition="outside"))
            fig.update_layout(template="plotly_white", title=dict(text="Polar level별 bit allocation (paper note)", font=dict(color="black")), height=300, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="레벨", yaxis_title="비트 수", plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"), showlegend=False)
            st.plotly_chart(fig, width="stretch", theme=None)


def attention_proxy_figure(true_scores: np.ndarray, est_scores: np.ndarray, title: str, top_k: int = 20) -> go.Figure:
    k = max(1, min(int(top_k), len(true_scores)))
    idx = np.argsort(true_scores)[-k:]
    idx = idx[np.argsort(true_scores[idx])[::-1]]
    xs = [f"k{int(i)}" for i in idx]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=true_scores[idx], name="true softmax score", opacity=0.8))
    fig.add_trace(go.Bar(x=xs, y=est_scores[idx], name="estimated softmax score", opacity=0.75))
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), barmode="group", height=340, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="상위 key 인덱스", yaxis_title="attention score", plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"))
    return fig


def qjl_single_sketch(vector: np.ndarray, m: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vec = np.asarray(vector, dtype=float)
    sketch = gaussian_sketch(int(m), vec.size, int(seed) + 1505 + int(m))
    projected = sketch @ vec
    signs = np.sign(projected)
    signs[signs == 0] = 1
    scaled = (np.linalg.norm(vec) / max(1, int(m))) * signs
    return projected, signs, scaled


def vector_bar_figure(values: np.ndarray, title: str, yaxis_title: str) -> go.Figure:
    arr = np.asarray(values, dtype=float)
    xs = [str(i) for i in range(arr.size)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=arr, text=[f"{v:.2f}" for v in arr], textposition="outside", cliponaxis=False))
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=280, margin=dict(l=10, r=10, t=46, b=10), xaxis_title="스케치 좌표", yaxis_title=yaxis_title, plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"), showlegend=False)
    return fig


def sign_heatmap_figure(signs: np.ndarray, title: str) -> go.Figure:
    arr = np.asarray(signs, dtype=float)
    z = ((arr + 1.0) / 2.0)[None, :]
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[str(i) for i in range(arr.size)],
        y=["sign(Sk)"],
        colorscale=[[0.0, QJL_NEGATIVE_COLOR], [0.499, QJL_NEGATIVE_COLOR], [0.5, QJL_POSITIVE_COLOR], [1.0, QJL_POSITIVE_COLOR]],
        showscale=False,
        hovertemplate="m=%{x}<br>sign=%{customdata}<extra></extra>",
        customdata=np.where(arr[None, :] > 0, "+1", "-1"),
    ))
    for i, value in enumerate(arr):
        fig.add_annotation(x=str(i), y="sign(Sk)", text="+1" if value > 0 else "-1", showarrow=False, font=dict(color=("#082f49" if value > 0 else "white"), size=12))
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=190, margin=dict(l=10, r=10, t=46, b=10), xaxis_title="스케치 좌표", yaxis_title="", plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"))
    return fig


def qjl_sign_space_3d(
    signs: np.ndarray,
    title: str,
    color_ids: np.ndarray,
    levels: int,
    point_indices: Optional[np.ndarray] = None,
    selected_index: Optional[int] = None,
) -> go.Figure:
    arr = np.asarray(signs, dtype=float)
    point_indices = np.arange(arr.shape[0], dtype=int) if point_indices is None else np.asarray(point_indices, dtype=int)
    dims = min(3, arr.shape[1])
    coords = np.zeros((arr.shape[0], 3), dtype=float)
    coords[:, :dims] = arr[:, :dims]
    colors = color_array_from_ids(color_ids, levels)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers",
        name="sign(Sk) 앞 3축",
        customdata=point_indices[:, None],
        marker=dict(size=6.0, opacity=0.78, color=colors, line=dict(width=0.5, color="white")),
        hovertemplate="벡터 #%{customdata[0]}<extra></extra>",
    ))
    if selected_index is not None:
        mask = point_indices == int(selected_index)
        if np.any(mask):
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
                mode="markers",
                name=f"선택 벡터 #{int(selected_index)}",
                customdata=point_indices[mask, None],
                marker=dict(size=10.0, opacity=1.0, color="#111827", line=dict(width=2.0, color="#f59e0b")),
                hovertemplate="벡터 #%{customdata[0]}<extra></extra>",
            ))
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(color="black")),
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        clickmode="event+select",
        scene=dict(
            aspectmode="cube",
            bgcolor="white",
            xaxis=dict(title="sign bit 1", range=[-1.15, 1.15], tickvals=[-1, 1], backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
            yaxis=dict(title="sign bit 2", range=[-1.15, 1.15], tickvals=[-1, 1], backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
            zaxis=dict(title="sign bit 3", range=[-1.15, 1.15], tickvals=[-1, 1], backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
        ),
        paper_bgcolor="white",
        font=dict(color="black"),
    )
    return fig


def qjl_sign_stage_coords(signs: np.ndarray) -> np.ndarray:
    arr = np.asarray(signs, dtype=float)
    dims = min(3, arr.shape[1])
    coords = np.zeros((arr.shape[0], 3), dtype=float)
    coords[:, :dims] = arr[:, :dims]
    return coords


def qjl_process_figure_3d(
    original_3d: np.ndarray,
    proxy_3d: np.ndarray,
    sign_3d: np.ndarray,
    surrogate_3d: np.ndarray,
    color_ids: np.ndarray,
    title: str,
    levels: int,
    point_indices: Optional[np.ndarray] = None,
    selected_index: Optional[int] = None,
) -> go.Figure:
    colors_final = color_array_from_ids(color_ids, levels)
    point_indices = np.arange(len(original_3d), dtype=int) if point_indices is None else np.asarray(point_indices, dtype=int)
    stages = [("원본", original_3d, [ORIGINAL_POINT_COLOR] * len(original_3d))]
    for t in np.linspace(0.2, 1.0, 5):
        cur = interpolate_points(original_3d, proxy_3d, float(t))
        stages.append((f"JL proxy {int(round(t * 100))}%", cur, [ORIGINAL_POINT_COLOR] * len(cur)))
    for t in np.linspace(0.2, 1.0, 5):
        cur = interpolate_points(proxy_3d, sign_3d, float(t))
        stages.append((f"sign bit snap {int(round(t * 100))}%", cur, colors_final))
    for t in np.linspace(0.2, 1.0, 5):
        cur = interpolate_points(sign_3d, surrogate_3d, float(t))
        stages.append((f"surrogate 역투영 {int(round(t * 100))}%", cur, colors_final))

    all_pts = np.vstack([original_3d, proxy_3d, sign_3d, surrogate_3d])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    pads = np.maximum((maxs - mins) * 0.15, 0.25)
    xr = [float(mins[0] - pads[0]), float(maxs[0] + pads[0])]
    yr = [float(mins[1] - pads[1]), float(maxs[1] + pads[1])]
    zr = [float(mins[2] - pads[2]), float(maxs[2] + pads[2])]
    selected_mask = np.asarray(point_indices == int(selected_index), dtype=bool) if selected_index is not None else np.zeros(len(point_indices), dtype=bool)

    frames = []
    for label, pts, colors in stages:
        frame_data = [
            go.Scatter3d(x=surrogate_3d[:, 0], y=surrogate_3d[:, 1], z=surrogate_3d[:, 2], mode="markers", name="최종 상태", customdata=point_indices[:, None], marker=dict(size=3.8, opacity=0.85, color=FINAL_REFERENCE_COLOR), visible="legendonly", hovertemplate="벡터 #%{customdata[0]}<extra></extra>"),
            go.Scatter3d(x=original_3d[:, 0], y=original_3d[:, 1], z=original_3d[:, 2], mode="markers", name="원본 기준", customdata=point_indices[:, None], marker=dict(size=3.8, opacity=0.85, color=ORIGINAL_REFERENCE_COLOR), visible="legendonly", hovertemplate="벡터 #%{customdata[0]}<extra></extra>"),
            go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name="현재 단계", customdata=point_indices[:, None], marker=dict(size=4.8, opacity=0.94, color=colors, line=dict(width=0.45, color="white")), hovertemplate="벡터 #%{customdata[0]}<extra></extra>"),
        ]
        if np.any(selected_mask):
            sel_pts = pts[selected_mask]
            frame_data.append(go.Scatter3d(x=sel_pts[:, 0], y=sel_pts[:, 1], z=sel_pts[:, 2], mode="markers", name=f"선택 벡터 #{int(selected_index)}", customdata=point_indices[selected_mask, None], marker=dict(size=8.5, opacity=1.0, color="#111827", line=dict(width=2.0, color="#f59e0b")), hovertemplate="벡터 #%{customdata[0]}<extra></extra>"))
        frames.append(go.Frame(name=label, data=frame_data, traces=list(range(len(frame_data)))))

    data = [
        go.Scatter3d(x=surrogate_3d[:, 0], y=surrogate_3d[:, 1], z=surrogate_3d[:, 2], mode="markers", name="최종 상태", customdata=point_indices[:, None], marker=dict(size=3.8, opacity=0.85, color=FINAL_REFERENCE_COLOR), visible="legendonly", hovertemplate="벡터 #%{customdata[0]}<extra></extra>"),
        go.Scatter3d(x=original_3d[:, 0], y=original_3d[:, 1], z=original_3d[:, 2], mode="markers", name="원본 기준", customdata=point_indices[:, None], marker=dict(size=3.8, opacity=0.85, color=ORIGINAL_REFERENCE_COLOR), visible="legendonly", hovertemplate="벡터 #%{customdata[0]}<extra></extra>"),
        go.Scatter3d(x=surrogate_3d[:, 0], y=surrogate_3d[:, 1], z=surrogate_3d[:, 2], mode="markers", name="현재 단계", customdata=point_indices[:, None], marker=dict(size=4.8, opacity=0.94, color=colors_final, line=dict(width=0.45, color="white")), hovertemplate="벡터 #%{customdata[0]}<extra></extra>"),
    ]
    if np.any(selected_mask):
        sel_pts = surrogate_3d[selected_mask]
        data.append(go.Scatter3d(x=sel_pts[:, 0], y=sel_pts[:, 1], z=sel_pts[:, 2], mode="markers", name=f"선택 벡터 #{int(selected_index)}", customdata=point_indices[selected_mask, None], marker=dict(size=8.5, opacity=1.0, color="#111827", line=dict(width=2.0, color="#f59e0b")), hovertemplate="벡터 #%{customdata[0]}<extra></extra>"))

    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        title=dict(text=title, font=dict(color="black")),
        template="plotly_white",
        height=560,
        margin=dict(l=0, r=0, t=40, b=0),
        clickmode="event+select",
        scene=dict(
            aspectmode="cube",
            bgcolor="white",
            xaxis=dict(title="투영축 1 / sign bit 1", range=xr, backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
            yaxis=dict(title="투영축 2 / sign bit 2", range=yr, backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
            zaxis=dict(title="투영축 3 / sign bit 3", range=zr, backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd", color="black"),
        ),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0,
            "y": 1.08,
            "font": {"color": "black"},
            "bgcolor": "white",
            "bordercolor": "#cbd5e1",
            "borderwidth": 1,
            "buttons": [
                {"label": "Play", "method": "animate", "args": [[label for label, _, _ in stages], {"frame": {"duration": 200, "redraw": True}, "fromcurrent": False, "mode": "immediate", "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": len(stages) - 1,
            "currentvalue": {"prefix": "3D 단계: ", "font": {"color": "black"}},
            "font": {"color": "black"},
            "bgcolor": "white",
            "activebgcolor": "#e2e8f0",
            "bordercolor": "#cbd5e1",
            "tickcolor": "black",
            "pad": {"t": 36},
            "steps": [{"label": label, "method": "animate", "args": [[label], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]} for label, _, _ in stages],
        }],
        legend=dict(bgcolor="rgba(255,255,255,0.82)", font=dict(color="black")),
        paper_bgcolor="white",
        selectionrevision=str(selected_index) if selected_index is not None else "none",
        font=dict(color="black"),
    )
    return fig


def render_qjl_sketch_panel(vector: np.ndarray, vector_index: int, seed: int, qjl_m: int) -> None:
    st.markdown("### QJL = 1-bit sign sketch for inner-product estimation")
    st.caption(f"선택 벡터 #{vector_index}를 기준으로 `Sk`, `sign(Sk)`, `(||k||₂ / m)·sign(Sk)`를 보여 줍니다. 현재 메인 QJL 스케치 차원은 m={int(qjl_m)} 입니다.")
    m_options = []
    for candidate in (4, 8, 16, int(qjl_m)):
        if candidate > 0 and candidate not in m_options:
            m_options.append(candidate)
    tabs = st.tabs([f"m={m}" for m in m_options])
    for tab, m in zip(tabs, m_options):
        with tab:
            projected, signs, scaled = qjl_single_sketch(vector, m, seed)
            col1, col2, col3 = st.columns([1.1, 0.95, 1.1])
            with col1:
                st.plotly_chart(vector_bar_figure(projected, "Projected vector: Sk", "실수 값"), width="stretch", theme=None)
            with col2:
                st.plotly_chart(sign_heatmap_figure(signs, "Sign sketch: sign(Sk)"), width="stretch", theme=None)
                st.caption(f"청록 = +1 (Sk>0), 주황 = -1 (Sk<0)")
                st.caption("아래 칩은 실제 저장되는 sign bit를 밝은 박스로 다시 적어 둔 것입니다.")
                st.markdown(sign_chip_box_html(signs), unsafe_allow_html=True)
            with col3:
                st.plotly_chart(vector_bar_figure(scaled, "Scaled sketch: (||k||₂ / m) · sign(Sk)", "scaled sign"), width="stretch", theme=None)
            st.caption("QJL의 실제 저장 표현은 `sign(Sk)`와 벡터 norm입니다. 왼쪽 `Sk`는 설명을 위한 중간 실수 벡터, 오른쪽 scaled sketch는 저장/추정에 쓰이는 방향성을 보여 주는 보조 그림입니다.")


def metric_reference_specs() -> List[Dict[str, str]]:
    return [
        {
            "metric": "MSE",
            "formula": r"\frac{1}{n}\sum_{i=1}^{n} \|x_i-\hat x_i\|_2^2",
            "meaning": "벡터 전체 오차 에너지. 큰 오차를 특히 크게 벌점합니다.",
            "good": "0에 가까울수록 좋음",
            "talk": "전체 복원 품질의 기본 점수로 소개하면 좋습니다.",
        },
        {
            "metric": "MAE",
            "formula": r"\frac{1}{nd}\sum_{i=1}^{n}\sum_{j=1}^{d} |x_{ij}-\hat x_{ij}|",
            "meaning": "좌표 하나하나의 평균 절대 오차입니다.",
            "good": "0에 가까울수록 좋음",
            "talk": "MSE보다 이상치 영향이 덜한 평균 오차라고 설명하기 좋습니다.",
        },
        {
            "metric": "Mean cosine",
            "formula": r"\frac{1}{n}\sum_{i=1}^{n} \frac{\langle x_i, \hat x_i\rangle}{\|x_i\|_2\,\|\hat x_i\|_2}",
            "meaning": "원본과 복원 벡터의 방향이 얼마나 비슷한지 봅니다.",
            "good": "1에 가까울수록 좋음",
            "talk": "벡터 방향 보존력이 중요할 때 해석하기 좋습니다.",
        },
        {
            "metric": "IP bias",
            "formula": r"\frac{1}{n}\sum_{i=1}^{n} (\langle q,\hat x_i\rangle-\langle q,x_i\rangle)",
            "meaning": "내적 추정이 평균적으로 한쪽으로 치우치는지 보여줍니다.",
            "good": "0에 가까울수록 좋음",
            "talk": "플러스면 과대추정, 마이너스면 과소추정이라고 말하면 됩니다.",
        },
        {
            "metric": "IP MAE",
            "formula": r"\frac{1}{n}\sum_{i=1}^{n} |\langle q,\hat x_i\rangle-\langle q,x_i\rangle|",
            "meaning": "내적 기준 평균 절대 오차입니다.",
            "good": "0에 가까울수록 좋음",
            "talk": "검색/어텐션처럼 내적 자체가 중요한 작업에서 핵심 지표입니다.",
        },
        {
            "metric": "IP corr",
            "formula": r"\mathrm{corr}(\langle q,x_i\rangle,\langle q,\hat x_i\rangle)",
            "meaning": "실제 내적 순서를 얼마나 잘 보존하는지 봅니다.",
            "good": "1에 가까울수록 좋음",
            "talk": "랭킹 보존 관점에서 발표할 때 직관이 좋습니다.",
        },
    ]



def render_metric_reference() -> None:
    specs = metric_reference_specs()
    with st.expander("지표 설명 / 공식 / 해석", expanded=False):
        overview_tab, formula_tab, guide_tab = st.tabs(["빠른 표", "공식", "발표용 해석"])
        with overview_tab:
            rows = [[item["metric"], item["meaning"], item["good"]] for item in specs]
            st.markdown(markdown_table(["지표", "무엇을 보는가", "좋은 방향"], rows))
            st.caption("요약: MSE · MAE · IP MAE는 낮을수록, Mean cosine · IP corr는 높을수록, IP bias는 0에 가까울수록 좋습니다.")
        with formula_tab:
            for item in specs:
                st.markdown(f"#### {item['metric']}")
                st.latex(item["formula"])
                st.markdown(f"- **의미:** {item['meaning']}\n- **좋은 방향:** {item['good']}")
        with guide_tab:
            rows = [[item["metric"], item["talk"], item["good"]] for item in specs]
            st.markdown(markdown_table(["지표", "발표에서 이렇게 읽기", "수렴 방향"], rows))
            st.markdown(
                "- **의미:** query는 `Sq` 형태로 실수 projection을 유지하고, key만 sign-bit sketch로 저장합니다.\n"
                "- **중요:** 이 식이 unbiased inner-product estimator의 핵심이라서, QJL은 Turbo/Polar처럼 복원 오차 그림만으로 판단하면 안 됩니다."
            )
        with guide_tab:
            st.markdown(markdown_table(["발표 포인트", "이렇게 말하면 쉬움"], [
                ["3D 그림의 역할", "왼쪽 3D는 복원형 quantizer 그림이 아니라, JL 투영 → sign sketch라는 흐름을 눈으로 보여 주는 보조 그림입니다."],
                ["진짜 핵심 지표", "QJL은 IP bias, IP MAE, IP corr 같은 내적 추정 지표로 읽는 편이 논문 취지에 맞습니다."],
                ["Value cache", "QJL 논문은 value cache까지 같은 방식으로 처리하는 것이 아니라, value는 표준 token-wise quantization을 사용합니다."],
            ]))


def render_alignment_reference() -> None:
    with st.expander("논문 원안 여부 / 비교 축 정리", expanded=False):
        align_tab, axis_tab = st.tabs(["논문 원안 여부", "비교 축 가이드"])
        with align_tab:
            st.markdown(markdown_table(["방법", "분류", "설명"], [
                ["TurboQuant", "Paper-aligned", "무작위 회전 뒤 좌표별 스칼라 양자화라는 큰 흐름을 따릅니다."],
                ["PolarQuant", "Paper-aligned (직관용 단순화 포함)", "재귀 polar 변환과 angle quantization 흐름을 따르지만, 앱의 방사형 bin 그림은 직관용으로 단순화했습니다."],
                ["QJL", "Paper-aligned", "1-bit sign sketch와 비대칭 inner-product estimator 관점을 중심에 두고 설명합니다."],
                ["Turbo + QJL", "Paper-aligned hybrid", "Turbo MSE base 뒤 residual에 1-bit QJL을 붙이는 논문식 2단계 구조입니다."],
                ["Polar + QJL", "Exploratory hybrid", "비교/교육용 탐색 하이브리드이며 PolarQuant 논문의 원안은 아닙니다."],
            ]))
        with axis_tab:
            st.markdown(markdown_table(["방법", "가장 잘 보여야 하는 것", "핵심 질문"], [
                ["TurboQuant", "좌표 스냅 / 코드북", "좌표별 quantization이 어떻게 일어나나?"],
                ["PolarQuant", "각도 / 반지름 / 재귀 구조", "polar 표현에서 무엇이 양자화되나?"],
                ["QJL", "sign sketch / IP 추정 / attention proxy", "복원이 아니라 무엇을 보존하나?"],
            ]))
            st.markdown(markdown_table(["비교 축", "여기에 두는 방법", "주요 지표"], [
                ["복원 / 구조 보존", "TurboQuant, PolarQuant", "MSE, MAE, Mean cosine, 단면 이동, 코드북 구조"],
                ["내적 / 추정 품질", "QJL, Turbo + QJL, Polar + QJL", "IP bias, IP MAE, IP corr, true-vs-est inner product, attention proxy"],
            ]))
            st.caption("발표 때는 Turbo/Polar를 먼저 geometry 관점에서 보여주고, QJL 계열은 inner-product estimator 관점으로 분리해서 설명하면 가장 덜 헷갈립니다.")


# -----------------------------
# Main
# -----------------------------


if "inspect_vector_value" not in st.session_state:
    st.session_state.inspect_vector_value = 0
if "inspect_vector_widget" not in st.session_state:
    st.session_state.inspect_vector_widget = 0
if "inspect_vector_pending" not in st.session_state:
    st.session_state.inspect_vector_pending = None
if "slice_pair_input" not in st.session_state:
    st.session_state.slice_pair_input = 0

inject_theme()
st.title("TurboQuant / PolarQuant / QJL Explorer")
st.caption("이 저장소는 TurboQuant, PolarQuant, QJL이 무엇을 보존하고 어떻게 달라지는지를 직관적으로 비교해 보여 주는 시각화 데모입니다.")

DIMENSION_PRESETS = [3, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


distribution_notes = {
    "Gaussian": "가장 기본적인 분포입니다. 좌표별 양자화와 내적 오차를 무난하게 비교할 때 적합합니다.",
    "Gaussian + outliers": "대부분은 중앙에 모여 있고 일부 이상치가 멀리 떨어져 있는 경우입니다. 이상치에 대한 민감도를 보기 좋습니다.",
    "Unit sphere": "모든 점의 길이가 거의 같은 구 표면 데이터입니다. TurboQuant의 회전 후 좌표 분포 해석과 잘 맞는 편입니다.",
    "Sphere shell + outliers": "거의 구 표면에 있으면서 일부 이상치가 더 바깥으로 튀는 경우입니다. 구형 구조 + 이상치를 함께 보고 싶을 때 적합합니다.",
    "Ball + outliers": "구 내부에 점이 퍼져 있고 일부 이상치가 바깥에 있는 경우입니다. PolarQuant의 반지름/각도 관찰에 특히 보기 좋습니다.",
}

WIDGET_DEFAULTS = {
    "app_mode": "Balanced",
    "n_points": 900,
    "dimension_input": 32,
    "distribution": "Gaussian",
    "precision": "fp16-like",
    "bit_width": 3,
    "precondition": True,
    "projection_mode": "Random projection",
    "seed_input": 7,
    "plot_points": 500,
    "process_points": 100,
}
for _state_key, _default_value in WIDGET_DEFAULTS.items():
    st.session_state.setdefault(_state_key, _default_value)

with st.sidebar:
    st.header("설정")
    mode = st.radio("앱 모드", ["Balanced", "Paper-aligned"], key="app_mode", help="Balanced는 설명을 우선하고, Paper-aligned는 논문 관점 설명을 조금 더 강조합니다.")
    n_points = st.slider("벡터 수", 300, 2200, step=100, key="n_points")
    st.caption("차원 d는 숫자로 직접 입력하고, 자주 쓰는 값은 아래 프리셋 버튼으로 바로 넣을 수 있습니다.")
    preset_cols = st.columns(4)
    for preset_index, preset_dimension in enumerate(DIMENSION_PRESETS):
        if preset_cols[preset_index % len(preset_cols)].button(str(preset_dimension), key=f"dimension_preset_{preset_dimension}", use_container_width=True):
            st.session_state.dimension_input = int(preset_dimension)
            st.rerun()
    dimension = int(st.number_input(
        "차원 d",
        min_value=3,
        max_value=4096,
        step=1,
        key="dimension_input",
        help="슬라이더 대신 직접 숫자를 넣어 큰 범위에서도 원하는 차원을 빠르게 고를 수 있습니다.",
    ))
    distribution = st.selectbox("데이터 분포", ["Gaussian", "Gaussian + outliers", "Unit sphere", "Sphere shell + outliers", "Ball + outliers"], key="distribution")
    precision = st.selectbox("입력 정밀도 시뮬레이션", ["fp16-like", "fp8-like", "int8-like"], key="precision")
    bit_width = st.slider("양자화 비트 수", 1, 6, help="각 방법이 한 좌표 또는 한 단계에서 대략 얼마나 촘촘하게 양자화하는지 보는 기준 비트 수입니다.", key="bit_width")
    precondition = st.toggle("랜덤 전처리 적용", key="precondition")
    projection_mode = st.selectbox("3D 공통 투영 방식", ["Random projection", "PCA", "First 3 coordinates"], key="projection_mode")
    seed = st.number_input("시드", min_value=0, max_value=999999, step=1, key="seed_input")
    plot_points = st.slider("비교용 표시 점 수", 200, 1200, step=100, key="plot_points")
    process_points = st.slider("3D 애니메이션 점 수", 40, 220, step=20, key="process_points")
    apply_pending_inspect_vector(max(0, n_points - 1))
    inspect_vector = st.number_input(
        "단면 예시 벡터 번호",
        min_value=0,
        max_value=max(0, n_points - 1),
        step=1,
        key="inspect_vector_widget",
        on_change=sync_inspect_vector_from_widget,
        help="단면 그림에서 자세히 볼 하나의 샘플 벡터 번호입니다. 3D나 구름도에서 점을 클릭하면 즉시 이 값으로 바뀌고, 다시 다른 점을 클릭해도 덮어씁니다.",
    )
    max_pair = max(0, dimension // 2 - 1)
    st.session_state.slice_pair_input = int(max(0, min(max_pair, st.session_state.get("slice_pair_input", 0))))
    if max_pair == 0:
        slice_pair = st.number_input(
            "단면 좌표쌍 번호",
            min_value=0,
            max_value=0,
            value=0,
            step=1,
            disabled=True,
            help="차원 d가 아주 작을 때는 볼 수 있는 2D 좌표쌍이 하나뿐이라 값이 0으로 고정됩니다.",
        )
    else:
        slice_pair = st.slider("단면 좌표쌍 번호", 0, max_pair, key="slice_pair_input", help="i를 고르면 (x[2i], x[2i+1]) 좌표쌍을 2D 단면으로 봅니다.")
    st.caption("단면 예시 벡터 번호 = 한 개 샘플 확대 보기 / 단면 좌표쌍 번호 = (x[2i], x[2i+1]) 2D 보기")
    st.caption(f"선택한 데이터 분포: {distribution_notes[distribution]}")
    st.caption(f"양자화 색상 그룹 수 = 2^{bit_width} = {2 ** bit_width}")

if mode == "Paper-aligned" and distribution in {"Gaussian + outliers", "Sphere shell + outliers", "Ball + outliers"}:
    st.info("Paper-aligned 모드에서는 Gaussian / Unit sphere가 논문 해석과 가장 직접적으로 맞습니다. 추가한 이상치 분포는 교육용 시각화에 가깝습니다.")

levels = max(1, 2 ** bit_width)
x = make_data(n_points, dimension, distribution, int(seed))
x = apply_precision_rounding(x, precision)
q = make_query(dimension, distribution, int(seed))

x_baseline, baseline_details = baseline_uniform_quantize(x, bit_width)
x_turbo, turbo_details = turbo_quantize_mse(x, bit_width, precondition, int(seed))
x_polar, polar_details = polar_quantize(x, bit_width, precondition, int(seed))
m_qjl = dimension if mode == "Paper-aligned" else max(8, dimension // 2)
x_qjl_vis, qjl_details = qjl_quantize(x, q, m_qjl, int(seed), bit_width)
x_turbo_prod, turbo_prod_details = turbo_quantize_prod(x, bit_width, precondition, int(seed))
x_polar_prod, polar_prod_details = polar_quantize_prod(x, bit_width, precondition, int(seed))

metrics_baseline = compute_metrics(x, x_baseline, q)
metrics_turbo = compute_metrics(x, x_turbo, q)
metrics_polar = compute_metrics(x, x_polar, q)
metrics_qjl_surrogate = compute_metrics(x, x_qjl_vis, q)
metrics_turbo_prod = compute_metrics(x, x_turbo_prod, q)
metrics_polar_prod = compute_metrics(x, x_polar_prod, q)

true_ip = x @ q
est_ip_baseline = x_baseline @ q
est_ip_turbo = x_turbo @ q
est_ip_polar = x_polar @ q
est_ip_qjl = qjl_details["ip_est"]
est_ip_turbo_prod = x_turbo_prod @ q
est_ip_polar_prod = x_polar_prod @ q

attn_temperature = max(float(np.std(true_ip)), 1.0)
attn_top_k = min(12, n_points)
qjl_attn = attention_proxy_metrics(true_ip, est_ip_qjl, attn_temperature, top_k=attn_top_k)
turbo_prod_attn = attention_proxy_metrics(true_ip, est_ip_turbo_prod, attn_temperature, top_k=attn_top_k)
polar_prod_attn = attention_proxy_metrics(true_ip, est_ip_polar_prod, attn_temperature, top_k=attn_top_k)

method_registry: Dict[str, MethodResult] = {
    "기존 양자화": MethodResult("기존 양자화", x_baseline, metrics_baseline, "원래 좌표계에서 바로 uniform scalar quantization", baseline_details),
    "TurboQuant": MethodResult("TurboQuant", x_turbo, metrics_turbo, "무작위 회전 후 좌표별 스칼라 양자화", turbo_details),
    "PolarQuant": MethodResult("PolarQuant", x_polar, metrics_polar, "재귀 polar 변환 후 각도 양자화", polar_details),
    "QJL": MethodResult("QJL", x_qjl_vis, metrics_qjl_surrogate, "1-bit sign sketch와 내적 추정이 본체이며 3D는 설명용", qjl_details),
    "Turbo + QJL": MethodResult("Turbo + QJL", x_turbo_prod, metrics_turbo_prod, "논문식 2단계: MSE base + residual QJL", turbo_prod_details),
    "Polar + QJL": MethodResult("Polar + QJL", x_polar_prod, metrics_polar_prod, "탐색적 비교용: Polar base + residual QJL", polar_prod_details),
}

static_idx = sample_indices(n_points, plot_points, int(seed) + 1001)
process_idx = sample_indices(n_points, min(process_points, plot_points), int(seed) + 1201)

qjl_mid_state = ((x @ qjl_details["sketch"].T) @ qjl_details["sketch"]) / max(1, qjl_details["sketch"].shape[0])
common_ref = np.vstack([
    x[static_idx],
    x_baseline[static_idx],
    x_turbo[static_idx],
    x_polar[static_idx],
    x_qjl_vis[static_idx],
    x_turbo_prod[static_idx],
    x_polar_prod[static_idx],
    qjl_mid_state[static_idx],
])
mean3, basis3 = fit_projector(common_ref, projection_mode, int(seed) + 10, 3)
proj_x = apply_projector(x[static_idx], mean3, basis3)
proj_qjl = apply_projector(x_qjl_vis[static_idx], mean3, basis3)

def build_projected_process_stages(stage_defs: List[Tuple[str, np.ndarray, Optional[List[str]]]], idx: np.ndarray, seed_offset: int) -> List[Dict[str, Any]]:
    ref = np.vstack([points[idx] for _, points, _ in stage_defs])
    mean_p, basis_p = fit_projector(ref, projection_mode, int(seed) + seed_offset, 3)
    projected: List[Dict[str, Any]] = []
    for label, points, colors in stage_defs:
        stage_colors = None if colors is None else list(np.asarray(colors, dtype=object)[idx])
        projected.append({
            "label": label,
            "points": apply_projector(points[idx], mean_p, basis_p),
            "colors": stage_colors,
        })
    return projected

process_registry = {
    "기존 양자화": {
        "stages": build_projected_process_stages([
            ("원본", x, None),
            ("클리핑", baseline_details["clipped"], None),
            ("균일 스냅", x_baseline, color_array_from_ids(baseline_details["color_ids"], levels)),
        ], process_idx, 11),
        "color_ids": baseline_details["color_ids"][process_idx],
    },
    "TurboQuant": {
        "stages": build_projected_process_stages([
            ("원본", x, None),
            ("회전", turbo_details["rot_scaled"], None),
            ("코드북 스냅", turbo_details["q_rot_scaled"], color_array_from_ids(turbo_details["color_ids"], levels)),
            ("최종 복원", x_turbo, color_array_from_ids(turbo_details["color_ids"], levels)),
        ], process_idx, 101),
        "color_ids": turbo_details["color_ids"][process_idx],
    },
    "PolarQuant": {
        "stages": build_projected_process_stages([
            ("원본", x, None),
            ("회전", polar_details["rot"], None),
            ("각도 스냅", polar_details["recon_rot"], color_array_from_ids(polar_details["color_ids"], levels)),
            ("최종 복원", x_polar, color_array_from_ids(polar_details["color_ids"], levels)),
        ], process_idx, 202),
        "color_ids": polar_details["color_ids"][process_idx],
    },
    "Turbo + QJL": {
        "stages": build_projected_process_stages([
            ("원본", x, None),
            ("회전", turbo_prod_details["rot_scaled"], None),
            ("base codebook snap", turbo_prod_details["q_rot_scaled"], color_array_from_ids(turbo_prod_details["color_ids"], levels)),
            ("base reconstruction", turbo_prod_details["base"], color_array_from_ids(turbo_prod_details["color_ids"], levels)),
            ("잔차 보정", turbo_prod_details["residual_mid"], color_array_from_ids(turbo_prod_details["color_ids"], levels)),
            ("최종 복원", x_turbo_prod, color_array_from_ids(turbo_prod_details["color_ids"], levels)),
        ], process_idx, 303),
        "color_ids": turbo_prod_details["color_ids"][process_idx],
    },
    "Polar + QJL": {
        "stages": build_projected_process_stages([
            ("원본", x, None),
            ("회전", polar_prod_details["rot"], None),
            ("base angle snap", polar_prod_details["recon_rot"], color_array_from_ids(polar_prod_details["color_ids"], levels)),
            ("base reconstruction", polar_prod_details["base"], color_array_from_ids(polar_prod_details["color_ids"], levels)),
            ("잔차 보정", polar_prod_details["residual_mid"], color_array_from_ids(polar_prod_details["color_ids"], levels)),
            ("최종 복원", x_polar_prod, color_array_from_ids(polar_prod_details["color_ids"], levels)),
        ], process_idx, 404),
        "color_ids": polar_prod_details["color_ids"][process_idx],
    },
    "QJL": {
        "stages": build_projected_process_stages([
            ("원본", x, None),
            ("JL proxy", qjl_mid_state, None),
            ("surrogate 역투영", x_qjl_vis, color_array_from_ids(qjl_details["color_ids"], levels)),
        ], process_idx, 505),
        "color_ids": qjl_details["color_ids"][process_idx],
    },
}

# UI labels keep the full method names, so provide explicit aliases as well.
process_registry["TurboQuant + QJL"] = process_registry["Turbo + QJL"]
process_registry["PolarQuant + QJL"] = process_registry["Polar + QJL"]

inspect_idx = int(st.session_state.get("inspect_vector_value", inspect_vector))
pair_start = 2 * int(slice_pair)
pair_end = pair_start + 2

baseline_original_pair = x[inspect_idx, pair_start:pair_end]
baseline_quant_pair = x_baseline[inspect_idx, pair_start:pair_end]
baseline_original_cloud = x[static_idx, pair_start:pair_end]
baseline_quant_cloud = x_baseline[static_idx, pair_start:pair_end]

polar_original_pair = polar_details["rot"][inspect_idx, pair_start:pair_end]
polar_quant_pair = polar_details["recon_rot"][inspect_idx, pair_start:pair_end]
polar_original_cloud = polar_details["rot"][static_idx, pair_start:pair_end]
polar_quant_cloud = polar_details["recon_rot"][static_idx, pair_start:pair_end]

turbo_original_pair = turbo_details["rot"][inspect_idx, pair_start:pair_end]
turbo_quant_pair = turbo_details["q_rot"][inspect_idx, pair_start:pair_end]
turbo_original_cloud = turbo_details["rot"][static_idx, pair_start:pair_end]
turbo_quant_cloud = turbo_details["q_rot"][static_idx, pair_start:pair_end]

st.markdown("### 지금 한 번에 보는 핵심")
summary_metrics = {
    "기존 MSE": metrics_baseline["MSE"],
    "Turbo MSE": metrics_turbo["MSE"],
    "Polar MSE": metrics_polar["MSE"],
    "QJL 내적 MAE": float(np.mean(np.abs(est_ip_qjl - true_ip))),
    "Turbo+QJL 내적 MAE": float(np.mean(np.abs(est_ip_turbo_prod - true_ip))),
    "Polar+QJL 내적 MAE": float(np.mean(np.abs(est_ip_polar_prod - true_ip))),
}
metric_cards(summary_metrics, accents=["blue", "blue", "red", "blue", "blue", "red"])

st.markdown(f'<div class="paper-card"><strong>선택한 데이터 분포</strong><br>{distribution_notes[distribution]}</div>', unsafe_allow_html=True)

with st.expander("논문 반영 범위 / 이 앱이 어디까지 paper-aligned 인가", expanded=False):
    st.markdown(
        """
- **기존 양자화**: 원래 Cartesian 좌표계에서 바로 uniform scalar codebook에 스냅하는 baseline을 추가했습니다.
- **TurboQuant**: 무작위 회전 후 좌표별 스칼라 양자화를 적용하는 구조를 반영했습니다. 여기서 보이는 격자형 그림은 **학습된 데이터 맵**이 아니라 **공통 scalar codebook**의 스냅 구조를 보여 주는 시각화입니다.
- **PolarQuant**: 재귀적 polar 변환, 첫 레벨 `[0, 2π)`, 상위 레벨 `[0, π/2]` 구조를 반영했습니다. 다만 앱의 방사형 angle bin은 **직관용 단순화 그림**이며, 논문 구현은 **optimized codebook**과 **level-dependent bit allocation**을 사용합니다.
- **QJL**: 논문의 본체는 **벡터 복원**이 아니라 **비대칭 inner-product estimator**입니다. 따라서 QJL 탭의 3D 그림은 설명용이고, 핵심 평가는 `IP bias / IP MAE / IP corr`입니다.
- **Turbo + QJL**: 먼저 MSE 양자화 후 residual에 1-bit QJL을 붙이는 **논문 친화적 2단계 구조**입니다.
- **Polar + QJL**: 비교/교육용 **탐색 하이브리드**이며 PolarQuant 논문의 원안은 아닙니다.
        """
    )

comparison_rows = [
    ["기존 양자화", f"{metrics_baseline['MSE']:.4f}", f"{metrics_baseline['IP MAE']:.4f}", f"{metrics_baseline['IP bias']:.4f}", "Cartesian uniform baseline"],
    ["TurboQuant", f"{metrics_turbo['MSE']:.4f}", f"{metrics_turbo['IP MAE']:.4f}", f"{metrics_turbo['IP bias']:.4f}", "좌표 기반 MSE"],
    ["PolarQuant", f"{metrics_polar['MSE']:.4f}", f"{metrics_polar['IP MAE']:.4f}", f"{metrics_polar['IP bias']:.4f}", "각도 기반"],
    ["QJL", f"{metrics_qjl_surrogate['MSE']:.4f}", f"{float(np.mean(np.abs(est_ip_qjl - true_ip))):.4f}", f"{float(np.mean(est_ip_qjl - true_ip)):.4f}", "내적 우선"],
    ["Turbo + QJL", f"{metrics_turbo_prod['MSE']:.4f}", f"{metrics_turbo_prod['IP MAE']:.4f}", f"{metrics_turbo_prod['IP bias']:.4f}", "논문식 하이브리드"],
    ["Polar + QJL", f"{metrics_polar_prod['MSE']:.4f}", f"{metrics_polar_prod['IP MAE']:.4f}", f"{metrics_polar_prod['IP bias']:.4f}", "비교용 하이브리드"],
]

st.caption("지표 공식과 3D 투영 방식 설명은 아래 `지표 / 투영 해설` 탭에 정리했습니다.")

baseline_tab, turbo_tab, polar_tab, qjl_tab, compare_tab, reference_tab = st.tabs(["기존 양자화", "TurboQuant", "PolarQuant", "QJL", "비교 / 하이브리드", "지표 / 투영 해설"])
with baseline_tab:
    one_line_box("첫 번째 탭은 비교 기준선입니다. 회전이나 polar 변환 없이, 원래 Cartesian 좌표에서 바로 uniform scalar quantization을 적용하면 어떤 격자형 왜곡이 생기는지 보여 줍니다.")
    note_card("Baseline note", "이 탭은 Turbo / Polar / QJL보다 앞에 두는 기준선입니다. 별도 preconditioning 없이 원래 좌표계에서 좌표를 균일 코드북에 직접 스냅합니다.")
    render_method_sequence_panel("방법 순서 / 핵심 수식", [("범위 잡기", "공통 범위 q_min, q_max를 정합니다."), ("클리핑", "범위를 넘는 값을 q_min~q_max로 잘라냅니다."), ("균일 코드북", "2^b개 레벨의 선형 코드북을 만듭니다."), ("좌표 스냅", "각 좌표를 가장 가까운 코드북 값으로 붙입니다.")], [r"x_{clip} = \mathrm{clip}(x, q_{min}, q_{max})", r"\mathcal{C} = \mathrm{linspace}(q_{min}, q_{max}, 2^b)", r"\hat x_j = Q_b(x_{clip,j})"], "이 baseline을 먼저 보면 이후 Turbo의 회전형 코드북, Polar의 각도형 코드북, QJL의 sign sketch가 왜 필요한지 더 잘 보입니다.")
    look_box([
        "3D 과정 보기에서 원본 → 클리핑 → 균일 스냅 순서를 먼저 보세요. 마지막 단계는 스냅된 복원점입니다.",
        "오른쪽 단면 예시에서는 좌표쌍이 공통 격자 위로 어떻게 이동하는지 볼 수 있습니다.",
        "이 탭이 기준선이고, 다음 Turbo/Polar/QJL 탭들이 각각 어떤 점을 더 잘 보존하려는지 비교하면 좋습니다.",
    ])
    metric_cards(metrics_baseline, accents=["blue", "red", "blue", "red", "red", "blue"])
    base_left, base_right = st.columns([0.55, 0.45])
    with base_left:
        st.caption("같은 색 점 = 같은 양자화 색상 그룹")
    with base_right:
        baseline_visible_bins = bit_pattern_multiselect("기존 양자화 색상 그룹 on/off", bit_width, key="baseline_visible_bins")
    baseline_process = process_registry["기존 양자화"]
    baseline_process_mask = ensure_nonempty_mask(filter_mask_from_bins(baseline_process["color_ids"], baseline_visible_bins))
    baseline_static_mask = ensure_nonempty_mask(filter_mask_from_bins(baseline_details["color_ids"][static_idx], baseline_visible_bins))
    baseline_ip_mask = ensure_nonempty_mask(filter_mask_from_bins(baseline_details["color_ids"], baseline_visible_bins))
    baseline_process_left, baseline_process_right = st.columns([1.25, 0.95])
    with baseline_process_left:
        base_pick = plotly_chart_pick(process_figure_3d(
            [
                {**stage, "points": stage["points"][baseline_process_mask], "colors": None if stage.get("colors") is None else list(np.asarray(stage["colors"], dtype=object)[baseline_process_mask])}
                for stage in baseline_process["stages"]
            ],
            baseline_process["color_ids"][baseline_process_mask],
            "기존 양자화 3D 과정",
            levels,
            point_indices=process_idx[baseline_process_mask],
            selected_index=inspect_idx,
        ), key="baseline_process_chart")
        update_inspect_vector(base_pick, n_points - 1)
        st.caption("기존 양자화 탭의 색상 그룹 on/off가 3D 과정 보기와 좌표쌍 구름도에도 같이 적용됩니다.")
    with baseline_process_right:
        slice_pick = plotly_chart_pick(pair_cloud_figure(
            baseline_original_cloud[baseline_static_mask],
            baseline_quant_cloud[baseline_static_mask],
            f"기존 양자화 좌표쌍 구름도 · 좌표쌍 {slice_pair}",
            color_ids=baseline_details["color_ids"][static_idx][baseline_static_mask],
            levels=levels,
            point_indices=static_idx[baseline_static_mask],
            selected_index=inspect_idx,
            x_grid=baseline_details["codebook"],
            y_grid=baseline_details["codebook"],
            quant_name="양자화 좌표쌍 (균일 격자)",
            selected_original_pair=baseline_original_pair,
            selected_quant_pair=baseline_quant_pair,
        ), key="baseline_pair_cloud")
        update_inspect_vector(slice_pick, n_points - 1)
        render_pair_summary("기존 양자화 단면 변화 표", baseline_original_pair, baseline_quant_pair, "좌표 변화, pair 반지름, pair 각도 변화량을 함께 보면 baseline 격자 스냅의 성질을 빠르게 설명할 수 있습니다.")
        render_baseline_slice_explainer()
    with st.expander("기존 양자화 추가 그래프", expanded=False):
        row1, row2, row3 = st.columns(3)
        with row1:
            st.plotly_chart(histogram_with_codebook(x[:, 0], baseline_details["codebook"], "좌표 1 분포와 baseline 코드북"), width="stretch", theme=None)
        with row2:
            st.plotly_chart(scatter_true_vs_est(true_ip[baseline_ip_mask], est_ip_baseline[baseline_ip_mask], "기존 양자화 true vs estimated IP", point_colors=color_array_from_ids(baseline_details["color_ids"][baseline_ip_mask], levels)), width="stretch", theme=None)
        with row3:
            st.plotly_chart(error_hist((est_ip_baseline - true_ip)[baseline_ip_mask], "기존 양자화 내적 오차", color="#2563eb"), width="stretch", theme=None)
        baseline_ideal_explainer(bit_width)

with turbo_tab:
    one_line_box("TurboQuant는 회전된 좌표를 공통 코드북에 스냅하는 방법입니다. 여기서 보이는 격자형 그림은 데이터셋별 learned map이 아니라 공통 scalar codebook 스냅을 설명하는 그림입니다.")
    note_card("Paper note", "TurboQuant는 PQ처럼 데이터셋별 codebook을 학습하는 그림보다, random rotation 뒤 좌표별 scalar codebook이 어떻게 작동하는지 보여 주는 data-oblivious / online quantizer로 이해하는 편이 논문 취지에 가깝습니다.")
    render_method_sequence_panel("방법 순서 / 핵심 수식", [("길이/방향 분리", "입력을 norm과 unit direction으로 나눕니다."), ("무작위 회전", "방향 벡터를 random rotation으로 섞습니다."), ("좌표별 코드북", "각 좌표를 common scalar codebook에 독립적으로 스냅합니다."), ("역회전 복원", "역회전 후 원래 norm을 다시 곱합니다.")], [r"u = x / \|x\|_2", r"z = Ru", r"\hat z_j = Q_b(z_j)", r"\hat x = \|x\|_2 R^\top \hat z"], "Turbo는 learned cluster map이 아니라 회전 뒤 좌표별 공통 코드북이 작동하는 구조로 설명하는 편이 정확합니다.")
    look_box([
        "회전 후 각 좌표가 코드북에 스냅되는 모습을 보세요.",
        "3D 과정 보기에서 원본 → 회전 → 코드북 스냅 → 최종 복원 순서를 따라가 보세요.",
        "오른쪽 단면 예시에서는 선택한 좌표쌍이 실제로 어디로 이동했는지 바로 볼 수 있습니다.",
        "추가 그래프에는 좌표쌍 구름도와 내적 비교만 남겨 복잡도를 줄였습니다.",
    ])
    metric_cards(metrics_turbo, accents=["blue", "red", "blue", "red", "red", "blue"])
    turbo_control_left, turbo_control_right = st.columns([0.55, 0.45])
    with turbo_control_left:
        st.caption("같은 색 점 = 같은 양자화 색상 그룹")
    with turbo_control_right:
        turbo_visible_bins = bit_pattern_multiselect("Turbo 색상 그룹 on/off", bit_width, key="turbo_visible_bins")
    turbo_process = process_registry["TurboQuant"]
    turbo_process_mask = ensure_nonempty_mask(filter_mask_from_bins(turbo_process["color_ids"], turbo_visible_bins))
    turbo_static_mask = ensure_nonempty_mask(filter_mask_from_bins(turbo_details["color_ids"][static_idx], turbo_visible_bins))
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        data = turbo_process
        turbo_pick = plotly_chart_pick(
            process_figure_3d(
                [
                    {**stage, "points": stage["points"][turbo_process_mask], "colors": None if stage.get("colors") is None else list(np.asarray(stage["colors"], dtype=object)[turbo_process_mask])}
                    for stage in data["stages"]
                ],
                data["color_ids"][turbo_process_mask],
                "TurboQuant 3D 과정",
                levels,
                point_indices=process_idx[turbo_process_mask],
                selected_index=inspect_idx,
            ),
            key="turbo_process_chart",
        )
        update_inspect_vector(turbo_pick, n_points - 1)
        st.caption("3D 점을 클릭하면 오른쪽 단면 예시 벡터 번호가 그 점으로 바뀝니다. 환경에 따라 3D 클릭이 덜 민감하면 아래 좌표쌍 구름도 클릭은 안정적으로 동작합니다.")
    with c2:
        st.plotly_chart(histogram_with_codebook(turbo_details["rot"].reshape(-1), turbo_details["codebook"], "회전 좌표 분포와 Turbo 코드북"), width="stretch", theme=None)
        st.plotly_chart(pair_vector_compare_figure(turbo_original_pair, turbo_quant_pair, f"Turbo 단면 예시 · 벡터 {inspect_idx}, 좌표쌍 {slice_pair}"), width="stretch", theme=None)
        render_pair_summary("Turbo 좌표 변화 표", turbo_original_pair, turbo_quant_pair, "Turbo는 좌표별 코드북 스냅이 핵심이라, 각 좌표와 pair 길이/각도가 얼마나 바뀌는지 같이 보는 것이 좋습니다.")
        st.caption(f"Turbo 단면 예시는 회전 정규화 좌표에서 보여 줍니다. 그래서 현재 {bit_width}비트면 {2 ** bit_width}×{2 ** bit_width} 코드북 격자가 눈에 더 잘 보입니다.")
        render_turbo_slice_explainer()
    with st.expander("Turbo 추가 그래프", expanded=False):
        extra_left, extra_right = st.columns([1.0, 0.95])
        with extra_left:
            turbo_pair_pick = plotly_chart_pick(
                pair_cloud_figure(
                    turbo_original_cloud[turbo_static_mask],
                    turbo_quant_cloud[turbo_static_mask],
                    f"Turbo 좌표쌍 구름도 · 좌표쌍 {slice_pair}",
                    color_ids=turbo_details["color_ids"][static_idx][turbo_static_mask],
                    levels=levels,
                    point_indices=static_idx[turbo_static_mask],
                    selected_index=inspect_idx,
                    x_grid=turbo_details["codebook"],
                    y_grid=turbo_details["codebook"],
                    quant_name="양자화 좌표쌍 (격자 스냅)",
                    selected_original_pair=turbo_original_pair,
                    selected_quant_pair=turbo_quant_pair,
                ),
                key="turbo_pair_cloud",
            )
            update_inspect_vector(turbo_pair_pick, n_points - 1)
            st.caption(f"세로/가로 점선이 Turbo 코드북 격자이고, 진한 파란/빨간 선은 선택 벡터의 원본/양자화 좌표쌍입니다. 현재 설정에서는 좌표당 {2 ** bit_width}레벨이므로 평면에서는 {2 ** bit_width}×{2 ** bit_width} 격자로 보입니다.")
        with extra_right:
            turbo_ip_mask = ensure_nonempty_mask(filter_mask_from_bins(turbo_details["color_ids"], turbo_visible_bins))
            st.plotly_chart(
                scatter_true_vs_est(
                    true_ip[turbo_ip_mask],
                    est_ip_turbo[turbo_ip_mask],
                    "TurboQuant 실제 vs 복원 내적",
                    point_colors=color_array_from_ids(turbo_details["color_ids"][turbo_ip_mask], levels),
                ),
                width="stretch",
                theme=None,
            )
        turbo_ideal_left, turbo_ideal_right = st.columns([1.0, 0.95])
        with turbo_ideal_left:
            turbo_ideal_kind = st.selectbox("Turbo 이론적 이상 분포 3D", ["구 표면", "구 표면 + 이상치", "구 내부", "구 내부 + 이상치"], index=1, key="turbo_ideal_kind")
            ideal_turbo = make_ideal_geometry(max(360, plot_points), dimension, turbo_ideal_kind, int(seed) + 7001)
            _, ideal_turbo_details = turbo_quantize_mse(ideal_turbo, bit_width, precondition, int(seed) + 7101)
            ideal_turbo_ref = np.vstack([ideal_turbo, ideal_turbo_details["rot_scaled"]])
            ideal_turbo_mean, ideal_turbo_basis = fit_projector(ideal_turbo_ref, projection_mode, int(seed) + 7201, 3)
            ideal_turbo_proj = apply_projector(ideal_turbo, ideal_turbo_mean, ideal_turbo_basis)
            ideal_turbo_mask = ensure_nonempty_mask(filter_mask_from_bins(ideal_turbo_details["color_ids"], turbo_visible_bins))
            st.plotly_chart(
                colored_map_3d(
                    ideal_turbo_proj[ideal_turbo_mask],
                    f"Turbo 이론적 이상 분포 · {turbo_ideal_kind}",
                    ideal_turbo_details["color_ids"][ideal_turbo_mask],
                    levels,
                    name="Turbo 이상 분포",
                ),
                width="stretch",
                theme=None,
            )
        with turbo_ideal_right:
            st.markdown('<div class="paper-card"><strong>Turbo 이상 분포 메모</strong><br>이 그림은 데이터셋을 따로 학습한 오프라인 맵이라기보다, 무작위 회전 뒤 좌표별 코드북 스냅이 잘 설명되는 이상화 분포를 3D로 단순화한 보기입니다.</div>', unsafe_allow_html=True)
            turbo_ideal_explainer(bit_width)
            st.plotly_chart(histogram_with_codebook(ideal_turbo_details["rot"].reshape(-1), ideal_turbo_details["codebook"], "이상 분포의 회전 좌표와 Turbo 코드북"), width="stretch", theme=None)

with polar_tab:
    one_line_box("PolarQuant는 좌표를 반지름과 각도로 바꾼 뒤 각도를 양자화합니다. 앱의 방사형 bin은 각도 스냅 직관용이며, 논문 구현은 분포 기반 optimized codebook과 level-dependent bit allocation을 사용합니다.")
    note_card("Paper note", "현재 방사형 그림은 이해를 돕기 위한 단순화 데모입니다. 논문 구현은 random preconditioning 뒤 각도 분포를 바탕으로 optimized codebook을 만들고, 실험에서는 4-level 재귀 구조와 첫 레벨 4비트 / 이후 2비트 배분을 사용합니다.")
    render_polar_paper_panel(first_bits=int(polar_details["first_bits"][0]), upper_bits=int(polar_details["upper_bits"][0]), bit_width=bit_width)
    look_box([
        "단면 예시에서 원본 각도 θ와 양자화 각도 θ̂를 바로 비교해 보세요.",
        "3D 과정 보기에서는 회전 → 각도 스냅 → 최종 복원 순서로 보시면 됩니다.",
        "오차 히스토그램과 내적 비교는 접어 두고 필요할 때만 펼치도록 정리했습니다.",
    ])
    metric_cards(metrics_polar, accents=["red", "blue", "red", "red", "red", "blue"])
    first_bits = int(polar_details["first_bits"][0])
    upper_bits = int(polar_details["upper_bits"][0])
    st.markdown(f'<span class="metric-chip">첫 레벨 비트 수 = {first_bits}</span><span class="metric-chip">상위 레벨 비트 수 = {upper_bits}</span><span class="metric-chip">단면 좌표쌍 = {slice_pair}</span>', unsafe_allow_html=True)
    polar_control_left, polar_control_right = st.columns([0.55, 0.45])
    with polar_control_left:
        st.caption("같은 색 점 = 같은 양자화 색상 그룹")
    with polar_control_right:
        polar_visible_bins = bit_pattern_multiselect("Polar 색상 그룹 on/off", bit_width, key="polar_visible_bins")
    polar_process = process_registry["PolarQuant"]
    polar_process_mask = ensure_nonempty_mask(filter_mask_from_bins(polar_process["color_ids"], polar_visible_bins))
    polar_static_mask = ensure_nonempty_mask(filter_mask_from_bins(polar_details["color_ids"][static_idx], polar_visible_bins))
    left, right = st.columns([1.15, 0.95])
    with left:
        data = polar_process
        polar_pick = plotly_chart_pick(
            process_figure_3d(
                [
                    {**stage, "points": stage["points"][polar_process_mask], "colors": None if stage.get("colors") is None else list(np.asarray(stage["colors"], dtype=object)[polar_process_mask])}
                    for stage in data["stages"]
                ],
                data["color_ids"][polar_process_mask],
                "PolarQuant 3D 과정",
                levels,
                point_indices=process_idx[polar_process_mask],
                selected_index=inspect_idx,
            ),
            key="polar_process_chart",
        )
        update_inspect_vector(polar_pick, n_points - 1)
        st.caption("3D 점을 클릭하면 오른쪽 단면 예시 벡터 번호가 그 점으로 바뀝니다. 환경에 따라 3D 클릭이 덜 민감하면 아래 좌표쌍 구름도 클릭은 안정적으로 동작합니다.")
    with right:
        st.plotly_chart(slice_geometry_figure(polar_original_pair, polar_quant_pair, f"Polar 단면 예시 · 벡터 {inspect_idx}, 좌표쌍 {slice_pair}"), width="stretch", theme=None)
        render_pair_summary("Polar 좌표 / 반지름 / 각도 변화 표", polar_original_pair, polar_quant_pair, "Polar에서는 각도 스냅이 핵심이지만, 상위 레벨 조합 때문에 pair 반지름도 함께 달라질 수 있습니다.")
        polar_pair_pick = plotly_chart_pick(
            pair_cloud_figure(
                polar_original_cloud[polar_static_mask],
                polar_quant_cloud[polar_static_mask],
                f"Polar 좌표쌍 구름도 · 좌표쌍 {slice_pair}",
                color_ids=polar_details["color_ids"][static_idx][polar_static_mask],
                levels=levels,
                point_indices=static_idx[polar_static_mask],
                selected_index=inspect_idx,
                radial_angles=polar_details["first_codebook"],
                quant_name="양자화 좌표쌍 (각도 스냅)",
                selected_original_pair=polar_original_pair,
                selected_quant_pair=polar_quant_pair,
                radius_rings=np.array([pair_radius(polar_original_pair), pair_radius(polar_quant_pair)]),
            ),
            key="polar_pair_cloud",
        )
        update_inspect_vector(polar_pair_pick, n_points - 1)
        st.caption(f"방사형 점선이 Polar 첫 레벨 각도 코드북이고, 동심원은 선택 벡터의 원본/양자화 후 반지름입니다. 지금 설정에서는 첫 레벨 {2 ** first_bits}개 각도 그룹으로 수렴합니다.")
        render_polar_slice_explainer()
    with st.expander("Polar 추가 그래프", expanded=False):
        radius_delta = np.linalg.norm(polar_details["recon_rot"][:, pair_start:pair_end], axis=1) - np.linalg.norm(polar_details["rot"][:, pair_start:pair_end], axis=1)
        bottom_left, bottom_mid, bottom_right = st.columns(3)
        with bottom_left:
            st.plotly_chart(error_hist(polar_details["lvl0_after"] - polar_details["lvl0_before"], "1단계 각도 양자화 오차"), width="stretch", theme=None)
        with bottom_mid:
            st.plotly_chart(error_hist(polar_details["lvllast_after"] - polar_details["lvllast_before"], "깊은 단계 각도 양자화 오차", color=BLUE), width="stretch", theme=None)
        with bottom_right:
            st.plotly_chart(error_hist(radius_delta, "선택 좌표쌍 반지름 변화", color="#0f766e"), width="stretch", theme=None)
        polar_ip_mask = ensure_nonempty_mask(filter_mask_from_bins(polar_details["color_ids"], polar_visible_bins))
        st.plotly_chart(
            scatter_true_vs_est(
                true_ip[polar_ip_mask],
                est_ip_polar[polar_ip_mask],
                "PolarQuant 실제 vs 복원 내적",
                point_colors=color_array_from_ids(polar_details["color_ids"][polar_ip_mask], levels),
            ),
            width="stretch",
            theme=None,
        )
        polar_ideal_left, polar_ideal_right = st.columns([1.0, 0.95])
        with polar_ideal_left:
            polar_ideal_kind = st.selectbox("Polar 이론적 이상 분포 3D", ["구 표면", "구 표면 + 이상치", "구 내부", "구 내부 + 이상치"], index=3, key="polar_ideal_kind")
            ideal_polar = make_ideal_geometry(max(360, plot_points), dimension, polar_ideal_kind, int(seed) + 7301)
            _, ideal_polar_details = polar_quantize(ideal_polar, bit_width, precondition, int(seed) + 7401)
            ideal_polar_ref = np.vstack([ideal_polar, ideal_polar_details["recon_rot"]])
            ideal_polar_mean, ideal_polar_basis = fit_projector(ideal_polar_ref, projection_mode, int(seed) + 7501, 3)
            ideal_polar_proj = apply_projector(ideal_polar, ideal_polar_mean, ideal_polar_basis)
            ideal_polar_mask = ensure_nonempty_mask(filter_mask_from_bins(ideal_polar_details["color_ids"], polar_visible_bins))
            st.plotly_chart(
                colored_map_3d(
                    ideal_polar_proj[ideal_polar_mask],
                    f"Polar 이론적 이상 분포 · {polar_ideal_kind}",
                    ideal_polar_details["color_ids"][ideal_polar_mask],
                    levels,
                    name="Polar 이상 분포",
                ),
                width="stretch",
                theme=None,
            )
        with polar_ideal_right:
            st.markdown('<div class="paper-card"><strong>Polar 이상 분포 메모</strong><br>이 그림은 오프라인 데이터 학습 맵이라기보다, polar 각도 스냅이 잘 설명되는 이상화 분포를 3D로 단순화한 보기입니다.</div>', unsafe_allow_html=True)
            polar_ideal_explainer(bit_width, first_bits, upper_bits)
            st.plotly_chart(error_hist(ideal_polar_details["lvl0_after"] - ideal_polar_details["lvl0_before"], "이상 분포의 1단계 각도 양자화 오차", color=BLUE), width="stretch", theme=None)

with qjl_tab:
    one_line_box("QJL의 핵심은 3D 벡터 복원이 아니라 asymmetric inner-product estimation입니다. 이 탭은 3D보다 sign sketch와 inner-product 추정 품질을 먼저 보이도록 재배치했습니다.")
    note_card("중요", "QJL은 Turbo/Polar처럼 복원형 quantizer로 읽기보다, query에는 JL transform을 적용하고 key는 sign-bit sketch와 norm만 저장해 inner product를 추정하는 방법으로 읽는 편이 논문 취지에 맞습니다.")
    look_box([
        "먼저 sign sketch 패널에서 Sk → sign(Sk) → scaled sketch 흐름을 보세요.",
        "그다음 오른쪽의 true vs estimated inner product와 attention score proxy를 보세요.",
        "아래 설명용 3D에서는 JL proxy → sign bit snap(±1) → surrogate 역투영 흐름을 보세요.",
    ])
    qjl_bias = float(np.mean(est_ip_qjl - true_ip))
    qjl_mae = float(np.mean(np.abs(est_ip_qjl - true_ip)))
    qjl_corr = float(np.corrcoef(true_ip, est_ip_qjl)[0, 1]) if np.std(est_ip_qjl) > EPS and np.std(true_ip) > EPS else 1.0
    qjl_metrics = {"IP bias": qjl_bias, "IP MAE": qjl_mae, "IP corr": qjl_corr, "score MAE": qjl_attn["score_mae"], "score TV": qjl_attn["score_tv"], "Top-k overlap": qjl_attn["topk_overlap"]}
    metric_cards(qjl_metrics, accents=["red", "blue", "blue", "red", "red", "blue"])
    st.caption(f"attention score proxy는 softmax(true IP / T)와 softmax(estimated IP / T)를 비교한 값이며, 현재 T={attn_temperature:.3f}, top-k={attn_top_k} 기준입니다.")
    render_qjl_core_explainer()
    left, right = st.columns([1.08, 0.92])
    with left:
        render_qjl_sketch_panel(x[inspect_idx], inspect_idx, int(seed), int(m_qjl))
    with right:
        qjl_result_tab, qjl_score_tab = st.tabs(["Inner product", "Attention score proxy"])
        with qjl_result_tab:
            st.plotly_chart(scatter_true_vs_est(true_ip, est_ip_qjl, "QJL true vs estimated inner product", name="QJL estimator", point_colors=color_array_from_ids(qjl_details["color_ids"], levels)), width="stretch", theme=None)
            st.plotly_chart(error_hist(est_ip_qjl - true_ip, "QJL inner-product error"), width="stretch", theme=None)
        with qjl_score_tab:
            st.plotly_chart(attention_proxy_figure(qjl_attn["true_scores"], qjl_attn["est_scores"], "QJL attention score proxy"), width="stretch", theme=None)
            score_rows = [
                ["score MAE", f"{qjl_attn['score_mae']:.4f}", "작을수록 좋음"],
                ["score TV", f"{qjl_attn['score_tv']:.4f}", "0에 가까울수록 attention score 분포가 유사"],
                ["IP error variance", f"{qjl_attn['ip_err_var']:.4f}", "작을수록 추정 흔들림이 작음"],
                [f"Top-{attn_top_k} overlap", f"{qjl_attn['topk_overlap']:.4f}", "클수록 중요한 key 순위가 더 잘 유지"],
            ]
            st.markdown(markdown_table(["지표", "값", "해석"], score_rows))
    with st.expander("QJL geometry / sign-space (설명용)", expanded=False):
        data = process_registry["QJL"]
        qjl_sign_stage = qjl_sign_stage_coords(qjl_details["signs"][process_idx])
        note_card("설명용 기하 시각화", "QJL의 실제 저장 표현은 sign(Sk)입니다. 그래서 아래 애니메이션은 원본 → JL proxy → sign bit snap(앞 3bit가 {-1,+1}로 수렴) → surrogate 역투영 흐름을 한 번에 보여 줍니다.")
        qjl_pick = plotly_chart_pick(
            qjl_process_figure_3d(
                data["stages"][0]["points"],
                data["stages"][1]["points"],
                qjl_sign_stage,
                data["stages"][2]["points"],
                data["color_ids"],
                "QJL surrogate / sign-bit process",
                levels,
                point_indices=process_idx,
                selected_index=inspect_idx,
            ),
            key="qjl_process_chart",
        )
        update_inspect_vector(qjl_pick, n_points - 1)
        st.caption("중간 단계에서 점들이 ±1 축으로 모이는 구간이 실제 sign(Sk)에 해당합니다. 마지막 단계는 그 sign sketch를 설명용 surrogate로 다시 본 모습입니다.")
        extra_mid, extra_right = st.columns(2)
        with extra_mid:
            st.plotly_chart(attention_proxy_figure(qjl_attn["true_scores"], turbo_prod_attn["est_scores"], "참고: Turbo + QJL attention proxy"), width="stretch", theme=None)
        with extra_right:
            st.plotly_chart(scatter_true_vs_est(true_ip, est_ip_turbo_prod, "참고: Turbo + QJL true vs estimated IP", point_colors=color_array_from_ids(turbo_prod_details["color_ids"], levels)), width="stretch", theme=None)

with compare_tab:
    one_line_box("비교 탭은 축을 둘로 나눴습니다. Turbo/Polar는 복원·구조 보존 관점, QJL 계열은 inner-product estimation 관점으로 보면 논문 취지와 가장 가깝습니다.")
    render_alignment_reference()
    st.markdown(markdown_table(["방법", "가장 잘 보여야 하는 것", "핵심 질문"], [["기존 양자화", "Cartesian 격자 / 공통 코드북", "아무 구조 변환 없이 quantization하면 어떤 기준선이 생기나"], ["TurboQuant", "좌표 스냅 / 코드북", "좌표별 quantization이 어떻게 일어나나"], ["PolarQuant", "각도 / 반지름 / 재귀 구조", "polar 표현에서 무엇이 양자화되나"], ["QJL", "sign sketch / IP 추정 / attention proxy", "복원이 아니라 무엇을 보존하나"]]))
    compare_geom_tab, compare_ip_tab, compare_hybrid_tab = st.tabs(["복원 / 구조 비교", "내적 / 추정 비교", "하이브리드 메모"])
    with compare_geom_tab:
        look_box([
            "Turbo와 Polar는 먼저 복원·구조 보존 축으로 비교하는 편이 자연스럽습니다.",
            "이 화면에는 같은 축 안에서 Turbo/Polar base와 Turbo/Polar + QJL residual을 함께 넣어 residual correction까지 바로 확인할 수 있게 했습니다.",
            "Turbo는 회전→코드북 스냅→최종 복원, Turbo + QJL은 base reconstruction→잔차 보정→최종 복원까지 읽어 주세요.",
        ])
        st.markdown(markdown_table(["방법", "MSE", "MAE", "Mean cosine", "메인 포인트"], [
            ["기존 양자화", f"{metrics_baseline['MSE']:.4f}", f"{metrics_baseline['MAE']:.4f}", f"{metrics_baseline['Mean cosine']:.4f}", "Cartesian uniform baseline / 격자형 스냅"],
            ["TurboQuant", f"{metrics_turbo['MSE']:.4f}", f"{metrics_turbo['MAE']:.4f}", f"{metrics_turbo['Mean cosine']:.4f}", "공통 scalar codebook / 격자형 스냅"],
            ["TurboQuant + QJL", f"{metrics_turbo_prod['MSE']:.4f}", f"{metrics_turbo_prod['MAE']:.4f}", f"{metrics_turbo_prod['Mean cosine']:.4f}", "base + residual QJL / 논문식 2단계"],
            ["PolarQuant", f"{metrics_polar['MSE']:.4f}", f"{metrics_polar['MAE']:.4f}", f"{metrics_polar['Mean cosine']:.4f}", "recursive polar / 각도형 스냅"],
            ["PolarQuant + QJL", f"{metrics_polar_prod['MSE']:.4f}", f"{metrics_polar_prod['MAE']:.4f}", f"{metrics_polar_prod['Mean cosine']:.4f}", "base + residual QJL / 비교용 하이브리드"],
        ]))
        compare_method = st.selectbox("복원 비교용 3D 방법", ["기존 양자화", "TurboQuant", "TurboQuant + QJL", "PolarQuant", "PolarQuant + QJL"], index=1)
        left, right = st.columns([1.15, 0.95])
        with left:
            data = process_registry[compare_method]
            compare_pick = plotly_chart_pick(process_figure_3d(data["stages"], data["color_ids"], f"{compare_method} 3D 과정", levels, point_indices=process_idx, selected_index=inspect_idx), key="compare_geom_process_chart")
            update_inspect_vector(compare_pick, n_points - 1)
            st.caption("복원 비교 탭에서도 점을 클릭하면 공통 단면 예시 벡터 번호가 업데이트됩니다.")
        with right:
            if compare_method == "기존 양자화":
                st.plotly_chart(pair_cloud_figure(
                    baseline_original_cloud,
                    baseline_quant_cloud,
                    f"기존 양자화 좌표쌍 구름도 · 좌표쌍 {slice_pair}",
                    color_ids=baseline_details["color_ids"][static_idx],
                    levels=levels,
                    point_indices=static_idx,
                    selected_index=inspect_idx,
                    x_grid=baseline_details["codebook"],
                    y_grid=baseline_details["codebook"],
                    quant_name="양자화 좌표쌍 (균일 격자)",
                    selected_original_pair=baseline_original_pair,
                    selected_quant_pair=baseline_quant_pair,
                ), width="stretch", theme=None)
            elif compare_method == "TurboQuant":
                st.plotly_chart(pair_cloud_figure(
                    turbo_original_cloud,
                    turbo_quant_cloud,
                    f"Turbo 좌표쌍 구름도 · 좌표쌍 {slice_pair}",
                    color_ids=turbo_details["color_ids"][static_idx],
                    levels=levels,
                    point_indices=static_idx,
                    selected_index=inspect_idx,
                    x_grid=turbo_details["codebook"],
                    y_grid=turbo_details["codebook"],
                    quant_name="양자화 좌표쌍 (격자 스냅)",
                    selected_original_pair=turbo_original_pair,
                    selected_quant_pair=turbo_quant_pair,
                ), width="stretch", theme=None)
            elif compare_method == "TurboQuant + QJL":
                turbo_prod_original_pair = turbo_prod_details["rot"][inspect_idx, pair_start:pair_end]
                turbo_prod_base_pair = turbo_prod_details["q_rot"][inspect_idx, pair_start:pair_end]
                turbo_prod_original_cloud = turbo_prod_details["rot"][static_idx, pair_start:pair_end]
                turbo_prod_base_cloud = turbo_prod_details["q_rot"][static_idx, pair_start:pair_end]
                st.plotly_chart(pair_cloud_figure(
                    turbo_prod_original_cloud,
                    turbo_prod_base_cloud,
                    f"Turbo + QJL 좌표쌍 구름도 · 좌표쌍 {slice_pair}",
                    color_ids=turbo_prod_details["color_ids"][static_idx],
                    levels=levels,
                    point_indices=static_idx,
                    selected_index=inspect_idx,
                    x_grid=turbo_prod_details["codebook"],
                    y_grid=turbo_prod_details["codebook"],
                    quant_name="base 좌표쌍 (residual QJL 전 스냅)",
                    selected_original_pair=turbo_prod_original_pair,
                    selected_quant_pair=turbo_prod_base_pair,
                ), width="stretch", theme=None)
                st.caption("이 단면도는 Turbo + QJL의 base snap을 보여 줍니다. residual QJL 보정은 왼쪽 3D에서 base reconstruction → 잔차 보정 → 최종 복원 단계로 확인하세요.")
            elif compare_method == "PolarQuant":
                st.plotly_chart(pair_cloud_figure(
                    polar_original_cloud,
                    polar_quant_cloud,
                    f"Polar 좌표쌍 구름도 · 좌표쌍 {slice_pair}",
                    color_ids=polar_details["color_ids"][static_idx],
                    levels=levels,
                    point_indices=static_idx,
                    selected_index=inspect_idx,
                    radial_angles=polar_details["first_codebook"],
                    quant_name="양자화 좌표쌍 (각도 스냅)",
                    selected_original_pair=polar_original_pair,
                    selected_quant_pair=polar_quant_pair,
                    radius_rings=np.array([pair_radius(polar_original_pair), pair_radius(polar_quant_pair)]),
                ), width="stretch", theme=None)
            else:
                polar_prod_original_pair = polar_prod_details["rot"][inspect_idx, pair_start:pair_end]
                polar_prod_base_pair = polar_prod_details["recon_rot"][inspect_idx, pair_start:pair_end]
                polar_prod_original_cloud = polar_prod_details["rot"][static_idx, pair_start:pair_end]
                polar_prod_base_cloud = polar_prod_details["recon_rot"][static_idx, pair_start:pair_end]
                st.plotly_chart(pair_cloud_figure(
                    polar_prod_original_cloud,
                    polar_prod_base_cloud,
                    f"Polar + QJL 좌표쌍 구름도 · 좌표쌍 {slice_pair}",
                    color_ids=polar_prod_details["color_ids"][static_idx],
                    levels=levels,
                    point_indices=static_idx,
                    selected_index=inspect_idx,
                    radial_angles=polar_prod_details["first_codebook"],
                    quant_name="base 좌표쌍 (residual QJL 전 각도 스냅)",
                    selected_original_pair=polar_prod_original_pair,
                    selected_quant_pair=polar_prod_base_pair,
                    radius_rings=np.array([pair_radius(polar_prod_original_pair), pair_radius(polar_prod_base_pair)]),
                ), width="stretch", theme=None)
                st.caption("이 단면도는 Polar + QJL의 base angle snap을 보여 줍니다. residual QJL 보정은 왼쪽 3D에서 base reconstruction → 잔차 보정 → 최종 복원 단계로 확인하세요.")
    with compare_ip_tab:
        look_box([
            "QJL 계열은 복원보다 true vs estimated inner product, bias, correlation으로 읽는 편이 정확합니다.",
            "Turbo + QJL은 논문 친화적 하이브리드이고, Polar + QJL은 비교용 탐색 하이브리드입니다.",
            "y=x에 가까울수록 내적 추정이 잘 된다고 설명하면 발표에서 이해가 빠릅니다.",
        ])
        st.markdown(markdown_table(["방법", "IP MAE", "IP bias", "IP corr", "score TV", "위치"], [
            ["QJL", f"{qjl_mae:.4f}", f"{qjl_bias:.4f}", f"{qjl_corr:.4f}", f"{qjl_attn['score_tv']:.4f}", "Inner-product estimator 본체"],
            ["Turbo + QJL", f"{metrics_turbo_prod['IP MAE']:.4f}", f"{metrics_turbo_prod['IP bias']:.4f}", f"{metrics_turbo_prod['IP corr']:.4f}", f"{turbo_prod_attn['score_tv']:.4f}", "Paper-aligned hybrid"],
            ["Polar + QJL", f"{metrics_polar_prod['IP MAE']:.4f}", f"{metrics_polar_prod['IP bias']:.4f}", f"{metrics_polar_prod['IP corr']:.4f}", f"{polar_prod_attn['score_tv']:.4f}", "Exploratory hybrid"],
        ]))
        left, right = st.columns([1.0, 1.0])
        with left:
            compare_ip_view = st.tabs(["True vs estimated IP", "Attention score proxy"])
            with compare_ip_view[0]:
                fig_multi = go.Figure()
                fig_multi.add_trace(go.Scatter(x=true_ip, y=est_ip_qjl, mode="markers", name="QJL", marker=dict(size=5, opacity=0.65, color="#0f766e")))
                fig_multi.add_trace(go.Scatter(x=true_ip, y=est_ip_turbo_prod, mode="markers", name="Turbo+QJL", marker=dict(size=5, opacity=0.72, color="#1d4ed8")))
                fig_multi.add_trace(go.Scatter(x=true_ip, y=est_ip_polar_prod, mode="markers", name="Polar+QJL", marker=dict(size=5, opacity=0.72, color="#b91c1c")))
                lo = float(min(true_ip.min(), est_ip_qjl.min(), est_ip_turbo_prod.min(), est_ip_polar_prod.min()))
                hi = float(max(true_ip.max(), est_ip_qjl.max(), est_ip_turbo_prod.max(), est_ip_polar_prod.max()))
                fig_multi.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="이상적인 y=x", line=dict(color="#64748b", dash="dash")))
                fig_multi.update_layout(template="plotly_white", title=dict(text="내적 추정 비교: QJL 계열", font=dict(color="black")), height=450, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="실제 내적", yaxis_title="추정 내적", plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"))
                st.plotly_chart(fig_multi, width="stretch", theme=None)
            with compare_ip_view[1]:
                score_compare = go.Figure()
                score_compare.add_trace(go.Bar(x=["QJL", "Turbo+QJL", "Polar+QJL"], y=[qjl_attn['score_tv'], turbo_prod_attn['score_tv'], polar_prod_attn['score_tv']], name="score TV"))
                score_compare.add_trace(go.Bar(x=["QJL", "Turbo+QJL", "Polar+QJL"], y=[qjl_attn['topk_overlap'], turbo_prod_attn['topk_overlap'], polar_prod_attn['topk_overlap']], name=f"Top-{attn_top_k} overlap"))
                score_compare.update_layout(template="plotly_white", title=dict(text="attention score proxy 비교", font=dict(color="black")), barmode="group", height=430, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="방법", yaxis_title="값", plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"))
                st.plotly_chart(score_compare, width="stretch", theme=None)
                st.plotly_chart(attention_proxy_figure(qjl_attn['true_scores'], qjl_attn['est_scores'], "QJL attention proxy 상세"), width="stretch", theme=None)
        with right:
            compare_ip_method = st.selectbox("내적 비교용 3D 방법", ["QJL", "Turbo + QJL", "Polar + QJL"], index=1)
            data = process_registry[compare_ip_method]
            compare_ip_pick = plotly_chart_pick(process_figure_3d(data["stages"], data["color_ids"], f"{compare_ip_method} geometry proxy", levels, point_indices=process_idx, selected_index=inspect_idx), key="compare_ip_process_chart")
            update_inspect_vector(compare_ip_pick, n_points - 1)
            st.caption("이 3D 그림은 QJL 계열의 sign sketch 흐름과 하이브리드의 base→residual→final 복원 경로를 설명하는 보조 시각화입니다. 핵심 평가는 왼쪽의 inner product / attention score 그래프입니다.")
        with st.expander("내적 비교 추가 그래프", expanded=False):
            bottom_left, bottom_mid, bottom_right = st.columns(3)
            with bottom_left:
                st.plotly_chart(error_hist(est_ip_qjl - true_ip, "QJL 내적 오차", color="#0f766e"), width="stretch", theme=None)
            with bottom_mid:
                st.plotly_chart(error_hist(est_ip_turbo_prod - true_ip, "Turbo + QJL 내적 오차", color=BLUE), width="stretch", theme=None)
            with bottom_right:
                st.plotly_chart(error_hist(est_ip_polar_prod - true_ip, "Polar + QJL 내적 오차", color=RED), width="stretch", theme=None)
    with compare_hybrid_tab:
        note_card("하이브리드 읽는 법", "Turbo + QJL은 논문 친화적 2단계 구조이고, Polar + QJL은 발표/비교를 위한 탐색 하이브리드입니다.")
        pipeline_box("하이브리드 비교", [("Base quantizer", "Turbo 또는 Polar base를 먼저 적용합니다."), ("Residual", "원본 - base 복원값 차이를 residual로 봅니다."), ("Residual QJL", "residual에 sign sketch를 적용합니다."), ("Final merge", "base 복원 + residual_hat 을 합쳐 최종점을 만듭니다.")])
        st.markdown(markdown_table(["방법", "분류", "발표할 때 이렇게 말하기"], [
            ["Turbo + QJL", "Paper-aligned hybrid", "Turbo MSE base가 구조를 먼저 잡고, residual만 1-bit QJL로 보정한다고 설명하면 됩니다."],
            ["Polar + QJL", "Exploratory hybrid", "PolarQuant 논문 원안은 아니고, base quantizer를 바꿨을 때 inner-product 품질이 어떻게 달라지는지 보는 비교용이라고 말하면 안전합니다."],
        ]))

with reference_tab:
    one_line_box("이 탭은 발표 때 자주 나오는 질문인 ‘이 지표가 뭔가요?’와 ‘3D 투영 방식은 왜 다르죠?’를 한곳에 모은 해설 탭입니다.")
    look_box([
        "지표 설명 expander에서는 각 카드 점수의 공식, 의미, 좋은 방향을 한 번에 볼 수 있습니다.",
        "투영 방식 expander에서는 Random projection / PCA / First 3 coordinates가 화면에 어떤 차이를 만드는지 바로 비교할 수 있습니다.",
        "여기서의 투영 방식은 시각화 도구이며, 양자화 알고리즘 자체와는 구분해서 보는 것이 좋습니다.",
    ])
    render_metric_reference()
    render_projection_reference(x[static_idx], x_turbo[static_idx], turbo_details["color_ids"][static_idx], levels, int(seed))

st.markdown("---")
st.markdown("**요약:** TurboQuant는 좌표 스냅, PolarQuant는 각도 스냅, QJL은 1-bit 내적 스케치, Turbo+QJL은 논문식 하이브리드, Polar+QJL은 비교용 탐색 하이브리드로 이해하면 됩니다.")
