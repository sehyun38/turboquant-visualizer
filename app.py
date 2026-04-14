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
            background: {BLUE};
            color: white;
            border: 1px solid {BLUE};
            border-radius: 12px;
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
            """
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
            """
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
            """
**핵심 한 줄**
- QJL의 본질은 **벡터 복원**이 아니라 **asymmetric inner-product estimation** 입니다.

**무엇을 저장하나**
- **query**: JL transform을 거친 실수 projection `Sq`
- **key**: sign-bit sketch `sign(Sk)` 와 key norm
- **value**: 논문 맥락에서는 별도 **token-wise quantization** 으로 처리

**직관 공식**
- `q \mapsto Sq`
- `k \mapsto QJL(S,k) = sign(Sk) \cdot ||k||_2 / \sqrt{m}`
- 이 둘의 내적으로 `\langle q, k \rangle` 를 추정

**이 탭을 읽는 법**
- 왼쪽 3D는 **JL 투영 → sign sketch** 흐름을 보여 주는 설명용 그림입니다.
- 진짜 평가는 오른쪽의 **실제 vs 추정 내적**, 그리고 아래의 **IP bias / IP MAE / IP corr** 로 보는 편이 정확합니다.
- 그래서 QJL은 Turbo/Polar와 같은 복원형 quantizer 비교축으로 놓기보다, **내적 추정기 축**으로 분리해서 설명하는 것이 논문 취지에 더 가깝습니다.
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
def discrete_palette(n: int) -> List[str]:
    if n <= 1:
        return [RED]
    colors = []
    import colorsys
    for i in range(n):
        h = i / max(1, n)
        r, g, b = colorsys.hsv_to_rgb(h, 0.78, 0.92)
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
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



def turbo_quantize_mse(x: np.ndarray, bits: int, precondition: bool, seed: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    d = x.shape[1]
    levels = max(1, 2 ** bits)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + EPS
    x_unit = x / norms
    rot = random_orthogonal(d, seed + 101) if precondition else np.eye(d)
    x_rot = x_unit @ rot.T
    codebook = turbo_codebook(d, bits, seed + 202)
    xq_rot, idx = quantize_by_codebook(x_rot, codebook)
    xq_unit = xq_rot @ rot
    xq = xq_unit * norms
    details = {
        "rot": x_rot,
        "rot_scaled": x_rot * norms,
        "q_rot": xq_rot,
        "q_unit": xq_unit,
        "norms": norms[:, 0],
        "codebook": codebook,
        "indices": idx,
        "color_ids": hash_ids(idx, levels),
        "rotation": rot,
    }
    return xq, details



def turbo_quantize_prod(x: np.ndarray, bits: int, precondition: bool, seed: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    levels = max(1, 2 ** bits)
    base_bits = max(1, bits - 1)
    x_base, details = turbo_quantize_mse(x, base_bits, precondition, seed)
    residual = x - x_base
    d = x.shape[1]
    s = gaussian_sketch(d, d, seed + 303)
    qjl = np.sign(residual @ s.T)
    qjl[qjl == 0] = 1
    gamma = np.linalg.norm(residual, axis=1, keepdims=True)
    residual_hat = math.sqrt(math.pi / 2.0) / d * gamma * (qjl @ s)
    x_hat = x_base + residual_hat
    details.update(
        {
            "base": x_base,
            "residual": residual,
            "qjl_sign": qjl,
            "gamma": gamma[:, 0],
            "sketch": s,
            "residual_hat": residual_hat,
            "base_bits": np.array([base_bits]),
            "color_ids": (details.get("color_ids", np.zeros(len(x), dtype=int)) + hash_ids((qjl > 0).astype(int), levels)) % levels,
        }
    )
    return x_hat, details


# -----------------------------
# PolarQuant helpers
# -----------------------------



def polar_forward_single(x: np.ndarray) -> Tuple[float, List[np.ndarray]]:
    cur = x.astype(float).copy()
    levels: List[np.ndarray] = []
    first_angles = []
    pair_radii = []
    for j in range(0, len(cur), 2):
        a, b = cur[j], cur[j + 1]
        first_angles.append(np.mod(np.arctan2(b, a), 2 * np.pi))
        pair_radii.append(math.hypot(a, b))
    levels.append(np.array(first_angles, dtype=float))
    radii = np.array(pair_radii, dtype=float)
    while len(radii) > 1:
        angles = []
        next_radii = []
        for j in range(0, len(radii), 2):
            left, right = radii[j], radii[j + 1]
            angles.append(math.atan2(right, left))
            next_radii.append(math.hypot(left, right))
        levels.append(np.array(angles, dtype=float))
        radii = np.array(next_radii, dtype=float)
    return float(radii[0]), levels



def polar_inverse_single(radius: float, levels: List[np.ndarray]) -> np.ndarray:
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



def polar_quantize(x: np.ndarray, bits: int, precondition: bool, seed: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    d = x.shape[1]
    levels = max(1, 2 ** bits)
    rot = random_orthogonal(d, seed + 404) if precondition else np.eye(d)
    x_rot = x @ rot.T
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
        reconstructed_rot.append(polar_inverse_single(radius, q_levels))
    x_hat_rot = np.vstack(reconstructed_rot)
    x_hat = x_hat_rot @ rot
    details = {
        "rot": x_rot,
        "recon_rot": x_hat_rot,
        "first_codebook": first_codebook,
        "upper_codebook": upper_codebook,
        "lvl0_before": np.array(lvl_before.get(0, [])),
        "lvl0_after": np.array(lvl_after.get(0, [])),
        "lvllast_before": np.array(lvl_before.get(int(math.log2(d)) - 1, [])),
        "lvllast_after": np.array(lvl_after.get(int(math.log2(d)) - 1, [])),
        "first_bits": np.array([first_bits]),
        "upper_bits": np.array([upper_bits]),
        "color_ids": np.array(color_ids, dtype=int),
        "rotation": rot,
    }
    return x_hat, details



def polar_quantize_prod(x: np.ndarray, bits: int, precondition: bool, seed: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    levels = max(1, 2 ** bits)
    base_bits = max(1, bits - 1)
    x_base, details = polar_quantize(x, base_bits, precondition, seed)
    residual = x - x_base
    d = x.shape[1]
    s = gaussian_sketch(d, d, seed + 909)
    qjl = np.sign(residual @ s.T)
    qjl[qjl == 0] = 1
    gamma = np.linalg.norm(residual, axis=1, keepdims=True)
    residual_hat = math.sqrt(math.pi / 2.0) / d * gamma * (qjl @ s)
    x_hat = x_base + residual_hat
    details.update(
        {
            "base": x_base,
            "residual": residual,
            "qjl_sign": qjl,
            "gamma": gamma[:, 0],
            "sketch": s,
            "residual_hat": residual_hat,
            "base_bits": np.array([base_bits]),
            "color_ids": (details.get("color_ids", np.zeros(len(x), dtype=int)) + hash_ids((qjl > 0).astype(int), levels)) % levels,
        }
    )
    return x_hat, details


# -----------------------------
# QJL helpers
# -----------------------------



def qjl_quantize(x: np.ndarray, q: np.ndarray, m: int, seed: int, bits: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    d = x.shape[1]
    levels = max(1, 2 ** bits)
    s = gaussian_sketch(m, d, seed + 505)
    signed = np.sign(x @ s.T)
    signed[signed == 0] = 1
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    sq = s @ q
    ip_est = math.sqrt(math.pi / 2.0) / m * norms[:, 0] * (signed @ sq)
    x_hat = math.sqrt(math.pi / 2.0) / m * norms * (signed @ s)
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
        "m": np.array([m]),
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
            marker=dict(size=3.6, opacity=0.34, color=BLUE)
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
    original_3d: np.ndarray,
    rotated_3d: np.ndarray,
    quantized_3d: np.ndarray,
    color_ids: np.ndarray,
    title: str,
    levels: int,
    stage1_label: str = "중간 단계",
    stage2_label: str = "최종 양자화",
    point_indices: Optional[np.ndarray] = None,
    selected_index: Optional[int] = None,
) -> go.Figure:
    colors_final = color_array_from_ids(color_ids, levels)
    point_indices = np.arange(len(original_3d), dtype=int) if point_indices is None else np.asarray(point_indices, dtype=int)
    stages = [("원본", original_3d, [BLUE] * len(original_3d))]
    for t in np.linspace(0.2, 1.0, 5):
        cur = interpolate_points(original_3d, rotated_3d, float(t))
        stages.append((f"{stage1_label} {int(round(t * 100))}%", cur, [BLUE] * len(cur)))
    for t in np.linspace(0.2, 1.0, 5):
        cur = interpolate_points(rotated_3d, quantized_3d, float(t))
        stages.append((f"{stage2_label} {int(round(t * 100))}%", cur, colors_final))

    all_pts = np.vstack([original_3d, rotated_3d, quantized_3d])
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
            go.Scatter3d(
                x=quantized_3d[:, 0], y=quantized_3d[:, 1], z=quantized_3d[:, 2],
                mode="markers",
                name="최종 상태",
                customdata=point_indices[:, None],
                marker=dict(size=3.2, opacity=0.18, color=colors_final),
                hovertemplate="벡터 #%{customdata[0]}<extra></extra>",
            ),
            go.Scatter3d(
                x=original_3d[:, 0], y=original_3d[:, 1], z=original_3d[:, 2],
                mode="markers",
                name="원본 기준",
                customdata=point_indices[:, None],
                marker=dict(size=3.0, opacity=0.16, color=BLUE),
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
                    mode="markers",
                    name=f"선택 벡터 #{int(selected_index)}",
                    customdata=point_indices[selected_mask, None],
                    marker=dict(size=8.5, opacity=1.0, color="#111827", line=dict(width=2.0, color="#f59e0b")),
                )
            )
        frames.append(go.Frame(name=label, data=frame_data, traces=list(range(len(frame_data)))))

    data = [
        go.Scatter3d(x=quantized_3d[:, 0], y=quantized_3d[:, 1], z=quantized_3d[:, 2], mode="markers", name="최종 상태", customdata=point_indices[:, None], marker=dict(size=3.2, opacity=0.18, color=colors_final), hovertemplate="벡터 #%{customdata[0]}<extra></extra>"),
        go.Scatter3d(x=original_3d[:, 0], y=original_3d[:, 1], z=original_3d[:, 2], mode="markers", name="원본 기준", customdata=point_indices[:, None], marker=dict(size=3.0, opacity=0.16, color=BLUE), hovertemplate="벡터 #%{customdata[0]}<extra></extra>"),
        go.Scatter3d(x=quantized_3d[:, 0], y=quantized_3d[:, 1], z=quantized_3d[:, 2], mode="markers", name="현재 단계", customdata=point_indices[:, None], marker=dict(size=4.5, opacity=0.94, color=colors_final, line=dict(width=0.45, color="white")), hovertemplate="벡터 #%{customdata[0]}<extra></extra>"),
    ]
    if np.any(selected_mask):
        sel_pts = quantized_3d[selected_mask]
        data.append(
            go.Scatter3d(
                x=sel_pts[:, 0], y=sel_pts[:, 1], z=sel_pts[:, 2],
                mode="markers",
                name=f"선택 벡터 #{int(selected_index)}",
                customdata=point_indices[selected_mask, None],
                marker=dict(size=8.5, opacity=1.0, color="#111827", line=dict(width=2.0, color="#f59e0b")),
                hovertemplate="벡터 #%{customdata[0]}<extra></extra>",
            )
        )

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
            "steps": [
                {"label": label, "method": "animate", "args": [[label], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                for label, _, _ in stages
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
    fig.add_trace(go.Scatter(x=[0, original_pair[0]], y=[0, original_pair[1]], mode="lines+markers", name="원본", line=dict(width=3, color=BLUE), marker=dict(size=8, color=BLUE)))
    fig.add_trace(go.Scatter(x=[0, quant_pair[0]], y=[0, quant_pair[1]], mode="lines+markers", name="양자화 후", line=dict(width=3, dash="dash", color=RED), marker=dict(size=8, color=RED)))
    fig.add_trace(go.Scatter(x=arc_o_x, y=arc_o_y, mode="lines", line=dict(color=BLUE, width=2), name="원본 각도 θ"))
    fig.add_trace(go.Scatter(x=arc_q_x, y=arc_q_y, mode="lines", line=dict(color=RED, width=2, dash="dot"), name="양자화 각도 θ̂"))
    fig.add_annotation(x=float(arc_o_x[-1]), y=float(arc_o_y[-1]), text="θ", showarrow=False, font=dict(color=BLUE, size=14))
    fig.add_annotation(x=float(arc_q_x[-1]), y=float(arc_q_y[-1]), text="θ̂", showarrow=False, font=dict(color=RED, size=14))
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=420, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="좌표축 1", yaxis_title="좌표축 2", xaxis=dict(range=[-1.2 * max_abs, 1.2 * max_abs]), yaxis=dict(range=[-1.2 * max_abs, 1.2 * max_abs], scaleanchor="x", scaleratio=1), plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"))
    return fig


def pair_vector_compare_figure(original_pair: np.ndarray, quant_pair: np.ndarray, title: str) -> go.Figure:
    max_abs = float(max(np.max(np.abs(original_pair)), np.max(np.abs(quant_pair)), 1.0))
    fig = go.Figure()
    fig.add_hline(y=0.0, line_color="#cbd5e1", line_width=1)
    fig.add_vline(x=0.0, line_color="#cbd5e1", line_width=1)
    fig.add_trace(go.Scatter(x=[0, original_pair[0]], y=[0, original_pair[1]], mode="lines+markers", name="원본 벡터", line=dict(width=3, color=BLUE), marker=dict(size=8, color=BLUE)))
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
    fig.add_trace(go.Scatter(x=original_pairs[:, 0], y=original_pairs[:, 1], mode="markers", customdata=point_indices[:, None], marker=dict(size=5, opacity=0.28, color=BLUE), name=original_name, hovertemplate="벡터 #%{customdata[0]}<extra></extra>"))
    fig.add_trace(go.Scatter(x=quant_pairs[:, 0], y=quant_pairs[:, 1], mode="markers", customdata=point_indices[:, None], marker=dict(size=5.5, opacity=0.9, color=q_colors, line=dict(width=0.3, color="white")), name=quant_name, hovertemplate="벡터 #%{customdata[0]}<extra></extra>"))
    if selected_index is not None:
        selected_mask = point_indices == int(selected_index)
        if np.any(selected_mask):
            fig.add_trace(go.Scatter(x=quant_pairs[selected_mask, 0], y=quant_pairs[selected_mask, 1], mode="markers", customdata=point_indices[selected_mask, None], marker=dict(size=10.5, color="#111827", line=dict(width=2.0, color="#f59e0b")), name=f"선택 벡터 #{int(selected_index)}", hovertemplate="벡터 #%{customdata[0]}<extra></extra>"))
    if selected_original_pair is not None:
        op = np.asarray(selected_original_pair, dtype=float)
        fig.add_trace(go.Scatter(x=[0.0, op[0]], y=[0.0, op[1]], mode="lines", line=dict(width=2, color=BLUE), name="선택 원본 반지름", hoverinfo="skip"))
    if selected_quant_pair is not None:
        qp = np.asarray(selected_quant_pair, dtype=float)
        fig.add_trace(go.Scatter(x=[0.0, qp[0]], y=[0.0, qp[1]], mode="lines", line=dict(width=2, color=RED, dash="dash"), name="선택 양자화 반지름", hoverinfo="skip"))
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=420, margin=dict(l=10, r=10, t=40, b=10), clickmode="event+select", xaxis_title="좌표축 1", yaxis_title="좌표축 2", yaxis=dict(scaleanchor="x", scaleratio=1), plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"), selectionrevision=str(selected_index) if selected_index is not None else "none")
    return fig


def bit_pattern_label(bit_id: int, bits: int) -> str:
    return f"{int(bit_id):0{max(1, bits)}b}"


def bit_pattern_multiselect(label: str, bits: int, key: str) -> List[int]:
    options = list(range(max(1, 2 ** bits)))
    mode = st.radio(
        label,
        ["전체", "bin 하나만 보기", "직접 여러 개 고르기"],
        horizontal=True,
        key=f"{key}_mode",
        help="하나만 보고 싶을 때는 'bin 하나만 보기'를 쓰면, 여러 개를 하나씩 지울 필요가 없습니다.",
    )
    if mode == "전체":
        st.caption(f"현재 {len(options)}개 bin 전체 표시")
        return options
    if mode == "bin 하나만 보기":
        focus = st.selectbox(
            "집중해서 볼 bin",
            options=options,
            index=0,
            format_func=lambda idx: f"{bit_pattern_label(int(idx), bits)}",
            key=f"{key}_single",
        )
        st.caption(f"현재 1개 bin만 표시: {bit_pattern_label(int(focus), bits)}")
        return [int(focus)]
    selected = st.multiselect(
        "직접 표시할 bin 선택",
        options=options,
        default=options,
        format_func=lambda idx: f"{bit_pattern_label(int(idx), bits)}",
        key=key,
        help="같은 색 점은 같은 양자화 비트 패턴 / bin을 뜻합니다. 표시할 패턴만 남길 수 있습니다.",
    )
    st.caption(f"현재 표시 중: {len(selected or options)} / {len(options)} bins")
    return selected or options


def extract_selected_point_index(event: Any) -> Optional[int]:
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

    if not candidate_points and isinstance(event, dict):
        selection = event.get("selection")
        point_indices = None
        if isinstance(selection, dict):
            point_indices = selection.get("point_indices") or selection.get("pointIndices")
        if point_indices:
            try:
                return int(point_indices[0])
            except Exception:
                pass

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
        point_number = None
        if hasattr(point, "point_number"):
            point_number = point.point_number
        elif isinstance(point, dict):
            point_number = point.get("point_number", point.get("pointNumber", point.get("pointIndex")))
        if point_number is not None:
            try:
                return int(point_number)
            except Exception:
                pass
    return None


def plotly_chart_pick(fig: go.Figure, key: str) -> Optional[int]:
    event = st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        key=key,
        on_select="rerun",
        selection_mode=("points",),
    )
    return extract_selected_point_index(event)


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


def colored_map_3d(points_3d: np.ndarray, title: str, color_ids: np.ndarray, levels: int, name: str = "양자화 bin 맵") -> go.Figure:
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
        st.latex(r"\hat x = \|x\|_2\, R^\top \hat z")
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
                ["QJL", "Paper-aligned", "비대칭 inner-product estimator 관점을 중심에 두고 설명합니다."],
                ["Turbo + QJL", "Paper-aligned hybrid", "Turbo MSE base 뒤 residual에 1-bit QJL을 붙이는 논문식 2단계 구조입니다."],
                ["Polar + QJL", "Exploratory hybrid", "비교/교육용 탐색 하이브리드이며 PolarQuant 논문의 원안은 아닙니다."],
            ]))
        with axis_tab:
            st.markdown(markdown_table(["비교 축", "여기에 두는 방법", "주요 지표"], [
                ["복원 / 구조 보존", "TurboQuant, PolarQuant", "MSE, MAE, Mean cosine, 단면 이동, 코드북 구조"],
                ["내적 / 추정 품질", "QJL, Turbo + QJL, Polar + QJL", "IP bias, IP MAE, IP corr, true-vs-est inner product"],
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
st.caption("흰 배경과 검은 제목을 유지하면서, TurboQuant / PolarQuant / QJL의 핵심 비교를 더 덜 복잡하게 보이도록 정리한 버전입니다.")

distribution_notes = {
    "Gaussian": "가장 기본적인 분포입니다. 좌표별 양자화와 내적 오차를 무난하게 비교할 때 적합합니다.",
    "Gaussian + outliers": "대부분은 중앙에 모여 있고 일부 이상치가 멀리 떨어져 있는 경우입니다. 이상치에 대한 민감도를 보기 좋습니다.",
    "Unit sphere": "모든 점의 길이가 거의 같은 구 표면 데이터입니다. TurboQuant의 회전 후 좌표 분포 해석과 잘 맞는 편입니다.",
    "Sphere shell + outliers": "거의 구 표면에 있으면서 일부 이상치가 더 바깥으로 튀는 경우입니다. 구형 구조 + 이상치를 함께 보고 싶을 때 적합합니다.",
    "Ball + outliers": "구 내부에 점이 퍼져 있고 일부 이상치가 바깥에 있는 경우입니다. PolarQuant의 반지름/각도 관찰에 특히 보기 좋습니다.",
}

with st.sidebar:
    st.header("설정")
    mode = st.radio("앱 모드", ["Balanced", "Paper-faithful"], index=0, help="Balanced는 설명을 우선하고, Paper-faithful은 논문식 해석을 조금 더 강조합니다.")
    n_points = st.slider("벡터 수", 300, 2200, 900, step=100)
    dimension = st.select_slider("차원 d", options=[8, 16, 32, 64, 128], value=32)
    distribution = st.selectbox("데이터 분포", ["Gaussian", "Gaussian + outliers", "Unit sphere", "Sphere shell + outliers", "Ball + outliers"])
    precision = st.selectbox("입력 정밀도 시뮬레이션", ["fp16-like", "fp8-like", "int8-like"])
    bit_width = st.slider("양자화 비트 수", 1, 6, 3, help="각 방법이 한 좌표 또는 한 단계에서 대략 얼마나 촘촘하게 양자화하는지 보는 기준 비트 수입니다.")
    precondition = st.toggle("랜덤 전처리 적용", value=True)
    projection_mode = st.selectbox("3D 공통 투영 방식", ["Random projection", "PCA", "First 3 coordinates"])
    seed = st.number_input("시드", min_value=0, max_value=999999, value=7, step=1)
    plot_points = st.slider("비교용 표시 점 수", 200, 1200, 500, step=100)
    process_points = st.slider("3D 애니메이션 점 수", 40, 220, 100, step=20)
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
    slice_pair = st.slider("단면 좌표쌍 번호", 0, max_pair, key="slice_pair_input", help="i를 고르면 (x[2i], x[2i+1]) 좌표쌍을 2D 단면으로 봅니다.")
    st.caption("단면 예시 벡터 번호 = 한 개 샘플 확대 보기 / 단면 좌표쌍 번호 = (x[2i], x[2i+1]) 2D 보기")
    st.caption(f"선택한 데이터 분포: {distribution_notes[distribution]}")
    st.caption(f"양자화 색상 bin 수 = 2^{bit_width} = {2 ** bit_width}")

if mode == "Paper-faithful" and distribution in {"Gaussian + outliers", "Sphere shell + outliers", "Ball + outliers"}:
    st.info("Paper-faithful 모드에서는 Gaussian / Unit sphere가 논문 해석과 가장 직접적으로 맞습니다. 추가한 이상치 분포는 교육용 시각화에 가깝습니다.")

levels = max(1, 2 ** bit_width)
x = make_data(n_points, dimension, distribution, int(seed))
x = apply_precision_rounding(x, precision)
q = make_query(dimension, distribution, int(seed))

x_turbo, turbo_details = turbo_quantize_mse(x, bit_width, precondition, int(seed))
x_polar, polar_details = polar_quantize(x, bit_width, precondition, int(seed))
m_qjl = dimension if mode == "Paper-faithful" else max(8, dimension // 2)
x_qjl_vis, qjl_details = qjl_quantize(x, q, m_qjl, int(seed), bit_width)
x_turbo_prod, turbo_prod_details = turbo_quantize_prod(x, bit_width, precondition, int(seed))
x_polar_prod, polar_prod_details = polar_quantize_prod(x, bit_width, precondition, int(seed))

metrics_turbo = compute_metrics(x, x_turbo, q)
metrics_polar = compute_metrics(x, x_polar, q)
metrics_qjl_surrogate = compute_metrics(x, x_qjl_vis, q)
metrics_turbo_prod = compute_metrics(x, x_turbo_prod, q)
metrics_polar_prod = compute_metrics(x, x_polar_prod, q)

true_ip = x @ q
est_ip_turbo = x_turbo @ q
est_ip_polar = x_polar @ q
est_ip_qjl = qjl_details["ip_est"]
est_ip_turbo_prod = x_turbo_prod @ q
est_ip_polar_prod = x_polar_prod @ q

method_registry: Dict[str, MethodResult] = {
    "TurboQuant": MethodResult("TurboQuant", x_turbo, metrics_turbo, "무작위 회전 후 좌표별 스칼라 양자화", turbo_details),
    "PolarQuant": MethodResult("PolarQuant", x_polar, metrics_polar, "재귀 polar 변환 후 각도 양자화", polar_details),
    "QJL": MethodResult("QJL", x_qjl_vis, metrics_qjl_surrogate, "내적 추정이 본체이며 3D 복원은 설명용", qjl_details),
    "Turbo + QJL": MethodResult("Turbo + QJL", x_turbo_prod, metrics_turbo_prod, "논문식 2단계: MSE base + residual QJL", turbo_prod_details),
    "Polar + QJL": MethodResult("Polar + QJL", x_polar_prod, metrics_polar_prod, "탐색적 비교용: Polar base + residual QJL", polar_prod_details),
}

static_idx = sample_indices(n_points, plot_points, int(seed) + 1001)
process_idx = sample_indices(n_points, min(process_points, plot_points), int(seed) + 1201)

qjl_mid_state = ((x @ qjl_details["sketch"].T) @ qjl_details["sketch"]) / max(1, qjl_details["sketch"].shape[0])
common_ref = np.vstack([
    x[static_idx],
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

def build_process_projection(mid_state: np.ndarray, final_state: np.ndarray, idx: np.ndarray, seed_offset: int):
    ref = np.vstack([x[idx], mid_state[idx], final_state[idx]])
    mean_p, basis_p = fit_projector(ref, projection_mode, int(seed) + seed_offset, 3)
    return apply_projector(x[idx], mean_p, basis_p), apply_projector(mid_state[idx], mean_p, basis_p), apply_projector(final_state[idx], mean_p, basis_p)

process_registry = {}
for offset, method_name, mid_state, stage1_label, stage2_label in [
    (101, "TurboQuant", turbo_details["rot_scaled"], "회전", "코드북 스냅"),
    (202, "PolarQuant", polar_details["rot"], "회전", "각도 스냅"),
    (303, "Turbo + QJL", turbo_prod_details["rot_scaled"], "회전", "잔차 보정"),
    (404, "Polar + QJL", polar_prod_details["rot"], "회전", "잔차 보정"),
    (505, "QJL", qjl_mid_state, "JL 투영", "부호 복원"),
]:
    orig3, mid3, fin3 = build_process_projection(mid_state, method_registry[method_name].reconstructed, process_idx, offset)
    process_registry[method_name] = {
        "orig": orig3,
        "mid": mid3,
        "final": fin3,
        "color_ids": method_registry[method_name].details["color_ids"][process_idx],
        "stage1_label": stage1_label,
        "stage2_label": stage2_label,
    }

inspect_idx = int(st.session_state.get("inspect_vector_value", inspect_vector))
pair_start = 2 * int(slice_pair)
pair_end = pair_start + 2

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
    "Turbo MSE": metrics_turbo["MSE"],
    "Polar MSE": metrics_polar["MSE"],
    "QJL 내적 MAE": float(np.mean(np.abs(est_ip_qjl - true_ip))),
    "Turbo+QJL 내적 MAE": float(np.mean(np.abs(est_ip_turbo_prod - true_ip))),
    "Polar+QJL 내적 MAE": float(np.mean(np.abs(est_ip_polar_prod - true_ip))),
}
metric_cards(summary_metrics, accents=["blue", "red", "blue", "blue", "red"])

st.markdown(f'<div class="paper-card"><strong>선택한 데이터 분포</strong><br>{distribution_notes[distribution]}</div>', unsafe_allow_html=True)

with st.expander("논문 반영 범위 / 이 앱이 어디까지 paper-faithful 인가", expanded=False):
    st.markdown(
        """
- **TurboQuant**: 무작위 회전 후 좌표별 스칼라 양자화를 적용하는 구조를 반영했습니다. 여기서 보이는 격자형 그림은 **학습된 데이터 맵**이 아니라 **공통 scalar codebook**의 스냅 구조를 보여 주는 시각화입니다.
- **PolarQuant**: 재귀적 polar 변환, 첫 레벨 `[0, 2π)`, 상위 레벨 `[0, π/2]` 구조를 반영했습니다. 다만 앱의 방사형 angle bin은 **직관용 단순화 그림**이며, 논문 구현은 **optimized codebook**과 **level-dependent bit allocation**을 사용합니다.
- **QJL**: 논문의 본체는 **벡터 복원**이 아니라 **비대칭 inner-product estimator**입니다. 따라서 QJL 탭의 3D 그림은 설명용이고, 핵심 평가는 `IP bias / IP MAE / IP corr`입니다.
- **Turbo + QJL**: 먼저 MSE 양자화 후 residual에 1-bit QJL을 붙이는 **논문 친화적 2단계 구조**입니다.
- **Polar + QJL**: 비교/교육용 **탐색 하이브리드**이며 PolarQuant 논문의 원안은 아닙니다.
        """
    )

comparison_rows = [
    ["TurboQuant", f"{metrics_turbo['MSE']:.4f}", f"{metrics_turbo['IP MAE']:.4f}", f"{metrics_turbo['IP bias']:.4f}", "좌표 기반 MSE"],
    ["PolarQuant", f"{metrics_polar['MSE']:.4f}", f"{metrics_polar['IP MAE']:.4f}", f"{metrics_polar['IP bias']:.4f}", "각도 기반"],
    ["QJL", f"{metrics_qjl_surrogate['MSE']:.4f}", f"{float(np.mean(np.abs(est_ip_qjl - true_ip))):.4f}", f"{float(np.mean(est_ip_qjl - true_ip)):.4f}", "내적 우선"],
    ["Turbo + QJL", f"{metrics_turbo_prod['MSE']:.4f}", f"{metrics_turbo_prod['IP MAE']:.4f}", f"{metrics_turbo_prod['IP bias']:.4f}", "논문식 하이브리드"],
    ["Polar + QJL", f"{metrics_polar_prod['MSE']:.4f}", f"{metrics_polar_prod['IP MAE']:.4f}", f"{metrics_polar_prod['IP bias']:.4f}", "비교용 하이브리드"],
]

st.caption("지표 공식과 3D 투영 방식 설명은 아래 `지표 / 투영 해설` 탭에 정리했습니다.")

turbo_tab, polar_tab, qjl_tab, compare_tab, reference_tab = st.tabs(["TurboQuant", "PolarQuant", "QJL", "비교 / 하이브리드", "지표 / 투영 해설"])
with turbo_tab:
    one_line_box("TurboQuant는 회전된 좌표를 공통 코드북에 스냅하는 방법입니다. 여기서 보이는 격자형 그림은 데이터셋별 learned map이 아니라 공통 scalar codebook 스냅을 설명하는 그림입니다.")
    note_card("Paper note", "TurboQuant 그림은 학습된 군집 지도가 아니라 random rotation 뒤 좌표별 scalar codebook이 어떻게 작동하는지 보여 주는 직관용 구조입니다.")
    look_box([
        "3D 과정 보기에서 원본 → 회전 → 코드북 스냅 순서를 따라가 보세요.",
        "오른쪽 단면 예시에서는 선택한 좌표쌍이 실제로 어디로 이동했는지 바로 볼 수 있습니다.",
        "추가 그래프에는 좌표쌍 구름도와 내적 비교만 남겨 복잡도를 줄였습니다.",
    ])
    metric_cards(metrics_turbo, accents=["blue", "red", "blue", "red", "red", "blue"])
    turbo_control_left, turbo_control_right = st.columns([0.55, 0.45])
    with turbo_control_left:
        st.caption("같은 색 점 = 같은 양자화 비트 패턴 / bin")
    with turbo_control_right:
        turbo_visible_bins = bit_pattern_multiselect("Turbo 색상 bin on/off", bit_width, key="turbo_visible_bins")
    turbo_process = process_registry["TurboQuant"]
    turbo_process_mask = ensure_nonempty_mask(filter_mask_from_bins(turbo_process["color_ids"], turbo_visible_bins))
    turbo_static_mask = ensure_nonempty_mask(filter_mask_from_bins(turbo_details["color_ids"][static_idx], turbo_visible_bins))
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        data = turbo_process
        turbo_pick = plotly_chart_pick(
            process_figure_3d(
                data["orig"][turbo_process_mask],
                data["mid"][turbo_process_mask],
                data["final"][turbo_process_mask],
                data["color_ids"][turbo_process_mask],
                "TurboQuant 3D 과정",
                levels,
                stage1_label=data["stage1_label"],
                stage2_label=data["stage2_label"],
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
    note_card("Paper note", "현재 방사형 그림은 이해를 돕기 위한 단순화 데모입니다. 논문 구현은 random preconditioning 뒤 각도 분포를 바탕으로 codebook을 만들고, 실험에서는 4-level 재귀 구조와 첫 레벨 4비트 / 이후 2비트 배분을 사용합니다.")
    look_box([
        "단면 예시에서 원본 각도 θ와 양자화 각도 θ̂를 바로 비교해 보세요.",
        "3D 과정 보기에서는 회전된 상태를 거쳐 최종 양자화점으로 이동합니다.",
        "오차 히스토그램과 내적 비교는 접어 두고 필요할 때만 펼치도록 정리했습니다.",
    ])
    metric_cards(metrics_polar, accents=["red", "blue", "red", "red", "red", "blue"])
    first_bits = int(polar_details["first_bits"][0])
    upper_bits = int(polar_details["upper_bits"][0])
    st.markdown(f'<span class="metric-chip">첫 레벨 비트 수 = {first_bits}</span><span class="metric-chip">상위 레벨 비트 수 = {upper_bits}</span><span class="metric-chip">단면 좌표쌍 = {slice_pair}</span>', unsafe_allow_html=True)
    polar_control_left, polar_control_right = st.columns([0.55, 0.45])
    with polar_control_left:
        st.caption("같은 색 점 = 같은 양자화 비트 패턴 / bin")
    with polar_control_right:
        polar_visible_bins = bit_pattern_multiselect("Polar 색상 bin on/off", bit_width, key="polar_visible_bins")
    polar_process = process_registry["PolarQuant"]
    polar_process_mask = ensure_nonempty_mask(filter_mask_from_bins(polar_process["color_ids"], polar_visible_bins))
    polar_static_mask = ensure_nonempty_mask(filter_mask_from_bins(polar_details["color_ids"][static_idx], polar_visible_bins))
    left, right = st.columns([1.15, 0.95])
    with left:
        data = polar_process
        polar_pick = plotly_chart_pick(
            process_figure_3d(
                data["orig"][polar_process_mask],
                data["mid"][polar_process_mask],
                data["final"][polar_process_mask],
                data["color_ids"][polar_process_mask],
                "PolarQuant 3D 과정",
                levels,
                stage1_label=data["stage1_label"],
                stage2_label=data["stage2_label"],
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
        st.caption(f"방사형 점선이 Polar 첫 레벨 각도 코드북이고, 동심원은 선택 벡터의 원본/양자화 후 반지름입니다. 지금 설정에서는 첫 레벨 {2 ** first_bits}개 각도 bin으로 수렴합니다.")
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
    one_line_box("QJL의 핵심은 3D 벡터 복원이 아니라 asymmetric inner-product estimation입니다. 이 탭의 3D 과정은 JL 투영과 sign sketch 흐름을 보여 주는 설명용 그림입니다.")
    note_card("중요", "QJL은 Turbo/Polar처럼 복원형 quantizer로 읽기보다, query에는 JL transform을 적용하고 key는 sign-bit sketch와 norm만 저장해 inner product를 추정하는 방법으로 읽는 편이 논문 취지에 맞습니다.")
    look_box([
        "왼쪽 3D는 설명용 과정 보기입니다. Play를 누르면 원본에서 다시 시작하고, 기본 화면은 마지막 단계로 보이게 했습니다.",
        "오른쪽 산점도가 y=x에 가까울수록 내적 추정이 잘 되는 것입니다.",
        "QJL은 본질적으로 비대칭 inner-product estimator라는 점을 함께 보세요.",
    ])
    qjl_bias = float(np.mean(est_ip_qjl - true_ip))
    qjl_mae = float(np.mean(np.abs(est_ip_qjl - true_ip)))
    qjl_corr = float(np.corrcoef(true_ip, est_ip_qjl)[0, 1]) if np.std(est_ip_qjl) > EPS and np.std(true_ip) > EPS else 1.0
    qjl_metrics = {"IP bias": qjl_bias, "IP MAE": qjl_mae, "IP corr": qjl_corr, "스케치 차원 m": float(m_qjl), "‖q‖ 평균": float(np.linalg.norm(q)), "‖k‖ 평균": float(np.mean(qjl_details["norms"]))}
    metric_cards(qjl_metrics, accents=["red", "blue", "blue", "blue", "red", "blue"])
    render_qjl_core_explainer()
    left, right = st.columns([1.15, 0.95])
    with left:
        data = process_registry["QJL"]
        qjl_pick = plotly_chart_pick(process_figure_3d(data["orig"], data["mid"], data["final"], data["color_ids"], "QJL 3D 과정 (설명용)", levels, stage1_label=data["stage1_label"], stage2_label=data["stage2_label"], point_indices=process_idx, selected_index=inspect_idx), key="qjl_process_chart")
        update_inspect_vector(qjl_pick, n_points - 1)
        st.caption("QJL 탭도 같은 방식으로 점을 클릭해 단면 예시 벡터 번호를 바꿀 수 있습니다.")
    with right:
        st.plotly_chart(scatter_true_vs_est(true_ip, est_ip_qjl, "QJL 실제 vs 추정 내적", name="QJL 추정기", point_colors=color_array_from_ids(qjl_details["color_ids"], levels)), width="stretch", theme=None)
        st.plotly_chart(error_hist(est_ip_qjl - true_ip, "QJL 추정 오차"), width="stretch", theme=None)
    with st.expander("QJL 추가 그래프", expanded=False):
        extra_left, extra_right = st.columns(2)
        with extra_left:
            st.plotly_chart(scatter_overlay_3d(proj_x, proj_qjl, "QJL 설명용 3D 비교", color_ids=qjl_details["color_ids"][static_idx], levels=levels, recon_name="QJL 설명용 복원"), width="stretch", theme=None)
        with extra_right:
            st.plotly_chart(scatter_true_vs_est(true_ip, est_ip_turbo_prod, "참고: Turbo + QJL 실제 vs 복원 내적", point_colors=color_array_from_ids(turbo_prod_details["color_ids"], levels)), width="stretch", theme=None)

with compare_tab:
    one_line_box("비교 탭은 축을 둘로 나눴습니다. Turbo/Polar는 복원·구조 보존 관점, QJL 계열은 inner-product estimation 관점으로 보면 논문 취지와 가장 가깝습니다.")
    render_alignment_reference()
    compare_geom_tab, compare_ip_tab, compare_hybrid_tab = st.tabs(["복원 / 구조 비교", "내적 / 추정 비교", "하이브리드 메모"])
    with compare_geom_tab:
        look_box([
            "Turbo와 Polar는 먼저 복원·구조 보존 축으로 비교하는 편이 자연스럽습니다.",
            "Turbo는 격자형 스냅, Polar는 각도형 스냅이라는 차이를 눈으로 먼저 보여 주세요.",
            "QJL은 이 축에서 보조로만 언급하고, 본격 비교는 다음 탭에서 하는 편이 덜 헷갈립니다.",
        ])
        st.markdown(markdown_table(["방법", "MSE", "MAE", "Mean cosine", "메인 포인트"], [
            ["TurboQuant", f"{metrics_turbo['MSE']:.4f}", f"{metrics_turbo['MAE']:.4f}", f"{metrics_turbo['Mean cosine']:.4f}", "공통 scalar codebook / 격자형 스냅"],
            ["PolarQuant", f"{metrics_polar['MSE']:.4f}", f"{metrics_polar['MAE']:.4f}", f"{metrics_polar['Mean cosine']:.4f}", "recursive polar / 각도형 스냅"],
        ]))
        compare_method = st.selectbox("복원 비교용 3D 방법", ["TurboQuant", "PolarQuant"], index=0)
        left, right = st.columns([1.15, 0.95])
        with left:
            data = process_registry[compare_method]
            compare_pick = plotly_chart_pick(process_figure_3d(data["orig"], data["mid"], data["final"], data["color_ids"], f"{compare_method} 3D 과정", levels, stage1_label=data["stage1_label"], stage2_label=data["stage2_label"], point_indices=process_idx, selected_index=inspect_idx), key="compare_geom_process_chart")
            update_inspect_vector(compare_pick, n_points - 1)
            st.caption("복원 비교 탭에서도 점을 클릭하면 공통 단면 예시 벡터 번호가 업데이트됩니다.")
        with right:
            if compare_method == "TurboQuant":
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
            else:
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
    with compare_ip_tab:
        look_box([
            "QJL 계열은 복원보다 true vs estimated inner product, bias, correlation으로 읽는 편이 정확합니다.",
            "Turbo + QJL은 논문 친화적 하이브리드이고, Polar + QJL은 비교용 탐색 하이브리드입니다.",
            "y=x에 가까울수록 내적 추정이 잘 된다고 설명하면 발표에서 이해가 빠릅니다.",
        ])
        st.markdown(markdown_table(["방법", "IP MAE", "IP bias", "IP corr", "위치"], [
            ["QJL", f"{qjl_mae:.4f}", f"{qjl_bias:.4f}", f"{qjl_corr:.4f}", "Inner-product estimator 본체"],
            ["Turbo + QJL", f"{metrics_turbo_prod['IP MAE']:.4f}", f"{metrics_turbo_prod['IP bias']:.4f}", f"{metrics_turbo_prod['IP corr']:.4f}", "Paper-aligned hybrid"],
            ["Polar + QJL", f"{metrics_polar_prod['IP MAE']:.4f}", f"{metrics_polar_prod['IP bias']:.4f}", f"{metrics_polar_prod['IP corr']:.4f}", "Exploratory hybrid"],
        ]))
        left, right = st.columns([1.0, 1.0])
        with left:
            fig_multi = go.Figure()
            fig_multi.add_trace(go.Scatter(x=true_ip, y=est_ip_qjl, mode="markers", name="QJL", marker=dict(size=5, opacity=0.65, color="#0f766e")))
            fig_multi.add_trace(go.Scatter(x=true_ip, y=est_ip_turbo_prod, mode="markers", name="Turbo+QJL", marker=dict(size=5, opacity=0.72, color="#1d4ed8")))
            fig_multi.add_trace(go.Scatter(x=true_ip, y=est_ip_polar_prod, mode="markers", name="Polar+QJL", marker=dict(size=5, opacity=0.72, color="#b91c1c")))
            lo = float(min(true_ip.min(), est_ip_qjl.min(), est_ip_turbo_prod.min(), est_ip_polar_prod.min()))
            hi = float(max(true_ip.max(), est_ip_qjl.max(), est_ip_turbo_prod.max(), est_ip_polar_prod.max()))
            fig_multi.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="이상적인 y=x", line=dict(color="#64748b", dash="dash")))
            fig_multi.update_layout(template="plotly_white", title=dict(text="내적 추정 비교: QJL 계열", font=dict(color="black")), height=450, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="실제 내적", yaxis_title="추정 내적", plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black"))
            st.plotly_chart(fig_multi, width="stretch", theme=None)
        with right:
            compare_ip_method = st.selectbox("내적 비교용 3D 방법", ["QJL", "Turbo + QJL", "Polar + QJL"], index=1)
            data = process_registry[compare_ip_method]
            compare_ip_pick = plotly_chart_pick(process_figure_3d(data["orig"], data["mid"], data["final"], data["color_ids"], f"{compare_ip_method} 3D 과정", levels, stage1_label=data["stage1_label"], stage2_label=data["stage2_label"], point_indices=process_idx, selected_index=inspect_idx), key="compare_ip_process_chart")
            update_inspect_vector(compare_ip_pick, n_points - 1)
            st.caption("이 3D 그림은 QJL 계열의 흐름을 설명하는 보조 시각화입니다. 핵심 평가는 왼쪽 내적 추정 그래프입니다.")
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
