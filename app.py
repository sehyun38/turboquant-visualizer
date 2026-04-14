import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    x = rng.normal(scale=0.35, size=(n, d))
    n_out = max(1, n // 20)
    idx = rng.choice(n, size=n_out, replace=False)
    x[idx] += rng.normal(scale=1.2, size=(n_out, d))
    return x


@st.cache_data(show_spinner=False)
def make_query(d: int, dist: str, seed: int) -> np.ndarray:
    q = make_data(1, d, "Gaussian" if dist != "Unit sphere" else "Unit sphere", seed + 777)[0]
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
    original_name: str = "Original",
    recon_name: str = "Quantized",
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
        title=title,
        template="plotly_white",
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            aspectmode="cube",
            xaxis=dict(backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd"),
            yaxis=dict(backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd"),
            zaxis=dict(backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd"),
        ),
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


def process_figure_3d(
    original_3d: np.ndarray,
    rotated_3d: np.ndarray,
    quantized_3d: np.ndarray,
    color_ids: np.ndarray,
    title: str,
    levels: int,
) -> go.Figure:
    colors_final = color_array_from_ids(color_ids, levels)
    stages = [("Original", original_3d, [BLUE] * len(original_3d))]
    for t in np.linspace(0.2, 1.0, 5):
        cur = interpolate_points(original_3d, rotated_3d, float(t))
        stages.append((f"Rotate {int(round(t * 100))}%", cur, [BLUE] * len(cur)))
    for t in np.linspace(0.2, 1.0, 5):
        cur = interpolate_points(rotated_3d, quantized_3d, float(t))
        stages.append((f"Quantize {int(round(t * 100))}%", cur, colors_final))

    all_pts = np.vstack([original_3d, rotated_3d, quantized_3d])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    pads = np.maximum((maxs - mins) * 0.15, 0.25)
    xr = [float(mins[0] - pads[0]), float(maxs[0] + pads[0])]
    yr = [float(mins[1] - pads[1]), float(maxs[1] + pads[1])]
    zr = [float(mins[2] - pads[2]), float(maxs[2] + pads[2])]

    frames = []
    for label, pts, colors in stages:
        frames.append(
            go.Frame(
                name=label,
                data=[
                    go.Scatter3d(
                        x=quantized_3d[:, 0], y=quantized_3d[:, 1], z=quantized_3d[:, 2],
                        mode="markers",
                        name="Quantized target",
                        marker=dict(size=3.2, opacity=0.18, color=colors_final),
                    ),
                    go.Scatter3d(
                        x=original_3d[:, 0], y=original_3d[:, 1], z=original_3d[:, 2],
                        mode="markers",
                        name="Original reference",
                        marker=dict(size=3.0, opacity=0.16, color=BLUE),
                    ),
                    go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode="markers",
                        name="Current state",
                        marker=dict(size=4.5, opacity=0.94, color=colors, line=dict(width=0.45, color="white")),
                    ),
                ],
                traces=[0, 1, 2],
            )
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(x=quantized_3d[:, 0], y=quantized_3d[:, 1], z=quantized_3d[:, 2], mode="markers", name="Quantized target", marker=dict(size=3.2, opacity=0.18, color=colors_final)),
            go.Scatter3d(x=original_3d[:, 0], y=original_3d[:, 1], z=original_3d[:, 2], mode="markers", name="Original reference", marker=dict(size=3.0, opacity=0.16, color=BLUE)),
            go.Scatter3d(x=original_3d[:, 0], y=original_3d[:, 1], z=original_3d[:, 2], mode="markers", name="Current state", marker=dict(size=4.5, opacity=0.94, color=BLUE, line=dict(width=0.45, color="white"))),
        ],
        frames=frames,
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=560,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title="x", range=xr, backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd"),
            yaxis=dict(title="y", range=yr, backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd"),
            zaxis=dict(title="z", range=zr, backgroundcolor="white", gridcolor="#dbeafe", zerolinecolor="#93c5fd"),
        ),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0,
            "y": 1.08,
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "3D process: "},
            "pad": {"t": 36},
            "steps": [
                {"label": label, "method": "animate", "args": [[label], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                for label, _, _ in stages
            ],
        }],
        legend=dict(bgcolor="rgba(255,255,255,0.82)"),
    )
    return fig


def histogram_with_codebook(values: np.ndarray, codebook: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=40, name="Distribution", opacity=0.8, marker=dict(color=BLUE, line=dict(width=0.4, color="white"))))
    for c in codebook:
        fig.add_vline(x=float(c), line_width=2, line_dash="dash", line_color=RED)
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=360, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white", paper_bgcolor="white")
    return fig


def scatter_true_vs_est(true_ip: np.ndarray, est_ip: np.ndarray, title: str, name: str = "Points", point_colors: List[str] | None = None) -> go.Figure:
    lo = float(min(true_ip.min(), est_ip.min()))
    hi = float(max(true_ip.max(), est_ip.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=true_ip, y=est_ip, mode="markers", marker=dict(size=6, opacity=0.78, color=point_colors or RED, line=dict(width=0.3, color="white")), name=name))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal y=x", line=dict(color=BLUE, dash="dash", width=2)))
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="True inner product", yaxis_title="Estimated / dequant inner product", plot_bgcolor="white", paper_bgcolor="white")
    return fig


def error_hist(errors: np.ndarray, title: str, color: str = RED) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=errors, nbinsx=40, opacity=0.82, name="Error", marker=dict(color=color, line=dict(width=0.4, color="white"))))
    fig.add_vline(x=0.0, line_width=2, line_color=BLUE)
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Estimate - truth", plot_bgcolor="white", paper_bgcolor="white")
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
    fig.add_trace(go.Scatter(x=[0, original_pair[0]], y=[0, original_pair[1]], mode="lines+markers", name="Original", line=dict(width=3, color=BLUE), marker=dict(size=8, color=BLUE)))
    fig.add_trace(go.Scatter(x=[0, quant_pair[0]], y=[0, quant_pair[1]], mode="lines+markers", name="Quantized", line=dict(width=3, dash="dash", color=RED), marker=dict(size=8, color=RED)))
    fig.add_trace(go.Scatter(x=arc_o_x, y=arc_o_y, mode="lines", line=dict(color=BLUE, width=2), name="θ original"))
    fig.add_trace(go.Scatter(x=arc_q_x, y=arc_q_y, mode="lines", line=dict(color=RED, width=2, dash="dot"), name="θ quantized"))
    fig.add_annotation(x=float(arc_o_x[-1]), y=float(arc_o_y[-1]), text="θ", showarrow=False, font=dict(color=BLUE, size=14))
    fig.add_annotation(x=float(arc_q_x[-1]), y=float(arc_q_y[-1]), text="θ̂", showarrow=False, font=dict(color=RED, size=14))
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(color="black")), height=420, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="x", yaxis_title="y", xaxis=dict(range=[-1.2 * max_abs, 1.2 * max_abs]), yaxis=dict(range=[-1.2 * max_abs, 1.2 * max_abs], scaleanchor="x", scaleratio=1), plot_bgcolor="white", paper_bgcolor="white")
    return fig


def pair_cloud_figure(original_pairs: np.ndarray, quant_pairs: np.ndarray, title: str, color_ids: np.ndarray | None = None, levels: int = 8) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=original_pairs[:, 0], y=original_pairs[:, 1], mode="markers", marker=dict(size=5, opacity=0.28, color=BLUE), name="Original slice cloud"))
    q_colors = color_array_from_ids(color_ids if color_ids is not None else np.zeros(len(quant_pairs), dtype=int), levels)
    fig.add_trace(go.Scatter(x=quant_pairs[:, 0], y=quant_pairs[:, 1], mode="markers", marker=dict(size=5.5, opacity=0.9, color=q_colors, line=dict(width=0.3, color="white")), name="Quantized slice cloud"))
    fig.update_layout(template="plotly_dark", title=dict(text=title, font=dict(color="black")), height=420, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="slice axis 1", yaxis_title="slice axis 2", yaxis=dict(scaleanchor="x", scaleratio=1), plot_bgcolor="white", paper_bgcolor="white")
    return fig


# -----------------------------
# Main
# -----------------------------


inject_theme()
st.title("TurboQuant / PolarQuant / QJL Explorer")
st.caption("흰 배경 위에서 TurboQuant / PolarQuant / QJL의 과정을 더 읽기 쉽게 보이도록 다시 다듬은 버전입니다. 원본 점은 파란색, 양자화된 상태는 빨강 계열 또는 2^bits 색 팔레트로 표현합니다.")

with st.sidebar:
    st.header("설정")
    mode = st.radio("앱 모드", ["Balanced", "Paper-faithful"], index=0, help="Balanced는 설명을 우선하고, Paper-faithful은 논문 구조를 더 강하게 반영합니다.")
    n_points = st.slider("벡터 수", 300, 2200, 900, step=100)
    dimension = st.select_slider("차원 d", options=[8, 16, 32, 64, 128], value=32)
    distribution = st.selectbox("데이터 분포", ["Gaussian", "Gaussian", "Unit sphere"])
    precision = st.selectbox("입력 정밀도 시뮬레이션", ["fp16-like", "fp8-like", "int8-like"])
    bit_width = st.slider("기준 비트 수", 1, 6, 3)
    precondition = st.toggle("랜덤 전처리 적용", value=True)
    projection_mode = st.selectbox("3D 공통 투영 방식", ["Random projection", "PCA", "First 3 coordinates"])
    seed = st.number_input("시드", min_value=0, max_value=999999, value=7, step=1)
    plot_points = st.slider("정적 비교 점 수", 200, 1200, 500, step=100)
    process_points = st.slider("3D 과정 점 수", 40, 220, 100, step=20)
    inspect_vector = st.number_input("단면 확인용 벡터 index", min_value=0, max_value=max(0, n_points - 1), value=0, step=1)
    max_pair = max(0, dimension // 2 - 1)
    slice_pair = st.slider("단면 pair index (x[2i], x[2i+1])", 0, max_pair, 0)
    st.caption(f"양자화 점 색상 수 = 2^{bit_width} = {2 ** bit_width}")

if mode == "Paper-faithful" and distribution == "Embedding-like mixture":
    st.info("Paper-faithful 모드에서는 Gaussian / sphere 가정이 논문 설명과 더 잘 맞습니다. mixture도 계산은 되지만 해석은 덜 깔끔합니다.")

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
    "QJL (surrogate)": MethodResult("QJL (surrogate)", x_qjl_vis, metrics_qjl_surrogate, "내적 추정이 본체이며 3D 복원은 설명용", qjl_details),
    "Turbo + QJL": MethodResult("Turbo + QJL", x_turbo_prod, metrics_turbo_prod, "논문식 2단계: MSE base + residual QJL", turbo_prod_details),
    "Polar + QJL": MethodResult("Polar + QJL", x_polar_prod, metrics_polar_prod, "탐색적 비교용: Polar base + residual QJL", polar_prod_details),
}

static_idx = sample_indices(n_points, plot_points, int(seed) + 1001)
process_idx = sample_indices(n_points, min(process_points, plot_points), int(seed) + 1201)

common_ref = np.vstack([x[static_idx], x_turbo[static_idx], x_polar[static_idx], x_qjl_vis[static_idx], x_turbo_prod[static_idx], x_polar_prod[static_idx]])
mean3, basis3 = fit_projector(common_ref, projection_mode, int(seed) + 10, 3)
proj_x = apply_projector(x[static_idx], mean3, basis3)
proj_qjl = apply_projector(x_qjl_vis[static_idx], mean3, basis3)

def build_process_projection(mid_state: np.ndarray, final_state: np.ndarray, idx: np.ndarray, seed_offset: int):
    ref = np.vstack([x[idx], mid_state[idx], final_state[idx]])
    mean_p, basis_p = fit_projector(ref, projection_mode, int(seed) + seed_offset, 3)
    return apply_projector(x[idx], mean_p, basis_p), apply_projector(mid_state[idx], mean_p, basis_p), apply_projector(final_state[idx], mean_p, basis_p)

process_registry = {}
for offset, method_name, mid_state in [
    (101, "TurboQuant", turbo_details["rot_scaled"]),
    (202, "PolarQuant", polar_details["rot"]),
    (303, "Turbo + QJL", turbo_prod_details["rot_scaled"]),
    (404, "Polar + QJL", polar_prod_details["rot"]),
]:
    orig3, mid3, fin3 = build_process_projection(mid_state, method_registry[method_name].reconstructed, process_idx, offset)
    process_registry[method_name] = {"orig": orig3, "mid": mid3, "final": fin3, "color_ids": method_registry[method_name].details["color_ids"][process_idx]}

st.markdown("### 지금 한 번에 보는 핵심")
summary_metrics = {
    "Turbo MSE": metrics_turbo["MSE"],
    "Polar MSE": metrics_polar["MSE"],
    "QJL IP MAE": float(np.mean(np.abs(est_ip_qjl - true_ip))),
    "Turbo+QJL IP MAE": float(np.mean(np.abs(est_ip_turbo_prod - true_ip))),
    "Polar+QJL IP MAE": float(np.mean(np.abs(est_ip_polar_prod - true_ip))),
}
metric_cards(summary_metrics, accents=["blue", "red", "blue", "blue", "red"])

with st.expander("Paper details / 이 앱이 논문과 어디까지 맞는지", expanded=False):
    st.markdown(
        """
- **TurboQuant**: 무작위 회전 후 좌표별 스칼라 양자화를 적용하는 구조를 반영했습니다. 앱에서는 Beta형 좌표 분포에 맞춘 코드북을 **샘플 기반 Max-Lloyd 근사**로 만듭니다.
- **PolarQuant**: 재귀적 polar 변환, 첫 레벨 `[0, 2π)`, 상위 레벨 `[0, π/2]` 구조를 반영했습니다.
- **QJL**: 논문의 본체는 **벡터 복원**이 아니라 **비대칭 inner-product estimator**입니다.
- **Turbo + QJL**: 먼저 MSE 양자화 후 residual에 1-bit QJL을 붙였습니다.
- **Polar + QJL**: 비교/교육용 탐색 하이브리드입니다.
        """
    )

comparison_rows = [
    ["TurboQuant", f"{metrics_turbo['MSE']:.4f}", f"{metrics_turbo['IP MAE']:.4f}", f"{metrics_turbo['IP bias']:.4f}", "paper-style MSE"],
    ["PolarQuant", f"{metrics_polar['MSE']:.4f}", f"{metrics_polar['IP MAE']:.4f}", f"{metrics_polar['IP bias']:.4f}", "paper-style angle"],
    ["QJL", f"{metrics_qjl_surrogate['MSE']:.4f}", f"{float(np.mean(np.abs(est_ip_qjl - true_ip))):.4f}", f"{float(np.mean(est_ip_qjl - true_ip)):.4f}", "inner-product first"],
    ["Turbo + QJL", f"{metrics_turbo_prod['MSE']:.4f}", f"{metrics_turbo_prod['IP MAE']:.4f}", f"{metrics_turbo_prod['IP bias']:.4f}", "paper hybrid"],
    ["Polar + QJL", f"{metrics_polar_prod['MSE']:.4f}", f"{metrics_polar_prod['IP MAE']:.4f}", f"{metrics_polar_prod['IP bias']:.4f}", "exploratory"],
]

turbo_tab, polar_tab, qjl_tab, compare_tab = st.tabs(["TurboQuant", "PolarQuant", "QJL", "Compare / Hybrid"])
with turbo_tab:
    one_line_box("TurboQuant는 회전된 좌표를 공통 코드북에 스냅하는 방법입니다. 3D reconstruction 안에서 바로 그 과정을 따라가게 바꿨습니다.")
    look_box(["3D reconstruction 슬라이더로 원본 → 회전 → 코드북 스냅 단계를 따라가 보세요.", "회전 좌표 histogram은 파란 분포와 빨간 코드북 선으로 읽기 쉽게 바꿨습니다.", "양자화 점 색상은 대표 codebook bin을 기준으로 2^bits 색으로 표시합니다."])
    metric_cards(metrics_turbo, accents=["blue", "red", "blue", "red", "red", "blue"])
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        data = process_registry["TurboQuant"]
        st.plotly_chart(process_figure_3d(data["orig"], data["mid"], data["final"], data["color_ids"], "TurboQuant reconstruction in 3D", levels), width="stretch", theme=None)
    with c2:
        st.plotly_chart(histogram_with_codebook(turbo_details["rot"].reshape(-1), turbo_details["codebook"], "Rotated coordinates + Turbo codebook"), width="stretch", theme=None)
        st.plotly_chart(scatter_true_vs_est(true_ip, est_ip_turbo, "True vs dequant inner products", point_colors=color_array_from_ids(turbo_details["color_ids"], levels)), width="stretch", theme=None)

with polar_tab:
    one_line_box("PolarQuant는 좌표를 반지름과 각도로 바꾼 뒤 각도를 양자화합니다. 단면도와 3D 과정을 같이 보도록 정리했습니다.")
    look_box(["단면도에서 원본 각도 θ와 양자화된 각도 θ̂를 파란색과 빨간색으로 비교하세요.", "3D reconstruction 슬라이더에서는 회전된 상태를 거쳐 최종 양자화점으로 이동합니다.", "양자화 점 색상은 대표 first-angle bin을 2^bits 색으로 압축해 보여줍니다."])
    metric_cards(metrics_polar, accents=["red", "blue", "red", "red", "red", "blue"])
    first_bits = int(polar_details["first_bits"][0])
    upper_bits = int(polar_details["upper_bits"][0])
    st.markdown(f'<span class="metric-chip">first-level bits = {first_bits}</span><span class="metric-chip">upper-level bits = {upper_bits}</span><span class="metric-chip">slice pair = {slice_pair}</span>', unsafe_allow_html=True)
    inspect_idx = int(inspect_vector)
    pair_start = 2 * int(slice_pair)
    pair_end = pair_start + 2
    original_rot_pair = polar_details["rot"][inspect_idx, pair_start:pair_end]
    recon_rot_pair = polar_details["recon_rot"][inspect_idx, pair_start:pair_end]
    original_slice_cloud = polar_details["rot"][static_idx, pair_start:pair_end]
    recon_slice_cloud = polar_details["recon_rot"][static_idx, pair_start:pair_end]
    left, right = st.columns([1.15, 0.95])
    with left:
        data = process_registry["PolarQuant"]
        st.plotly_chart(process_figure_3d(data["orig"], data["mid"], data["final"], data["color_ids"], "PolarQuant reconstruction in 3D", levels), width="stretch", theme=None)
    with right:
        st.plotly_chart(slice_geometry_figure(original_rot_pair, recon_rot_pair, f"Slice geometry for vector {inspect_idx} / pair {slice_pair}"), width="stretch", theme=None)
        st.plotly_chart(pair_cloud_figure(original_slice_cloud, recon_slice_cloud, f"Pair slice cloud (rotated space) for pair {slice_pair}", color_ids=polar_details["color_ids"][static_idx], levels=levels), width="stretch", theme=None)
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.plotly_chart(error_hist(polar_details["lvl0_after"] - polar_details["lvl0_before"], "Level-1 angle quantization error", color=RED), width="stretch", theme=None)
    with bottom_right:
        st.plotly_chart(error_hist(polar_details["lvllast_after"] - polar_details["lvllast_before"], "Deep-level angle quantization error", color=BLUE), width="stretch", theme=None)
    st.plotly_chart(scatter_true_vs_est(true_ip, est_ip_polar, "PolarQuant: true vs dequant inner products", point_colors=color_array_from_ids(polar_details["color_ids"], levels)), width="stretch", theme=None)

with qjl_tab:
    one_line_box("QJL의 핵심은 벡터 복원이 아니라 inner-product estimator입니다. 그래서 이 탭은 지표와 산점도를 가장 앞에 둡니다.")
    look_box(["왼쪽 scatter가 y=x에 가까울수록 내적 추정이 잘 되는 것입니다.", "QJL surrogate 3D는 직관용 보조 시각화이고, 논문적 핵심은 true vs estimated inner product입니다.", "색상은 sign sketch의 앞부분 패턴을 2^bits 색으로 해시해서 보여줍니다."])
    qjl_bias = float(np.mean(est_ip_qjl - true_ip))
    qjl_mae = float(np.mean(np.abs(est_ip_qjl - true_ip)))
    qjl_corr = float(np.corrcoef(true_ip, est_ip_qjl)[0, 1]) if np.std(est_ip_qjl) > EPS and np.std(true_ip) > EPS else 1.0
    qjl_metrics = {"IP bias": qjl_bias, "IP MAE": qjl_mae, "IP corr": qjl_corr, "Sketch dim m": float(m_qjl), "Mean |q|": float(np.linalg.norm(q)), "Mean |k|": float(np.mean(qjl_details["norms"]))}
    metric_cards(qjl_metrics, accents=["red", "blue", "blue", "blue", "red", "blue"])
    left, right = st.columns([1.0, 1.0])
    with left:
        st.plotly_chart(scatter_true_vs_est(true_ip, est_ip_qjl, "QJL: true vs estimated inner products", name="QJL estimator", point_colors=color_array_from_ids(qjl_details["color_ids"], levels)), width="stretch", theme=None)
        st.plotly_chart(error_hist(est_ip_qjl - true_ip, "QJL estimator error", color=RED), width="stretch", theme=None)
    with right:
        st.plotly_chart(scatter_overlay_3d(proj_x, proj_qjl, "Visualization-only QJL surrogate in 3D", color_ids=qjl_details["color_ids"][static_idx], levels=levels, recon_name="QJL surrogate"), width="stretch", theme=None)
        st.plotly_chart(scatter_true_vs_est(true_ip, est_ip_turbo_prod, "Reference: Turbo + QJL inner products", point_colors=color_array_from_ids(turbo_prod_details["color_ids"], levels)), width="stretch", theme=None)

with compare_tab:
    one_line_box("5번째 탭에는 Turbo+QJL뿐 아니라 Polar+QJL도 넣었습니다. 같은 원본 점이 어떤 방식으로 이동하고, 내적 오차가 어떻게 달라지는지 바로 비교할 수 있습니다.")
    look_box(["Turbo 계열은 파란 계열, Polar 계열은 빨간 계열로 보되 최종 양자화점은 모두 2^bits 색 팔레트로 표시합니다.", "3D reconstruction 안의 슬라이더로 한 방법의 원본 → 중간 상태 → 최종 상태를 단계별로 확인하세요.", "inner-product scatter에서 y=x에 더 가깝고 error histogram이 더 좁으면 더 좋은 쪽입니다."])
    pipeline_box("Hybrid compare", [("Base quantizer", "Turbo 또는 Polar base를 먼저 적용합니다."), ("Residual", "원본 - base 복원값 차이를 residual로 봅니다."), ("Residual QJL", "residual에 sign sketch를 적용합니다."), ("Final merge", "base 복원 + residual_hat 을 합쳐 최종점을 만듭니다.")])
    st.markdown(markdown_table(["Method", "MSE", "IP MAE", "IP bias", "Role"], comparison_rows))
    compare_method = st.selectbox("비교용 3D reconstruction 방법", ["TurboQuant", "PolarQuant", "Turbo + QJL", "Polar + QJL"], index=2)
    left, right = st.columns([1.15, 0.95])
    with left:
        data = process_registry[compare_method]
        st.plotly_chart(process_figure_3d(data["orig"], data["mid"], data["final"], data["color_ids"], f"{compare_method} 3D reconstruction process", levels), width="stretch", theme=None, key="unique_chart_id_01")
    with right:
        fig_multi = go.Figure()
        fig_multi.add_trace(go.Scatter(x=true_ip, y=est_ip_turbo, mode="markers", name="Turbo", marker=dict(size=4, opacity=0.35, color=BLUE)))
        fig_multi.add_trace(go.Scatter(x=true_ip, y=est_ip_polar, mode="markers", name="Polar", marker=dict(size=4, opacity=0.35, color=RED)))
        fig_multi.add_trace(go.Scatter(x=true_ip, y=est_ip_turbo_prod, mode="markers", name="Turbo+QJL", marker=dict(size=5, opacity=0.72, color="#1d4ed8")))
        fig_multi.add_trace(go.Scatter(x=true_ip, y=est_ip_polar_prod, mode="markers", name="Polar+QJL", marker=dict(size=5, opacity=0.72, color="#b91c1c")))
        lo = float(min(true_ip.min(), est_ip_turbo.min(), est_ip_polar.min(), est_ip_turbo_prod.min(), est_ip_polar_prod.min()))
        hi = float(max(true_ip.max(), est_ip_turbo.max(), est_ip_polar.max(), est_ip_turbo_prod.max(), est_ip_polar_prod.max()))
        fig_multi.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal y=x", line=dict(color="#64748b", dash="dash")))
        fig_multi.update_layout(template="plotly_white", title="Method compare: inner products", height=450, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="True inner product", yaxis_title="Estimated / dequant inner product", plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_multi, width="stretch", theme=None)
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.plotly_chart(error_hist(est_ip_turbo_prod - true_ip, "Turbo + QJL inner-product error", color=BLUE), width="stretch", theme=None)
    with bottom_right:
        st.plotly_chart(error_hist(est_ip_polar_prod - true_ip, "Polar + QJL inner-product error", color=RED), width="stretch", theme=None)

st.markdown("---")
st.markdown("**요약:** TurboQuant는 좌표 스냅, PolarQuant는 각도 스냅, QJL은 1-bit inner-product sketch, Turbo+QJL은 논문식 하이브리드, Polar+QJL은 비교용 탐색 하이브리드로 이해하면 됩니다.")
