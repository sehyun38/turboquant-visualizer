import ast
import math
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_functions(*names: str):
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(APP_PATH))
    wanted = []
    wanted_set = set(names)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_set:
            node.decorator_list = []
            wanted.append(node)
    missing = wanted_set - {node.name for node in wanted}
    if missing:
        raise RuntimeError(f"Missing functions: {sorted(missing)}")
    module = ast.Module(body=wanted, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "math": math,
        "EPS": 1e-12,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "go": go,
        "ORIGINAL_POINT_COLOR": "#9ca3af",
        "ORIGINAL_REFERENCE_COLOR": "#dc2626",
        "FINAL_REFERENCE_COLOR": "#2563eb",
    }
    exec(compile(module, filename=str(APP_PATH), mode="exec"), namespace)
    return namespace


class QuantizationLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_functions(
            "gaussian_sketch",
            "build_preconditioner",
            "apply_preconditioner_rows",
            "invert_preconditioner_rows",
            "effective_sketch_dim",
            "qjl_quantize",
            "softmax_stable",
            "attention_proxy_metrics",
            "polar_forward_single",
            "polar_inverse_single",
            "polar_quantize",
            "polar_level_bits",
            "uniform_angle_codebook",
            "quantize_angles",
            "random_orthogonal",
            "sample_unit_coordinate_distribution",
            "lloyd_max_1d",
            "baseline_uniform_quantize",
            "turbo_codebook",
            "quantize_by_codebook",
            "hash_ids",
            "turbo_quantize_mse",
            "turbo_quantize_prod",
            "polar_quantize_prod",
            "discrete_palette",
            "color_array_from_ids",
            "interpolate_points",
            "process_figure_3d",
            "build_plot_point_index_map",
            "extract_selected_point_index",
        )

    def test_qjl_sign_output_is_binary(self):
        qjl_quantize = self.ns["qjl_quantize"]
        rng = np.random.default_rng(7)
        x = rng.normal(size=(32, 16))
        q = rng.normal(size=(16,))
        _, details = qjl_quantize(x, q, m=8, seed=11, bits=3)
        unique = set(np.unique(details["signs"]).tolist())
        self.assertTrue(unique.issubset({-1.0, 1.0}))

    def test_qjl_output_shapes_match(self):
        qjl_quantize = self.ns["qjl_quantize"]
        rng = np.random.default_rng(9)
        x = rng.normal(size=(20, 12))
        q = rng.normal(size=(12,))
        x_hat, details = qjl_quantize(x, q, m=6, seed=13, bits=2)
        self.assertEqual(x_hat.shape, x.shape)
        self.assertEqual(details["signs"].shape, (20, 6))
        self.assertEqual(details["sketch"].shape, (6, 12))
        self.assertEqual(details["ip_est"].shape, (20,))

    def test_qjl_inner_product_estimator_smoke(self):
        qjl_quantize = self.ns["qjl_quantize"]
        attention_proxy_metrics = self.ns["attention_proxy_metrics"]
        rng = np.random.default_rng(17)
        x = rng.normal(size=(256, 32))
        q = rng.normal(size=(32,))
        _, details = qjl_quantize(x, q, m=32, seed=5, bits=3)
        true_ip = details["ip_true"]
        est_ip = details["ip_est"]
        corr = float(np.corrcoef(true_ip, est_ip)[0, 1])
        self.assertGreater(corr, 0.45)
        metrics = attention_proxy_metrics(true_ip, est_ip, temperature=max(float(np.std(true_ip)), 1.0), top_k=10)
        self.assertTrue(0.0 <= metrics["score_tv"] <= 1.0)
        self.assertTrue(0.0 <= metrics["topk_overlap"] <= 1.0)

    def test_polar_angle_ranges_follow_definition(self):
        polar_forward_single = self.ns["polar_forward_single"]
        rng = np.random.default_rng(21)
        x = rng.normal(size=(8,))
        radius, levels = polar_forward_single(x)
        self.assertGreater(radius, 0.0)
        first = levels[0]
        self.assertTrue(np.all(first >= 0.0))
        self.assertTrue(np.all(first < 2.0 * math.pi + 1e-12))
        for upper in levels[1:]:
            self.assertTrue(np.all(upper >= 0.0))
            self.assertTrue(np.all(upper <= math.pi / 2.0 + 1e-12))


    def test_baseline_quantized_values_snap_to_uniform_codebook(self):
        baseline_uniform_quantize = self.ns["baseline_uniform_quantize"]
        rng = np.random.default_rng(41)
        x = rng.normal(size=(20, 8))
        x_hat, details = baseline_uniform_quantize(x, bits=3)
        codebook = details["codebook"]
        membership = np.isclose(x_hat[..., None], codebook[None, None, :], atol=1e-8)
        self.assertTrue(np.all(np.any(membership, axis=-1)))

    def test_turbo_quantized_values_snap_to_codebook(self):
        turbo_quantize_mse = self.ns["turbo_quantize_mse"]
        rng = np.random.default_rng(29)
        x = rng.normal(size=(24, 8))
        _, details = turbo_quantize_mse(x, bits=2, precondition=True, seed=3)
        codebook = details["codebook"]
        q_rot = details["q_rot"]
        membership = np.isclose(q_rot[..., None], codebook[None, None, :], atol=1e-8)
        self.assertTrue(np.all(np.any(membership, axis=-1)))

    def test_turbo_mse_details_include_inverse_rotation_reconstruction_state(self):
        turbo_quantize_mse = self.ns["turbo_quantize_mse"]
        rng = np.random.default_rng(30)
        x = rng.normal(size=(10, 8))
        x_hat, details = turbo_quantize_mse(x, bits=3, precondition=True, seed=5)
        self.assertIn("q_rot_scaled", details)
        self.assertIn("reconstructed", details)
        self.assertEqual(details["q_rot_scaled"].shape, x.shape)
        self.assertTrue(np.allclose(details["reconstructed"], x_hat))

    def test_prod_details_include_base_residual_and_final_states(self):
        turbo_quantize_prod = self.ns["turbo_quantize_prod"]
        polar_quantize_prod = self.ns["polar_quantize_prod"]
        rng = np.random.default_rng(32)
        x = rng.normal(size=(12, 8))
        turbo_hat, turbo_details = turbo_quantize_prod(x, bits=3, precondition=True, seed=7)
        polar_hat, polar_details = polar_quantize_prod(x, bits=3, precondition=True, seed=7)
        for x_hat, details in [(turbo_hat, turbo_details), (polar_hat, polar_details)]:
            self.assertIn("base", details)
            self.assertIn("residual", details)
            self.assertIn("residual_hat", details)
            self.assertIn("residual_mid", details)
            self.assertIn("reconstructed", details)
            self.assertEqual(details["base"].shape, x.shape)
            self.assertEqual(details["residual_mid"].shape, x.shape)
            self.assertTrue(np.allclose(details["reconstructed"], x_hat))

    def test_process_figure_supports_multi_stage_sequences(self):
        process_figure_3d = self.ns["process_figure_3d"]
        rng = np.random.default_rng(37)
        original = rng.normal(size=(6, 3))
        stage_a = original + 0.2
        stage_b = original + 0.4
        stage_c = original + 0.6
        fig = process_figure_3d(
            [
                {"label": "원본", "points": original, "colors": None},
                {"label": "중간 A", "points": stage_a, "colors": None},
                {"label": "중간 B", "points": stage_b, "colors": None},
                {"label": "최종", "points": stage_c, "colors": ["#111111"] * len(original)},
            ],
            np.arange(len(original)),
            "multi-stage",
            levels=8,
            point_indices=np.arange(len(original)),
            selected_index=2,
        )
        self.assertGreaterEqual(len(fig.frames), 10)
        slider_labels = [step["label"] for step in fig.layout.sliders[0]["steps"]]
        self.assertIn("원본", slider_labels)
        self.assertIn("최종", slider_labels)
        self.assertTrue(any("중간 A" in label and "중간 B" in label for label in slider_labels))
        self.assertEqual(fig.data[0].visible, "legendonly")
        self.assertEqual(fig.data[1].visible, "legendonly")
        self.assertEqual(fig.data[0].marker.color, self.ns["FINAL_REFERENCE_COLOR"])
        self.assertEqual(fig.data[1].marker.color, self.ns["ORIGINAL_REFERENCE_COLOR"])

    def test_polar_roundtrip_supports_odd_dimensions_when_cropped(self):
        polar_forward_single = self.ns["polar_forward_single"]
        polar_inverse_single = self.ns["polar_inverse_single"]
        rng = np.random.default_rng(31)
        x = rng.normal(size=(3,))
        radius, levels = polar_forward_single(x)
        x_hat = polar_inverse_single(radius, levels, output_dim=3)
        self.assertEqual(x_hat.shape, (3,))
        self.assertTrue(np.allclose(x, x_hat, atol=1e-8))

    def test_polar_quantize_handles_min_dimension_three(self):
        polar_quantize = self.ns["polar_quantize"]
        rng = np.random.default_rng(35)
        x = rng.normal(size=(12, 3))
        x_hat, details = polar_quantize(x, bits=3, precondition=True, seed=7)
        self.assertEqual(x_hat.shape, x.shape)
        self.assertEqual(details["rot"].shape, x.shape)
        self.assertEqual(details["recon_rot"].shape, x.shape)

    def test_polar_inverse_roundtrip_without_quantization(self):
        polar_forward_single = self.ns["polar_forward_single"]
        polar_inverse_single = self.ns["polar_inverse_single"]
        rng = np.random.default_rng(33)
        x = rng.normal(size=(8,))
        radius, levels = polar_forward_single(x)
        x_hat = polar_inverse_single(radius, levels)
        self.assertTrue(np.allclose(x, x_hat, atol=1e-8))

    def test_click_index_resolution_prefers_custom_point_map_over_helper_lines(self):
        build_plot_point_index_map = self.ns["build_plot_point_index_map"]
        extract_selected_point_index = self.ns["extract_selected_point_index"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0.0, 1.0], y=[0.0, 1.0], mode="lines", name="helper"))
        fig.add_trace(go.Scatter(
            x=[0.1, 0.2],
            y=[0.3, 0.4],
            mode="markers",
            customdata=np.array([[16], [42]]),
            name="points",
        ))
        point_index_map = build_plot_point_index_map(fig)
        event = {
            "selection": {
                "points": [
                    {"curve_number": 0, "point_number": 1},
                    {"curve_number": 1, "point_number": 1},
                ]
            }
        }
        self.assertEqual(extract_selected_point_index(event, point_index_map=point_index_map), 42)


    def test_source_uses_explicit_widget_keys_for_independent_sidebar_state(self):
        source = APP_PATH.read_text(encoding="utf-8")
        for token in [
            'key="distribution"',
            'key="precision"',
            'key="bit_width"',
            'key="precondition"',
            'key="projection_mode"',
            'key="seed_input"',
            'key="plot_points"',
            'key="process_points"',
        ]:
            self.assertIn(token, source)

    def test_high_dimension_4096_runtime_smoke(self):
        turbo_quantize_mse = self.ns["turbo_quantize_mse"]
        polar_quantize = self.ns["polar_quantize"]
        qjl_quantize = self.ns["qjl_quantize"]
        rng = np.random.default_rng(53)
        x = rng.normal(size=(24, 4096))
        q = rng.normal(size=(4096,))

        turbo_hat, turbo_details = turbo_quantize_mse(x, bits=2, precondition=True, seed=11)
        polar_hat, polar_details = polar_quantize(x, bits=2, precondition=True, seed=11)
        qjl_hat, qjl_details = qjl_quantize(x, q, m=2048, seed=11, bits=2)

        self.assertEqual(turbo_hat.shape, x.shape)
        self.assertEqual(turbo_details["q_rot"].shape, x.shape)
        self.assertEqual(polar_hat.shape, x.shape)
        self.assertEqual(polar_details["recon_rot"].shape, x.shape)
        self.assertEqual(qjl_hat.shape, x.shape)
        self.assertEqual(qjl_details["sketch"].shape[1], 4096)
        self.assertLessEqual(int(qjl_details["m"][0]), 64)

    def test_dimension_input_range_is_opened_to_3_and_4096(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('st.number_input(', source)
        self.assertIn('"차원 d"', source)
        self.assertIn('min_value=3', source)
        self.assertIn('max_value=4096', source)
        self.assertIn('key="dimension_input"', source)
        self.assertIn('DIMENSION_PRESETS = [3, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]', source)



if __name__ == "__main__":
    unittest.main()
