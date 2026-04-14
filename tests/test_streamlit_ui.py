import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except Exception:  # pragma: no cover
    AppTest = None

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@unittest.skipIf(AppTest is None, "streamlit.testing is not available")
class StreamlitUISmokeTests(unittest.TestCase):
    def make_app(self) -> AppTest:
        return AppTest.from_file(str(APP_PATH), default_timeout=180)

    def test_app_renders_without_exceptions_and_shows_main_tabs(self):
        at = self.make_app()
        at.run(timeout=180)
        self.assertEqual(len(at.exception), 0)
        labels = [tab.label for tab in at.tabs]
        for expected in ["기존 양자화", "TurboQuant", "PolarQuant", "QJL", "비교 / 하이브리드", "지표 / 투영 해설"]:
            self.assertIn(expected, labels)

    def test_inspect_vector_widget_updates_session_state_on_rerun(self):
        at = self.make_app()
        at.run(timeout=180)
        target = 5
        inspect_input = next(widget for widget in at.number_input if widget.label == "단면 예시 벡터 번호")
        inspect_input.set_value(target)
        at.run(timeout=180)
        self.assertEqual(next(widget for widget in at.number_input if widget.label == "단면 예시 벡터 번호").value, target)
        self.assertEqual(at.session_state["inspect_vector_widget"], target)
        self.assertEqual(at.session_state["inspect_vector_value"], target)

    def test_pending_click_like_update_path_applies_to_widget(self):
        at = self.make_app()
        at.session_state["inspect_vector_pending"] = 7
        at.run(timeout=180)
        inspect_input = next(widget for widget in at.number_input if widget.label == "단면 예시 벡터 번호")
        self.assertEqual(inspect_input.value, 7)
        self.assertEqual(at.session_state["inspect_vector_value"], 7)
        self.assertEqual(at.session_state["inspect_vector_widget"], 7)

    def test_dimension_number_input_accepts_new_min_and_exposes_max_bound(self):
        at = self.make_app()
        at.run(timeout=180)
        dim_input = next(widget for widget in at.number_input if widget.label == "차원 d")
        self.assertEqual(int(dim_input.min), 3)
        self.assertEqual(int(dim_input.max), 4096)

        sliders = {widget.label: widget for widget in at.slider}
        sliders["벡터 수"].set_value(300)
        sliders["비교용 표시 점 수"].set_value(200)
        sliders["3D 애니메이션 점 수"].set_value(40)
        dim_input.set_value(3)
        at.run(timeout=180)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(next(widget for widget in at.number_input if widget.label == "차원 d").value, 3)


    def test_dimension_change_preserves_other_sidebar_conditions(self):
        at = self.make_app()
        at.run(timeout=180)
        distribution = next(widget for widget in at.selectbox if widget.label == "데이터 분포")
        projection = next(widget for widget in at.selectbox if widget.label == "3D 공통 투영 방식")
        precision = next(widget for widget in at.selectbox if widget.label == "입력 정밀도 시뮬레이션")
        distribution.set_value("Unit sphere")
        projection.set_value("PCA")
        precision.set_value("fp8-like")
        at.run(timeout=180)
        dim_input = next(widget for widget in at.number_input if widget.label == "차원 d")
        dim_input.set_value(8)
        at.run(timeout=180)
        self.assertEqual(next(widget for widget in at.selectbox if widget.label == "데이터 분포").value, "Unit sphere")
        self.assertEqual(next(widget for widget in at.selectbox if widget.label == "3D 공통 투영 방식").value, "PCA")
        self.assertEqual(next(widget for widget in at.selectbox if widget.label == "입력 정밀도 시뮬레이션").value, "fp8-like")

    def test_qjl_sign_sketch_copy_is_visible(self):
        at = self.make_app()
        at.run(timeout=180)
        markdown_text = "\n".join(getattr(node, "value", "") for node in at.markdown)
        self.assertIn("QJL = 1-bit sign sketch for inner-product estimation", markdown_text)


    def test_source_contains_reconstruction_stage_copy_for_mse_and_prod(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("코드북 스냅", source)
        self.assertIn("최종 복원", source)
        self.assertIn("base reconstruction", source)
        self.assertIn("잔차 보정", source)

if __name__ == "__main__":
    unittest.main()
