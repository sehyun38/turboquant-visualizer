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

    def test_qjl_sign_sketch_copy_is_visible(self):
        at = self.make_app()
        at.run(timeout=180)
        markdown_text = "\n".join(getattr(node, "value", "") for node in at.markdown)
        self.assertIn("QJL = 1-bit sign sketch for inner-product estimation", markdown_text)


if __name__ == "__main__":
    unittest.main()
