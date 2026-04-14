# TurboQuant / PolarQuant / QJL Explorer

Streamlit app for comparing **TurboQuant**, **PolarQuant**, and **QJL** with a white-background presentation and explanation-focused layouts.

## This revision

- kept the existing app structure and math flow mostly intact
- unified chart backgrounds to **white** and chart titles to **black**
- changed confusing sidebar labels:
  - `기준 비트 수` → `양자화 비트 수`
  - `단면 확인용 벡터 index` → `단면 예시 벡터 번호`
  - `단면 pair index` → `단면 좌표쌍 번호`
- added clearer sidebar help text for slice controls
- added new data distributions:
  - `Gaussian + outliers`
  - `Sphere shell + outliers`
  - `Ball + outliers`
- added a **TurboQuant slice view** so Turbo also has a 2D slice-style inspection path
- changed 3D process views so they:
  - open on the **final quantized state** by default
  - restart from **original** when pressing **Play**
- reorganized each tab to reduce clutter by moving secondary charts into expanders
- changed many chart titles/labels to more natural Korean wording
- added a **QJL 3D process** view with Play/Pause controls on the QJL tab

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Internal computations still run in the selected high dimension `d`.
- 3D views are explanatory projections, not literal full-dimensional geometry.
- QJL remains primarily an **inner-product estimator** view; its 3D panel is educational.
- `Polar + QJL` is still an exploratory comparison view, not a paper-claimed official method.
