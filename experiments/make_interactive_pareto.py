import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vf_core.paths import OUTPUT_DIR


SERIES_BY_CASE = {
    "fixed_light": [
        {
            "label": "SP-S1",
            "file": "SP_baseline.csv",
            "topsis_file": "SP_baseline_topsis.csv",
            "marker": "circle",
            "color": "#1F78B4",
        },
        {
            "label": "Fixed-light 14812",
            "file": "SP_fixed_light_14812.csv",
            "topsis_file": "SP_fixed_light_14812_topsis.csv",
            "marker": "triangle",
            "color": "#E31A1C",
        },
        {
            "label": "Fixed-light 18517",
            "file": "SP_fixed_light_18517.csv",
            "topsis_file": "SP_fixed_light_18517_topsis.csv",
            "marker": "square",
            "color": "#33A02C",
        },
        {
            "label": "Fixed-light 21481",
            "file": "SP_fixed_light_21481.csv",
            "topsis_file": "SP_fixed_light_21481_topsis.csv",
            "marker": "diamond",
            "color": "#6A3D9A",
        },
    ],
    "price": [
        {
            "label": "SP-S1",
            "file": "SP_baseline.csv",
            "topsis_file": "SP_baseline_topsis.csv",
            "marker": "circle",
            "color": "#1F78B4",
        },
        {
            "label": "Price 80%",
            "file": "SP_price_080.csv",
            "topsis_file": "SP_price_080_topsis.csv",
            "marker": "triangle",
            "color": "#33A02C",
        },
        {
            "label": "Price 120%",
            "file": "SP_price_120.csv",
            "topsis_file": "SP_price_120_topsis.csv",
            "marker": "square",
            "color": "#E31A1C",
        },
    ],
    "export": [
        {
            "label": "SP-S1",
            "file": "SP_baseline.csv",
            "topsis_file": "SP_baseline_topsis.csv",
            "marker": "circle",
            "color": "#1F78B4",
        },
        {
            "label": "Export 10%",
            "file": "SP_export_010.csv",
            "topsis_file": "SP_export_010_topsis.csv",
            "marker": "triangle",
            "color": "#33A02C",
        },
        {
            "label": "Export 50%",
            "file": "SP_export_050.csv",
            "topsis_file": "SP_export_050_topsis.csv",
            "marker": "square",
            "color": "#E31A1C",
        },
    ],
}

DEFAULT_OUTPUT_HTML = {
    "fixed_light": "SP_fixed_light_pareto_compare_interactive.html",
    "price": "SP_price_sensitivity_pareto_updated_interactive.html",
    "export": "SP_export_limit_pareto_interactive.html",
}


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    for candidate in candidates:
        matches = [column for column in df.columns if candidate in column]
        if matches:
            return matches[0]
    raise KeyError(f"None of these columns were found: {candidates}")


def objective_columns(df: pd.DataFrame) -> tuple[str, str]:
    co2_col = find_column(df, ["Grid CO2 Emission (ton/yr)", "grid_co2_emission"])
    cash_col = find_column(df, ["Total annual net cash flow", "annual_net_cash_flow"])
    return co2_col, cash_col


def objective_values(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    co2_col, cash_col = objective_columns(df)
    co2 = df[co2_col].to_numpy(dtype=float)
    cash = df[cash_col].to_numpy(dtype=float)
    if co2_col == "grid_co2_emission":
        co2 = co2 * 1e4
    if cash_col == "annual_net_cash_flow":
        cash = cash * 1e6
    return co2, cash / 1e6


def ppfd_from_num_light(num_light: float) -> float:
    return 10.0 * (float(num_light) - 14812.0) / 741.0 + 200.0


def row_value(row: pd.Series, name: str, default=np.nan):
    return row[name] if name in row.index else default


def design_values(row: pd.Series) -> tuple[float, int, int, int]:
    if "num_light" in row.index:
        num_light = float(row["num_light"])
    else:
        num_light = 14812.0 + 741.0 * float(row_value(row, "num_light_index", 0))
    return (
        ppfd_from_num_light(num_light),
        int(round(float(row_value(row, "num_pv", 0)))),
        int(round(float(row_value(row, "num_wind", 0)))),
        int(round(float(row_value(row, "num_batt", 0)))),
    )


def marker_path(marker: str, x: float, y: float, size: float) -> str:
    if marker == "triangle":
        h = size * 1.2
        return f"M {x:.3f} {y - h:.3f} L {x - size:.3f} {y + h * 0.65:.3f} L {x + size:.3f} {y + h * 0.65:.3f} Z"
    if marker == "square":
        return f"M {x - size:.3f} {y - size:.3f} H {x + size:.3f} V {y + size:.3f} H {x - size:.3f} Z"
    if marker == "diamond":
        return f"M {x:.3f} {y - size * 1.25:.3f} L {x - size:.3f} {y:.3f} L {x:.3f} {y + size * 1.25:.3f} L {x + size:.3f} {y:.3f} Z"
    return ""


def star_path(x: float, y: float, outer: float, inner: float) -> str:
    points = []
    for i in range(10):
        angle = -np.pi / 2 + i * np.pi / 5
        radius = outer if i % 2 == 0 else inner
        points.append((x + radius * np.cos(angle), y + radius * np.sin(angle)))
    return "M " + " L ".join(f"{px:.3f} {py:.3f}" for px, py in points) + " Z"


def build_points(output_dir: Path, series_list: list[dict]) -> list[dict]:
    points = []
    for series in series_list:
        csv_path = output_dir / series["file"]
        df = pd.read_csv(csv_path)
        co2, cash_million = objective_values(df)
        for index, row in df.iterrows():
            if not (np.isfinite(co2[index]) and np.isfinite(cash_million[index])):
                continue
            ppfd, pv, wt, batt = design_values(row)
            points.append({
                "series": series["label"],
                "kind": "Pareto solution",
                "marker": series["marker"],
                "color": series["color"],
                "co2": float(co2[index]),
                "anp": float(cash_million[index]),
                "ppfd": float(ppfd),
                "pv": pv,
                "wt": wt,
                "batt": batt,
            })

        topsis_path = output_dir / series["topsis_file"]
        if topsis_path.exists():
            topsis_df = pd.read_csv(topsis_path)
            topsis_co2, topsis_cash_million = objective_values(topsis_df)
            row = topsis_df.iloc[0]
            ppfd, pv, wt, batt = design_values(row)
            points.append({
                "series": series["label"],
                "kind": "TOPSIS optimal",
                "marker": "star",
                "color": series["color"],
                "co2": float(topsis_co2[0]),
                "anp": float(topsis_cash_million[0]),
                "ppfd": float(ppfd),
                "pv": pv,
                "wt": wt,
                "batt": batt,
            })
    return points


def nice_domain(values: list[float], pad_ratio: float = 0.06) -> tuple[float, float]:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    pad = (hi - lo) * pad_ratio if hi > lo else 1.0
    return lo - pad, hi + pad


def render_html(points: list[dict], series_list: list[dict], title: str) -> str:
    width, height = 980, 620
    margin = {"left": 82, "right": 230, "top": 36, "bottom": 76}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    x_min, x_max = nice_domain([p["co2"] for p in points])
    y_min, y_max = nice_domain([p["anp"] for p in points])

    def sx(value):
        return margin["left"] + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value):
        return margin["top"] + (y_max - value) / (y_max - y_min) * plot_h

    x_ticks = np.linspace(np.ceil(x_min / 50) * 50, np.floor(x_max / 50) * 50, 5)
    if len(x_ticks) == 0:
        x_ticks = np.linspace(x_min, x_max, 5)
    y_ticks = np.linspace(np.round(y_min, 2), np.round(y_max, 2), 5)

    svg_parts = [
        f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_w}" height="{plot_h}" fill="white" stroke="black" stroke-width="1.2"/>'
    ]
    for tick in x_ticks:
        x = sx(float(tick))
        svg_parts.append(f'<line x1="{x:.2f}" y1="{margin["top"] + plot_h}" x2="{x:.2f}" y2="{margin["top"] + plot_h + 6}" stroke="black"/>')
        svg_parts.append(f'<text x="{x:.2f}" y="{margin["top"] + plot_h + 24}" text-anchor="middle">{tick:.0f}</text>')
    for tick in y_ticks:
        y = sy(float(tick))
        svg_parts.append(f'<line x1="{margin["left"] - 6}" y1="{y:.2f}" x2="{margin["left"]}" y2="{y:.2f}" stroke="black"/>')
        svg_parts.append(f'<text x="{margin["left"] - 12}" y="{y + 4:.2f}" text-anchor="end">{tick:.2f}</text>')
    svg_parts.append(f'<text x="{margin["left"] + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle">Grid CO2 Emissions (ton/year)</text>')
    svg_parts.append(f'<text transform="translate(24 {margin["top"] + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle">Annual Net Profit (10^6 EUR/year)</text>')

    for point in points:
        x, y = sx(point["co2"]), sy(point["anp"])
        payload = html.escape(json.dumps(point), quote=True)
        if point["marker"] == "circle":
            svg_parts.append(
                f'<circle class="point" cx="{x:.3f}" cy="{y:.3f}" r="4.5" fill="white" stroke="{point["color"]}" stroke-width="1.3" data-point="{payload}"/>'
            )
        elif point["marker"] == "star":
            svg_parts.append(
                f'<path class="point" d="{star_path(x, y, 8.5, 3.8)}" fill="{point["color"]}" stroke="black" stroke-width="0.8" data-point="{payload}"/>'
            )
        else:
            svg_parts.append(
                f'<path class="point" d="{marker_path(point["marker"], x, y, 5.2)}" fill="white" stroke="{point["color"]}" stroke-width="1.3" data-point="{payload}"/>'
            )

    legend_x = margin["left"] + plot_w + 36
    legend_y = margin["top"] + 20
    svg_parts.append(f'<text x="{legend_x}" y="{legend_y}" font-weight="700">Scenarios</text>')
    for i, series in enumerate(series_list):
        y = legend_y + 28 + i * 28
        if series["marker"] == "circle":
            svg_parts.append(f'<circle cx="{legend_x + 8}" cy="{y - 4}" r="5" fill="white" stroke="{series["color"]}" stroke-width="1.3"/>')
        else:
            svg_parts.append(f'<path d="{marker_path(series["marker"], legend_x + 8, y - 4, 5.5)}" fill="white" stroke="{series["color"]}" stroke-width="1.3"/>')
        svg_parts.append(f'<text x="{legend_x + 24}" y="{y}">{html.escape(series["label"])}</text>')
    svg_parts.append(f'<text x="{legend_x}" y="{legend_y + 128}" font-weight="700">Notation</text>')
    svg_parts.append(f'<circle cx="{legend_x + 8}" cy="{legend_y + 154}" r="5" fill="white" stroke="gray" stroke-width="1.3"/>')
    svg_parts.append(f'<text x="{legend_x + 24}" y="{legend_y + 158}">Pareto solutions</text>')
    svg_parts.append(f'<path d="{star_path(legend_x + 8, legend_y + 184, 8.5, 3.8)}" fill="black" stroke="black" stroke-width="0.8"/>')
    svg_parts.append(f'<text x="{legend_x + 24}" y="{legend_y + 188}">TOPSIS optimal</text>')

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{ margin: 0; padding: 24px; font-family: "Times New Roman", Georgia, serif; background: #fff; color: #111; }}
  .wrap {{ width: {width}px; }}
  svg {{ max-width: 100%; height: auto; }}
  text {{ font-size: 15px; }}
  .point {{ cursor: pointer; vector-effect: non-scaling-stroke; }}
  .point:hover {{ stroke-width: 2.4; }}
  #tooltip {{
    position: fixed; display: none; pointer-events: none; background: white;
    border: 1px solid #222; padding: 8px 10px; font-size: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.16); line-height: 1.35; min-width: 210px;
  }}
</style>
</head>
<body>
<div class="wrap">
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Interactive Pareto comparison">
{''.join(svg_parts)}
</svg>
</div>
<div id="tooltip"></div>
<script>
const tooltip = document.getElementById('tooltip');
function fmt(value, digits) {{
  return Number(value).toLocaleString(undefined, {{maximumFractionDigits: digits, minimumFractionDigits: digits}});
}}
document.querySelectorAll('.point').forEach((el) => {{
  el.addEventListener('mousemove', (event) => {{
    const p = JSON.parse(el.dataset.point);
    tooltip.innerHTML = `
      <strong>${{p.series}}</strong><br>
      ${{p.kind}}<br>
      ANP: ${{fmt(p.anp, 4)}} &times;10<sup>6</sup> EUR/year<br>
      CO2 emission: ${{fmt(p.co2, 2)}} ton/year<br>
      PPFD: ${{fmt(p.ppfd, 1)}} &micro;mol m<sup>-2</sup> s<sup>-1</sup><br>
      PV: ${{p.pv}}<br>
      WT: ${{p.wt}}<br>
      Battery: ${{p.batt}}
    `;
    tooltip.style.display = 'block';
    tooltip.style.left = `${{event.clientX + 14}}px`;
    tooltip.style.top = `${{event.clientY + 14}}px`;
  }});
  el.addEventListener('mouseleave', () => {{
    tooltip.style.display = 'none';
  }});
}});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(SERIES_BY_CASE), default="fixed_light")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--output-html", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    series_list = SERIES_BY_CASE[args.case]
    output_html = args.output_html or DEFAULT_OUTPUT_HTML[args.case]
    points = build_points(output_dir, series_list)
    html_text = render_html(points, series_list, output_html.replace("_", " ").replace(".html", ""))
    html_path = output_dir / output_html
    html_path.write_text(html_text, encoding="utf-8")
    print(f"Saved interactive figure: {html_path}")
    print(f"Points included: {len(points)}")


if __name__ == "__main__":
    main()
