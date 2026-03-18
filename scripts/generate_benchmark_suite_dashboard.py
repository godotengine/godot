#!/usr/bin/env python3
"""Render a lightweight HTML/SVG dashboard from benchmark_suite_report.json."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


SVG_WIDTH = 1120
SVG_HEIGHT = 420
SVG_MARGIN_LEFT = 180
SVG_MARGIN_RIGHT = 40
SVG_MARGIN_TOP = 50
SVG_MARGIN_BOTTOM = 70
BAR_GAP = 12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark suite HTML/SVG dashboard.")
    parser.add_argument("--suite-report", type=Path, required=True, help="Path to benchmark_suite_report.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated dashboard assets")
    return parser.parse_args()


def _read_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        metric = float(value)
        if math.isfinite(metric):
            return metric
    return None


def _relative_path(path_text: str, output_dir: Path) -> str:
    path = Path(path_text)
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _svg_bar_chart(
    title: str,
    subtitle: str,
    lane_results: list[dict[str, Any]],
    metric_key: str,
    value_suffix: str,
    fill_color: str,
) -> str:
    values = [_safe_float(lane.get(metric_key)) for lane in lane_results]
    numeric_values = [value for value in values if value is not None]
    max_value = max(numeric_values) if numeric_values else 1.0
    plot_width = SVG_WIDTH - SVG_MARGIN_LEFT - SVG_MARGIN_RIGHT
    plot_height = SVG_HEIGHT - SVG_MARGIN_TOP - SVG_MARGIN_BOTTOM
    bar_height = max(18, int((plot_height - BAR_GAP * max(len(lane_results) - 1, 0)) / max(len(lane_results), 1)))
    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        f'<rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="#f8f6f1"/>',
        f'<text x="{SVG_MARGIN_LEFT}" y="28" font-family="Georgia, serif" font-size="24" fill="#182126">{html.escape(title)}</text>',
        f'<text x="{SVG_MARGIN_LEFT}" y="48" font-family="Menlo, Consolas, monospace" font-size="12" fill="#51606b">{html.escape(subtitle)}</text>',
    ]
    for idx, lane in enumerate(lane_results):
        lane_id = str(lane.get("lane_id", f"lane_{idx + 1}"))
        value = values[idx]
        top = SVG_MARGIN_TOP + idx * (bar_height + BAR_GAP)
        label_y = top + bar_height * 0.72
        elements.append(
            f'<text x="{SVG_MARGIN_LEFT - 12}" y="{label_y:.1f}" text-anchor="end" font-family="Menlo, Consolas, monospace" font-size="13" fill="#182126">{html.escape(lane_id)}</text>'
        )
        elements.append(
            f'<rect x="{SVG_MARGIN_LEFT}" y="{top}" width="{plot_width}" height="{bar_height}" fill="#e5ddd0" rx="6" ry="6"/>'
        )
        if value is None or max_value <= 0.0:
            elements.append(
                f'<text x="{SVG_MARGIN_LEFT + 8}" y="{label_y:.1f}" font-family="Menlo, Consolas, monospace" font-size="12" fill="#7b6f63">n/a</text>'
            )
            continue
        width = max(2.0, (value / max_value) * plot_width)
        elements.append(
            f'<rect x="{SVG_MARGIN_LEFT}" y="{top}" width="{width:.1f}" height="{bar_height}" fill="{fill_color}" rx="6" ry="6"/>'
        )
        elements.append(
            f'<text x="{SVG_MARGIN_LEFT + width + 8:.1f}" y="{label_y:.1f}" font-family="Menlo, Consolas, monospace" font-size="12" fill="#182126">{value:.2f}{html.escape(value_suffix)}</text>'
        )
    elements.append("</svg>")
    return "\n".join(elements)


def _write_svg(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"[dashboard] wrote {path}")


def _format_metric(value: Any, decimals: int = 2) -> str:
    metric = _safe_float(value)
    if metric is None:
        return "n/a"
    return f"{metric:.{decimals}f}"


def _render_html(report: dict[str, Any], output_dir: Path, chart_files: list[Path]) -> str:
    lane_results = list(report.get("lane_results", []))
    aggregate_score = _format_metric(report.get("aggregate_score"))
    capture_dir = str(report.get("capture_dir", ""))
    charts_html = "\n".join(
        f'<section class="panel"><h2>{html.escape(chart.stem.replace("_", " ").title())}</h2><img src="{html.escape(_relative_path(str(chart), output_dir))}" alt="{html.escape(chart.name)}"></section>'
        for chart in chart_files
    )
    table_rows: list[str] = []
    gallery_items: list[str] = []
    for lane in lane_results:
        table_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(lane.get('lane_id', 'n/a')))}</code></td>"
            f"<td>{html.escape(_format_metric(lane.get('score')))}</td>"
            f"<td>{html.escape(_format_metric(lane.get('avg_fps')))}</td>"
            f"<td>{html.escape(_format_metric(lane.get('p99_frame_ms')))}</td>"
            f"<td>{html.escape(_format_metric(lane.get('gpu_time_frame_ms')))}</td>"
            f"<td>{html.escape(_format_metric(lane.get('capture_ssim_min'), 3))}</td>"
            f"<td>{html.escape(_format_metric(lane.get('capture_psnr_min')))}</td>"
            "</tr>"
        )
        report_dict = lane.get("report")
        if not isinstance(report_dict, dict):
            continue
        for capture in report_dict.get("captures", []):
            if not isinstance(capture, dict) or not capture.get("saved"):
                continue
            capture_path = str(capture.get("capture_path", ""))
            if not capture_path:
                continue
            gallery_items.append(
                "<figure class=\"capture\">"
                f"<img src=\"{html.escape(_relative_path(capture_path, output_dir))}\" alt=\"{html.escape(str(capture.get('capture_id', 'capture')))}\">"
                f"<figcaption><code>{html.escape(str(lane.get('lane_id', 'n/a')))}</code> "
                f"{html.escape(str(capture.get('capture_id', 'capture')))} "
                f"SSIM {html.escape(_format_metric(capture.get('ssim'), 3))} "
                f"PSNR {html.escape(_format_metric(capture.get('psnr')))}</figcaption>"
                "</figure>"
            )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GodotGS Benchmark Dashboard</title>
  <style>
    :root {{
      --paper: #f8f6f1;
      --ink: #182126;
      --muted: #51606b;
      --accent: #1f8f6b;
      --line: #d7cec2;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #ffffff 0%, transparent 35%),
        linear-gradient(135deg, #f8f6f1 0%, #efe7db 100%);
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card, .panel {{
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px 20px;
      box-shadow: 0 10px 30px rgba(24, 33, 38, 0.08);
      backdrop-filter: blur(10px);
    }}
    .metric {{
      font-size: 34px;
      line-height: 1;
      margin-bottom: 8px;
      color: var(--accent);
      font-weight: 700;
    }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
    }}
    img {{
      max-width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
    .capture {{
      margin: 0;
    }}
    figcaption {{
      margin-top: 8px;
      font-size: 13px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>
    <section class="summary">
      <div class="card">
        <div class="metric">{aggregate_score}</div>
        <div class="meta">Aggregate score</div>
      </div>
      <div class="card">
        <div class="metric">{html.escape(str(report.get('profile', 'n/a')))}</div>
        <div class="meta">Profile</div>
      </div>
      <div class="card">
        <div class="metric">{len(lane_results)}</div>
        <div class="meta">Lanes</div>
      </div>
      <div class="card">
        <div class="metric">{html.escape(Path(capture_dir).name if capture_dir else 'disabled')}</div>
        <div class="meta">Capture output</div>
      </div>
    </section>
    {charts_html}
    <section class="panel">
      <h2>Lane Metrics</h2>
      <table>
        <thead>
          <tr><th>Lane</th><th>Score</th><th>Avg FPS</th><th>P99 ms</th><th>GPU ms</th><th>SSIM min</th><th>PSNR min</th></tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </section>
    <section class="panel">
      <h2>Captured Frames</h2>
      <div class="gallery">
        {''.join(gallery_items) if gallery_items else '<p>No captures were generated for this run.</p>'}
      </div>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = _read_report(args.suite_report)
    lane_results = list(report.get("lane_results", []))
    charts = [
        ("benchmark_suite_scores.svg", "Lane Score", "Weighted throughput/fidelity score per lane", "score", "", "#1f8f6b"),
        ("benchmark_suite_fps.svg", "Average FPS", "Mean FPS by lane", "avg_fps", " fps", "#c96b2c"),
        ("benchmark_suite_gpu_ms.svg", "GPU Frame Time", "Lower is better", "gpu_time_frame_ms", " ms", "#355caa"),
        ("benchmark_suite_ssim.svg", "SSIM Minimum", "Visual similarity for captured lanes", "capture_ssim_min", "", "#7a4ea3"),
        ("benchmark_suite_psnr.svg", "PSNR Minimum", "Visual error envelope for captured lanes", "capture_psnr_min", " dB", "#9f3d3d"),
    ]
    chart_files: list[Path] = []
    for filename, title, subtitle, metric_key, suffix, color in charts:
        chart_path = args.output_dir / filename
        svg = _svg_bar_chart(title, subtitle, lane_results, metric_key, suffix, color)
        _write_svg(chart_path, svg)
        chart_files.append(chart_path)
    dashboard_path = args.output_dir / "benchmark_suite_dashboard.html"
    dashboard_path.write_text(_render_html(report, args.output_dir, chart_files), encoding="utf-8")
    print(f"[dashboard] wrote {dashboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
