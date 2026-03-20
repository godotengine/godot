# Performance Dashboard

This page displays interactive benchmark results from the latest suite run.

!!! note "Data freshness"
    Charts use `assets/data/benchmark_latest.json` generated during docs build. Run `python scripts/export_benchmark_vegalite.py` locally to update from your latest benchmark run.

## Lane Scores Overview

```vegalite
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "data": {"url": "../assets/data/benchmark_latest.json"},
  "mark": {"type": "bar", "cornerRadiusEnd": 4, "tooltip": true},
  "encoding": {
    "y": {"field": "lane_id", "type": "nominal", "sort": "-x", "title": "Lane"},
    "x": {"field": "score", "type": "quantitative", "title": "Score"},
    "color": {"field": "score", "type": "quantitative", "scale": {"scheme": "viridis"}, "legend": null},
    "tooltip": [
      {"field": "lane_id", "title": "Lane"},
      {"field": "lane_name", "title": "Description"},
      {"field": "score", "title": "Score", "format": ".1f"},
      {"field": "weight", "title": "Weight", "format": ".1f"}
    ]
  },
  "width": "container",
  "height": 250,
  "title": "Weighted Lane Scores"
}
```

## Frame Timing

```vegalite
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "data": {"url": "../assets/data/benchmark_latest.json"},
  "mark": {"type": "bar", "cornerRadiusEnd": 4, "tooltip": true},
  "encoding": {
    "y": {"field": "lane_id", "type": "nominal", "sort": "-x", "title": "Lane"},
    "x": {"field": "p99_frame_ms", "type": "quantitative", "title": "Frame Time (ms)"},
    "color": {"value": "#355caa"},
    "tooltip": [
      {"field": "lane_id", "title": "Lane"},
      {"field": "p99_frame_ms", "title": "P99 Frame (ms)", "format": ".2f"},
      {"field": "avg_fps", "title": "Avg FPS", "format": ".1f"},
      {"field": "gpu_time_frame_ms", "title": "GPU Time (ms)", "format": ".2f"}
    ]
  },
  "width": "container",
  "height": 250,
  "title": "P99 Frame Time by Lane (lower is better)"
}
```

## How to Update

1. Run a benchmark: `python tests/runtime/run_benchmark.py --profile everything`
2. Export data: `python scripts/export_benchmark_vegalite.py`
3. Build docs: `python scripts/build_docs_site.py --strict`

See [Benchmark Suite Runner](../testing/benchmark-suite.md) for full benchmark documentation.
