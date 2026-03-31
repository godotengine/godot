# Performance Dashboard

This page surfaces the current published benchmark snapshot and the suite lanes that are expected to grow it.

Charts use `assets/data/benchmark_latest.json` generated during docs build.
The current public dataset contains one committed result row, and the coverage table below shows the user-relevant benchmark lanes already defined in the suite.

## Current Public Snapshot

| Lane | Purpose | Score | Avg FPS | P99 Frame (ms) | GPU Time (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `static_baseline` | Low-noise raster baseline | 90.7 | 74.0 | 15.62 | 0.0 |

The snapshot above is the current committed public result. It is the reference row used by the charts below until more published scenarios are added.

## Coverage Map

| Lane | Purpose | Status |
| --- | --- | --- |
| `static_baseline` | Low-noise raster baseline | Published in `benchmark_latest.json` |
| `streaming_corridor` | Camera sweep stressing chunk turnover | Defined in the benchmark suite, not yet published |
| `city_flyover` | High-altitude visibility-change stress | Defined in the benchmark suite, not yet published |
| `instance_storm` | Many-instance submission pressure | Defined in the benchmark suite, not yet published |
| `lighting_stress` | Animated light and shading stress | Defined in the benchmark suite, not yet published |
| `unified_composite` | Integrated all-systems composite lane | Defined in the benchmark suite, not yet published |

## Lane Scores Overview

```vegalite
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "data": {"url": "../assets/data/benchmark_latest.json"},
  "mark": {"type": "bar", "cornerRadiusEnd": 4, "tooltip": true},
  "encoding": {
    "y": {"field": "lane_id", "type": "nominal", "sort": "-x", "title": "Lane"},
    "x": {"field": "score", "type": "quantitative", "title": "Score"},
    "color": {"value": "#355caa"},
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
3. Update the current snapshot table above when the published lane set changes.
4. Build docs: `python scripts/build_docs_site.py --strict`

See [Benchmark Suite Runner](../testing/benchmark-suite.md) for full benchmark documentation.
