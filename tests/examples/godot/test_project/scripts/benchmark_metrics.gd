class_name BenchmarkMetrics
extends RefCounted

static func _sorted_copy(samples: Array) -> Array:
	var sorted: Array = samples.duplicate()
	sorted.sort()
	return sorted

static func percentile(samples: Array, percentile_value: float) -> float:
	if samples.is_empty():
		return 0.0
	var sorted := _sorted_copy(samples)
	var pct := clampf(percentile_value, 0.0, 100.0)
	var idx := int(round((pct / 100.0) * float(sorted.size() - 1)))
	return float(sorted[idx])

static func summarize_samples(frame_ms_samples: Array, fps_samples: Array) -> Dictionary:
	var total_frame_ms := 0.0
	var avg_frame_ms := 0.0
	for frame_ms in frame_ms_samples:
		total_frame_ms += float(frame_ms)
	if not frame_ms_samples.is_empty():
		avg_frame_ms = total_frame_ms / float(frame_ms_samples.size())
	# Use harmonic mean: total_frames / total_time. This gives true throughput
	# and avoids inflated averages when frame times are bimodal.
	var avg_fps := 0.0
	if total_frame_ms > 0.0:
		avg_fps = float(frame_ms_samples.size()) / (total_frame_ms / 1000.0)

	var min_fps := 0.0 if fps_samples.is_empty() else float(fps_samples.min())
	var max_fps := 0.0 if fps_samples.is_empty() else float(fps_samples.max())
	var p1_fps := percentile(fps_samples, 1.0)
	var p5_fps := percentile(fps_samples, 5.0)
	var p95_frame_ms := percentile(frame_ms_samples, 95.0)
	var p99_frame_ms := percentile(frame_ms_samples, 99.0)
	var max_frame_ms := 0.0 if frame_ms_samples.is_empty() else float(frame_ms_samples.max())

	var stability := 0.0
	if avg_fps > 0.0:
		stability = clampf(p1_fps / avg_fps, 0.0, 1.0)

	return {
		"sample_count": frame_ms_samples.size(),
		"avg_fps": avg_fps,
		"min_fps": min_fps,
		"max_fps": max_fps,
		"p1_fps": p1_fps,
		"p5_fps": p5_fps,
		"avg_frame_ms": avg_frame_ms,
		"p95_frame_ms": p95_frame_ms,
		"p99_frame_ms": p99_frame_ms,
		"max_frame_ms": max_frame_ms,
		"stability": stability,
	}

static func has_samples(summary: Dictionary) -> bool:
	return int(summary.get("sample_count", 0)) > 0

static func compute_score(overall_summary: Dictionary, monitor_max: Dictionary) -> float:
	var avg_fps: float = float(overall_summary.get("avg_fps", 0.0))
	var p1_fps: float = float(overall_summary.get("p1_fps", 0.0))
	var p99_frame_ms: float = maxf(float(overall_summary.get("p99_frame_ms", 0.0)), 0.001)
	var stability: float = float(overall_summary.get("stability", 0.0))

	var fps_component: float = clampf(avg_fps / 90.0, 0.0, 1.0) * 45.0
	var low_percentile_component: float = clampf(p1_fps / 60.0, 0.0, 1.0) * 25.0
	var frame_component: float = clampf(16.6 / p99_frame_ms, 0.0, 1.0) * 20.0
	var stability_component: float = clampf(stability, 0.0, 1.0) * 10.0

	var penalty: float = 0.0
	if float(monitor_max.get("streaming_queue_pressure_active", 0.0)) > 0.0:
		penalty += 3.0
	if float(monitor_max.get("streaming_upload_bandwidth_cap_hit", 0.0)) > 0.0:
		penalty += 2.0
	if float(monitor_max.get("streaming_chunk_load_cap_hit", 0.0)) > 0.0:
		penalty += 2.0

	return clampf(fps_component + low_percentile_component + frame_component + stability_component - penalty, 0.0, 100.0)

static func _setting_value(settings: Dictionary, key: String, fallback: Variant) -> Variant:
	if settings.has(key):
		return settings[key]
	return fallback

static func build_recommendations(report: Dictionary) -> Array[Dictionary]:
	var recommendations: Array[Dictionary] = []
	var overall: Dictionary = report.get("overall", {})
	var monitor_max: Dictionary = report.get("monitor_max", {})
	var settings: Dictionary = report.get("project_settings", {})

	var avg_fps: float = float(overall.get("avg_fps", 0.0))
	var p1_fps: float = float(overall.get("p1_fps", 0.0))
	var p99_frame_ms: float = float(overall.get("p99_frame_ms", 0.0))
	var streaming_vram_usage_mb: float = float(monitor_max.get("streaming_vram_usage_mb", 0.0))
	var streaming_visible_change_ratio: float = float(monitor_max.get("streaming_visible_change_ratio", 0.0))
	var queue_pressure: float = float(monitor_max.get("streaming_queue_pressure_active", 0.0))
	var upload_cap_hit: float = float(monitor_max.get("streaming_upload_bandwidth_cap_hit", 0.0))
	var chunk_load_cap_hit: float = float(monitor_max.get("streaming_chunk_load_cap_hit", 0.0))
	var lod_reduction_ratio: float = float(monitor_max.get("lod_reduction_ratio_pct", 0.0))
	var lod_transitions: float = float(monitor_max.get("lod_transitions_this_frame", 0.0))

	if avg_fps < 45.0 or p1_fps < 28.0 or p99_frame_ms > 33.3:
		recommendations.append({
			"setting": "rendering/gaussian_splatting/lod/max_distance",
			"current": _setting_value(settings, "rendering/gaussian_splatting/lod/max_distance", 50.0),
			"suggested": 35.0,
			"reason": "Frame consistency indicates splat density is too high at distance.",
			"tradeoff": "Slightly earlier distance culling for far splats.",
		})
		recommendations.append({
			"setting": "rendering/gaussian_splatting/lod/bias",
			"current": _setting_value(settings, "rendering/gaussian_splatting/lod/bias", 1.0),
			"suggested": 1.15,
			"reason": "Higher LOD bias reduces per-frame shading/raster pressure.",
			"tradeoff": "Marginal detail loss on medium/far splats.",
		})

	if queue_pressure > 0.0 or upload_cap_hit > 0.0 or chunk_load_cap_hit > 0.0 or streaming_visible_change_ratio > 0.8:
		recommendations.append({
			"setting": "rendering/gaussian_splatting/streaming/max_chunk_loads_per_frame",
			"current": _setting_value(settings, "rendering/gaussian_splatting/streaming/max_chunk_loads_per_frame", 8),
			"suggested": 12,
			"reason": "Streaming queue pressure/cap hits suggest chunk ingress is rate-limited.",
			"tradeoff": "Higher upload bursts can increase transient frame spikes.",
		})
		recommendations.append({
			"setting": "rendering/gaussian_splatting/streaming/pack_worker_threads",
			"current": _setting_value(settings, "rendering/gaussian_splatting/streaming/pack_worker_threads", 4),
			"suggested": 6,
			"reason": "Packing throughput appears to be the bottleneck during streaming transitions.",
			"tradeoff": "Additional CPU thread pressure on low-core systems.",
		})

	if streaming_vram_usage_mb > 1800.0:
		recommendations.append({
			"setting": "rendering/gaussian_splatting/streaming/vram_budget_mb",
			"current": _setting_value(settings, "rendering/gaussian_splatting/streaming/vram_budget_mb", 2048),
			"suggested": 1536,
			"reason": "VRAM usage approached high-water mark, increasing eviction risk on smaller GPUs.",
			"tradeoff": "More aggressive streaming may increase pop-in if camera moves quickly.",
		})

	if p1_fps < 35.0:
		recommendations.append({
			"setting": "rendering/gaussian_splatting/animation/wind_strength",
			"current": _setting_value(settings, "rendering/gaussian_splatting/animation/wind_strength", 0.0),
			"suggested": 0.4,
			"reason": "Low percentile FPS indicates animation deformation cost is too high for current hardware.",
			"tradeoff": "Reduced wind motion amplitude.",
		})
		recommendations.append({
			"setting": "rendering/gaussian_splatting/effects/max_effectors",
			"current": _setting_value(settings, "rendering/gaussian_splatting/effects/max_effectors", 0),
			"suggested": 2,
			"reason": "Effector-heavy frames are likely contributing to frame-time outliers.",
			"tradeoff": "Fewer concurrent procedural effect regions.",
		})

	if lod_transitions > 12.0 and lod_reduction_ratio > 40.0:
		recommendations.append({
			"setting": "rendering/gaussian_splatting/lod/hysteresis_zone",
			"current": _setting_value(settings, "rendering/gaussian_splatting/lod/hysteresis_zone", 0.5),
			"suggested": 0.8,
			"reason": "Frequent LOD swaps indicate transition hysteresis is too tight for motion profile.",
			"tradeoff": "Slightly slower response to rapid camera distance changes.",
		})

	if recommendations.is_empty():
		recommendations.append({
			"setting": "none",
			"current": "n/a",
			"suggested": "keep current settings",
			"reason": "Benchmark stayed within target envelope for this workload.",
			"tradeoff": "No change required.",
		})

	return recommendations
