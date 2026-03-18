extends SceneTree

const METRICS_MARKER := "[RUNTIME_METRICS]"
const FAIL_MARKER := "[RUNTIME_FAIL]"

const TIERS := [
    {"name": "tier_2_5m", "size": 2500000},
    {"name": "tier_5m", "size": 5000000},
    {"name": "tier_6_5m", "size": 6500000}
]
const SAMPLE_FRAMES := 90
const WARMUP_FRAMES := 3
const FIRST_VISIBLE_TIMEOUT_FRAMES := 180

var renderer: GaussianSplatRenderer
var failures: Array[String] = []

func _init() -> void:
    call_deferred("_run")

func _record_failure(message: String, context: Dictionary = {}) -> void:
    if not context.is_empty():
        message = "%s | context=%s" % [message, str(context)]
    failures.append(message)
    push_error("%s %s" % [FAIL_MARKER, message])

func _average(values: Array) -> float:
    if values.is_empty():
        return 0.0
    var total := 0.0
    for v in values:
        total += float(v)
    return total / float(values.size())

func _percentile(values: Array, percentile: float) -> float:
    if values.is_empty():
        return 0.0
    var sorted_values: Array = values.duplicate()
    sorted_values.sort()
    var idx := int(round(clampf(percentile, 0.0, 1.0) * float(sorted_values.size() - 1)))
    return float(sorted_values[idx])

func _build_dataset(size: int) -> GaussianData:
    var data := GaussianData.new()
    var positions := PackedVector3Array()
    var colors := PackedColorArray()
    var scales := PackedVector3Array()
    var sh_dc := PackedFloat32Array()
    var opacities := PackedFloat32Array()

    var rng := RandomNumberGenerator.new()
    rng.seed = 0x5A17A11 ^ size

    data.resize(size)
    positions.resize(size)
    colors.resize(size)
    scales.resize(size)
    sh_dc.resize(size * 3)
    opacities.resize(size)

    for i in range(size):
        var pos = Vector3(
            rng.randf_range(-80.0, 80.0),
            rng.randf_range(-80.0, 80.0),
            rng.randf_range(-80.0, 80.0)
        )
        positions[i] = pos
        colors[i] = Color(rng.randf(), rng.randf(), rng.randf(), 0.85)
        var sh_idx := i * 3
        sh_dc[sh_idx + 0] = colors[i].r
        sh_dc[sh_idx + 1] = colors[i].g
        sh_dc[sh_idx + 2] = colors[i].b
        opacities[i] = colors[i].a
        scales[i] = Vector3.ONE * rng.randf_range(0.1, 1.0)

    data.set_positions(positions)
    data.set_scales(scales)
    data.set_opacities(opacities)
    data.set_spherical_harmonics(sh_dc)
    return data

func _read_int(stats: Dictionary, keys: Array[String], default_value: int = 0) -> int:
    for key in keys:
        if stats.has(key):
            return int(stats.get(key, default_value))
    return default_value

func _read_float(stats: Dictionary, keys: Array[String], default_value: float = 0.0) -> float:
    for key in keys:
        if stats.has(key):
            return float(stats.get(key, default_value))
    return default_value

func _extract_streaming_state(stats: Dictionary) -> Dictionary:
    if stats.has("streaming_state") and stats["streaming_state"] is Dictionary:
        return stats["streaming_state"]
    return {}

func _extract_streaming_diagnostics(stream_state: Dictionary) -> Dictionary:
    if stream_state.has("diagnostics") and stream_state["diagnostics"] is Dictionary:
        return stream_state["diagnostics"]
    return {}

func _read_optional_float(stats: Dictionary, keys: Array[String]) -> Variant:
    for key in keys:
        if stats.has(key):
            return float(stats.get(key, 0.0))
    return null

func _read_optional_string(stats: Dictionary, keys: Array[String]) -> Variant:
    for key in keys:
        if stats.has(key):
            return String(stats.get(key, ""))
    return null

func _read_optional_bool(stats: Dictionary, keys: Array[String]) -> Variant:
    for key in keys:
        if stats.has(key):
            return bool(stats.get(key, false))
    return null

func _run_tier(name: String, size: int) -> Dictionary:
    print("\n[Probe] Running %s (%d splats)" % [name, size])
    if renderer.get_max_splats() < size:
        renderer.set_max_splats(size)

    var dataset := _build_dataset(size)
    var set_err := renderer.set_gaussian_data(dataset)
    if set_err != OK:
        _record_failure("set_gaussian_data failed", {"tier": name, "size": size, "err": set_err})
        return {"name": name, "size": size, "set_error": set_err}

    for i in range(WARMUP_FRAMES):
        await process_frame
        renderer.force_sort_for_view(Transform3D.IDENTITY)

    var frame_ms_values: Array = []
    var first_visible_ms := -1.0
    var bench_start_usec := Time.get_ticks_usec()
    var fallback_frames := 0
    var source_frames := 0

    var max_pack_q := 0
    var max_upload_q := 0
    var max_sync_fallback_q := 0
    var max_chunks_evicted_frame := 0
    var max_chunks_loaded_frame := 0
    var max_scheduler_update_ms := 0.0
    var max_visibility_ms := 0.0
    var max_load_ms := 0.0
    var max_prefetch_ms := 0.0
    var max_upload_mb_frame := 0.0
    var sum_upload_mb := 0.0
    var sum_chunks_evicted := 0
    var sum_chunks_loaded := 0

    var diagnostics_categories: Dictionary = {}
    var queue_pressure_reason_sources: Dictionary = {}
    var last_queue_pressure_reason_source: Variant = null
    var max_pack_queue_latency_max_ms: Variant = null
    var max_upload_queue_latency_max_ms: Variant = null
    var max_pack_mutex_wait_max_ms: Variant = null
    var saw_scheduler_queue_pressure_scan_throttle_active := false
    var scheduler_queue_pressure_scan_throttle_active_known := false
    var last_stats: Dictionary = {}

    for frame_idx in range(SAMPLE_FRAMES):
        var frame_start := Time.get_ticks_usec()
        await process_frame
        renderer.force_sort_for_view(Transform3D.IDENTITY)
        var stats: Dictionary = renderer.get_render_stats()
        last_stats = stats

        var frame_ms := float(Time.get_ticks_usec() - frame_start) / 1000.0
        frame_ms_values.append(frame_ms)

        var visible := _read_int(stats, ["visible_after_culling", "visible_splats", "cull_cpu_visible_count"])
        if first_visible_ms < 0.0 and visible > 0:
            first_visible_ms = float(Time.get_ticks_usec() - bench_start_usec) / 1000.0

        var data_source := String(stats.get("data_source", ""))
        if not data_source.is_empty():
            source_frames += 1
            if data_source.findn("stream") == -1:
                fallback_frames += 1

        var s := _extract_streaming_state(stats)
        if not s.is_empty():
            var diagnostics := _extract_streaming_diagnostics(s)
            max_pack_q = max(max_pack_q, _read_int(s, ["scheduler_pack_queue_depth"]))
            max_upload_q = max(max_upload_q, _read_int(s, ["scheduler_upload_queue_depth", "pending_uploads"]))
            max_sync_fallback_q = max(max_sync_fallback_q, _read_int(s, ["scheduler_sync_fallback_queue_depth"]))
            var evicted_frame := _read_int(s, ["chunks_evicted_this_frame"])
            var loaded_frame := _read_int(s, ["chunks_loaded_this_frame"])
            max_chunks_evicted_frame = max(max_chunks_evicted_frame, evicted_frame)
            max_chunks_loaded_frame = max(max_chunks_loaded_frame, loaded_frame)
            sum_chunks_evicted += evicted_frame
            sum_chunks_loaded += loaded_frame

            var upload_mb_frame := _read_float(s, ["upload_mb_this_frame"])
            max_upload_mb_frame = maxf(max_upload_mb_frame, upload_mb_frame)
            sum_upload_mb += upload_mb_frame

            max_scheduler_update_ms = maxf(max_scheduler_update_ms, _read_float(s, ["scheduler_update_cpu_ms"]))
            max_visibility_ms = maxf(max_visibility_ms, _read_float(s, ["scheduler_visibility_cpu_ms"]))
            max_load_ms = maxf(max_load_ms, _read_float(s, ["scheduler_load_cpu_ms"]))
            max_prefetch_ms = maxf(max_prefetch_ms, _read_float(s, ["scheduler_prefetch_cpu_ms"]))

            var category := String(s.get("diagnostics_category", ""))
            if not category.is_empty():
                diagnostics_categories[category] = int(diagnostics_categories.get(category, 0)) + 1

            var queue_pressure_reason_source: Variant = _read_optional_string(
                diagnostics,
                ["queue_pressure_reason_source", "queue_pressure_source"]
            )
            if queue_pressure_reason_source == null:
                queue_pressure_reason_source = _read_optional_string(
                    s,
                    ["queue_pressure_reason_source", "queue_pressure_source"]
                )
            if queue_pressure_reason_source != null and not String(queue_pressure_reason_source).is_empty():
                last_queue_pressure_reason_source = queue_pressure_reason_source
                queue_pressure_reason_sources[String(queue_pressure_reason_source)] = int(
                    queue_pressure_reason_sources.get(queue_pressure_reason_source, 0)
                ) + 1

            var pack_queue_latency_max_ms: Variant = _read_optional_float(
                diagnostics,
                ["pack_queue_latency_max_ms"]
            )
            if pack_queue_latency_max_ms == null:
                pack_queue_latency_max_ms = _read_optional_float(s, ["pack_queue_latency_max_ms"])
            if pack_queue_latency_max_ms != null:
                max_pack_queue_latency_max_ms = float(pack_queue_latency_max_ms) if max_pack_queue_latency_max_ms == null else maxf(
                    float(max_pack_queue_latency_max_ms),
                    float(pack_queue_latency_max_ms)
                )

            var upload_queue_latency_max_ms: Variant = _read_optional_float(
                diagnostics,
                ["upload_queue_latency_max_ms"]
            )
            if upload_queue_latency_max_ms == null:
                upload_queue_latency_max_ms = _read_optional_float(s, ["upload_queue_latency_max_ms"])
            if upload_queue_latency_max_ms != null:
                max_upload_queue_latency_max_ms = float(
                    upload_queue_latency_max_ms
                ) if max_upload_queue_latency_max_ms == null else maxf(
                    float(max_upload_queue_latency_max_ms),
                    float(upload_queue_latency_max_ms)
                )

            var pack_mutex_wait_max_ms: Variant = _read_optional_float(
                diagnostics,
                ["pack_mutex_wait_max_ms"]
            )
            if pack_mutex_wait_max_ms == null:
                pack_mutex_wait_max_ms = _read_optional_float(s, ["pack_mutex_wait_max_ms"])
            if pack_mutex_wait_max_ms != null:
                max_pack_mutex_wait_max_ms = float(pack_mutex_wait_max_ms) if max_pack_mutex_wait_max_ms == null else maxf(
                    float(max_pack_mutex_wait_max_ms),
                    float(pack_mutex_wait_max_ms)
                )

            var throttle_active: Variant = _read_optional_bool(
                s,
                ["scheduler_queue_pressure_scan_throttle_active"]
            )
            if throttle_active == null:
                throttle_active = _read_optional_bool(
                    diagnostics,
                    ["scheduler_queue_pressure_scan_throttle_active"]
                )
            if throttle_active != null:
                scheduler_queue_pressure_scan_throttle_active_known = true
                if bool(throttle_active):
                    saw_scheduler_queue_pressure_scan_throttle_active = true

    if first_visible_ms < 0.0:
        for i in range(max(0, FIRST_VISIBLE_TIMEOUT_FRAMES - SAMPLE_FRAMES)):
            await process_frame
            renderer.force_sort_for_view(Transform3D.IDENTITY)
            var late_stats: Dictionary = renderer.get_render_stats()
            var late_visible := _read_int(late_stats, ["visible_after_culling", "visible_splats", "cull_cpu_visible_count"])
            if late_visible > 0:
                first_visible_ms = float(Time.get_ticks_usec() - bench_start_usec) / 1000.0
                last_stats = late_stats
                break

    var frame_avg_ms: float = _average(frame_ms_values)
    var frame_p95_ms: float = _percentile(frame_ms_values, 0.95)
    var fps_avg: float = 1000.0 / maxf(0.001, frame_avg_ms)
    var fps_p95: float = 1000.0 / maxf(0.001, frame_p95_ms)

    var stream_state := _extract_streaming_state(last_stats)
    var scheduler_stalls := _read_int(stream_state, ["scheduler_stall_frames"])
    var upload_stalls := _read_int(stream_state, ["upload_stall_frames"])
    var sync_fallback_stalls := _read_int(stream_state, ["sync_fallback_stall_frames"])

    return {
        "name": name,
        "size": size,
        "frame_avg_ms": frame_avg_ms,
        "frame_p95_ms": frame_p95_ms,
        "fps_avg": fps_avg,
        "fps_p95": fps_p95,
        "first_visible_ms": first_visible_ms,
        "fallback_rate": 0.0 if source_frames <= 0 else float(fallback_frames) / float(source_frames),
        "source_frames": source_frames,
        "max_pack_queue_depth": max_pack_q,
        "max_upload_queue_depth": max_upload_q,
        "max_sync_fallback_queue_depth": max_sync_fallback_q,
        "max_chunks_loaded_this_frame": max_chunks_loaded_frame,
        "max_chunks_evicted_this_frame": max_chunks_evicted_frame,
        "sum_chunks_loaded": sum_chunks_loaded,
        "sum_chunks_evicted": sum_chunks_evicted,
        "max_upload_mb_this_frame": max_upload_mb_frame,
        "sum_upload_mb": sum_upload_mb,
        "max_scheduler_update_cpu_ms": max_scheduler_update_ms,
        "max_scheduler_visibility_cpu_ms": max_visibility_ms,
        "max_scheduler_load_cpu_ms": max_load_ms,
        "max_scheduler_prefetch_cpu_ms": max_prefetch_ms,
        "scheduler_stall_frames": scheduler_stalls,
        "upload_stall_frames": upload_stalls,
        "sync_fallback_stall_frames": sync_fallback_stalls,
        "queue_pressure_reason_source": last_queue_pressure_reason_source,
        "queue_pressure_reason_sources": queue_pressure_reason_sources,
        "pack_queue_latency_max_ms": max_pack_queue_latency_max_ms,
        "upload_queue_latency_max_ms": max_upload_queue_latency_max_ms,
        "pack_mutex_wait_max_ms": max_pack_mutex_wait_max_ms,
        "scheduler_queue_pressure_scan_throttle_active": saw_scheduler_queue_pressure_scan_throttle_active,
        "scheduler_queue_pressure_scan_throttle_active_optional":
            saw_scheduler_queue_pressure_scan_throttle_active if scheduler_queue_pressure_scan_throttle_active_known else null,
        "diagnostics_categories": diagnostics_categories,
        "last_data_source": String(last_stats.get("data_source", "")),
        "last_visible_splats": _read_int(last_stats, ["visible_after_culling", "visible_splats", "cull_cpu_visible_count"]),
        "last_total_splats": _read_int(last_stats, ["total_splats", "uploaded_splat_count", "buffer_manager_count"]),
        "last_sort_avg_ms": _read_float(last_stats, ["gpu_sorter_avg_sort_ms", "gpu_sorter_avg_sort_time_ms"])
    }

func _run() -> void:
    renderer = GaussianSplatRenderer.new()
    if renderer == null:
        _record_failure("Failed to instantiate GaussianSplatRenderer")
        print("%s %s" % [METRICS_MARKER, JSON.stringify({"status": "failed", "failures": failures})])
        quit(1)
        return
    renderer.initialize()

    var tier_results: Array = []
    for tier in TIERS:
        var result := await _run_tier(String(tier["name"]), int(tier["size"]))
        tier_results.append(result)

    var payload := {
        "status": "passed" if failures.is_empty() else "failed",
        "failures": failures,
        "tiers": tier_results
    }
    print("%s %s" % [METRICS_MARKER, JSON.stringify(payload)])
    quit(0 if failures.is_empty() else 1)
