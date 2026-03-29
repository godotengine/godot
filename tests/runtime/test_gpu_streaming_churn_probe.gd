extends SceneTree

const METRICS_MARKER := "[RUNTIME_METRICS]"
const FAIL_MARKER := "[RUNTIME_FAIL]"

const DATASET_SIZE := 5000000
const SAMPLE_FRAMES := 180
const WARMUP_FRAMES := 5
const TAIL_VISIBILITY_CONFIRMATION_FRAMES := 6
# Keep in sync with GaussianStreamingSystem::DiagnosticsState::STALL_THRESHOLD_FRAMES.
const SCHEDULER_STALL_FAIL_FRAMES := 30
const FRAME_P95_SPIKE_MIN_MS := 20.0
const FRAME_P95_TO_AVG_SPIKE_FAIL_RATIO := 2.2
const QUEUE_PRESSURE_STARVATION_MIN_FRAMES := 32
const QUEUE_PRESSURE_STARVATION_FAIL_RATIO := 0.85
const SCHEDULER_CPU_P95_TO_FRAME_P95_FAIL_RATIO := 0.85
const QUEUE_PRESSURE_HIGH_MUTEX_WAIT_MIN_MS := 8.0
const QUEUE_PRESSURE_HIGH_MUTEX_WAIT_FAIL_MIN_FRAMES := 30
const QUEUE_PRESSURE_HIGH_MUTEX_WAIT_FAIL_RATIO := 0.80
const UPLOAD_PROGRESS_EPSILON_MB := 0.01
const CAMERA_PATH_POINTS := [
    Vector3(-180.0, 35.0, -180.0),
    Vector3(180.0, 35.0, -180.0),
    Vector3(180.0, 35.0, 180.0),
    Vector3(-180.0, 35.0, 180.0)
]

# Force constrained residency to expose load/evict churn behaviour.
const OVERRIDES := {
    "rendering/gaussian_splatting/streaming/vram_budget_mb": 768,
    "rendering/gaussian_splatting/streaming/min_chunks_in_vram": 4,
    "rendering/gaussian_splatting/streaming/max_chunks_in_vram": 14,
    "rendering/gaussian_splatting/streaming/max_chunk_loads_per_frame": 8,
    "rendering/gaussian_splatting/streaming/max_prefetch_loads_per_frame": 4,
    "rendering/gaussian_splatting/streaming/queue_pressure_candidate_scan_throttle_enabled": true,
    "rendering/gaussian_splatting/streaming/queue_pressure_candidate_scan_throttle_min_queue_depth": 1,
    "rendering/gaussian_splatting/streaming/queue_pressure_visible_scan_cap": 24,
    "rendering/gaussian_splatting/streaming/queue_pressure_prefetch_scan_cap": 24,
    "rendering/gaussian_splatting/streaming/max_upload_mb_per_frame": 48,
    "rendering/gaussian_splatting/streaming/max_upload_mb_per_slice": 8
}

var renderer: GaussianSplatRenderer
var failures: Array[String] = []
var previous_settings: Dictionary = {}

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
    for value in values:
        total += float(value)
    return total / float(values.size())

func _percentile(values: Array, percentile: float) -> float:
    if values.is_empty():
        return 0.0
    var sorted_values: Array = values.duplicate()
    sorted_values.sort()
    var idx := int(round(clampf(percentile, 0.0, 1.0) * float(sorted_values.size() - 1)))
    return float(sorted_values[idx])

func _read_int(stats: Dictionary, key: String, default_value: int = 0) -> int:
    if stats.has(key):
        return int(stats.get(key, default_value))
    return default_value

func _read_float(stats: Dictionary, key: String, default_value: float = 0.0) -> float:
    if stats.has(key):
        return float(stats.get(key, default_value))
    return default_value

func _read_optional_float(stats: Dictionary, keys: Array[String]) -> Variant:
    for key in keys:
        if stats.has(key):
            return float(stats.get(key, 0.0))
    return null

func _read_optional_int(stats: Dictionary, keys: Array[String]) -> Variant:
    for key in keys:
        if stats.has(key):
            return int(stats.get(key, 0))
    return null

func _read_optional_bool(stats: Dictionary, keys: Array[String]) -> Variant:
    for key in keys:
        if stats.has(key):
            return bool(stats.get(key, false))
    return null

func _read_optional_string(stats: Dictionary, keys: Array[String]) -> Variant:
    for key in keys:
        if stats.has(key):
            return String(stats.get(key, ""))
    return null

func _extract_diagnostics(stream_state: Dictionary) -> Dictionary:
    if stream_state.has("diagnostics") and stream_state["diagnostics"] is Dictionary:
        return stream_state["diagnostics"]
    return {}

func _build_dataset(size: int) -> GaussianData:
    var data := GaussianData.new()
    var positions := PackedVector3Array()
    var scales := PackedVector3Array()
    var sh_dc := PackedFloat32Array()
    var opacities := PackedFloat32Array()

    var rng := RandomNumberGenerator.new()
    rng.seed = 0xC117E22 ^ size

    data.resize(size)
    positions.resize(size)
    scales.resize(size)
    sh_dc.resize(size * 3)
    opacities.resize(size)

    for i in range(size):
        positions[i] = Vector3(
            rng.randf_range(-90.0, 90.0),
            rng.randf_range(-40.0, 40.0),
            rng.randf_range(-90.0, 90.0)
        )
        var r := rng.randf()
        var g := rng.randf()
        var b := rng.randf()
        var sh_idx := i * 3
        sh_dc[sh_idx + 0] = r
        sh_dc[sh_idx + 1] = g
        sh_dc[sh_idx + 2] = b
        opacities[i] = 0.85
        scales[i] = Vector3.ONE * rng.randf_range(0.1, 1.0)

    data.set_positions(positions)
    data.set_scales(scales)
    data.set_opacities(opacities)
    data.set_spherical_harmonics(sh_dc)
    return data

func _capture_settings() -> void:
    previous_settings.clear()
    for key in OVERRIDES.keys():
        previous_settings[key] = ProjectSettings.get_setting(key)

func _apply_overrides() -> void:
    _capture_settings()
    for key in OVERRIDES.keys():
        ProjectSettings.set_setting(key, OVERRIDES[key])

func _restore_settings() -> void:
    for key in previous_settings.keys():
        ProjectSettings.set_setting(key, previous_settings[key])

func _camera_transform(frame_idx: int) -> Transform3D:
    var segment_count: int = CAMERA_PATH_POINTS.size()
    var segment_length: int = max(1, SAMPLE_FRAMES / segment_count)
    var index: int = int(frame_idx / segment_length) % segment_count
    var next_index: int = (index + 1) % segment_count
    var segment_t: float = float(frame_idx % segment_length) / float(segment_length)
    var pos: Vector3 = CAMERA_PATH_POINTS[index].lerp(CAMERA_PATH_POINTS[next_index], segment_t)
    var forward: Vector3 = (Vector3.ZERO - pos).normalized()
    var basis: Basis = Basis.looking_at(forward, Vector3.UP)
    return Transform3D(basis, pos)

func _run() -> void:
    _apply_overrides()

    renderer = GaussianSplatRenderer.new()
    if renderer == null:
        _record_failure("Failed to create renderer")
        print("%s %s" % [METRICS_MARKER, JSON.stringify({"status": "failed", "failures": failures})])
        _restore_settings()
        quit(1)
        return
    renderer.initialize()

    renderer.set_max_splats(DATASET_SIZE)
    var dataset := _build_dataset(DATASET_SIZE)
    var set_err := renderer.set_gaussian_data(dataset)
    if set_err != OK:
        _record_failure("set_gaussian_data failed", {"err": set_err})
        print("%s %s" % [METRICS_MARKER, JSON.stringify({"status": "failed", "failures": failures})])
        _restore_settings()
        quit(1)
        return

    for i in range(WARMUP_FRAMES):
        await process_frame
        renderer.force_sort_for_view(_camera_transform(i))

    var frame_ms_values: Array = []
    var scheduler_update_cpu_ms_values: Array = []
    var scheduler_cpu_total_attributed_ms_values: Array = []
    var sum_loaded := 0
    var sum_evicted := 0
    var sum_visible_evicted := 0
    var sum_upload_mb := 0.0
    var sum_visible_splats := 0
    var max_visible_splats := 0
    var min_visible_splats := -1
    var visible_splats_positive_frames := 0
    var visible_splats_zero_frames := 0
    var first_visible_frame := -1
    var last_visible_frame := -1
    var raster_error_frames := 0
    var composite_error_frames := 0
    var last_stage_raster_status := "unknown"
    var last_stage_composite_status := "unknown"
    var last_stage_raster_reason := ""
    var last_stage_composite_reason := ""
    var max_pack_q := 0
    var max_upload_q := 0
    var max_sync_fallback_q := 0
    var max_loaded_frame := 0
    var max_evicted_frame := 0
    var max_scheduler_update_ms := 0.0
    var max_scheduler_load_ms := 0.0
    var max_scheduler_visibility_ms := 0.0
    var max_scheduler_prefetch_ms := 0.0
    var max_scheduler_cpu_total_attributed_ms := 0.0
    var max_scheduler_cpu_unattributed_ms := 0.0
    var max_scheduler_visible_scan_budget_effective := 0
    var max_scheduler_prefetch_scan_budget_effective := 0
    var saw_scheduler_queue_pressure_scan_throttle_active := false
    var scheduler_queue_pressure_scan_throttle_active_known := false
    var max_scheduler_queue_pressure_scan_throttle_queue_depth := 0
    var scheduler_queue_pressure_scan_throttle_enabled := false
    var max_scheduler_stall_frames := 0
    var max_scheduler_load_candidates := 0
    var min_scheduler_visible_scan_budget_effective_under_pressure_candidates := -1
    var max_upload_mb_frame := 0.0
    var queue_pressure_active_frames := 0
    var queue_pressure_candidate_frames := 0
    var queue_pressure_no_progress_frames := 0
    var queue_pressure_scan_starved_frames := 0
    var queue_pressure_scan_starvation_frames := 0
    var queue_pressure_throttle_active_frames := 0
    var queue_pressure_high_mutex_wait_no_upload_frames := 0
    var queue_pressure_high_mutex_wait_samples := 0
    var queue_pressure_reason_sources: Dictionary = {}
    var last_queue_pressure_reason_source: Variant = null
    var max_pack_queue_latency_max_ms: Variant = null
    var max_upload_queue_latency_max_ms: Variant = null
    var max_pack_mutex_wait_max_ms: Variant = null

    var diagnostics_categories: Dictionary = {}
    var diagnostics_reasons: Dictionary = {}
    var last_stats: Dictionary = {}

    for frame_idx in range(SAMPLE_FRAMES):
        var frame_start_usec := Time.get_ticks_usec()
        await process_frame
        renderer.force_sort_for_view(_camera_transform(frame_idx + WARMUP_FRAMES))
        var stats: Dictionary = renderer.get_render_stats()
        last_stats = stats
        frame_ms_values.append(float(Time.get_ticks_usec() - frame_start_usec) / 1000.0)
        var visible_splats := _read_int(stats, "visible_splats")
        sum_visible_splats += visible_splats
        max_visible_splats = max(max_visible_splats, visible_splats)
        min_visible_splats = visible_splats if min_visible_splats < 0 else min(min_visible_splats, visible_splats)
        if visible_splats > 0:
            visible_splats_positive_frames += 1
            if first_visible_frame < 0:
                first_visible_frame = frame_idx
            last_visible_frame = frame_idx
        else:
            visible_splats_zero_frames += 1

        last_stage_raster_status = String(stats.get("stage_raster_status", last_stage_raster_status))
        last_stage_composite_status = String(stats.get("stage_composite_status", last_stage_composite_status))
        last_stage_raster_reason = String(stats.get("stage_raster_reason", last_stage_raster_reason))
        last_stage_composite_reason = String(stats.get("stage_composite_reason", last_stage_composite_reason))
        if bool(stats.get("stage_raster_is_error", false)):
            raster_error_frames += 1
        if bool(stats.get("stage_composite_is_error", false)):
            composite_error_frames += 1

        var stream_state: Variant = stats.get("streaming_state", {})
        if stream_state is Dictionary:
            var s: Dictionary = stream_state
            var diagnostics := _extract_diagnostics(s)
            var loaded_frame := _read_int(s, "chunks_loaded_this_frame")
            var evicted_frame := _read_int(s, "chunks_evicted_this_frame")
            var visible_evicted_frame := _read_int(s, "visible_chunks_evicted_this_frame")
            var upload_mb_frame := _read_float(s, "upload_mb_this_frame")
            var scheduler_update_cpu_ms := _read_float(s, "scheduler_update_cpu_ms")
            var scheduler_cpu_total_attributed_ms := _read_float(s, "scheduler_cpu_total_attributed_ms")
            var load_candidates := _read_int(s, "scheduler_load_candidates")
            var visible_scan_budget_effective := _read_int(s, "scheduler_visible_scan_budget_effective")

            sum_loaded += loaded_frame
            sum_evicted += evicted_frame
            sum_visible_evicted += visible_evicted_frame
            sum_upload_mb += upload_mb_frame
            scheduler_update_cpu_ms_values.append(scheduler_update_cpu_ms)
            scheduler_cpu_total_attributed_ms_values.append(scheduler_cpu_total_attributed_ms)

            max_loaded_frame = max(max_loaded_frame, loaded_frame)
            max_evicted_frame = max(max_evicted_frame, evicted_frame)
            max_upload_mb_frame = maxf(max_upload_mb_frame, upload_mb_frame)
            max_scheduler_load_candidates = max(max_scheduler_load_candidates, load_candidates)

            max_pack_q = max(max_pack_q, _read_int(s, "scheduler_pack_queue_depth"))
            max_upload_q = max(max_upload_q, _read_int(s, "scheduler_upload_queue_depth"))
            max_sync_fallback_q = max(max_sync_fallback_q, _read_int(s, "scheduler_sync_fallback_queue_depth"))

            max_scheduler_update_ms = maxf(max_scheduler_update_ms, scheduler_update_cpu_ms)
            max_scheduler_load_ms = maxf(max_scheduler_load_ms, _read_float(s, "scheduler_load_cpu_ms"))
            max_scheduler_visibility_ms = maxf(max_scheduler_visibility_ms, _read_float(s, "scheduler_visibility_cpu_ms"))
            max_scheduler_prefetch_ms = maxf(max_scheduler_prefetch_ms, _read_float(s, "scheduler_prefetch_cpu_ms"))
            max_scheduler_cpu_total_attributed_ms = maxf(
                max_scheduler_cpu_total_attributed_ms,
                scheduler_cpu_total_attributed_ms
            )
            max_scheduler_cpu_unattributed_ms = maxf(
                max_scheduler_cpu_unattributed_ms,
                _read_float(s, "scheduler_cpu_unattributed_ms")
            )
            max_scheduler_visible_scan_budget_effective = max(
                max_scheduler_visible_scan_budget_effective,
                visible_scan_budget_effective
            )
            max_scheduler_prefetch_scan_budget_effective = max(
                max_scheduler_prefetch_scan_budget_effective,
                _read_int(s, "scheduler_prefetch_scan_budget_effective")
            )

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

            var throttle_active_value: Variant = _read_optional_bool(
                s,
                ["scheduler_queue_pressure_scan_throttle_active"]
            )
            if throttle_active_value == null:
                throttle_active_value = _read_optional_bool(
                    diagnostics,
                    ["scheduler_queue_pressure_scan_throttle_active"]
                )
            if throttle_active_value != null:
                scheduler_queue_pressure_scan_throttle_active_known = true
                if bool(throttle_active_value):
                    saw_scheduler_queue_pressure_scan_throttle_active = true
                    queue_pressure_throttle_active_frames += 1

            if bool(s.get("queue_pressure_active", false)):
                queue_pressure_active_frames += 1
                if load_candidates > 0:
                    queue_pressure_candidate_frames += 1
                    if min_scheduler_visible_scan_budget_effective_under_pressure_candidates < 0:
                        min_scheduler_visible_scan_budget_effective_under_pressure_candidates = visible_scan_budget_effective
                    else:
                        min_scheduler_visible_scan_budget_effective_under_pressure_candidates = min(
                            min_scheduler_visible_scan_budget_effective_under_pressure_candidates,
                            visible_scan_budget_effective
                        )
                    var no_progress := loaded_frame == 0
                    var scan_starved := visible_scan_budget_effective <= 1
                    if no_progress:
                        queue_pressure_no_progress_frames += 1
                    if scan_starved:
                        queue_pressure_scan_starved_frames += 1
                    if no_progress and scan_starved:
                        queue_pressure_scan_starvation_frames += 1
                    if pack_mutex_wait_max_ms != null:
                        queue_pressure_high_mutex_wait_samples += 1
                        var chunks_uploaded_this_frame: Variant = _read_optional_int(
                            diagnostics,
                            ["chunks_uploaded_this_frame"]
                        )
                        if chunks_uploaded_this_frame == null:
                            chunks_uploaded_this_frame = _read_optional_int(s, ["chunks_uploaded_this_frame"])
                        var upload_mb_this_frame: Variant = _read_optional_float(
                            s,
                            ["upload_mb_this_frame"]
                        )
                        if upload_mb_this_frame == null:
                            upload_mb_this_frame = _read_optional_float(diagnostics, ["upload_mb_this_frame"])
                        var has_progress_signal := s.has("chunks_loaded_this_frame") or \
                            chunks_uploaded_this_frame != null or upload_mb_this_frame != null
                        var has_upload_progress := loaded_frame > 0
                        if chunks_uploaded_this_frame != null and int(chunks_uploaded_this_frame) > 0:
                            has_upload_progress = true
                        if upload_mb_this_frame != null and float(upload_mb_this_frame) > UPLOAD_PROGRESS_EPSILON_MB:
                            has_upload_progress = true
                        if float(pack_mutex_wait_max_ms) >= QUEUE_PRESSURE_HIGH_MUTEX_WAIT_MIN_MS and \
                                has_progress_signal and \
                                not has_upload_progress:
                            queue_pressure_high_mutex_wait_no_upload_frames += 1
            scheduler_queue_pressure_scan_throttle_enabled = bool(
                s.get("scheduler_queue_pressure_scan_throttle_enabled", scheduler_queue_pressure_scan_throttle_enabled)
            )
            max_scheduler_queue_pressure_scan_throttle_queue_depth = max(
                max_scheduler_queue_pressure_scan_throttle_queue_depth,
                _read_int(s, "scheduler_queue_pressure_scan_throttle_queue_depth")
            )
            max_scheduler_stall_frames = max(max_scheduler_stall_frames, _read_int(s, "scheduler_stall_frames"))

            var category := String(s.get("diagnostics_category", ""))
            if not category.is_empty():
                diagnostics_categories[category] = int(diagnostics_categories.get(category, 0)) + 1

            if not diagnostics.is_empty():
                var reason := String(diagnostics.get("reason", ""))
                if not reason.is_empty():
                    diagnostics_reasons[reason] = int(diagnostics_reasons.get(reason, 0)) + 1

    var frame_avg_ms: float = _average(frame_ms_values)
    var frame_p95_ms: float = _percentile(frame_ms_values, 0.95)
    var frame_max_ms := 0.0
    for value in frame_ms_values:
        frame_max_ms = maxf(frame_max_ms, float(value))
    var frame_p95_to_avg_ratio := frame_p95_ms / maxf(0.001, frame_avg_ms)
    var frame_max_to_p95_ratio := frame_max_ms / maxf(0.001, frame_p95_ms)
    var scheduler_update_cpu_p95_ms := _percentile(scheduler_update_cpu_ms_values, 0.95)
    var scheduler_cpu_total_attributed_p95_ms := _percentile(scheduler_cpu_total_attributed_ms_values, 0.95)
    var scheduler_update_cpu_p95_to_frame_p95_ratio := scheduler_update_cpu_p95_ms / maxf(0.001, frame_p95_ms)
    var queue_pressure_active_ratio := float(queue_pressure_active_frames) / float(max(1, SAMPLE_FRAMES))
    var queue_pressure_no_progress_ratio := 0.0 if queue_pressure_candidate_frames <= 0 else float(
        queue_pressure_no_progress_frames
    ) / float(queue_pressure_candidate_frames)
    var queue_pressure_scan_starved_ratio := 0.0 if queue_pressure_candidate_frames <= 0 else float(
        queue_pressure_scan_starved_frames
    ) / float(queue_pressure_candidate_frames)
    var queue_pressure_scan_starvation_ratio := 0.0 if queue_pressure_candidate_frames <= 0 else float(
        queue_pressure_scan_starvation_frames
    ) / float(queue_pressure_candidate_frames)
    var queue_pressure_high_mutex_wait_no_upload_ratio := 0.0 if queue_pressure_high_mutex_wait_samples <= 0 else float(
        queue_pressure_high_mutex_wait_no_upload_frames
    ) / float(queue_pressure_high_mutex_wait_samples)
    var fps_avg: float = 1000.0 / maxf(0.001, frame_avg_ms)
    var fps_p95: float = 1000.0 / maxf(0.001, frame_p95_ms)
    var visible_splats_avg: float = float(sum_visible_splats) / float(max(1, SAMPLE_FRAMES))
    var last_visible_splats := _read_int(last_stats, "visible_splats")
    var last_total_splats := int(last_stats.get("total_splats", 0))
    var tail_visibility_confirmation_frames := 0
    var tail_visibility_recovered := false

    if last_visible_splats <= 0:
        var confirmation_transform := _camera_transform(SAMPLE_FRAMES + WARMUP_FRAMES - 1)
        for i in range(TAIL_VISIBILITY_CONFIRMATION_FRAMES):
            tail_visibility_confirmation_frames += 1
            await process_frame
            renderer.force_sort_for_view(confirmation_transform)
            last_stats = renderer.get_render_stats()
            last_visible_splats = _read_int(last_stats, "visible_splats")
            last_total_splats = int(last_stats.get("total_splats", last_total_splats))
            last_stage_raster_status = String(last_stats.get("stage_raster_status", last_stage_raster_status))
            last_stage_composite_status = String(last_stats.get("stage_composite_status", last_stage_composite_status))
            last_stage_raster_reason = String(last_stats.get("stage_raster_reason", last_stage_raster_reason))
            last_stage_composite_reason = String(last_stats.get("stage_composite_reason", last_stage_composite_reason))
            if bool(last_stats.get("stage_raster_is_error", false)):
                raster_error_frames += 1
            if bool(last_stats.get("stage_composite_is_error", false)):
                composite_error_frames += 1
            if last_visible_splats > 0:
                tail_visibility_recovered = true
                break
        if tail_visibility_recovered and first_visible_frame < 0:
            first_visible_frame = SAMPLE_FRAMES + WARMUP_FRAMES - 1
        if tail_visibility_recovered:
            last_visible_frame = SAMPLE_FRAMES + WARMUP_FRAMES - 1 + tail_visibility_confirmation_frames - 1

    if max_visible_splats <= 0:
        _record_failure(
            "Churn probe never rendered visible splats",
            {
                "visible_splats_max": max_visible_splats,
                "visible_splats_avg": visible_splats_avg,
                "visible_splats_positive_frames": visible_splats_positive_frames,
                "visible_splats_zero_frames": visible_splats_zero_frames,
                "first_visible_frame": first_visible_frame,
                "last_visible_frame": last_visible_frame,
                "last_visible_splats": last_visible_splats,
                "last_total_splats": last_total_splats,
                "sum_chunks_loaded": sum_loaded,
                "sum_upload_mb": sum_upload_mb,
                "tail_visibility_confirmation_frames": tail_visibility_confirmation_frames
            }
        )
    elif last_visible_splats <= 0 and not tail_visibility_recovered:
        _record_failure(
            "Churn probe ended with sustained zero visible splats",
            {
                "visible_splats_max": max_visible_splats,
                "visible_splats_avg": visible_splats_avg,
                "visible_splats_positive_frames": visible_splats_positive_frames,
                "visible_splats_zero_frames": visible_splats_zero_frames,
                "first_visible_frame": first_visible_frame,
                "last_visible_frame": last_visible_frame,
                "last_visible_splats": last_visible_splats,
                "last_total_splats": last_total_splats,
                "sum_chunks_loaded": sum_loaded,
                "sum_upload_mb": sum_upload_mb,
                "tail_visibility_confirmation_frames": tail_visibility_confirmation_frames
            }
        )
    if visible_splats_zero_frames > visible_splats_positive_frames:
        _record_failure(
            "Churn probe spent most sampled frames with zero visible splats",
            {
                "visible_splats_positive_frames": visible_splats_positive_frames,
                "visible_splats_zero_frames": visible_splats_zero_frames,
                "visible_splats_avg": visible_splats_avg,
                "visible_splats_max": max_visible_splats,
                "tail_visibility_recovered": tail_visibility_recovered
            }
        )
    if raster_error_frames > 0 or composite_error_frames > 0:
        _record_failure(
            "Raster/composite stage errors occurred during churn probe",
            {
                "raster_error_frames": raster_error_frames,
                "composite_error_frames": composite_error_frames,
                "last_stage_raster_status": last_stage_raster_status,
                "last_stage_composite_status": last_stage_composite_status,
                "last_stage_raster_reason": last_stage_raster_reason,
                "last_stage_composite_reason": last_stage_composite_reason
            }
        )

    var stream_last: Variant = last_stats.get("streaming_state", {})
    var scheduler_stalls := 0
    var upload_stalls := 0
    var sync_fallback_stalls := 0
    if stream_last is Dictionary:
        scheduler_stalls = _read_int(stream_last, "scheduler_stall_frames")
        upload_stalls = _read_int(stream_last, "upload_stall_frames")
        sync_fallback_stalls = _read_int(stream_last, "sync_fallback_stall_frames")

    var scheduler_stalled_diagnostic_frames := int(diagnostics_categories.get("scheduler_stalled", 0))
    if max_scheduler_stall_frames >= SCHEDULER_STALL_FAIL_FRAMES or scheduler_stalled_diagnostic_frames > 0:
        _record_failure(
            "Scheduler forward progress stalled during churn probe",
            {
                "scheduler_stall_fail_threshold": SCHEDULER_STALL_FAIL_FRAMES,
                "max_scheduler_stall_frames": max_scheduler_stall_frames,
                "scheduler_stalled_diagnostic_frames": scheduler_stalled_diagnostic_frames,
                "scheduler_stall_frames_last": scheduler_stalls,
                "upload_stall_frames_last": upload_stalls,
                "sync_fallback_stall_frames_last": sync_fallback_stalls
            }
        )
    if queue_pressure_candidate_frames >= QUEUE_PRESSURE_STARVATION_MIN_FRAMES and \
            queue_pressure_scan_starvation_ratio >= QUEUE_PRESSURE_STARVATION_FAIL_RATIO and \
            frame_p95_ms >= FRAME_P95_SPIKE_MIN_MS and \
            frame_p95_to_avg_ratio >= FRAME_P95_TO_AVG_SPIKE_FAIL_RATIO:
        _record_failure(
            "Queue-pressure scan starvation correlated with p95 frame spikes during churn probe",
            {
                "queue_pressure_starvation_min_frames": QUEUE_PRESSURE_STARVATION_MIN_FRAMES,
                "queue_pressure_starvation_fail_ratio": QUEUE_PRESSURE_STARVATION_FAIL_RATIO,
                "frame_p95_spike_min_ms": FRAME_P95_SPIKE_MIN_MS,
                "frame_p95_to_avg_spike_fail_ratio": FRAME_P95_TO_AVG_SPIKE_FAIL_RATIO,
                "queue_pressure_candidate_frames": queue_pressure_candidate_frames,
                "queue_pressure_scan_starvation_frames": queue_pressure_scan_starvation_frames,
                "queue_pressure_scan_starvation_ratio": queue_pressure_scan_starvation_ratio,
                "frame_avg_ms": frame_avg_ms,
                "frame_p95_ms": frame_p95_ms,
                "frame_p95_to_avg_ratio": frame_p95_to_avg_ratio
            }
        )
    if queue_pressure_active_frames >= QUEUE_PRESSURE_STARVATION_MIN_FRAMES and \
            frame_p95_ms >= FRAME_P95_SPIKE_MIN_MS and \
            frame_p95_to_avg_ratio >= FRAME_P95_TO_AVG_SPIKE_FAIL_RATIO and \
            scheduler_update_cpu_p95_to_frame_p95_ratio >= SCHEDULER_CPU_P95_TO_FRAME_P95_FAIL_RATIO:
        _record_failure(
            "Scheduler CPU time dominated churn-probe p95 frame spikes under queue pressure",
            {
                "queue_pressure_active_frames": queue_pressure_active_frames,
                "frame_avg_ms": frame_avg_ms,
                "frame_p95_ms": frame_p95_ms,
                "frame_p95_to_avg_ratio": frame_p95_to_avg_ratio,
                "scheduler_update_cpu_p95_ms": scheduler_update_cpu_p95_ms,
                "scheduler_cpu_total_attributed_p95_ms": scheduler_cpu_total_attributed_p95_ms,
                "scheduler_update_cpu_p95_to_frame_p95_ratio": scheduler_update_cpu_p95_to_frame_p95_ratio,
                "scheduler_cpu_p95_to_frame_p95_fail_ratio": SCHEDULER_CPU_P95_TO_FRAME_P95_FAIL_RATIO
            }
        )
    if queue_pressure_candidate_frames >= QUEUE_PRESSURE_STARVATION_MIN_FRAMES and \
            queue_pressure_high_mutex_wait_samples >= QUEUE_PRESSURE_HIGH_MUTEX_WAIT_FAIL_MIN_FRAMES and \
            queue_pressure_high_mutex_wait_no_upload_ratio >= QUEUE_PRESSURE_HIGH_MUTEX_WAIT_FAIL_RATIO and \
            queue_pressure_no_progress_ratio >= QUEUE_PRESSURE_STARVATION_FAIL_RATIO:
        _record_failure(
            "Persistent queue pressure had high pack mutex wait with no upload progress during churn probe",
            {
                "queue_pressure_candidate_frames": queue_pressure_candidate_frames,
                "queue_pressure_no_progress_frames": queue_pressure_no_progress_frames,
                "queue_pressure_no_progress_ratio": queue_pressure_no_progress_ratio,
                "queue_pressure_high_mutex_wait_samples": queue_pressure_high_mutex_wait_samples,
                "queue_pressure_high_mutex_wait_no_upload_frames": queue_pressure_high_mutex_wait_no_upload_frames,
                "queue_pressure_high_mutex_wait_no_upload_ratio": queue_pressure_high_mutex_wait_no_upload_ratio,
                "queue_pressure_high_mutex_wait_min_ms": QUEUE_PRESSURE_HIGH_MUTEX_WAIT_MIN_MS,
                "queue_pressure_high_mutex_wait_fail_min_frames": QUEUE_PRESSURE_HIGH_MUTEX_WAIT_FAIL_MIN_FRAMES,
                "queue_pressure_high_mutex_wait_fail_ratio": QUEUE_PRESSURE_HIGH_MUTEX_WAIT_FAIL_RATIO,
                "pack_mutex_wait_max_ms": max_pack_mutex_wait_max_ms
            }
        )

    var payload := {
        "status": "passed" if failures.is_empty() else "failed",
        "failures": failures,
        "dataset_size": DATASET_SIZE,
        "sample_frames": SAMPLE_FRAMES,
        "override_settings": OVERRIDES,
        "frame_avg_ms": frame_avg_ms,
        "frame_p95_ms": frame_p95_ms,
        "frame_max_ms": frame_max_ms,
        "frame_p95_to_avg_ratio": frame_p95_to_avg_ratio,
        "frame_max_to_p95_ratio": frame_max_to_p95_ratio,
        "fps_avg": fps_avg,
        "fps_p95": fps_p95,
        "visible_splats_max": max_visible_splats,
        "visible_splats_min": min_visible_splats if min_visible_splats >= 0 else 0,
        "visible_splats_avg": visible_splats_avg,
        "visible_splats_positive_frames": visible_splats_positive_frames,
        "visible_splats_zero_frames": visible_splats_zero_frames,
        "visible_splats_first_visible_frame": first_visible_frame,
        "visible_splats_last_visible_frame": last_visible_frame,
        "tail_visibility_confirmation_frames": tail_visibility_confirmation_frames,
        "tail_visibility_recovered": tail_visibility_recovered,
        "raster_error_frames": raster_error_frames,
        "composite_error_frames": composite_error_frames,
        "last_stage_raster_status": last_stage_raster_status,
        "last_stage_composite_status": last_stage_composite_status,
        "last_stage_raster_reason": last_stage_raster_reason,
        "last_stage_composite_reason": last_stage_composite_reason,
        "sum_chunks_loaded": sum_loaded,
        "sum_chunks_evicted": sum_evicted,
        "sum_visible_chunks_evicted": sum_visible_evicted,
        "churn_ratio_evicted_to_loaded": 0.0 if sum_loaded == 0 else float(sum_evicted) / float(sum_loaded),
        "sum_upload_mb": sum_upload_mb,
        "max_upload_mb_this_frame": max_upload_mb_frame,
        "max_chunks_loaded_this_frame": max_loaded_frame,
        "max_chunks_evicted_this_frame": max_evicted_frame,
        "max_pack_queue_depth": max_pack_q,
        "max_upload_queue_depth": max_upload_q,
        "max_sync_fallback_queue_depth": max_sync_fallback_q,
        "max_scheduler_update_cpu_ms": max_scheduler_update_ms,
        "max_scheduler_load_cpu_ms": max_scheduler_load_ms,
        "max_scheduler_visibility_cpu_ms": max_scheduler_visibility_ms,
        "max_scheduler_prefetch_cpu_ms": max_scheduler_prefetch_ms,
        "max_scheduler_cpu_total_attributed_ms": max_scheduler_cpu_total_attributed_ms,
        "max_scheduler_cpu_unattributed_ms": max_scheduler_cpu_unattributed_ms,
        "scheduler_update_cpu_p95_ms": scheduler_update_cpu_p95_ms,
        "scheduler_cpu_total_attributed_p95_ms": scheduler_cpu_total_attributed_p95_ms,
        "scheduler_update_cpu_p95_to_frame_p95_ratio": scheduler_update_cpu_p95_to_frame_p95_ratio,
        "max_scheduler_visible_scan_budget_effective": max_scheduler_visible_scan_budget_effective,
        "max_scheduler_prefetch_scan_budget_effective": max_scheduler_prefetch_scan_budget_effective,
        "max_scheduler_load_candidates": max_scheduler_load_candidates,
        "min_scheduler_visible_scan_budget_effective_under_pressure_candidates":
            min_scheduler_visible_scan_budget_effective_under_pressure_candidates,
        "scheduler_queue_pressure_scan_throttle_enabled": scheduler_queue_pressure_scan_throttle_enabled,
        "scheduler_queue_pressure_scan_throttle_active": saw_scheduler_queue_pressure_scan_throttle_active,
        "scheduler_queue_pressure_scan_throttle_active_optional":
            saw_scheduler_queue_pressure_scan_throttle_active if scheduler_queue_pressure_scan_throttle_active_known else null,
        "queue_pressure_throttle_active_frames": queue_pressure_throttle_active_frames,
        "max_scheduler_queue_pressure_scan_throttle_queue_depth": max_scheduler_queue_pressure_scan_throttle_queue_depth,
        "queue_pressure_active_frames": queue_pressure_active_frames,
        "queue_pressure_active_ratio": queue_pressure_active_ratio,
        "queue_pressure_candidate_frames": queue_pressure_candidate_frames,
        "queue_pressure_no_progress_frames": queue_pressure_no_progress_frames,
        "queue_pressure_no_progress_ratio": queue_pressure_no_progress_ratio,
        "queue_pressure_scan_starved_frames": queue_pressure_scan_starved_frames,
        "queue_pressure_scan_starved_ratio": queue_pressure_scan_starved_ratio,
        "queue_pressure_scan_starvation_frames": queue_pressure_scan_starvation_frames,
        "queue_pressure_scan_starvation_ratio": queue_pressure_scan_starvation_ratio,
        "queue_pressure_high_mutex_wait_samples": queue_pressure_high_mutex_wait_samples,
        "queue_pressure_high_mutex_wait_no_upload_frames": queue_pressure_high_mutex_wait_no_upload_frames,
        "queue_pressure_high_mutex_wait_no_upload_ratio": queue_pressure_high_mutex_wait_no_upload_ratio,
        "queue_pressure_high_mutex_wait_min_ms": QUEUE_PRESSURE_HIGH_MUTEX_WAIT_MIN_MS,
        "queue_pressure_high_mutex_wait_fail_min_frames": QUEUE_PRESSURE_HIGH_MUTEX_WAIT_FAIL_MIN_FRAMES,
        "queue_pressure_high_mutex_wait_fail_ratio": QUEUE_PRESSURE_HIGH_MUTEX_WAIT_FAIL_RATIO,
        "queue_pressure_starvation_min_frames": QUEUE_PRESSURE_STARVATION_MIN_FRAMES,
        "queue_pressure_starvation_fail_ratio": QUEUE_PRESSURE_STARVATION_FAIL_RATIO,
        "queue_pressure_reason_source": last_queue_pressure_reason_source,
        "queue_pressure_reason_sources": queue_pressure_reason_sources,
        "pack_queue_latency_max_ms": max_pack_queue_latency_max_ms,
        "upload_queue_latency_max_ms": max_upload_queue_latency_max_ms,
        "pack_mutex_wait_max_ms": max_pack_mutex_wait_max_ms,
        "frame_p95_spike_min_ms": FRAME_P95_SPIKE_MIN_MS,
        "frame_p95_to_avg_spike_fail_ratio": FRAME_P95_TO_AVG_SPIKE_FAIL_RATIO,
        "scheduler_cpu_p95_to_frame_p95_fail_ratio": SCHEDULER_CPU_P95_TO_FRAME_P95_FAIL_RATIO,
        "scheduler_stall_fail_threshold": SCHEDULER_STALL_FAIL_FRAMES,
        "max_scheduler_stall_frames": max_scheduler_stall_frames,
        "scheduler_stall_frames": scheduler_stalls,
        "upload_stall_frames": upload_stalls,
        "sync_fallback_stall_frames": sync_fallback_stalls,
        "scheduler_stalled_diagnostic_frames": scheduler_stalled_diagnostic_frames,
        "diagnostics_categories": diagnostics_categories,
        "diagnostics_reasons": diagnostics_reasons,
        "last_data_source": String(last_stats.get("data_source", "")),
        "last_visible_splats": last_visible_splats,
        "last_total_splats": last_total_splats,
        "last_sort_avg_ms": float(last_stats.get("gpu_sorter_avg_sort_ms", 0.0))
    }

    print("%s %s" % [METRICS_MARKER, JSON.stringify(payload)])
    _restore_settings()
    quit(0 if failures.is_empty() else 1)
