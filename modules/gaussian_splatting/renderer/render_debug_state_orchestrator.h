#ifndef GAUSSIAN_RENDER_DEBUG_STATE_ORCHESTRATOR_H
#define GAUSSIAN_RENDER_DEBUG_STATE_ORCHESTRATOR_H

#include "gaussian_splat_renderer.h"

struct RenderRouteUID {
	static constexpr const char *COMMON_UNSET_ROUTE = "COMMON.UNSET.ROUTE";
	static constexpr const char *COMMON_UNKNOWN_ROUTE = "COMMON.UNKNOWN.ROUTE";
	static constexpr const char *COMMON_UNSET_SORT_ROUTE = "COMMON.UNSET.SORT_ROUTE";
	static constexpr const char *COMMON_UNKNOWN_SORT_ROUTE = "COMMON.UNKNOWN.SORT_ROUTE";
	static constexpr const char *COMMON_SKIP_NO_DATA = "COMMON.SKIP.NO_DATA";
	static constexpr const char *COMMON_SKIP_NO_VISIBLE = "COMMON.SKIP.NO_VISIBLE";
	static constexpr const char *COMMON_SKIP_CAMERA_STABLE = "COMMON.SKIP.CAMERA_STABLE";
	static constexpr const char *COMMON_SKIP_STREAMING_NOT_READY = "COMMON.SKIP.STREAMING_NOT_READY";
	static constexpr const char *COMMON_FAIL_NO_DEVICE = "COMMON.FAIL.NO_DEVICE";
	static constexpr const char *COMMON_FAIL_SORT_FAILED = "COMMON.FAIL.SORT_FAILED";
	static constexpr const char *COMMON_FAIL_NO_OUTPUT = "COMMON.FAIL.NO_OUTPUT";

	static constexpr const char *INSTANCE_ENTRY_INSTANCED_FAST = "INSTANCE.ENTRY.INSTANCED_FAST";
	static constexpr const char *INSTANCE_STREAMING = "INSTANCE.STREAMING";
	static constexpr const char *INSTANCE_CULL_GPU = "INSTANCE.CULL.GPU";
	static constexpr const char *INSTANCE_SORT_GPU = "INSTANCE.SORT.GPU";
	static constexpr const char *INSTANCE_SORT_CPU_FALLBACK = "INSTANCE.SORT.CPU_FALLBACK";
	static constexpr const char *INSTANCE_SORT_CACHED = "INSTANCE.SORT.CACHED";
	static constexpr const char *INSTANCE_SORT_IDENTITY_FALLBACK = "INSTANCE.SORT.IDENTITY_FALLBACK";
	static constexpr const char *INSTANCE_RASTER_COMPUTE = "INSTANCE.RASTER.COMPUTE";
	static constexpr const char *INSTANCE_RASTER_FRAGMENT = "INSTANCE.RASTER.FRAGMENT";
	static constexpr const char *INSTANCE_RASTER_CACHED = "INSTANCE.RASTER.CACHED";
	static constexpr const char *INSTANCE_RASTER_PAINTERLY = "INSTANCE.RASTER.PAINTERLY";

	static bool is_route_uid_missing(const String &p_route_uid) {
		return p_route_uid.is_empty() ||
				p_route_uid == COMMON_UNSET_ROUTE ||
				p_route_uid == COMMON_UNKNOWN_ROUTE;
	}

	static bool is_sort_route_uid_missing(const String &p_sort_route_uid) {
		return p_sort_route_uid.is_empty() ||
				p_sort_route_uid == COMMON_UNSET_SORT_ROUTE ||
				p_sort_route_uid == COMMON_UNKNOWN_SORT_ROUTE;
	}
};

class RenderDebugStateOrchestrator {
public:
	struct RuntimePorts {
		Error (GaussianSplatRenderer::*dump_pipeline_trace_to_file)(const String &p_path) const = &GaussianSplatRenderer::dump_pipeline_trace_to_file;
		RenderingDevice *(GaussianSplatRenderer::*resolve_resource_owner)(const RID &p_rid, RenderingDevice *p_fallback) const = &GaussianSplatRenderer::get_resource_owner;
	};

	struct Dependencies {
		GaussianSplatRenderer *renderer = nullptr;
		Ref<TileRenderer> *tile_renderer = nullptr;
		Ref<DebugOverlaySystem> *debug_overlay_system = nullptr;
		GaussianSplatRenderer::JacobianDebugConfig *jacobian_debug = nullptr;
		RuntimePorts runtime_ports;
	};

	explicit RenderDebugStateOrchestrator(const Dependencies &p_dependencies);

	GaussianSplatRenderer::DebugConfig &get_config() { return debug_config; }
	const GaussianSplatRenderer::DebugConfig &get_config() const { return debug_config; }
	GaussianSplatRenderer::DebugState &get_state() { return debug_state; }
	const GaussianSplatRenderer::DebugState &get_state() const { return debug_state; }

	void set_debug_preview_mode(GaussianSplatRenderer::DebugPreviewMode p_mode);
	Dictionary get_binning_debug_counters() const;
	void store_stage_metrics(const GaussianSplatRenderer::StageMetrics &p_metrics);
	void clear_stage_metrics();
	void update_frame_times(float p_render_ms, float p_sort_ms);
	void reset_debug_overlay_metrics(float p_sort_ms);
	void update_raster_metrics(const RasterPerformance &p_perf, const RasterStats &p_stats);
	void clear_overlay_dirty_flags();
	void apply_debug_options_to_render_params(TileRenderer::RenderParams &r_params) const;
	void set_debug_show_tile_grid(bool p_enabled);
	void set_debug_show_density_heatmap(bool p_enabled);
	void set_debug_show_shadow_opacity(bool p_enabled);
	void set_debug_show_performance_hud(bool p_enabled);
	void set_debug_show_residency_hud(bool p_enabled);
	void set_debug_show_device_boundaries(bool p_enabled);
	void set_debug_show_texture_states(bool p_enabled);
	void set_debug_dump_gpu_counters(bool p_enabled);
	bool get_debug_dump_gpu_counters() const;
	void set_debug_binning_counters_enabled(bool p_enabled);
	bool get_debug_binning_counters_enabled() const;
	void set_debug_pipeline_trace_enabled(bool p_enabled);
	bool get_debug_pipeline_trace_enabled() const;
	void set_debug_state_guardrails_enabled(bool p_enabled);
	bool get_debug_state_guardrails_enabled() const;
	void set_debug_cull_guardrails_enabled(bool p_enabled);
	bool get_debug_cull_guardrails_enabled() const;
	void set_debug_splat_audit_enabled(bool p_enabled);
	bool get_debug_splat_audit_enabled() const;
	void set_debug_splat_audit_sample_count(int p_count);
	int get_debug_splat_audit_sample_count() const;
	void set_debug_overlay_opacity(float p_opacity);
	void set_debug_compute_raster_policy(int p_policy);
	int get_debug_compute_raster_policy() const;
	void set_jacobian_bypass_radius_depth_floor(bool p_enabled);
	void set_jacobian_bypass_j_col2_clamp(bool p_enabled);
	void set_jacobian_invert_j_col2_sign(bool p_enabled);
	void set_max_conic_aspect(float p_aspect);

private:
	GaussianSplatRenderer *renderer = nullptr;
	Ref<TileRenderer> *tile_renderer = nullptr;
	Ref<DebugOverlaySystem> *debug_overlay_system = nullptr;
	GaussianSplatRenderer::JacobianDebugConfig *jacobian_debug = nullptr;
	RuntimePorts runtime_ports;
	GaussianSplatRenderer::DebugConfig debug_config;
	GaussianSplatRenderer::DebugState debug_state;
	bool has_prev_visible = false;
	uint32_t last_visible_count = 0;
	bool has_prev_contrib = false;
	uint32_t last_contrib_count = 0;
	bool has_anomaly_dump_frame = false;
	uint64_t last_anomaly_dump_frame = 0;
	struct CullGuardrailSample {
		uint64_t key = 0;
		uint32_t visible_count = 0;
		uint32_t static_visible_chunks = 0;
		uint32_t static_chunk_total = 0;
		uint32_t streaming_visible = 0;
		uint32_t gpu_visible = 0;
		uint32_t cpu_visible = 0;
		uint32_t frame_id = 0;
		bool streaming_active = false;
		bool valid = false;
	};
	static constexpr uint32_t kCullGuardrailSamples = 64;
	CullGuardrailSample cull_guardrail_samples[kCullGuardrailSamples];
	uint32_t cull_guardrail_cursor = 0;

	bool _check_cull_guardrails(uint64_t p_frame_id, uint32_t p_visible_count, String &r_message);
};

#endif
