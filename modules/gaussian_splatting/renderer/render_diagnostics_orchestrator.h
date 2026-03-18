#ifndef GAUSSIAN_RENDER_DIAGNOSTICS_ORCHESTRATOR_H
#define GAUSSIAN_RENDER_DIAGNOSTICS_ORCHESTRATOR_H

#include "gaussian_splat_renderer.h"

#include <functional>

class RenderDebugStateOrchestrator;

class RenderDiagnosticsOrchestrator {
public:
	using BuildDeviceCapabilityReportFn = std::function<Dictionary()>;

	RenderDiagnosticsOrchestrator(GaussianSplatRenderer *p_renderer,
			RenderDebugStateOrchestrator *p_debug_state_orchestrator,
			BuildDeviceCapabilityReportFn p_build_device_capability_report);

	GaussianSplatRenderer::DiagnosticsState &get_state() { return diagnostics_state; }
	const GaussianSplatRenderer::DiagnosticsState &get_state() const { return diagnostics_state; }

	void record_rendering_error(const RenderingError &p_error);
	void transition_recovery_state(GaussianSplatRenderer::ErrorRecoveryStateMachine::State p_state, const String &p_reason);
	void record_cross_device_operation(const GaussianSplatRenderer::CrossDeviceOperation &p_operation);
	void capture_frame_timing_sample();
	void increment_frame_counter();
	void emit_runtime_diagnostics_if_requested();

	Array serialize_texture_trace() const;
	Array serialize_cross_device_operations() const;
	Array serialize_frame_timing() const;
	Dictionary serialize_error_statistics() const;

	Dictionary build_render_stats() const;
	float get_sort_time_ms_internal() const;
	float get_render_time_ms_internal() const;
	Dictionary get_last_sort_metrics_internal() const;
	Array get_sort_metrics_history_internal() const;
	void record_sort_sample(const GaussianSplatRenderer::SortFrameMetrics &p_sample);
	void finalize_frame_metrics(uint64_t p_frame_start_usec);

	Dictionary get_runtime_diagnostic_snapshot() const;

private:
	GaussianSplatRenderer *renderer = nullptr;
	RenderDebugStateOrchestrator *debug_state_orchestrator = nullptr;
	BuildDeviceCapabilityReportFn build_device_capability_report;
	GaussianSplatRenderer::DiagnosticsState diagnostics_state;
};

#endif
