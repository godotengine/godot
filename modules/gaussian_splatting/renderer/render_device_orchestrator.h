#ifndef GAUSSIAN_RENDER_DEVICE_ORCHESTRATOR_H
#define GAUSSIAN_RENDER_DEVICE_ORCHESTRATOR_H

#include "gaussian_splat_renderer.h"

#include <functional>

class RenderDeviceOrchestrator {
public:
	using RecordCrossDeviceOperationFn = std::function<void(const GaussianSplatRenderer::CrossDeviceOperation &)>;
	using RecordRenderingErrorFn = std::function<void(const RenderingError &)>;
	using EmitRuntimeDiagnosticsFn = std::function<void()>;

	RenderDeviceOrchestrator(GaussianSplatRenderer *p_renderer, RenderDeviceManager *p_device_manager,
			GPUSortingPipeline *p_sorting_pipeline, RecordCrossDeviceOperationFn p_record_cross_device_operation,
			RecordRenderingErrorFn p_record_rendering_error, EmitRuntimeDiagnosticsFn p_emit_runtime_diagnostics);

	GaussianSplatRenderer::DeviceState &get_device_state() { return device_state; }
	const GaussianSplatRenderer::DeviceState &get_device_state() const { return device_state; }

	void initialize_device_state(RenderingDevice *p_device);
	void set_rendering_device(RenderingDevice *p_device);

	bool is_main_rendering_device(RenderingDevice *p_device) const;
	void safe_submit_sync(RenderingDevice *p_device);
	void push_cross_device_operation(const String &p_context, RenderingDevice *p_source, RenderingDevice *p_target);
	Dictionary build_device_capability_report() const;

	void track_resource_owner(const RID &p_rid, RenderingDevice *p_device, bool p_owned, const char *p_label);
	void forget_resource_owner(const RID &p_rid);
	RenderingDevice *get_resource_owner(const RID &p_rid, RenderingDevice *p_fallback) const;
	RenderingDevice *get_texture_owner_device(const RID &p_texture) const;
	RenderingDevice *acquire_submission_device_for(RenderingDevice *p_device,
			GaussianSplatManager::ScopedSubmissionLock &r_lock) const;
	RD::TextureFormat get_texture_format(RenderingDevice *p_device, RID p_texture) const;
	void update_tile_renderer_output_tracking(const RID &p_color_output, RenderingDevice *p_color_device,
			const RID &p_depth_output, RenderingDevice *p_depth_device);
	void free_owned_resource(RenderingDevice *p_fallback_device, RID &p_rid);

	RenderingDevice *get_submission_device();
	RenderingDevice *peek_submission_device() const;
	RenderingDevice *acquire_rendering_device();
	RenderingDevice *get_main_rendering_device() const;
	bool ensure_rendering_device(const char *p_context);
	bool ensure_submission_device(const char *p_context);
	void synchronize_tile_submission(RenderingDevice *p_device, const char *p_context);

private:
	GaussianSplatRenderer::DeviceState device_state;
	GaussianSplatRenderer *renderer = nullptr;
	RenderDeviceManager *device_manager = nullptr;
	GPUSortingPipeline *sorting_pipeline = nullptr;
	RecordCrossDeviceOperationFn record_cross_device_operation;
	RecordRenderingErrorFn record_rendering_error;
	EmitRuntimeDiagnosticsFn emit_runtime_diagnostics;
};

#endif
