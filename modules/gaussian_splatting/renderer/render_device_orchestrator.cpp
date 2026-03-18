#include "render_device_orchestrator.h"
#include "render_diagnostics_orchestrator.h"

#include "core/error/error_macros.h"
#include "core/os/os.h"
#include "core/string/ustring.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering_server.h"
#include "../interfaces/gpu_sorting_pipeline.h"
#include "../interfaces/sync_policy.h"
#include "../logger/gs_logger.h"

namespace {

static bool _is_main_rendering_device(RenderingDevice *p_device) {
	if (!p_device) {
		return false;
	}

	if (RenderingServer *rs = RenderingServer::get_singleton()) {
		if (RenderingDevice *server_device = rs->get_rendering_device()) {
			if (p_device == server_device) {
				return true;
			}
		}
	}

	if (RenderingDevice *singleton_device = RenderingDevice::get_singleton()) {
		return p_device == singleton_device;
	}

	return false;
}

static RenderingDevice *_ensure_local_device(RenderingDevice *p_candidate) {
	return p_candidate;
}

static RenderingDevice *_acquire_manager_local_device() {
	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		if (RenderingDevice *primary = _ensure_local_device(manager->get_primary_rendering_device())) {
			return primary;
		}
		if (RenderingDevice *shared = _ensure_local_device(manager->get_shared_submission_device())) {
			return shared;
		}
	}

	return nullptr;
}

static bool _submit_and_sync_local_device(RenderingDevice *p_device) {
	if (!p_device) {
		return false;
	}

	// Main RenderingDevice synchronization is owned by Godot's frame lifecycle.
	if (_is_main_rendering_device(p_device)) {
		return false;
	}
	if (!gs_device_utils::is_local_device(p_device)) {
		return false;
	}

	gs_device_utils::safe_submit_and_sync(p_device);
	return true;
}

static String _sync_context_or_default(const char *p_context, const char *p_fallback) {
	if (p_context && p_context[0] != '\0') {
		return String(p_context);
	}
	return String(p_fallback);
}

} // namespace

RenderDeviceOrchestrator::RenderDeviceOrchestrator(GaussianSplatRenderer *p_renderer, RenderDeviceManager *p_device_manager,
		GPUSortingPipeline *p_sorting_pipeline, RecordCrossDeviceOperationFn p_record_cross_device_operation,
		RecordRenderingErrorFn p_record_rendering_error, EmitRuntimeDiagnosticsFn p_emit_runtime_diagnostics) :
		renderer(p_renderer),
		device_manager(p_device_manager),
		sorting_pipeline(p_sorting_pipeline),
		record_cross_device_operation(p_record_cross_device_operation),
		record_rendering_error(p_record_rendering_error),
		emit_runtime_diagnostics(p_emit_runtime_diagnostics) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_NULL(device_manager);
	ERR_FAIL_COND_MSG(!record_cross_device_operation, "RenderDeviceOrchestrator requires a cross-device callback.");
	ERR_FAIL_COND_MSG(!record_rendering_error, "RenderDeviceOrchestrator requires an error callback.");
	ERR_FAIL_COND_MSG(!emit_runtime_diagnostics, "RenderDeviceOrchestrator requires diagnostics callback.");
}

void RenderDeviceOrchestrator::initialize_device_state(RenderingDevice *p_device) {
	device_state = GaussianSplatRenderer::DeviceState();
	device_state.rd = p_device;
}

void RenderDeviceOrchestrator::set_rendering_device(RenderingDevice *p_device) {
	if (device_state.rd == p_device) {
		return;
	}
	device_state.rd = p_device;
	device_state.reported_missing_render_device = false;
	device_state.reported_missing_submission_device = false;
}

bool RenderDeviceOrchestrator::is_main_rendering_device(RenderingDevice *p_device) const {
	return _is_main_rendering_device(p_device);
}

void RenderDeviceOrchestrator::safe_submit_sync(RenderingDevice *p_device) {
	if (!p_device) {
		return;
	}

	const String sync_context("safe_submit_sync");
	RenderingDevice *sync_device = p_device;
	GaussianSplatManager::ScopedSubmissionLock submission_lock;
	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		RenderingDevice *acquired = manager->acquire_submission_device(sync_device, submission_lock);
		if (acquired) {
			if (sync_device != acquired) {
				push_cross_device_operation(sync_context, sync_device, acquired);
			}
			sync_device = acquired;
		}
	}

	bool synced = _submit_and_sync_local_device(sync_device);
	if (!synced && sync_device != p_device) {
		push_cross_device_operation(sync_context + String(":fallback_submit_sync"), sync_device, p_device);
		synced = _submit_and_sync_local_device(p_device);
	}

	if (synced) {
		return;
	}

	// No explicit submit/sync on main RenderingDevice.
	if (_is_main_rendering_device(sync_device) || _is_main_rendering_device(p_device)) {
		return;
	}

	RenderingError error(RenderingErrorCodes::command_buffer_synchronization_failed(), RenderingError::Severity::WARNING,
			"Unable to submit and synchronize non-main RenderingDevice");
	error.add_context("context", sync_context);
	error.add_context("frame", static_cast<int64_t>(renderer->get_frame_state().frame_counter));
	error.add_context("sync_device_instance_id", static_cast<int64_t>(sync_device->get_device_instance_id()));
	error.add_context("source_device_instance_id", static_cast<int64_t>(p_device->get_device_instance_id()));
	if (record_rendering_error) {
		record_rendering_error(error);
	}
	if (emit_runtime_diagnostics) {
		emit_runtime_diagnostics();
	}
}

void RenderDeviceOrchestrator::push_cross_device_operation(const String &p_context, RenderingDevice *p_source, RenderingDevice *p_target) {
	if (!p_source && !p_target) {
		return;
	}
	GaussianSplatRenderer::CrossDeviceOperation op;
	op.timestamp_usec = OS::get_singleton()->get_ticks_usec();
	op.context = p_context;
	if (p_source) {
		op.source_device = p_source->get_device_instance_id();
	}
	if (p_target) {
		op.target_device = p_target->get_device_instance_id();
	}
	if (record_cross_device_operation) {
		record_cross_device_operation(op);
	}
}

Dictionary RenderDeviceOrchestrator::build_device_capability_report() const {
	Dictionary report;
	RenderingDevice *device = get_main_rendering_device();
	if (!device) {
		report["status"] = "unavailable";
		return report;
	}

	report["status"] = "available";
	report["device_name"] = device->get_device_name();
	report["device_vendor"] = device->get_device_vendor_name();
	report["device_api"] = device->get_device_api_name();
	report["api_version"] = device->get_device_api_version();
	report["device_type"] = (int)device->get_device_type();
	report["total_memory"] = static_cast<int64_t>(device->get_device_total_memory());
	report["allocation_count"] = static_cast<int64_t>(device->get_device_allocation_count());
	report["driver_memory_report"] = device->get_driver_and_device_memory_report();
	return report;
}

void RenderDeviceOrchestrator::track_resource_owner(const RID &p_rid, RenderingDevice *p_device, bool p_owned, const char *p_label) {
	// Phase 8/C: Delegate to RenderDeviceManager (inline fallback removed)
	ERR_FAIL_NULL_MSG(device_manager, "RenderDeviceManager not initialized");
	device_manager->track_resource(p_rid, p_device, p_owned, p_label);
}

RenderingDevice *RenderDeviceOrchestrator::get_resource_owner(const RID &p_rid, RenderingDevice *p_fallback) const {
	// Phase 8/C: Delegate to RenderDeviceManager (inline fallback removed)
	if (device_manager) {
		return device_manager->get_resource_owner(p_rid, p_fallback);
	}
	return p_fallback;
}

RenderingDevice *RenderDeviceOrchestrator::get_texture_owner_device(const RID &p_texture) const {
	// Phase 8/C: Delegate to RenderDeviceManager (inline fallback removed)
	if (device_manager) {
		return device_manager->get_texture_owner(p_texture);
	}
	return nullptr;
}

RenderingDevice *RenderDeviceOrchestrator::acquire_submission_device_for(RenderingDevice *p_device,
		GaussianSplatManager::ScopedSubmissionLock &r_lock) const {
	if (!p_device) {
		return nullptr;
	}

	RenderingDevice *device = p_device;
	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		RenderingDevice *acquired = manager->acquire_submission_device(device, r_lock);
		if (acquired) {
			device = acquired;
		}
	}

	return device;
}

RD::TextureFormat RenderDeviceOrchestrator::get_texture_format(RenderingDevice *p_device, RID p_texture) const {
	if (!p_texture.is_valid()) {
		return RD::TextureFormat();
	}

	LocalVector<RenderingDevice *> candidates;
	auto add_candidate = [&candidates](RenderingDevice *p_candidate) {
		if (!p_candidate) {
			return;
		}
		for (RenderingDevice *existing : candidates) {
			if (existing == p_candidate) {
				return;
			}
		}
		candidates.push_back(p_candidate);
	};

	add_candidate(get_texture_owner_device(p_texture));
	add_candidate(p_device);
	add_candidate(device_state.rd);
	add_candidate(peek_submission_device());
	add_candidate(get_main_rendering_device());
	if (sorting_pipeline) {
		add_candidate(sorting_pipeline->get_depth_submission_device());
	}

	for (RenderingDevice *candidate : candidates) {
		GaussianSplatManager::ScopedSubmissionLock submission_lock;
		RenderingDevice *device = acquire_submission_device_for(candidate, submission_lock);
		if (!device) {
			device = candidate;
		}
		if (!device) {
			continue;
		}
		if (!device->texture_is_valid(p_texture)) {
			continue;
		}
		return device->texture_get_format(p_texture);
	}

	return RD::TextureFormat();
}

void RenderDeviceOrchestrator::update_tile_renderer_output_tracking(const RID &p_color_output, RenderingDevice *p_color_device,
		const RID &p_depth_output, RenderingDevice *p_depth_device) {
	if (renderer->get_subsystem_state().rasterizer.is_valid()) {
		renderer->get_subsystem_state().rasterizer->track_output_resources(p_color_output, p_color_device, p_depth_output, p_depth_device);
	}
}

void RenderDeviceOrchestrator::free_owned_resource(RenderingDevice *p_fallback_device, RID &p_rid) {
	if (!p_rid.is_valid()) {
		return;
	}
	ERR_FAIL_NULL_MSG(device_manager,
			vformat("[GaussianSplatRenderer] Missing RenderDeviceManager while freeing owned RID %s",
					String::num_uint64(p_rid.get_id())));
	device_manager->free_owned_resource(p_fallback_device, p_rid);
}

RenderingDevice *RenderDeviceOrchestrator::get_submission_device() {
	GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
	if (manager) {
		RenderingDevice *shared_device = _ensure_local_device(manager->get_shared_submission_device());
		if (shared_device) {
			device_state.reported_missing_submission_device = false;
			return shared_device;
		}
	}

	if (device_state.rd) {
		device_state.reported_missing_submission_device = false;
		return device_state.rd;
	}

	if (!device_state.reported_missing_submission_device) {
		GS_LOG_WARN_DEFAULT("[GaussianSplatRenderer] Unable to acquire RenderingDevice for submissions");
		device_state.reported_missing_submission_device = true;
	}

	return device_state.rd;
}

RenderingDevice *RenderDeviceOrchestrator::peek_submission_device() const {
	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		RenderingDevice *shared_device = _ensure_local_device(manager->get_shared_submission_device());
		if (shared_device) {
			return shared_device;
		}
	}

	return device_state.rd;
}

RenderingDevice *RenderDeviceOrchestrator::acquire_rendering_device() {
	if (device_state.rd) {
		return device_state.rd;
	}

	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		device_state.rd = _ensure_local_device(manager->get_primary_rendering_device());
		if (!device_state.rd) {
			device_state.rd = _acquire_manager_local_device();
		}
	}

	if (!device_state.rd) {
		device_state.rd = get_main_rendering_device();
	}

	return device_state.rd;
}

RenderingDevice *RenderDeviceOrchestrator::get_main_rendering_device() const {
	if (RenderingServer *rs = RenderingServer::get_singleton()) {
		RenderingDevice *server_device = rs->get_rendering_device();
		if (server_device) {
			return server_device;
		}
	}

	if (RenderingDevice *singleton_device = RenderingDevice::get_singleton()) {
		return singleton_device;
	}

	return device_state.rd;
}

bool RenderDeviceOrchestrator::ensure_rendering_device(const char *p_context) {
	if (acquire_rendering_device()) {
		return true;
	}

	WARN_PRINT_ONCE(String("[GaussianSplatRenderer] ") + p_context + ": RenderingDevice unavailable; deferring operation");
	RenderingError error(RenderingErrorCodes::device_unavailable(), RenderingError::Severity::RECOVERABLE,
			vformat("RenderingDevice unavailable during %s", p_context));
	error.add_context("context", p_context);
	error.add_context("frame", static_cast<int64_t>(renderer->get_frame_state().frame_counter));
	if (record_rendering_error) {
		record_rendering_error(error);
	}
	if (emit_runtime_diagnostics) {
		emit_runtime_diagnostics();
	}
	return false;
}

bool RenderDeviceOrchestrator::ensure_submission_device(const char *p_context) {
	RenderingDevice *submission_device = get_submission_device();
	if (submission_device) {
		return true;
	}

	WARN_PRINT_ONCE(String("[GaussianSplatRenderer] ") + p_context + ": Submission device unavailable; deferring operation");
	RenderingError error(RenderingErrorCodes::submission_device_unavailable(), RenderingError::Severity::RECOVERABLE,
			vformat("Submission device unavailable during %s", p_context));
	error.add_context("context", p_context);
	error.add_context("frame", static_cast<int64_t>(renderer->get_frame_state().frame_counter));
	if (record_rendering_error) {
		record_rendering_error(error);
	}
	if (emit_runtime_diagnostics) {
		emit_runtime_diagnostics();
	}
	return false;
}

void RenderDeviceOrchestrator::synchronize_tile_submission(RenderingDevice *p_device, const char *p_context) {
	if (!p_device) {
		return;
	}

	const String sync_context = _sync_context_or_default(p_context, "tile_sync");
	RenderingDevice *sync_device = p_device;
	GaussianSplatManager::ScopedSubmissionLock submission_lock;
	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		RenderingDevice *acquired = manager->acquire_submission_device(sync_device, submission_lock);
		if (acquired) {
			if (sync_device != acquired) {
				push_cross_device_operation(sync_context, sync_device, acquired);
			}
			sync_device = acquired;
		}
	}

	bool synced = _submit_and_sync_local_device(sync_device);
	if (!synced && sync_device != p_device) {
		push_cross_device_operation(sync_context + String(":fallback_submit_sync"), sync_device, p_device);
		synced = _submit_and_sync_local_device(p_device);
	}

	if (synced) {
		return;
	}

	if (_is_main_rendering_device(sync_device) || _is_main_rendering_device(p_device)) {
		// Main RenderingDevice path is frame-synchronized by the engine.
		return;
	}

	WARN_PRINT_ONCE(vformat("[GaussianSplatRenderer] %s: Unable to synchronize RenderingDevice before tile fallback", sync_context));
	RenderingError error(RenderingErrorCodes::command_buffer_synchronization_failed(), RenderingError::Severity::WARNING,
			vformat("Unable to synchronize RenderingDevice before tile fallback (%s)", sync_context));
	error.add_context("context", sync_context);
	error.add_context("frame", static_cast<int64_t>(renderer->get_frame_state().frame_counter));
	if (record_rendering_error) {
		record_rendering_error(error);
	}
	if (emit_runtime_diagnostics) {
		emit_runtime_diagnostics();
	}
}

// Getter definitions moved to header (inline)

bool GaussianSplatRenderer::is_main_rendering_device(RenderingDevice *p_device) const {
	return device_orchestrator->is_main_rendering_device(p_device);
}

void GaussianSplatRenderer::_safe_submit_sync(RenderingDevice *p_device) {
	device_orchestrator->safe_submit_sync(p_device);
}

void GaussianSplatRenderer::_push_cross_device_operation(const String &p_context, RenderingDevice *p_source, RenderingDevice *p_target) {
	device_orchestrator->push_cross_device_operation(p_context, p_source, p_target);
}

Dictionary GaussianSplatRenderer::_build_device_capability_report() const {
	return device_orchestrator->build_device_capability_report();
}

void GaussianSplatRenderer::_track_resource_owner(const RID &p_rid, RenderingDevice *p_device, bool p_owned, const char *p_label) {
	device_orchestrator->track_resource_owner(p_rid, p_device, p_owned, p_label);
}

RenderingDevice *GaussianSplatRenderer::_get_resource_owner(const RID &p_rid, RenderingDevice *p_fallback) const {
	return device_orchestrator->get_resource_owner(p_rid, p_fallback);
}

RenderingDevice *GaussianSplatRenderer::get_texture_owner_device(const RID &p_texture) const {
	return device_orchestrator->get_texture_owner_device(p_texture);
}

RenderingDevice *GaussianSplatRenderer::_acquire_submission_device_for(RenderingDevice *p_device,
		GaussianSplatManager::ScopedSubmissionLock &r_lock) const {
	return device_orchestrator->acquire_submission_device_for(p_device, r_lock);
}

RD::TextureFormat GaussianSplatRenderer::_get_texture_format(RenderingDevice *p_device, RID p_texture) const {
	return device_orchestrator->get_texture_format(p_device, p_texture);
}

void GaussianSplatRenderer::_update_tile_renderer_output_tracking(const RID &p_color_output, RenderingDevice *p_color_device,
		const RID &p_depth_output, RenderingDevice *p_depth_device) {
	device_orchestrator->update_tile_renderer_output_tracking(p_color_output, p_color_device, p_depth_output, p_depth_device);
}

void GaussianSplatRenderer::_free_owned_resource(RenderingDevice *p_fallback_device, RID &p_rid) {
	device_orchestrator->free_owned_resource(p_fallback_device, p_rid);
}

RenderingDevice *GaussianSplatRenderer::_get_submission_device() {
	return device_orchestrator->get_submission_device();
}

RenderingDevice *GaussianSplatRenderer::_peek_submission_device() const {
	return device_orchestrator->peek_submission_device();
}

RenderingDevice *GaussianSplatRenderer::_acquire_rendering_device() {
	return device_orchestrator->acquire_rendering_device();
}

RenderingDevice *GaussianSplatRenderer::_get_main_rendering_device() const {
	return device_orchestrator->get_main_rendering_device();
}

bool GaussianSplatRenderer::_ensure_rendering_device(const char *p_context) {
	return device_orchestrator->ensure_rendering_device(p_context);
}

bool GaussianSplatRenderer::_ensure_submission_device(const char *p_context) {
	return device_orchestrator->ensure_submission_device(p_context);
}

void GaussianSplatRenderer::_synchronize_tile_submission(RenderingDevice *p_device, const char *p_context) {
	device_orchestrator->synchronize_tile_submission(p_device, p_context);
}
