#include "render_resource_orchestrator.h"

#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/math/math_defs.h"
#include "core/string/ustring.h"
#include "servers/rendering/rendering_device.h"
#include "gpu_sorting_config.h"
#include "../interfaces/interactive_state_manager.h"
#include "../logger/gs_logger.h"
#include "../shaders/gaussian_splat.glsl.gen.h"
#include <cstring>

RenderResourceOrchestrator::RenderResourceOrchestrator(const Dependencies &p_dependencies) :
		renderer(p_dependencies.renderer),
		device_state(p_dependencies.device_state),
		pipeline_features_effective(p_dependencies.pipeline_features_effective),
		pipeline_features_warning_cache(p_dependencies.pipeline_features_warning_cache) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_NULL(device_state);
	ERR_FAIL_NULL(pipeline_features_effective);
	ERR_FAIL_NULL(pipeline_features_warning_cache);
	resource_state.gpu_resources_initialized = false;
	resource_state.gpu_initialization_pending = false;
	resource_state.buffer_manager.instantiate();
	resource_state.buffer_manager_initialized = false;
}

void RenderResourceOrchestrator::initialize_shaders() {
	// Skip shader initialization if no RenderingDevice (headless mode)
	if (!renderer->ensure_rendering_device("_initialize_shaders")) {
		// This is OK in headless mode or when using fallback rendering
		return;
	}

	GaussianSplatRenderer::FrameStateProvider state_provider(renderer);
	const GaussianSplatRenderer::IFrameStateView &state_view = state_provider;
	const GaussianSplatRenderer::SubsystemState &subsystem_state_view = state_view.get_subsystem_state_view();

	if (pipeline_state.gaussian_shader_source == nullptr) {
		pipeline_state.gaussian_shader_source = memnew(GaussianSplatShaderRD);
	}
	ERR_FAIL_NULL_MSG(pipeline_state.gaussian_shader_source, "[Hello Splat] Gaussian shader source is not initialized");

	PackedStringArray painterly_defines;
	Ref<PainterlyMaterial> painterly_material;
	if (subsystem_state_view.painterly_renderer.is_valid()) {
		painterly_material = subsystem_state_view.painterly_renderer->get_material();
	}
	if (painterly_material.is_valid()) {
		painterly_defines = painterly_material->get_shader_define_strings();
	}

	// Initialize state shaders for interactive system
	if (subsystem_state_view.interactive_state_manager.is_valid()) {
		subsystem_state_view.interactive_state_manager->initialize_renderer_state_shaders(renderer);
	}

	// Always build a baseline variant along with an optional painterly override so we can
	// select the correct embedded shader at runtime without touching the filesystem.
	Vector<String> shader_variants;
	shader_variants.push_back("");

	int active_variant = 0;
	if (!painterly_defines.is_empty()) {
		String define_block;
		const int defines_count = painterly_defines.size();
		for (int i = 0; i < defines_count; i++) {
			define_block += "\n#define ";
			define_block += painterly_defines[i];
			define_block += "\n";
		}
		shader_variants.push_back(define_block);
		active_variant = shader_variants.size() - 1;
	}

	pipeline_state.gaussian_shader_source->initialize(shader_variants);
	pipeline_state.gaussian_shader_initialized = true;

	if (pipeline_state.gaussian_shader_version.is_valid()) {
		pipeline_state.gaussian_shader_source->version_free(pipeline_state.gaussian_shader_version);
		pipeline_state.gaussian_shader_version = RID();
	}

	pipeline_state.gaussian_shader_version = pipeline_state.gaussian_shader_source->version_create();
	if (!pipeline_state.gaussian_shader_version.is_valid()) {
		GS_LOG_WARN_DEFAULT("[Hello Splat] Failed to create Gaussian splat shader version");
		return;
	}

	pipeline_state.gaussian_shader = pipeline_state.gaussian_shader_source->version_get_shader(
			pipeline_state.gaussian_shader_version, active_variant);
	if (!pipeline_state.gaussian_shader.is_valid()) {
		pipeline_state.gaussian_shader_source->version_free(pipeline_state.gaussian_shader_version);
		pipeline_state.gaussian_shader_version = RID();
		GS_LOG_WARN_DEFAULT("[Hello Splat] Failed to retrieve Gaussian splat shader");
		return;
	}

	if (subsystem_state_view.interactive_state_manager.is_valid()) {
		subsystem_state_view.interactive_state_manager->ensure_renderer_state_shader_cache(renderer);
	}
}

void RenderResourceOrchestrator::create_gpu_resources_safe() {
	// Safe GPU resource creation - called during first process frame
	GaussianSplatRenderer::FrameStateProvider state_provider(renderer);
	const GaussianSplatRenderer::IFrameStateView &state_view = state_provider;
	const GaussianSplatRenderer::SubsystemState &subsystem_state_view = state_view.get_subsystem_state_view();

	bool needs_buffer_resize = false;
	if (resource_state.gpu_resources_initialized && !resource_state.gpu_initialization_pending) {
		if (resource_state.buffer_manager.is_valid() && resource_state.buffer_manager_initialized) {
			const uint32_t current_capacity = resource_state.buffer_manager->get_buffer_capacity();
			const uint32_t desired_capacity = static_cast<uint32_t>(MAX(0, renderer->get_performance_settings().max_splats));
			needs_buffer_resize = desired_capacity > current_capacity;
		}
		if (!needs_buffer_resize) {
			return; // Already initialized
		}
	}

	resource_state.gpu_initialization_pending = true;

	if (!renderer->ensure_rendering_device("_create_gpu_resources_safe")) {
		if (!device_state->reported_missing_render_device) {
			GS_LOG_RENDERER_WARN("[Hello Splat] RenderingDevice not available (headless mode or no GPU)");
			device_state->reported_missing_render_device = true;
		}
		return;
	}

	device_state->reported_missing_render_device = false;

	// Initialize local submission device (Issue #142: Only local devices can submit and sync)
	if (!GaussianSplatManager::get_singleton()) {
		GS_LOG_WARN_DEFAULT("[Hello Splat] GaussianSplatManager unavailable; GPU submissions may fail");
	}

	renderer->get_submission_device();

	bool buffer_manager_ready = true;

	// Initialize GPU buffer manager
	if (resource_state.buffer_manager.is_valid() && (!resource_state.buffer_manager_initialized || needs_buffer_resize)) {
		int max_splats_for_buffer = renderer->get_performance_settings().max_splats;
		Error err = resource_state.buffer_manager->initialize(
				device_state->rd, max_splats_for_buffer);
		if (err == OK) {
			resource_state.buffer_manager_initialized = true;
		} else {
			GS_LOG_GPU_MEMORY_WARN("[GPU Buffer] Failed to initialize GPU buffer manager: " + itos(err));
			buffer_manager_ready = false;
		}
	}

	if (resource_state.buffer_manager.is_valid() && resource_state.buffer_manager_initialized) {
		RID test_buffer = resource_state.buffer_manager->get_current_read_buffer();
		if (!test_buffer.is_valid()) {
			GS_LOG_WARN_DEFAULT("[GPU Buffer] GPU buffer manager initialized without a readable buffer RID");
			buffer_manager_ready = false;
		}
	}

	RenderingDevice *painterly_device = renderer->get_main_rendering_device();
	if (subsystem_state_view.painterly_renderer.is_valid()) {
		PainterlyPassGraph *pass_graph = subsystem_state_view.painterly_renderer->get_pass_graph();
		if (pass_graph) {
			pass_graph->setup(painterly_device);
		}

		// Initialize painterly pipeline resources
		if (renderer->get_painterly_config().enabled && pass_graph) {
			Size2i default_size(1280, 720);
			pass_graph->configure(default_size, renderer->get_painterly_config().internal_scale,
					renderer->get_painterly_config().enable_strokes, renderer->get_painterly_config().low_end_mode);
		}

		if (painterly_device) {
			Error err = subsystem_state_view.painterly_renderer->initialize(painterly_device, Size2i(1280, 720));
			if (err == OK) {
				// Configure with current settings
				::PainterlyConfig config;
				config.viewport_size = Size2i(1280, 720);
				config.internal_scale = renderer->get_painterly_config().internal_scale;
				config.enable_stylization = renderer->get_painterly_config().enable_strokes;
				config.enable_strokes = renderer->get_painterly_config().enable_strokes;
				config.low_end_mode = renderer->get_painterly_config().low_end_mode;
				config.edge_threshold = renderer->get_painterly_config().edge_threshold;
				config.edge_intensity = renderer->get_painterly_config().edge_intensity;
				config.stroke_length = renderer->get_painterly_config().stroke_length;
				config.stroke_opacity = renderer->get_painterly_config().stroke_opacity;
				config.gamma = renderer->get_painterly_config().gamma;
				subsystem_state_view.painterly_renderer->configure(config);

				// Compile shaders
				err = subsystem_state_view.painterly_renderer->compile_shaders();
				if (err != OK) {
					GS_LOG_WARN_DEFAULT("[Painterly] Failed to compile PainterlyRenderer shaders");
				}
			} else {
				GS_LOG_WARN_DEFAULT("[Painterly] Failed to initialize modular PainterlyRenderer");
			}
		}
	}

	// Initialize shaders and GPU sorter infrastructure
	initialize_shaders();
	if (buffer_manager_ready) {
		renderer->refresh_gpu_sorter("create_gpu_resources_safe");
	}
	if (!buffer_manager_ready && g_gpu_sorting_config.enable_performance_logging) {
		static uint32_t sorter_skip_log_counter = 0;
		if (++sorter_skip_log_counter % 60u == 1u) {
			GS_LOG_GPU_SORT_WARN(vformat("[GPU Sort] Skipping sorter init; buffer manager not ready (max_splats=%d)",
					renderer->get_performance_settings().max_splats));
		}
	}

	// Create vertex buffer for test splats
	const int splat_count = renderer->get_test_data_state().positions.size();
	if (splat_count > 0) {
		auto &test_state = renderer->get_test_data_state();
		const bool buffers_valid = test_state.vertex_buffer.is_valid() &&
				test_state.position_buffer.is_valid() &&
				test_state.scale_buffer.is_valid() &&
				test_state.rotation_buffer.is_valid() &&
				test_state.sh_buffer.is_valid();
		const bool buffers_up_to_date = buffers_valid &&
				test_state.uploaded_generation == test_state.content_generation &&
				test_state.uploaded_count == static_cast<uint32_t>(splat_count);
		if (!buffers_up_to_date) {
		// Create vertex data (position, color, scale per splat)
		Vector<uint8_t> vertex_data;
		vertex_data.resize(splat_count * (sizeof(float) * 10)); // 3 pos + 4 color + 3 scale

		uint8_t *w = vertex_data.ptrw();
		for (int i = 0; i < splat_count; i++) {
			float *vertex = (float *)(w + i * sizeof(float) * 10);
			// Position
			vertex[0] = renderer->get_test_data_state().positions[i].x;
			vertex[1] = renderer->get_test_data_state().positions[i].y;
			vertex[2] = renderer->get_test_data_state().positions[i].z;
			// Color
			vertex[3] = renderer->get_test_data_state().colors[i].r;
			vertex[4] = renderer->get_test_data_state().colors[i].g;
			vertex[5] = renderer->get_test_data_state().colors[i].b;
			vertex[6] = renderer->get_test_data_state().colors[i].a;
			// Scale
			vertex[7] = renderer->get_test_data_state().scales[i].x;
			vertex[8] = renderer->get_test_data_state().scales[i].y;
			vertex[9] = renderer->get_test_data_state().scales[i].z;
		}

		renderer->get_test_data_state().vertex_buffer = device_state->rd->vertex_buffer_create(vertex_data.size(), vertex_data);
		if (renderer->get_test_data_state().vertex_buffer.is_valid()) {
			device_state->rd->set_resource_name(renderer->get_test_data_state().vertex_buffer, "GS_RenderResourceOrchestrator_VertexBuffer");
		}
		renderer->track_resource_owner(renderer->get_test_data_state().vertex_buffer, device_state->rd);
		if (!renderer->get_test_data_state().vertex_buffer.is_valid()) {
			GS_LOG_RENDERER_WARN("[Hello Splat] Failed to create vertex buffer");
		}

		// Populate structured buffers for the splat shader path.
		Vector<float> position_data;
		position_data.resize(splat_count * 4);
		Vector<float> scale_data;
		scale_data.resize(splat_count * 4);
		Vector<float> rotation_data;
		rotation_data.resize(splat_count * 4);
		Vector<float> sh_data;
		sh_data.resize(splat_count * 16);

		float *position_ptr = position_data.ptrw();
		float *scale_ptr = scale_data.ptrw();
		float *rotation_ptr = rotation_data.ptrw();
		float *sh_ptr_base = sh_data.ptrw();

		for (int i = 0; i < splat_count; i++) {
			const Vector3 &pos = renderer->get_test_data_state().positions[i];
			const Color &col = renderer->get_test_data_state().colors[i];
			const Vector3 &scl = renderer->get_test_data_state().scales[i];

			float *pos_out = position_ptr + i * 4;
			pos_out[0] = pos.x;
			pos_out[1] = pos.y;
			pos_out[2] = pos.z;
			pos_out[3] = col.a;

			float *scale_out = scale_ptr + i * 4;
			scale_out[0] = scl.x;
			scale_out[1] = scl.y;
			scale_out[2] = scl.z;
			scale_out[3] = 0.0f;

			float *rotation_out = rotation_ptr + i * 4;
			rotation_out[0] = 0.0f;
			rotation_out[1] = 0.0f;
			rotation_out[2] = 0.0f;
			rotation_out[3] = 1.0f;

			float *sh_ptr = sh_ptr_base + i * 16;
			sh_ptr[0] = col.r;
			sh_ptr[1] = col.g;
			sh_ptr[2] = col.b;
			uint32_t splat_metadata = 0u;
			memcpy(&sh_ptr[3], &splat_metadata, sizeof(uint32_t));
			for (int k = 4; k < 16; k++) {
				sh_ptr[k] = 0.0f;
			}
		}

		Vector<uint8_t> position_bytes;
		position_bytes.resize(position_data.size() * sizeof(float));
		memcpy(position_bytes.ptrw(), position_data.ptr(), position_bytes.size());

		Vector<uint8_t> scale_bytes;
		scale_bytes.resize(scale_data.size() * sizeof(float));
		memcpy(scale_bytes.ptrw(), scale_data.ptr(), scale_bytes.size());

		Vector<uint8_t> rotation_bytes;
		rotation_bytes.resize(rotation_data.size() * sizeof(float));
		memcpy(rotation_bytes.ptrw(), rotation_data.ptr(), rotation_bytes.size());

		Vector<uint8_t> sh_bytes;
		sh_bytes.resize(sh_data.size() * sizeof(float));
		memcpy(sh_bytes.ptrw(), sh_data.ptr(), sh_bytes.size());

		if (renderer->get_test_data_state().position_buffer.is_valid()) {
			renderer->free_owned_resource(device_state->rd, renderer->get_test_data_state().position_buffer);
		}
		renderer->get_test_data_state().position_buffer = device_state->rd->storage_buffer_create(position_bytes.size(), position_bytes);
		if (renderer->get_test_data_state().position_buffer.is_valid()) {
			device_state->rd->set_resource_name(renderer->get_test_data_state().position_buffer, "GS_RenderResourceOrchestrator_PositionBuffer");
		}
		renderer->track_resource_owner(renderer->get_test_data_state().position_buffer, device_state->rd);

		if (renderer->get_test_data_state().scale_buffer.is_valid()) {
			renderer->free_owned_resource(device_state->rd, renderer->get_test_data_state().scale_buffer);
		}
		renderer->get_test_data_state().scale_buffer = device_state->rd->storage_buffer_create(scale_bytes.size(), scale_bytes);
		if (renderer->get_test_data_state().scale_buffer.is_valid()) {
			device_state->rd->set_resource_name(renderer->get_test_data_state().scale_buffer, "GS_RenderResourceOrchestrator_ScaleBuffer");
		}
		renderer->track_resource_owner(renderer->get_test_data_state().scale_buffer, device_state->rd);

		if (renderer->get_test_data_state().rotation_buffer.is_valid()) {
			renderer->free_owned_resource(device_state->rd, renderer->get_test_data_state().rotation_buffer);
		}
		renderer->get_test_data_state().rotation_buffer = device_state->rd->storage_buffer_create(rotation_bytes.size(), rotation_bytes);
		if (renderer->get_test_data_state().rotation_buffer.is_valid()) {
			device_state->rd->set_resource_name(renderer->get_test_data_state().rotation_buffer, "GS_RenderResourceOrchestrator_RotationBuffer");
		}
		renderer->track_resource_owner(renderer->get_test_data_state().rotation_buffer, device_state->rd);

		if (renderer->get_test_data_state().sh_buffer.is_valid()) {
			renderer->free_owned_resource(device_state->rd, renderer->get_test_data_state().sh_buffer);
		}
		renderer->get_test_data_state().sh_buffer = device_state->rd->storage_buffer_create(sh_bytes.size(), sh_bytes);
		if (renderer->get_test_data_state().sh_buffer.is_valid()) {
			device_state->rd->set_resource_name(renderer->get_test_data_state().sh_buffer, "GS_RenderResourceOrchestrator_SHBuffer");
		}
		renderer->track_resource_owner(renderer->get_test_data_state().sh_buffer, device_state->rd);

		test_state.uploaded_generation = test_state.content_generation;
		test_state.uploaded_count = static_cast<uint32_t>(splat_count);
		}

	if (!renderer->get_test_data_state().position_buffer.is_valid() || !renderer->get_test_data_state().scale_buffer.is_valid() ||
			!renderer->get_test_data_state().rotation_buffer.is_valid() || !renderer->get_test_data_state().sh_buffer.is_valid()) {
		GS_LOG_WARN_DEFAULT("[Hello Splat] Failed to allocate splat attribute buffers for shader path");
	}
	}

	// Prepare painterly material GPU resources if available
	if (renderer->get_subsystem_state().painterly_renderer.is_valid()) {
		renderer->get_subsystem_state().painterly_renderer->update_painterly_gpu_resources(renderer);
	}

	// Initialize TileRenderer and TileRasterizer (proper initialization, not lazy)
	// This ensures the tile rendering pipeline is ready before first render call
	bool tile_renderer_exists = renderer->get_tile_renderer_state().renderer.is_valid();
	bool tile_init_failed = renderer->get_tile_renderer_state().init_failed;
	const auto &debug_config = renderer->get_debug_config();
	const bool log_enabled = debug_config.enable_data_logging ||
			debug_config.enable_frame_logging ||
			debug_config.enable_tile_pipeline_logs ||
			debug_config.enable_all_debug;
#ifdef DEBUG_ENABLED
	if (log_enabled) {
		GS_LOG_RENDERER_DEBUG(vformat("[GPU-RESOURCES] TileRenderer check: exists=%s init_failed=%s",
				tile_renderer_exists ? "yes" : "no", tile_init_failed ? "yes" : "no"));
	}
#endif
	if (!tile_renderer_exists && !tile_init_failed) {
#ifdef DEBUG_ENABLED
		if (log_enabled) {
			GS_LOG_RENDERER_DEBUG("[GPU-RESOURCES] Creating TileRenderer...");
		}
#endif
		renderer->get_tile_renderer_state().renderer.instantiate();
		renderer->get_tile_renderer_state().renderer->set_performance_monitor(&renderer->get_tile_renderer_state().gpu_performance_monitor);
		Size2i default_viewport(1280, 720);
#ifdef DEBUG_ENABLED
		if (log_enabled) {
			GS_LOG_RENDERER_DEBUG(vformat("[GPU-RESOURCES] Calling TileRenderer::initialize with rd=%s",
					device_state->rd ? "valid" : "null"));
		}
#endif
		Error init_err = renderer->get_tile_renderer_state().renderer->initialize(
				device_state->rd,
				default_viewport,
				TileRenderer::DEFAULT_TILE_SIZE,
				RD::DATA_FORMAT_R8G8B8A8_UNORM,
				device_state->rd);
#ifdef DEBUG_ENABLED
		if (log_enabled) {
			GS_LOG_RENDERER_DEBUG(vformat("[GPU-RESOURCES] TileRenderer::initialize returned: %d", init_err));
		}
#endif
		if (init_err == OK) {
			bool is_init = renderer->get_tile_renderer_state().renderer->is_initialized();
#ifdef DEBUG_ENABLED
			if (log_enabled) {
				GS_LOG_RENDERER_DEBUG(vformat("[GPU-RESOURCES] TileRenderer is_initialized after init: %s", is_init ? "yes" : "no"));
			}
#endif
			if (!subsystem_state_view.rasterizer.is_valid()) {
				renderer->get_subsystem_state().rasterizer.instantiate();
			}
			renderer->get_subsystem_state().rasterizer->set_device_manager(subsystem_state_view.device_manager);
			renderer->get_subsystem_state().rasterizer->set_tile_renderer(renderer->get_tile_renderer_state().renderer);
			GS_LOG_INFO_DEFAULT("[TileRenderer] Initialized during GPU resource setup");
		} else {
			GS_LOG_WARN_DEFAULT(vformat("[TileRenderer] Failed to initialize: %d", init_err));
			renderer->get_tile_renderer_state().renderer.unref();
			renderer->get_tile_renderer_state().init_failed = true;
		}
	} else {
#ifdef DEBUG_ENABLED
		if (log_enabled) {
			GS_LOG_RENDERER_DEBUG("[GPU-RESOURCES] TileRenderer NOT initialized (already exists or init_failed)");
		}
#endif
	}

	// Initialize InteractiveStateManager now that we have a RenderingDevice
	if (subsystem_state_view.interactive_state_manager.is_valid() &&
			!subsystem_state_view.interactive_state_manager->is_initialized()) {
		Error state_err = subsystem_state_view.interactive_state_manager->initialize(device_state->rd);
		if (state_err != OK) {
			GS_LOG_WARN_DEFAULT(vformat("[InteractiveStateManager] Failed to initialize: %d", state_err));
		}
	}

	if (buffer_manager_ready) {
		resource_state.gpu_resources_initialized = true;
		resource_state.gpu_initialization_pending = false;
	} else {
		resource_state.gpu_resources_initialized = false;
		// Keep pending true so that we retry initialization on the next frame once the device becomes available.
	}
}

void RenderResourceOrchestrator::update_pipeline_features(RenderingDevice *p_device) {
	ERR_FAIL_NULL(pipeline_features_effective);
	ERR_FAIL_NULL(pipeline_features_warning_cache);

	String warnings;
	PipelineFeatureSet effective = g_pipeline_feature_set.get_effective(
			p_device,
			g_gpu_sorting_config.enable_compute_raster,
			true,
			&warnings);

	if (!warnings.is_empty() && warnings != *pipeline_features_warning_cache) {
		GS_LOG_WARN_DEFAULT("[Pipeline Feature Set] Capability validation warnings:");
		PackedStringArray lines = warnings.split("\n", false);
		for (int i = 0; i < lines.size(); ++i) {
			const String &line = lines[i];
			if (!line.is_empty()) {
				GS_LOG_WARN_DEFAULT(vformat("[Pipeline Feature Set] %s", line));
			}
		}
	}

	*pipeline_features_warning_cache = warnings;
	*pipeline_features_effective = effective;
}

void RenderResourceOrchestrator::update_gpu_pass_metrics_from_tile_renderer() {
	GaussianSplatRenderer::FrameStateProvider state_provider(renderer);
	GaussianSplatRenderer::IFrameMutationAccess &state_mut = state_provider;
	const GaussianSplatRenderer::IFrameStateView &state_view = state_provider;
	GaussianSplatRenderer::PerformanceMetrics &metrics = state_mut.get_performance_state_mut().metrics;

	if (!renderer->get_tile_renderer_state().renderer.is_valid()) {
		metrics.gpu_tile_binning_time_ms = 0.0f;
		metrics.gpu_tile_raster_time_ms = 0.0f;
		metrics.gpu_tile_prefix_time_ms = 0.0f;
		metrics.gpu_tile_resolve_time_ms = 0.0f;
		metrics.gpu_frame_time_ms = 0.0f;
		metrics.gpu_utilization = 0.0f;
		metrics.gpu_timing_frame_serial = 0;
		metrics.gpu_timing_frames_behind = 0;
		metrics.gpu_timeline_inflight_frames = 0;
		metrics.gpu_timeline_completed_frames = 0;
		metrics.gpu_timeline_stall_count = 0;
		metrics.gpu_timeline_stall_ms = 0.0f;
		metrics.gpu_timeline_last_value = 0;
		metrics.tile_sort_sync_fallback_count = 0;
		return;
	}

	// Use subsystem_state.rasterizer interface for GPU timing (Phase 8 migration)
	state_view.get_subsystem_state_view().rasterizer->resolve_gpu_timestamps_async();
	RasterPerformance perf = state_view.get_subsystem_state_view().rasterizer->get_performance();

	metrics.gpu_tile_binning_time_ms = perf.binning_gpu_ms;
	metrics.gpu_tile_raster_time_ms = perf.raster_gpu_ms;
	metrics.gpu_tile_prefix_time_ms = perf.prefix_gpu_ms;
	metrics.gpu_tile_resolve_time_ms = perf.resolve_gpu_ms;
	metrics.gpu_frame_time_ms = perf.frame_gpu_ms;
	metrics.tile_sort_sync_fallback_count = perf.sort_sync_fallback_count;
	metrics.gpu_timing_frame_serial = perf.timing_frame_serial;
	metrics.gpu_timing_frames_behind = perf.timing_frames_behind;

	GPUPerformanceMonitor::SummaryMetrics timeline_summary =
		renderer->get_tile_renderer_state().gpu_performance_monitor.get_summary_metrics();
	metrics.gpu_timeline_inflight_frames = timeline_summary.inflight_frames;
	metrics.gpu_timeline_completed_frames = timeline_summary.completed_frames;
	metrics.gpu_timeline_stall_count = timeline_summary.stall_count;
	metrics.gpu_timeline_stall_ms = float(timeline_summary.total_stall_ns) / 1000000.0f;
	metrics.gpu_timeline_last_value = timeline_summary.last_frame_index;

	float utilization = renderer->get_tile_renderer_state().gpu_performance_monitor.get_gpu_utilization_async();
	metrics.gpu_utilization = utilization * 100.0f;
}

RID RenderResourceOrchestrator::load_graphics_shader(const Vector<String> &p_vertex_paths,
		const Vector<String> &p_fragment_paths) {
	RenderingDevice *device = device_state->rd;
	if (!device) {
		if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
			device = manager->get_primary_rendering_device();
		}
	}
	if (!device) {
		return RID();
	}

	Vector<uint8_t> vertex_spirv;
	const int vertex_path_count = p_vertex_paths.size();
	for (int i = 0; i < vertex_path_count; i++) {
		const String &path = p_vertex_paths[i];
		if (path.is_empty()) {
			continue;
		}

		Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
		if (file.is_null()) {
			continue;
		}

		String source = file->get_as_text();
		if (source.is_empty()) {
			continue;
		}

		vertex_spirv = device->shader_compile_spirv_from_source(RD::SHADER_STAGE_VERTEX, source);
		if (!vertex_spirv.is_empty()) {
			break;
		}
	}

	if (vertex_spirv.is_empty()) {
		return RID();
	}

	Vector<uint8_t> fragment_spirv;
	const int fragment_path_count = p_fragment_paths.size();
	for (int i = 0; i < fragment_path_count; i++) {
		const String &path = p_fragment_paths[i];
		if (path.is_empty()) {
			continue;
		}

		Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
		if (file.is_null()) {
			continue;
		}

		String source = file->get_as_text();
		if (source.is_empty()) {
			continue;
		}

		fragment_spirv = device->shader_compile_spirv_from_source(RD::SHADER_STAGE_FRAGMENT, source);
		if (!fragment_spirv.is_empty()) {
			break;
		}
	}

	if (fragment_spirv.is_empty()) {
		return RID();
	}

	Vector<RD::ShaderStageSPIRVData> stages;
	RD::ShaderStageSPIRVData stage;
	stage.shader_stage = RD::SHADER_STAGE_VERTEX;
	stage.spirv = vertex_spirv;
	stages.push_back(stage);

	stage.shader_stage = RD::SHADER_STAGE_FRAGMENT;
	stage.spirv = fragment_spirv;
	stages.push_back(stage);

	RID shader = device->shader_create_from_spirv(stages);
	if (shader.is_valid()) {
		renderer->track_resource_owner(shader, device);
	}
	return shader;
}

void GaussianSplatRenderer::_initialize_shaders() {
	resource_orchestrator->initialize_shaders();
}

void GaussianSplatRenderer::_create_gpu_resources_safe() {
	resource_orchestrator->create_gpu_resources_safe();
}

void GaussianSplatRenderer::_update_gpu_pass_metrics_from_tile_renderer() {
	resource_orchestrator->update_gpu_pass_metrics_from_tile_renderer();
}

void GaussianSplatRenderer::update_pipeline_features(RenderingDevice *p_device) {
	resource_orchestrator->update_pipeline_features(p_device);
}

RID GaussianSplatRenderer::_load_graphics_shader(const Vector<String> &p_vertex_paths,
		const Vector<String> &p_fragment_paths) {
	return resource_orchestrator->load_graphics_shader(p_vertex_paths, p_fragment_paths);
}
