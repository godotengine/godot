/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "test_macros.h"

#include <cstring>
#include <cstddef>

#include "core/config/project_settings.h"
#include "core/variant/variant.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_manager.h"
#include "../core/gaussian_streaming.h"
#include "../renderer/gaussian_gpu_layout.h"
#include "../renderer/gpu_sorting_config.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../renderer/render_debug_state_orchestrator.h"
#include "../renderer/render_instancing_orchestrator.h"
#include "../renderer/render_pipeline_stages.h"
#include "../interfaces/gpu_sorting_pipeline.h"
#include "../interfaces/output_compositor.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering_server.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_data_rd.h"
#include "servers/rendering/storage/render_scene_buffers.h"

namespace TestGaussianSplatting {

static_assert(GS_RENDER_PARAMS_LAYOUT_VERSION == 16, "Render params layout version mismatch");

static_assert(sizeof(InstanceDataGPU) == 96, "InstanceDataGPU size contract changed");
static_assert(offsetof(InstanceDataGPU, lod) == 72, "InstanceDataGPU.lod offset contract changed");
static_assert(offsetof(InstanceDataGPU, wind_params) == 80, "InstanceDataGPU.wind_params offset contract changed");
static_assert(sizeof(AssetMetaGPU) == 112, "AssetMetaGPU size contract changed");
static_assert(offsetof(AssetMetaGPU, lod_ranges) == 48, "AssetMetaGPU.lod_ranges offset contract changed");
static_assert(sizeof(ChunkMetaGPU) == 64, "ChunkMetaGPU size contract changed");
static_assert(offsetof(ChunkMetaGPU, sh_limit) == 44, "ChunkMetaGPU.sh_limit offset contract changed");
static_assert(sizeof(SplatRefGPU) == 8, "SplatRefGPU size contract changed");
static_assert(sizeof(PackedGaussian) == 144, "PackedGaussian size contract changed");
static_assert(offsetof(PackedGaussian, rotation) == 32, "PackedGaussian.rotation offset contract changed");
static_assert(offsetof(PackedGaussian, sh) == 48, "PackedGaussian.sh offset contract changed");
static_assert(offsetof(PackedGaussian, sh_metadata) == 140, "PackedGaussian.sh_metadata offset contract changed");
static_assert(sizeof(PackedGaussianF16) == 144, "PackedGaussianF16 size contract changed");
static_assert(sizeof(PackedGaussianQuantized) == 80, "PackedGaussianQuantized size contract changed");
static_assert(sizeof(TileRenderParamsGPU) == 720, "TileRenderParamsGPU size contract changed");
static_assert(offsetof(TileRenderParamsGPU, viewport_size) == 256, "TileRenderParamsGPU.viewport_size offset contract changed");
static_assert(offsetof(TileRenderParamsGPU, camera_position) == 320, "TileRenderParamsGPU.camera_position offset contract changed");
static_assert(offsetof(TileRenderParamsGPU, lighting_mode) == 560, "TileRenderParamsGPU.lighting_mode offset contract changed");
static_assert(offsetof(TileRenderParamsGPU, instance_rotation_inv_col2) == 640, "TileRenderParamsGPU.instance_rotation_inv_col2 offset contract changed");
static_assert(offsetof(TileRenderParamsGPU, wind_dir_strength) == 656, "TileRenderParamsGPU.wind_dir_strength offset contract changed");
static_assert(offsetof(TileRenderParamsGPU, wind_time_config) == 672, "TileRenderParamsGPU.wind_time_config offset contract changed");
static_assert(offsetof(TileRenderParamsGPU, effector_sphere) == 688, "TileRenderParamsGPU.effector_sphere offset contract changed");
static_assert(offsetof(TileRenderParamsGPU, effector_config) == 704, "TileRenderParamsGPU.effector_config offset contract changed");

// Utility to ensure we have a GaussianSplatManager available during the test run.
// Note: Named with "Pipeline" suffix to avoid redefinition with test_render_validation.h
class ScopedGaussianManagerPipeline {
    GaussianSplatManager *manager = nullptr;
    bool owns_instance = false;

public:
    ScopedGaussianManagerPipeline() {
        manager = GaussianSplatManager::get_singleton();
        if (!manager) {
            manager = memnew(GaussianSplatManager);
            owns_instance = true;
        }
    }

    ~ScopedGaussianManagerPipeline() {
        if (owns_instance && manager) {
            memdelete(manager);
        }
    }

    GaussianSplatManager *get() const { return manager; }
};

class ScopedProjectSetting {
    ProjectSettings *settings = nullptr;
    String setting_path;
    Variant previous_value;
    bool had_previous_value = false;

public:
    ScopedProjectSetting(ProjectSettings *p_settings, const String &p_setting_path) :
            settings(p_settings),
            setting_path(p_setting_path) {
        if (settings && settings->has_setting(setting_path)) {
            previous_value = settings->get_setting(setting_path);
            had_previous_value = true;
        }
    }

    ~ScopedProjectSetting() {
        if (!settings) {
            return;
        }

        if (had_previous_value) {
            settings->set_setting(setting_path, previous_value);
        } else {
            settings->clear(setting_path);
        }
        settings->emit_signal("settings_changed");
    }
};

class ScopedGpuSortingConfigReload {
public:
    ~ScopedGpuSortingConfigReload() {
        g_gpu_sorting_config.load_from_project_settings();
    }
};

class ScopedRenderingDeviceLease {
    RenderingDevice *device = nullptr;
    bool owns_device = false;

public:
    RenderingDevice *acquire(RenderingServer *p_rendering_server, GaussianSplatManager *p_manager) {
        if (p_manager) {
            device = p_manager->get_primary_rendering_device();
        }
        if (!device && p_rendering_server) {
            device = p_rendering_server->create_local_rendering_device();
            owns_device = device != nullptr;
        }
        return device;
    }

    ~ScopedRenderingDeviceLease() {
        if (owns_device && device) {
            memdelete(device);
        }
    }
};

static void fill_gaussians(LocalVector<Gaussian> &p_gaussians, uint32_t p_count) {
    p_gaussians.resize(p_count);
    for (uint32_t i = 0; i < p_count; i++) {
        Gaussian &g = p_gaussians[i];
        g = Gaussian{};
        const float ring = float(i % 64) * 0.02f;
        const float layer = float(i / 64) * 0.05f;
        g.position = Vector3(ring * Math::sin((float)i * 0.1f), layer, -5.0f - layer);
        g.scale = Vector3(0.08f, 0.08f, 0.08f);
        g.opacity = 0.9f;
        g.sh_dc = Color(0.4f + 0.6f * Math::sin(i * 0.05f), 0.5f, 0.6f + 0.3f * Math::cos(i * 0.07f), g.opacity);
        g.normal = Vector3(0, 1, 0);
        g.area = 0.01f;
        g.brush_axes = Vector2(1.0f, 0.0f);
        g.painterly_meta = gaussian_pack_painterly_meta(i % 8);
    }
}

static Vector<GaussianSplatRenderer::StaticChunk> make_single_static_chunk(uint32_t p_count, const AABB &p_bounds) {
    Vector<GaussianSplatRenderer::StaticChunk> chunks;
    chunks.resize(1);

    GaussianSplatRenderer::StaticChunk chunk;
    chunk.bounds = p_bounds;
    chunk.center = p_bounds.get_center();
    chunk.radius = MAX(MAX(p_bounds.size.x, p_bounds.size.y), p_bounds.size.z) * 0.5f;
    if (chunk.radius <= 0.0f) {
        chunk.radius = 1.0f;
    }

    chunk.indices.resize(p_count);
    for (uint32_t i = 0; i < p_count; i++) {
        chunk.indices.write[i] = i;
    }

    chunks.write[0] = chunk;
    return chunks;
}

static Vector<GaussianSplatRenderer::StaticChunk> make_overlapping_static_chunks(uint32_t p_chunk_size, const AABB &p_bounds) {
    Vector<GaussianSplatRenderer::StaticChunk> chunks;
    chunks.resize(2);

    auto make_chunk = [&](uint32_t p_start_index) {
        GaussianSplatRenderer::StaticChunk chunk;
        chunk.bounds = p_bounds;
        chunk.center = p_bounds.get_center();
        chunk.radius = MAX(MAX(p_bounds.size.x, p_bounds.size.y), p_bounds.size.z) * 0.5f;
        if (chunk.radius <= 0.0f) {
            chunk.radius = 1.0f;
        }
        chunk.indices.resize(p_chunk_size);
        for (uint32_t i = 0; i < p_chunk_size; i++) {
            chunk.indices.write[i] = p_start_index + i;
        }
        return chunk;
    };

    chunks.write[0] = make_chunk(0);
    chunks.write[1] = make_chunk(p_chunk_size / 2);
    return chunks;
}

static Vector<uint32_t> read_renderer_sort_indices(const Ref<GaussianSplatRenderer> &p_renderer, uint32_t p_count) {
    Vector<uint32_t> indices;
    indices.resize(p_count);
    const Vector<uint8_t> &bytes = p_renderer->get_sorting_state().sort_index_bytes;
    const int required_bytes = int(p_count * sizeof(uint32_t));
    if (bytes.size() < required_bytes) {
        indices.resize(0);
        return indices;
    }

    memcpy(indices.ptrw(), bytes.ptr(), required_bytes);
    return indices;
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Global CPU cull telemetry overrides retired legacy disabled route state") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();
    renderer->test_release_current_streaming_system();

    Vector<Vector3> positions;
    positions.push_back(Vector3(0.0f, 0.0f, -2.0f));
    positions.push_back(Vector3(0.0f, 0.0f, -4.0f));
    renderer->test_set_test_splats(positions);

    const RID legacy_buffer = RID::from_uint64(0x19u);
    renderer->track_resource_owner(legacy_buffer, primary_rd, false, "test_legacy_gpu_cull_buffer");
    GaussianSplatRenderer::StreamingState &streaming_state = renderer->get_streaming_state();
    streaming_state.registered_gaussian_buffer = legacy_buffer;

    Projection projection;
    projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

	const int visible = renderer->test_cull_visible_count(Transform3D(), projection, Size2i(1280, 720));
	CHECK(visible > 0);

	Dictionary stats = renderer->get_render_stats();
	CHECK_MESSAGE(stats.get("cull_route_uid", String()) == String(RenderRouteUID::GLOBAL_CULL_CPU),
			"Expected executed global CPU culling to publish the global CPU route");
	CHECK_MESSAGE(stats.get("cull_route_reason", String()) == String("global_cpu_path"),
			"Expected executed global CPU culling to publish the global CPU reason");
	CHECK_MESSAGE(!bool(stats.get("cull_route_uid_missing", true)),
			"Expected render stats to mark cull_route_uid as present");

	Dictionary snapshot = renderer->get_runtime_diagnostic_snapshot();
	Dictionary telemetry = snapshot.get("telemetry", Dictionary());
	CHECK_MESSAGE(telemetry.get("cull_route_uid", String()) == String(RenderRouteUID::GLOBAL_CULL_CPU),
			"Expected runtime diagnostic telemetry to preserve the executed global CPU route");
	CHECK_MESSAGE(telemetry.get("cull_route_reason", String()) == String("global_cpu_path"),
			"Expected runtime diagnostic telemetry to preserve the executed global CPU reason");

	renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Instance cull failures that fall through to global CPU publish global CPU routes") {
    Ref<GPUCuller> culler;
    culler.instantiate();
    CHECK(culler.is_valid());
    if (!culler.is_valid()) {
        return;
    }

    GPUCuller::InstancePipelineInputs instance_inputs;
    culler->set_instance_pipeline_inputs(instance_inputs);

    LocalVector<Vector3> positions;
    positions.push_back(Vector3(0.0f, 0.0f, -2.0f));

    GPUCuller::CullingInputs inputs;
    inputs.test_positions = &positions;

    Projection projection;
    projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    GPUCuller::CullingSummary summary =
            culler->cull_for_view(Transform3D(), projection, Size2i(1280, 720), inputs);

    CHECK(summary.visible_after_culling > 0);
    CHECK(summary.cull_route_uid == String(RenderRouteUID::GLOBAL_CULL_CPU));
    CHECK(summary.cull_route_reason == String("global_cpu_path"));
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Instance cull failures without fallback preserve explicit no-data skip routes") {
    Ref<GPUCuller> culler;
    culler.instantiate();
    CHECK(culler.is_valid());
    if (!culler.is_valid()) {
        return;
    }

    GPUCuller::InstancePipelineInputs instance_inputs;
    culler->set_instance_pipeline_inputs(instance_inputs);

    Projection projection;
    projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    GPUCuller::CullingInputs inputs;
    GPUCuller::CullingSummary summary =
            culler->cull_for_view(Transform3D(), projection, Size2i(1280, 720), inputs);

    CHECK(summary.visible_after_culling == 0);
    CHECK(summary.cull_route_uid == String(RenderRouteUID::COMMON_SKIP_NO_DATA));
    CHECK(summary.cull_route_reason == String("instance_pipeline_failed_no_fallback"));
}

TEST_CASE("[GaussianSplatting] GPU layout contract invariants remain stable") {
    CHECK(GS_RENDER_PARAMS_LAYOUT_VERSION == 16u);
    CHECK(sizeof(InstanceDataGPU) == size_t(96));
    CHECK(sizeof(AssetMetaGPU) == size_t(112));
    CHECK(sizeof(ChunkMetaGPU) == size_t(64));
    CHECK(sizeof(SplatRefGPU) == size_t(8));
    CHECK(sizeof(PackedGaussian) == size_t(144));
    CHECK(sizeof(PackedGaussianF16) == size_t(144));
    CHECK(sizeof(PackedGaussianQuantized) == size_t(80));
    CHECK(sizeof(TileRenderParamsGPU) == size_t(720));

    CHECK(offsetof(PackedGaussian, rotation) == size_t(32));
    CHECK(offsetof(PackedGaussian, sh) == size_t(48));
    CHECK(offsetof(PackedGaussian, sh_metadata) == size_t(140));

    CHECK(offsetof(TileRenderParamsGPU, viewport_size) == size_t(256));
    CHECK(offsetof(TileRenderParamsGPU, camera_position) == size_t(320));
    CHECK(offsetof(TileRenderParamsGPU, lighting_mode) == size_t(560));
    CHECK(offsetof(TileRenderParamsGPU, instance_rotation_inv_col2) == size_t(640));
    CHECK(offsetof(TileRenderParamsGPU, wind_dir_strength) == size_t(656));
    CHECK(offsetof(TileRenderParamsGPU, wind_time_config) == size_t(672));
    CHECK(offsetof(TileRenderParamsGPU, effector_sphere) == size_t(688));
    CHECK(offsetof(TileRenderParamsGPU, effector_config) == size_t(704));
}

static GaussianRenderPipeline::InstancePipelineBuffers make_ready_instance_pipeline_buffers(bool p_quantization_required) {
    GaussianRenderPipeline::InstancePipelineBuffers buffers;
    uint64_t rid_id = 1;
    auto next_rid = [&rid_id]() {
        return RID::from_uint64(rid_id++);
    };

    buffers.instance_buffer = next_rid();
    buffers.asset_meta_buffer = next_rid();
    buffers.asset_chunk_index_buffer = next_rid();
    buffers.chunk_meta_buffer = next_rid();
    buffers.visible_chunk_buffer = next_rid();
    buffers.splat_ref_buffer = next_rid();
    buffers.sort_key_buffer = next_rid();
    buffers.sort_value_buffer = next_rid();
    buffers.atlas_gaussian_buffer = next_rid();
    buffers.counter_buffer = next_rid();
    buffers.chunk_dispatch_buffer = next_rid();
    buffers.indirect_count_buffer = next_rid();
    buffers.instance_count_buffer = next_rid();
    buffers.quantization_required = p_quantization_required;
    if (p_quantization_required) {
        buffers.quantization_buffer = next_rid();
    }

    buffers.instance_count = 1;
    buffers.dispatch_chunk_count = 1;
    buffers.max_visible_chunks = 1;
    buffers.max_visible_splats = 1;
    buffers.max_chunk_splats = 1;
    return buffers;
}

TEST_CASE("[GaussianSplatting] Instanced readiness gate reports deterministic buffer failure modes") {
    GaussianRenderPipeline::InstancePipelineBuffers ready_buffers = make_ready_instance_pipeline_buffers(false);

    RenderInstancingOrchestrator::InstanceReadinessResult missing_streaming =
            RenderInstancingOrchestrator::evaluate_instance_pipeline_readiness(
                    false, true, ready_buffers);
    CHECK(!missing_streaming.ready);
    CHECK(missing_streaming.failure_mode ==
            RenderInstancingOrchestrator::InstanceReadinessFailureMode::STREAMING_SYSTEM_UNAVAILABLE);

    RenderInstancingOrchestrator::InstanceReadinessResult missing_pipeline_buffers =
            RenderInstancingOrchestrator::evaluate_instance_pipeline_readiness(
                    true, false, ready_buffers);
    CHECK(!missing_pipeline_buffers.ready);
    CHECK(missing_pipeline_buffers.failure_mode ==
            RenderInstancingOrchestrator::InstanceReadinessFailureMode::INSTANCE_PIPELINE_BUFFERS_UNAVAILABLE);

    GaussianRenderPipeline::InstancePipelineBuffers invalid_buffers = ready_buffers;
    invalid_buffers.splat_ref_buffer = RID();
    RenderInstancingOrchestrator::InstanceReadinessResult invalid_pipeline_buffers =
            RenderInstancingOrchestrator::evaluate_instance_pipeline_readiness(
                    true, true, invalid_buffers);
    CHECK(!invalid_pipeline_buffers.ready);
    CHECK(invalid_pipeline_buffers.failure_mode ==
            RenderInstancingOrchestrator::InstanceReadinessFailureMode::INSTANCE_PIPELINE_BUFFERS_INVALID);

    RenderInstancingOrchestrator::InstanceReadinessResult ready =
            RenderInstancingOrchestrator::evaluate_instance_pipeline_readiness(
                    true, true, ready_buffers);
    CHECK(ready.ready);
    CHECK(ready.failure_mode == RenderInstancingOrchestrator::InstanceReadinessFailureMode::NONE);
}

TEST_CASE("[GaussianSplatting] Cull projection contract applies flip_y consistently") {
	Ref<GaussianSplatRenderer> renderer;
	renderer.instantiate();
	CHECK(renderer.is_valid());
	if (!renderer.is_valid()) {
		return;
	}

	Projection projection;
	projection.columns[1][1] = 1.75f;

	RenderDataRD render_data;
	RenderSceneDataRD scene_data;
	render_data.scene_data = &scene_data;

	scene_data.flip_y = false;
	const Projection unflipped = renderer->build_cull_projection(&render_data, projection);
	CHECK(unflipped.columns[1][1] == doctest::Approx(1.75f));

	scene_data.flip_y = true;
	const Projection flipped = renderer->build_cull_projection(&render_data, projection);
	CHECK(flipped.columns[1][1] == doctest::Approx(-1.75f));

	CHECK(renderer->validate_cull_projection_contract(&render_data, projection, flipped, "unit_test"));
	CHECK(renderer->get_performance_state().metrics.cull_projection_contract_mismatch_count == 0);

	CHECK(!renderer->validate_cull_projection_contract(&render_data, projection, unflipped, "unit_test_mismatch"));
	CHECK(renderer->get_performance_state().metrics.cull_projection_contract_mismatch_count == 1);
}

TEST_CASE("[GaussianSplatting] Instanced readiness gate requires quantization buffer when enabled") {
    GaussianRenderPipeline::InstancePipelineBuffers missing_quantization =
            make_ready_instance_pipeline_buffers(true);
    missing_quantization.quantization_buffer = RID();
    RenderInstancingOrchestrator::InstanceReadinessResult quantization_missing =
            RenderInstancingOrchestrator::evaluate_instance_pipeline_readiness(
                    true, true, missing_quantization);
    CHECK(!quantization_missing.ready);
    CHECK(quantization_missing.failure_mode ==
            RenderInstancingOrchestrator::InstanceReadinessFailureMode::INSTANCE_PIPELINE_BUFFERS_INVALID);

    GaussianRenderPipeline::InstancePipelineBuffers ready_quantization =
            make_ready_instance_pipeline_buffers(true);
    RenderInstancingOrchestrator::InstanceReadinessResult quantization_ready =
            RenderInstancingOrchestrator::evaluate_instance_pipeline_readiness(
                    true, true, ready_quantization);
    CHECK(quantization_ready.ready);
    CHECK(quantization_ready.failure_mode ==
            RenderInstancingOrchestrator::InstanceReadinessFailureMode::NONE);
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Instanced render skips callbacks when readiness preconditions fail") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    Ref<OutputCompositor> output_compositor;
    output_compositor.instantiate();
    CHECK(output_compositor.is_valid());
    if (!output_compositor.is_valid()) {
        renderer.unref();
        return;
    }

    RenderPipelineStages pipeline_stages(renderer.ptr());
    bool prepare_called = false;
    bool render_called = false;

    RenderInstancingOrchestrator::Dependencies instancing_dependencies;
    instancing_dependencies.renderer = renderer.ptr();
    instancing_dependencies.output_compositor = output_compositor.ptr();
    instancing_dependencies.pipeline_stages = &pipeline_stages;
    instancing_dependencies.prepare_render_frame_context =
            [&prepare_called](RenderDataRD *, const Transform3D &, const Projection &, const Projection &, bool,
                    GaussianSplatRenderer::RenderFrameContext &) {
                prepare_called = true;
            };
    instancing_dependencies.render_sorted_splats =
            [&render_called](RenderDataRD *, const Transform3D &, const Projection &, const Projection &, bool) {
                render_called = true;
            };
    RenderInstancingOrchestrator orchestrator(instancing_dependencies);

    GaussianRenderPipeline::InstancePipelineBuffers ready_buffers = make_ready_instance_pipeline_buffers(false);
    RenderInstancingOrchestrator::InstanceReadinessResult missing_streaming =
            RenderInstancingOrchestrator::evaluate_instance_pipeline_readiness(
                    false, true, ready_buffers);
    CHECK(!missing_streaming.ready);
    CHECK(missing_streaming.failure_mode ==
            RenderInstancingOrchestrator::InstanceReadinessFailureMode::STREAMING_SYSTEM_UNAVAILABLE);

    LocalVector<Transform3D> instance_transforms;
    instance_transforms.push_back(Transform3D());
    orchestrator.render_instanced(nullptr, GaussianSplatManager::SharedDynamicAssetHandle(),
            Transform3D(), Projection(), Projection(), instance_transforms);

    CHECK_MESSAGE(!prepare_called, "Expected readiness gate to skip frame preparation callback");
    CHECK_MESSAGE(!render_called, "Expected readiness gate to skip render callback");

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] RenderSceneInstance drives GPU streaming + sorting") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    const uint32_t chunk_size = GaussianStreamingSystem::CHUNK_SIZE;
    const uint32_t total_gaussians = chunk_size * 3;

    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, total_gaussians);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(total_gaussians);
    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        return;
    }

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
    scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = Ref<RenderSceneBuffersRD>();

    // Warm up streaming and ensure at least one frame has real data.
    for (int i = 0; i < 3; i++) {
        renderer->render_scene_instance(&render_data);
    }

    CHECK(renderer->has_rendered_content());
    CHECK(renderer->get_final_texture().is_valid());
    CHECK(renderer->get_visible_splat_count() > 0);

    Dictionary stats = renderer->get_render_stats();
    CHECK(stats.get("using_real_data", false));
    const String data_source = stats.get("data_source", String());
    bool valid_data_source = (data_source == String("StreamingGPU")) || (data_source == String("GPUBufferManager"));
    CHECK_MESSAGE(valid_data_source, vformat("Unexpected render stats data_source: %s", data_source));
    CHECK(stats.get("gpu_sorter_ready", false));

    Dictionary sort_metrics = renderer->get_last_sort_metrics();
    CHECK(sort_metrics.has("elements"));
    if (sort_metrics.has("elements")) {
        CHECK(int(sort_metrics["elements"]) > 0);
    }
    if (sort_metrics.has("used_gpu")) {
        CHECK(bool(sort_metrics["used_gpu"]));
    }

    renderer->commit_to_render_buffers(&render_data);

	renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] World static chunks keep streaming instance buffers ready without SceneDirector instances") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (project_settings == nullptr) {
        MESSAGE("Skipping test - ProjectSettings unavailable");
        return;
    }

    const String streaming_setting = "rendering/gaussian_splatting/streaming/enabled";
    const String instance_pipeline_setting = "rendering/gaussian_splatting/instance_pipeline/enabled";
    ScopedProjectSetting streaming_guard(project_settings, streaming_setting);
    ScopedProjectSetting instance_pipeline_guard(project_settings, instance_pipeline_setting);
    project_settings->set_setting(streaming_setting, true);
    project_settings->set_setting(instance_pipeline_setting, true);

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    const uint32_t total_gaussians = GaussianStreamingSystem::CHUNK_SIZE * 2;
    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, total_gaussians);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(total_gaussians);
    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        return;
    }

    renderer->set_static_chunks(make_single_static_chunk(total_gaussians, data->get_aabb()));

    // Simulate a world/static setup where gaussian data is present but the
    // streaming system was not bootstrapped yet (e.g. data set before RD ready).
    renderer->test_release_current_streaming_system();
    CHECK_MESSAGE(!renderer->test_has_current_streaming_system(),
            "Expected precondition: streaming system starts invalid before render bootstrap");

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
    scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = Ref<RenderSceneBuffersRD>();

    bool streaming_system_ready = false;
    bool instance_buffers_ready = false;
    for (int i = 0; i < 8; i++) {
        renderer->render_scene_instance(&render_data);
        const Dictionary frame_stats = renderer->get_render_stats();
        const String frame_cull_route_uid = frame_stats.get("cull_route_uid", String());
        CHECK_MESSAGE(frame_cull_route_uid != String(RenderRouteUID::GLOBAL_CULL_CPU),
                "Expected streaming warmup not to fall through to the resident/global CPU path");
        CHECK_MESSAGE(frame_cull_route_uid != String(RenderRouteUID::COMMON_SKIP_NO_DATA),
                "Expected streaming warmup with gaussian data to stay on instance/not-ready telemetry, not no-data");
        if (frame_cull_route_uid.begins_with("COMMON.SKIP.STREAMING_NOT_READY.")) {
            CHECK_MESSAGE(String(frame_stats.get("cull_route_reason", String())).begins_with("streaming_not_ready_"),
                    "Expected typed streaming-not-ready frames to publish a typed cull_route_reason");
            CHECK_MESSAGE(frame_stats.get("stage_cull_status", String()) == String("skipped"),
                    "Expected streaming-not-ready warmup frames to publish skipped cull stage metrics");
        }
        streaming_system_ready = streaming_system_ready || renderer->test_has_current_streaming_system();
        instance_buffers_ready = instance_buffers_ready || renderer->has_instance_pipeline_buffers();
        if (streaming_system_ready && instance_buffers_ready) {
            break;
        }
    }

    if (!streaming_system_ready) {
        MESSAGE("Skipping test - streaming system unavailable");
        renderer.unref();
        return;
    }

    Dictionary stats = renderer->get_render_stats();
    CHECK_MESSAGE(stats.get("cull_route_uid", String()) == String(RenderRouteUID::INSTANCE_CULL_GPU),
            "Expected cull route UID to report the instance pipeline");
    CHECK_MESSAGE(stats.get("cull_route_reason", String()) == String("instance_pipeline_active"),
            "Expected cull route reason to report the instance pipeline");

    CHECK_MESSAGE(instance_buffers_ready,
            "Expected instance pipeline buffers to become ready for world/static-chunk streaming without SceneDirector instances");
    const GaussianRenderPipeline::InstancePipelineBuffers &buffers = renderer->get_instance_pipeline_buffers();
    CHECK_MESSAGE(buffers.instance_count > 0, "Expected instance pipeline to synthesize at least one instance for world/static-chunk streaming");
    CHECK_MESSAGE(buffers.max_visible_chunks > 0, "Expected world/static-chunk streaming buffers to expose visible chunk capacity");

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Dynamic handle warmup publishes explicit streaming-not-ready skips") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();
    renderer->test_release_current_streaming_system();

    GaussianSplatManager::SharedDynamicAssetHandle handle;
    handle.asset_rid = RID::from_uint64(0x177u);
    handle.gaussian_buffer = RID::from_uint64(0x178u);
    handle.gaussian_count = GaussianStreamingSystem::CHUNK_SIZE;
    renderer->get_streaming_state().shared_dynamic_asset_handle = handle;

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
    scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = Ref<RenderSceneBuffersRD>();

    renderer->render_scene_instance(&render_data);

    const Dictionary stats = renderer->get_render_stats();
    const String cull_route_uid = stats.get("cull_route_uid", String());
    CHECK_MESSAGE(cull_route_uid.begins_with("COMMON.SKIP.STREAMING_NOT_READY."),
            vformat("Expected dynamic-handle warmup to publish a typed streaming-not-ready route, got '%s'", cull_route_uid));
    CHECK_MESSAGE(cull_route_uid != String(RenderRouteUID::COMMON_SKIP_NO_DATA),
            "Expected dynamic-handle warmup not to degrade to a no-data skip");
    CHECK_MESSAGE(cull_route_uid != String(RenderRouteUID::GLOBAL_CULL_CPU),
            "Expected dynamic-handle warmup not to fall through to the resident/global CPU path");
    CHECK_MESSAGE(String(stats.get("cull_route_reason", String())).begins_with("streaming_not_ready_"),
            "Expected dynamic-handle warmup to publish a typed streaming-not-ready reason");
    CHECK_MESSAGE(stats.get("stage_cull_status", String()) == String("skipped"),
            "Expected dynamic-handle warmup to publish skipped cull stage metrics");
    CHECK_MESSAGE(String(stats.get("stage_cull_reason", String())).find("streaming path not ready") != -1,
            "Expected dynamic-handle warmup to report a streaming-not-ready cull skip reason");

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Missing renderer data publishes an explicit streaming-not-ready skip") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();
    renderer->test_release_current_streaming_system();

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
    scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = Ref<RenderSceneBuffersRD>();

    renderer->render_scene_instance(&render_data);

    const Dictionary stats = renderer->get_render_stats();
    const String cull_route_uid = stats.get("cull_route_uid", String());
    CHECK_MESSAGE(cull_route_uid.begins_with(String(RenderRouteUID::COMMON_SKIP_STREAMING_NOT_READY)),
            vformat("Expected missing renderer data to publish a typed streaming-not-ready route, got '%s'", cull_route_uid));
    CHECK_MESSAGE(String(stats.get("cull_route_reason", String())).begins_with(String("streaming_not_ready_")),
            "Expected missing renderer data to publish a typed streaming-not-ready reason");
    CHECK_MESSAGE(stats.get("stage_cull_status", String()) == String("skipped"),
            "Expected missing renderer data to publish skipped cull stage metrics");
    CHECK_MESSAGE(String(stats.get("stage_cull_reason", String())).find("streaming path not ready") != -1,
            "Expected missing renderer data to report a streaming-not-ready cull skip reason");

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Static layout fallback publishes typed validator-aligned diagnostics") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (project_settings == nullptr) {
        MESSAGE("Skipping test - ProjectSettings unavailable");
        return;
    }

    const String streaming_setting = "rendering/gaussian_splatting/streaming/enabled";
    const String instance_pipeline_setting = "rendering/gaussian_splatting/instance_pipeline/enabled";
    ScopedProjectSetting streaming_guard(project_settings, streaming_setting);
    ScopedProjectSetting instance_pipeline_guard(project_settings, instance_pipeline_setting);
    project_settings->set_setting(streaming_setting, true);
    project_settings->set_setting(instance_pipeline_setting, true);

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    const uint32_t chunk_size = GaussianStreamingSystem::CHUNK_SIZE;
    const uint32_t total_gaussians = chunk_size * 2;
    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, total_gaussians);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(total_gaussians);
    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        return;
    }

    renderer->set_static_chunks(make_overlapping_static_chunks(chunk_size, data->get_aabb()));

    renderer->test_release_current_streaming_system();

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
    scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = Ref<RenderSceneBuffersRD>();

    bool streaming_system_ready = false;
    for (int i = 0; i < 8; i++) {
        renderer->render_scene_instance(&render_data);
        streaming_system_ready = streaming_system_ready || renderer->test_has_current_streaming_system();
        if (streaming_system_ready) {
            break;
        }
    }

    if (!streaming_system_ready) {
        MESSAGE("Skipping test - streaming system unavailable");
        renderer.unref();
        return;
    }

    Dictionary stats = renderer->get_render_stats();
    CHECK_MESSAGE(stats.has("streaming_state"), "Expected render stats to expose streaming_state diagnostics");
    if (!stats.has("streaming_state")) {
        renderer.unref();
        return;
    }

    const Dictionary streaming_state = stats["streaming_state"];
    const Dictionary layout_hint_validation = streaming_state.get("layout_hint_validation", Dictionary());
    const int64_t orchestrator_fallback_total =
            layout_hint_validation.get("orchestrator_fallback_total", int64_t(0));
    CHECK_MESSAGE(orchestrator_fallback_total > 0,
            "Expected static-layout fallback to be captured in orchestrator layout-hint diagnostics");
    CHECK(layout_hint_validation.get("orchestrator_last_usage", String("none")) == String("io"));
    CHECK(layout_hint_validation.get("orchestrator_last_reason", String("none")) ==
            String("hint_overlapping_ranges"));
    CHECK(layout_hint_validation.get("orchestrator_last_reason_category", String("other")) ==
            String("non_contiguous"));

    const Dictionary orchestrator_reason_counts =
            layout_hint_validation.get("orchestrator_reason_counts", Dictionary());
    const int64_t overlap_reason_count =
            orchestrator_reason_counts.get("hint_overlapping_ranges", int64_t(0));
    CHECK(overlap_reason_count > 0);
    const Dictionary orchestrator_category_counts =
            layout_hint_validation.get("orchestrator_category_counts", Dictionary());
    const int64_t non_contiguous_category_count =
            orchestrator_category_counts.get("non_contiguous", int64_t(0));
    CHECK(non_contiguous_category_count > 0);

    CHECK(streaming_state.get("layout_hint_orchestrator_last_reason", String("none")) ==
            String("hint_overlapping_ranges"));
    CHECK(streaming_state.get("layout_hint_orchestrator_last_reason_category", String("other")) ==
            String("non_contiguous"));
    CHECK(streaming_state.get("layout_hint_last_reason", String("none")) == String("hint_overlapping_ranges"));
    CHECK(streaming_state.get("layout_hint_last_reason_category", String("other")) == String("non_contiguous"));

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Global CPU cull telemetry ignores missing registered legacy buffer state") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    Vector<Vector3> positions;
    positions.push_back(Vector3(0.0f, 0.0f, -2.0f));
    positions.push_back(Vector3(0.0f, 0.0f, -4.0f));
    renderer->test_set_test_splats(positions);

    Projection projection;
    projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

	const int visible = renderer->test_cull_visible_count(Transform3D(), projection, Size2i(1280, 720));
	CHECK(visible > 0);

	Dictionary stats = renderer->get_render_stats();
	CHECK_MESSAGE(stats.get("cull_route_uid", String()) == String(RenderRouteUID::GLOBAL_CULL_CPU),
			"Expected executed global CPU culling to ignore missing legacy buffer state");
	CHECK_MESSAGE(stats.get("cull_route_reason", String()) == String("global_cpu_path"),
			"Expected executed global CPU culling to publish the global CPU reason");
	CHECK_MESSAGE(!bool(stats.get("cull_route_uid_missing", true)),
			"Expected cull route UID to be normalized and marked present");

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Streaming indices validate against buffer capacity") {
	RenderingServer *rs = RenderingServer::get_singleton();
	if (rs == nullptr) {
		MESSAGE("Skipping test - Rendering server unavailable");
		return;
	}

	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	if (project_settings == nullptr) {
		MESSAGE("Skipping test - ProjectSettings unavailable");
		return;
	}

	const String streaming_setting = "rendering/gaussian_splatting/streaming/enabled";
	ScopedProjectSetting streaming_guard(project_settings, streaming_setting);
	project_settings->set_setting(streaming_setting, true);

	ScopedGaussianManagerPipeline manager_scope;
	GaussianSplatManager *manager = manager_scope.get();
	if (manager == nullptr) {
		MESSAGE("Skipping test - GaussianSplatManager unavailable");
		return;
	}

	RenderingDevice *primary_rd = manager->get_primary_rendering_device();
	if (!primary_rd) {
		primary_rd = rs->create_local_rendering_device();
	}
	if (primary_rd == nullptr) {
		MESSAGE("Skipping test - Rendering device unavailable");
		return;
	}

	Ref<GaussianSplatRenderer> renderer;
	renderer.instantiate(primary_rd);
	CHECK(renderer.is_valid());
	if (!renderer.is_valid()) {
		return;
	}
	renderer->initialize();
	renderer->set_debug_binning_counters_enabled(true);

	const uint32_t total_gaussians = 2000;
	LocalVector<Gaussian> gaussians;
	fill_gaussians(gaussians, total_gaussians);

	Ref<::GaussianData> data;
	data.instantiate();
	data->set_gaussians(gaussians);

	renderer->set_max_splats(total_gaussians);
	Error set_data_err = renderer->set_gaussian_data(data);
	CHECK(set_data_err == OK);
	if (set_data_err != OK) {
		return;
	}

	RenderSceneDataRD scene_data;
	scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
	scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

	RenderDataRD render_data;
	render_data.scene_data = &scene_data;
	render_data.render_buffers = Ref<RenderSceneBuffersRD>();

	Dictionary counters;
	bool streaming_active = false;
	bool counters_ready = false;
	uint32_t streaming_capacity = 0;

	for (int i = 0; i < 6; i++) {
		renderer->render_scene_instance(&render_data);
		const GaussianSplatRenderer::StreamingState &streaming_state = renderer->get_streaming_state();
		streaming_active = streaming_state.use_streamed_data &&
				streaming_state.streamed_indices_are_local &&
				streaming_state.current_stream_gpu_buffer.is_valid();
		streaming_capacity = streaming_state.streaming_gpu_total_capacity;

		counters = renderer->get_binning_debug_counters();
		const int64_t iterated = counters.get("raster_splats_iterated", int64_t(0));
		if (streaming_active && iterated > 0) {
			counters_ready = true;
			break;
		}
	}

	if (!streaming_active) {
		MESSAGE("Skipping test - streaming buffer indices not active");
		renderer.unref();
		return;
	}
	if (streaming_capacity <= total_gaussians) {
		MESSAGE("Skipping test - streaming buffer capacity not larger than source data");
		renderer.unref();
		return;
	}
	if (!counters_ready) {
		MESSAGE("Skipping test - binning debug counters unavailable");
		renderer.unref();
		return;
	}

	const int64_t oob_rejects = counters.get("raster_reject_gaussian_idx_oob", int64_t(0));
	CHECK_MESSAGE(oob_rejects == 0, vformat("Expected no gaussian index OOB rejects, got %d", oob_rejects));

	renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] RenderSceneInstance supports forced CPU sorting") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (project_settings == nullptr) {
        MESSAGE("Skipping test - ProjectSettings unavailable");
        return;
    }

    const String force_cpu_setting = "rendering/gaussian_splatting/sorting/force_cpu_sort";
    ScopedProjectSetting force_cpu_guard(project_settings, force_cpu_setting);
    project_settings->set_setting(force_cpu_setting, true);

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    const uint32_t total_gaussians = GaussianStreamingSystem::CHUNK_SIZE * 2;
    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, total_gaussians);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(total_gaussians);
    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        return;
    }

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
    scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = Ref<RenderSceneBuffersRD>();

    for (int i = 0; i < 2; i++) {
        renderer->render_scene_instance(&render_data);
    }

    Dictionary sort_metrics = renderer->get_last_sort_metrics();
    CHECK_MESSAGE(sort_metrics.has("cpu_fallback"), "Expected sort metrics to include cpu_fallback");
    if (sort_metrics.has("cpu_fallback")) {
        CHECK_MESSAGE(bool(sort_metrics["cpu_fallback"]), "Expected CPU sorting fallback to be active");
    }
    if (sort_metrics.has("used_gpu")) {
        CHECK_MESSAGE(!bool(sort_metrics["used_gpu"]), "Expected GPU sorting to be disabled");
    }
	if (sort_metrics.has("algorithm")) {
		String algorithm = sort_metrics["algorithm"];
		CHECK_MESSAGE(algorithm.find("CPU") != -1, vformat("Expected CPU algorithm, got '%s'", algorithm));
	}

	// Regression check: forcing CPU sort with an empty frame must stay on the CPU fallback path
	// instead of dropping to FAIL just because there are zero visible splats.
	Ref<::GaussianData> empty_data;
	empty_data.instantiate();
	LocalVector<Gaussian> empty_gaussians;
	empty_data->set_gaussians(empty_gaussians);

	Error set_empty_err = renderer->set_gaussian_data(empty_data);
	CHECK(set_empty_err == OK);
	if (set_empty_err == OK) {
		for (int i = 0; i < 2; i++) {
			renderer->render_scene_instance(&render_data);
		}

		Dictionary empty_sort_metrics = renderer->get_last_sort_metrics();
		CHECK_MESSAGE(empty_sort_metrics.has("cpu_fallback"), "Expected sort metrics to include cpu_fallback for empty frame");
		if (empty_sort_metrics.has("cpu_fallback")) {
			CHECK_MESSAGE(bool(empty_sort_metrics["cpu_fallback"]), "Expected CPU fallback to remain active for empty frame");
		}
		if (empty_sort_metrics.has("used_gpu")) {
			CHECK_MESSAGE(!bool(empty_sort_metrics["used_gpu"]), "Expected GPU sorting to stay disabled for empty frame");
		}
		if (empty_sort_metrics.has("elements")) {
			CHECK_MESSAGE(int64_t(empty_sort_metrics["elements"]) == 0, "Expected empty frame to report zero sorted elements");
		}
	}

	renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Strict global sort forces a real CPU sort when the camera-stable fallback buffer is missing") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (project_settings == nullptr) {
        MESSAGE("Skipping test - ProjectSettings unavailable");
        return;
    }

    const String force_cpu_setting = "rendering/gaussian_splatting/sorting/force_cpu_sort";
    const String strict_sort_setting = "rendering/gaussian_splatting/sorting/strict_global_sort";
    ScopedGpuSortingConfigReload strict_reload_guard;
    ScopedProjectSetting force_cpu_guard(project_settings, force_cpu_setting);
    ScopedProjectSetting strict_guard(project_settings, strict_sort_setting);
    project_settings->set_setting(force_cpu_setting, true);
    project_settings->set_setting(strict_sort_setting, true);
    project_settings->emit_signal("settings_changed");
    g_gpu_sorting_config.load_from_project_settings();

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    Vector<Vector3> positions;
    positions.push_back(Vector3(0.0f, 0.0f, -2.0f));
    positions.push_back(Vector3(0.0f, 0.0f, -8.0f));
    renderer->test_set_test_splats(positions);

    GPUCuller::CullingState &cull_state = renderer->get_subsystem_state().gpu_culler->get_state();
    cull_state.culled_indices.resize(2);
    cull_state.culled_indices[0] = 1;
    cull_state.culled_indices[1] = 0;

    GaussianSplatRenderer::SortingState &sorting_state = renderer->get_sorting_state();
    const Transform3D world_to_camera = Transform3D();
    sorting_state.sorted_splat_count = 2;
    sorting_state.last_sort_world_to_camera_transform = world_to_camera;
    sorting_state.last_sort_transform_valid = true;
    renderer->get_subsystem_state().gpu_culler->get_config().cull_params_dirty = false;
    renderer->get_subsystem_state().sorting_pipeline->release_sort_buffers();

    GaussianSplatRenderer::SortStageSummary summary =
            renderer->test_sort_for_view(world_to_camera, GaussianRenderState::IndexDomain::GAUSSIAN_GLOBAL);

    CHECK(summary.did_execute);
    CHECK(summary.sorted_count == 2);
    CHECK(renderer->get_debug_state().sort_route_uid == String(RenderRouteUID::INSTANCE_SORT_CPU_FALLBACK));
    CHECK(renderer->get_debug_state().sort_route_uid != String(RenderRouteUID::COMMON_SKIP_CAMERA_STABLE));
    CHECK(cull_state.culled_indices[0] == 0);
    CHECK(cull_state.culled_indices[1] == 1);

    const Vector<uint32_t> sorted_indices = read_renderer_sort_indices(renderer, 2);
    REQUIRE(sorted_indices.size() == 2);
    CHECK(sorted_indices[0] == 0);
    CHECK(sorted_indices[1] == 1);

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Camera-stable global fallback publishes current cull order when strict sort is disabled") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (project_settings == nullptr) {
        MESSAGE("Skipping test - ProjectSettings unavailable");
        return;
    }

    const String force_cpu_setting = "rendering/gaussian_splatting/sorting/force_cpu_sort";
    const String strict_sort_setting = "rendering/gaussian_splatting/sorting/strict_global_sort";
    ScopedGpuSortingConfigReload strict_reload_guard;
    ScopedProjectSetting force_cpu_guard(project_settings, force_cpu_setting);
    ScopedProjectSetting strict_guard(project_settings, strict_sort_setting);
    project_settings->set_setting(force_cpu_setting, false);
    project_settings->set_setting(strict_sort_setting, false);
    project_settings->emit_signal("settings_changed");
    g_gpu_sorting_config.load_from_project_settings();

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    GPUCuller::CullingState &cull_state = renderer->get_subsystem_state().gpu_culler->get_state();
    cull_state.culled_indices.resize(2);
    cull_state.culled_indices[0] = 1;
    cull_state.culled_indices[1] = 0;

    GaussianSplatRenderer::SortingState &sorting_state = renderer->get_sorting_state();
    const Transform3D world_to_camera = Transform3D();
    sorting_state.sorted_splat_count = 2;
    sorting_state.last_sort_world_to_camera_transform = world_to_camera;
    sorting_state.last_sort_transform_valid = true;
    renderer->get_subsystem_state().gpu_culler->get_config().cull_params_dirty = false;
    renderer->get_subsystem_state().sorting_pipeline->release_sort_buffers();

    GaussianSplatRenderer::SortStageSummary summary =
            renderer->test_sort_for_view(world_to_camera, GaussianRenderState::IndexDomain::GAUSSIAN_GLOBAL);

    CHECK_FALSE(summary.did_execute);
    CHECK(summary.sorted_count == 2);
    CHECK(renderer->get_debug_state().sort_route_uid == String(RenderRouteUID::COMMON_SKIP_CAMERA_STABLE));
    CHECK_FALSE(renderer->get_sorting_state().last_sort_transform_valid);
    CHECK(cull_state.culled_indices[0] == 1);
    CHECK(cull_state.culled_indices[1] == 0);

    const Vector<uint32_t> sorted_indices = read_renderer_sort_indices(renderer, 2);
    REQUIRE(sorted_indices.size() == 2);
    CHECK(sorted_indices[0] == 1);
    CHECK(sorted_indices[1] == 0);

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Strict global sort refuses unsorted CPU fallback when positions are unavailable") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (project_settings == nullptr) {
        MESSAGE("Skipping test - ProjectSettings unavailable");
        return;
    }

    const String force_cpu_setting = "rendering/gaussian_splatting/sorting/force_cpu_sort";
    const String strict_sort_setting = "rendering/gaussian_splatting/sorting/strict_global_sort";
    ScopedGpuSortingConfigReload strict_reload_guard;
    ScopedProjectSetting force_cpu_guard(project_settings, force_cpu_setting);
    ScopedProjectSetting strict_guard(project_settings, strict_sort_setting);
    project_settings->set_setting(force_cpu_setting, true);
    project_settings->set_setting(strict_sort_setting, true);
    project_settings->emit_signal("settings_changed");
    g_gpu_sorting_config.load_from_project_settings();

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    GPUCuller::CullingState &cull_state = renderer->get_subsystem_state().gpu_culler->get_state();
    cull_state.culled_indices.resize(2);
    cull_state.culled_indices[0] = 1;
    cull_state.culled_indices[1] = 0;
    renderer->get_subsystem_state().gpu_culler->get_config().cull_params_dirty = false;

    GaussianSplatRenderer::SortStageSummary summary =
            renderer->test_sort_for_view(Transform3D(), GaussianRenderState::IndexDomain::GAUSSIAN_GLOBAL);

    CHECK_FALSE(summary.did_execute);
    CHECK(summary.sorted_count == 0);
    CHECK(renderer->get_debug_state().sort_route_uid == String(RenderRouteUID::COMMON_FAIL_SORT_FAILED));
    CHECK(renderer->get_visible_splat_count() == 0);
    CHECK_FALSE(renderer->get_sorting_state().last_sort_transform_valid);
    CHECK(renderer->get_sorting_state().sort_index_bytes.is_empty());
    CHECK(cull_state.culled_indices[0] == 1);
    CHECK(cull_state.culled_indices[1] == 0);

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] CPU fallback publishes unsorted cull order when positions are unavailable and strict sort is disabled") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (project_settings == nullptr) {
        MESSAGE("Skipping test - ProjectSettings unavailable");
        return;
    }

    const String force_cpu_setting = "rendering/gaussian_splatting/sorting/force_cpu_sort";
    const String strict_sort_setting = "rendering/gaussian_splatting/sorting/strict_global_sort";
    ScopedGpuSortingConfigReload strict_reload_guard;
    ScopedProjectSetting force_cpu_guard(project_settings, force_cpu_setting);
    ScopedProjectSetting strict_guard(project_settings, strict_sort_setting);
    project_settings->set_setting(force_cpu_setting, true);
    project_settings->set_setting(strict_sort_setting, false);
    project_settings->emit_signal("settings_changed");
    g_gpu_sorting_config.load_from_project_settings();

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    GPUCuller::CullingState &cull_state = renderer->get_subsystem_state().gpu_culler->get_state();
    cull_state.culled_indices.resize(2);
    cull_state.culled_indices[0] = 1;
    cull_state.culled_indices[1] = 0;
    renderer->get_subsystem_state().gpu_culler->get_config().cull_params_dirty = false;

    GaussianSplatRenderer::SortStageSummary summary =
            renderer->test_sort_for_view(Transform3D(), GaussianRenderState::IndexDomain::GAUSSIAN_GLOBAL);

    CHECK(summary.did_execute);
    CHECK(summary.sorted_count == 2);
    CHECK(renderer->get_debug_state().sort_route_uid == String(RenderRouteUID::INSTANCE_SORT_CPU_FALLBACK));
    CHECK(renderer->get_visible_splat_count() == 2);
    CHECK(cull_state.culled_indices[0] == 1);
    CHECK(cull_state.culled_indices[1] == 0);

    const Vector<uint32_t> sorted_indices = read_renderer_sort_indices(renderer, 2);
    REQUIRE(sorted_indices.size() == 2);
    CHECK(sorted_indices[0] == 1);
    CHECK(sorted_indices[1] == 0);

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Camera-stable instance path publishes identity fallback when strict sort is disabled") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (project_settings == nullptr) {
        MESSAGE("Skipping test - ProjectSettings unavailable");
        return;
    }

    const String force_cpu_setting = "rendering/gaussian_splatting/sorting/force_cpu_sort";
    const String strict_sort_setting = "rendering/gaussian_splatting/sorting/strict_global_sort";
    ScopedGpuSortingConfigReload strict_reload_guard;
    ScopedProjectSetting force_cpu_guard(project_settings, force_cpu_setting);
    ScopedProjectSetting strict_guard(project_settings, strict_sort_setting);
    project_settings->set_setting(force_cpu_setting, false);
    project_settings->set_setting(strict_sort_setting, false);
    project_settings->emit_signal("settings_changed");
    g_gpu_sorting_config.load_from_project_settings();

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    ScopedRenderingDeviceLease device_lease;
    RenderingDevice *primary_rd = device_lease.acquire(rs, manager);
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    renderer->set_instance_pipeline_buffers(make_ready_instance_pipeline_buffers(false));
    renderer->get_subsystem_state().sorting_pipeline->test_set_last_instance_visible_splat_count(
            1, renderer->get_frame_state().frame_counter);
    renderer->get_subsystem_state().sorting_pipeline->release_sort_buffers();

    GPUCuller::CullingState &cull_state = renderer->get_subsystem_state().gpu_culler->get_state();
    cull_state.culled_indices.resize(1);
    cull_state.culled_indices[0] = 0;

    GaussianSplatRenderer::SortingState &sorting_state = renderer->get_sorting_state();
    sorting_state.sorted_splat_count = 0;
    sorting_state.last_sort_world_to_camera_transform = Transform3D();
    sorting_state.last_sort_transform_valid = true;
    renderer->get_subsystem_state().gpu_culler->get_config().cull_params_dirty = false;

    GaussianSplatRenderer::SortStageSummary summary =
            renderer->test_sort_for_view(Transform3D(), GaussianRenderState::IndexDomain::CHUNK_REF);

    CHECK_FALSE(summary.did_execute);
    CHECK(summary.sorted_count == 1);
    CHECK(renderer->get_debug_state().sort_route_uid == String(RenderRouteUID::INSTANCE_SORT_IDENTITY_FALLBACK));
    CHECK(renderer->get_visible_splat_count() == 1);
    const Vector<uint32_t> sorted_indices = read_renderer_sort_indices(renderer, 1);
    REQUIRE(sorted_indices.size() == 1);
    CHECK(sorted_indices[0] == 0);

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Production metrics contract and perf gate reporting") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (project_settings == nullptr) {
        MESSAGE("Skipping test - ProjectSettings unavailable");
        return;
    }

    const String validate_setting = "rendering/gaussian_splatting/diagnostics/validate_production_metrics";
    const String summary_interval_setting = "rendering/gaussian_splatting/diagnostics/summary_interval_frames";
    const String summary_history_setting = "rendering/gaussian_splatting/diagnostics/summary_history_size";
    const String gate_enabled_setting = "rendering/gaussian_splatting/diagnostics/perf_gate_enabled";
    const String gate_splats_setting = "rendering/gaussian_splatting/diagnostics/perf_gate_splat_threshold";
    const String gate_budget_setting = "rendering/gaussian_splatting/diagnostics/perf_gate_budget_ms";
    const String tier_preset_setting = "rendering/gaussian_splatting/quality/tier_preset";
    const String tier_apply_streaming_setting = "rendering/gaussian_splatting/quality/tier_apply_streaming_budgets";
    const String upload_frame_cap_setting = "rendering/gaussian_splatting/streaming/max_upload_mb_per_frame";
    const String vram_budget_setting = "rendering/gaussian_splatting/streaming/vram_budget_mb";

    ScopedProjectSetting validate_guard(project_settings, validate_setting);
    ScopedProjectSetting summary_interval_guard(project_settings, summary_interval_setting);
    ScopedProjectSetting summary_history_guard(project_settings, summary_history_setting);
    ScopedProjectSetting gate_enabled_guard(project_settings, gate_enabled_setting);
    ScopedProjectSetting gate_splats_guard(project_settings, gate_splats_setting);
    ScopedProjectSetting gate_budget_guard(project_settings, gate_budget_setting);
    ScopedProjectSetting tier_preset_guard(project_settings, tier_preset_setting);
    ScopedProjectSetting tier_apply_streaming_guard(project_settings, tier_apply_streaming_setting);
    ScopedProjectSetting upload_frame_cap_guard(project_settings, upload_frame_cap_setting);
    ScopedProjectSetting vram_budget_guard(project_settings, vram_budget_setting);

    project_settings->set_setting(validate_setting, true);
    project_settings->set_setting(summary_interval_setting, 1);
    project_settings->set_setting(summary_history_setting, 4);
    project_settings->set_setting(gate_enabled_setting, true);
    project_settings->set_setting(gate_splats_setting, 1);
    project_settings->set_setting(gate_budget_setting, 1000.0f);
    project_settings->set_setting(tier_apply_streaming_setting, true);
    project_settings->set_setting(tier_preset_setting, "low");
    project_settings->set_setting(upload_frame_cap_setting, 128);
    project_settings->set_setting(vram_budget_setting, 12288);

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    RenderingDevice *primary_rd = manager->get_primary_rendering_device();
    if (!primary_rd) {
        primary_rd = rs->create_local_rendering_device();
    }
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    const uint32_t total_gaussians = GaussianStreamingSystem::CHUNK_SIZE * 2;
    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, total_gaussians);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(total_gaussians);
    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        return;
    }

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
    scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = Ref<RenderSceneBuffersRD>();

    for (int i = 0; i < 2; i++) {
        renderer->render_scene_instance(&render_data);
    }

    Dictionary stats = renderer->get_render_stats();
    CHECK_MESSAGE(stats.has("production_metrics"), "Expected production_metrics in render stats");
    CHECK_MESSAGE(stats.has("production_metrics_validation"), "Expected production_metrics_validation in render stats");
    CHECK_MESSAGE(stats.has("perf_gate"), "Expected perf_gate in render stats");

    Dictionary production_metrics = stats.get("production_metrics", Dictionary());
    CHECK_MESSAGE(production_metrics.has("frame"), "Expected frame in production_metrics");
    CHECK_MESSAGE(production_metrics.has("cull_ms"), "Expected cull_ms in production_metrics");
    CHECK_MESSAGE(production_metrics.has("cull_route_uid"), "Expected cull_route_uid in production_metrics");
    CHECK_MESSAGE(production_metrics.has("cull_route_reason"), "Expected cull_route_reason in production_metrics");
    CHECK_MESSAGE(production_metrics.has("stage_total_ms"), "Expected stage_total_ms in production_metrics");
    CHECK_MESSAGE(stats.has("streaming_effective_upload_cap_mb_per_frame"), "Expected streaming effective upload cap in render stats");
    CHECK_MESSAGE(stats.has("streaming_effective_vram_budget_mb"), "Expected streaming effective VRAM budget in render stats");
    CHECK_MESSAGE(stats.has("streaming_cap_source_upload_mb_per_frame"), "Expected streaming upload cap source in render stats");
    CHECK_MESSAGE(stats.has("streaming_cap_source_vram_budget_mb"), "Expected streaming VRAM cap source in render stats");
    CHECK_MESSAGE(stats.has("streaming_upload_frame_cap_hit"), "Expected streaming upload frame cap indicator in render stats");
    CHECK_MESSAGE(stats.has("streaming_queue_pressure_active"), "Expected streaming queue pressure indicator in render stats");
    CHECK_MESSAGE(int64_t(stats.get("streaming_effective_upload_cap_mb_per_frame", int64_t(-1))) == 32,
            "Expected low-tier effective upload frame cap");
    CHECK_MESSAGE(int64_t(stats.get("streaming_effective_vram_budget_mb", int64_t(-1))) == 2048,
            "Expected low-tier effective VRAM budget");
    CHECK_MESSAGE(String(stats.get("streaming_cap_source_upload_mb_per_frame", String())) == String("tier_preset"),
            "Expected upload cap source to report tier preset");
    CHECK_MESSAGE(String(stats.get("streaming_cap_source_vram_budget_mb", String())) == String("tier_preset"),
            "Expected VRAM cap source to report tier preset");

    Dictionary validation = stats.get("production_metrics_validation", Dictionary());
    CHECK_MESSAGE(bool(validation.get("valid", false)), "Expected production_metrics_validation to be valid");
    CHECK_MESSAGE(String(production_metrics.get("cull_route_uid", String())).length() > 0,
            "Expected production_metrics to publish a non-empty cull route UID");
    CHECK_MESSAGE(String(production_metrics.get("cull_route_reason", String())).length() > 0,
            "Expected production_metrics to publish a non-empty cull route reason");
    CHECK_MESSAGE(!bool(stats.get("cull_route_uid_missing", true)),
            "Expected render stats to mark cull_route_uid as present");

    Dictionary perf_gate = stats.get("perf_gate", Dictionary());
    CHECK_MESSAGE(bool(perf_gate.get("enabled", false)), "Expected perf gate enabled");
    if (perf_gate.get("applicable", false)) {
        CHECK_MESSAGE(bool(perf_gate.get("passed", false)), "Expected perf gate to pass with high budget");
    }

    Dictionary snapshot = renderer->get_runtime_diagnostic_snapshot();
    CHECK_MESSAGE(snapshot.has("production_metrics_contract"), "Expected production_metrics_contract in snapshot");
    Dictionary snapshot_metrics = snapshot.get("production_metrics", Dictionary());
    CHECK_MESSAGE(snapshot_metrics.has("cull_route_uid"), "Expected snapshot production_metrics to expose cull_route_uid");
    CHECK_MESSAGE(snapshot_metrics.has("cull_route_reason"), "Expected snapshot production_metrics to expose cull_route_reason");
    Dictionary telemetry = snapshot.get("telemetry", Dictionary());
    CHECK_MESSAGE(telemetry.has("cull_route_uid"), "Expected runtime telemetry snapshot to expose cull_route_uid");
    CHECK_MESSAGE(telemetry.has("cull_route_reason"), "Expected runtime telemetry snapshot to expose cull_route_reason");
    Array summaries = snapshot.get("production_metrics_summaries", Array());
    CHECK_MESSAGE(summaries.size() >= 1, "Expected at least one production metrics summary");

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Production metrics validation marks diagnostics-disabled mode") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (project_settings == nullptr) {
        MESSAGE("Skipping test - ProjectSettings unavailable");
        return;
    }

    const String validate_setting = "rendering/gaussian_splatting/diagnostics/validate_production_metrics";
    const String gate_enabled_setting = "rendering/gaussian_splatting/diagnostics/perf_gate_enabled";
    ScopedProjectSetting validate_guard(project_settings, validate_setting);
    ScopedProjectSetting gate_enabled_guard(project_settings, gate_enabled_setting);
    project_settings->set_setting(validate_setting, false);
    project_settings->set_setting(gate_enabled_setting, false);

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    RenderingDevice *primary_rd = manager->get_primary_rendering_device();
    if (!primary_rd) {
        primary_rd = rs->create_local_rendering_device();
    }
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    LocalVector<Gaussian> gaussians;
    const uint32_t total_gaussians = 512;
    fill_gaussians(gaussians, total_gaussians);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(total_gaussians);
    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        return;
    }

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
    scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = Ref<RenderSceneBuffersRD>();

    renderer->render_scene_instance(&render_data);

    Dictionary stats = renderer->get_render_stats();
    CHECK_MESSAGE(stats.has("production_metrics_validation"),
            "Expected production_metrics_validation in render stats");
    Dictionary validation = stats.get("production_metrics_validation", Dictionary());
    CHECK_MESSAGE(bool(validation.get("valid", false)),
            "Expected disabled diagnostics validation path to still report valid contract");
    CHECK_MESSAGE(bool(validation.get("disabled", false)),
            "Expected production metrics validation to mark diagnostics as disabled");

    Dictionary snapshot = renderer->get_runtime_diagnostic_snapshot();
    Dictionary snapshot_validation = snapshot.get("production_metrics_validation", Dictionary());
    CHECK_MESSAGE(bool(snapshot_validation.get("disabled", false)),
            "Expected runtime snapshot validation to mark diagnostics as disabled");
    CHECK_MESSAGE(int64_t(snapshot.get("production_metrics_invalid_count", int64_t(-1))) == 0,
            "Expected no production metrics invalid count increments when validation is disabled");

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Stage results report cull/sort skipped when GPU culler unavailable") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    RenderingDevice *primary_rd = manager->get_primary_rendering_device();
    if (!primary_rd) {
        primary_rd = rs->create_local_rendering_device();
    }
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    const uint32_t total_gaussians = GaussianStreamingSystem::CHUNK_SIZE * 2;
    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, total_gaussians);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(total_gaussians);
    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        return;
    }

    renderer->test_disable_gpu_culler();

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
    scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = Ref<RenderSceneBuffersRD>();

    renderer->render_scene_instance(&render_data);

    Dictionary stats = renderer->get_render_stats();
    CHECK_MESSAGE(stats.get("stage_metrics_valid", false), "Expected stage metrics to be valid");
    CHECK_MESSAGE(stats.get("cull_route_uid", String()) == String(RenderRouteUID::COMMON_SKIP_GPU_CULLER_UNAVAILABLE),
            "Expected cull route UID to report the GPU-culler-disabled bypass");
    CHECK_MESSAGE(stats.get("cull_route_reason", String()) == String("gpu_culler_unavailable"),
            "Expected cull route reason to report the missing GPU culler");
    const String cull_status = stats.get("stage_cull_status", String());
    CHECK_MESSAGE(cull_status == String("skipped"),
            vformat("Expected cull stage skipped, got '%s'", cull_status));
    const String cull_reason = stats.get("stage_cull_reason", String());
    CHECK_MESSAGE(cull_reason.find("GPU culler unavailable") != -1,
            vformat("Expected cull reason to mention GPU culler unavailable, got '%s'", cull_reason));
    const String sort_status = stats.get("stage_sort_status", String());
    CHECK_MESSAGE(sort_status == String("skipped"),
            vformat("Expected sort stage skipped, got '%s'", sort_status));
    const String sort_reason = stats.get("stage_sort_reason", String());
    CHECK_MESSAGE(sort_reason.find("GPU culler unavailable") != -1,
            vformat("Expected sort reason to mention GPU culler unavailable, got '%s'", sort_reason));

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Stage results report raster failure when rasterizer missing") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    RenderingDevice *primary_rd = manager->get_primary_rendering_device();
    if (!primary_rd) {
        primary_rd = rs->create_local_rendering_device();
    }
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    const uint32_t total_gaussians = GaussianStreamingSystem::CHUNK_SIZE * 2;
    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, total_gaussians);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(total_gaussians);
    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        return;
    }

    renderer->test_disable_rasterizer();

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
    scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = Ref<RenderSceneBuffersRD>();

    renderer->render_scene_instance(&render_data);

    Dictionary stats = renderer->get_render_stats();
    CHECK_MESSAGE(stats.get("stage_metrics_valid", false), "Expected stage metrics to be valid");
    const String raster_status = stats.get("stage_raster_status", String());
    CHECK_MESSAGE(raster_status == String("failed"),
            vformat("Expected raster stage failed, got '%s'", raster_status));
    const String raster_reason = stats.get("stage_raster_reason", String());
    CHECK_MESSAGE(raster_reason.find("Tile fallback failed") != -1,
            vformat("Expected raster reason to mention tile fallback failure, got '%s'", raster_reason));
    const String composite_status = stats.get("stage_composite_status", String());
    CHECK_MESSAGE(composite_status == String("skipped"),
            vformat("Expected composite stage skipped, got '%s'", composite_status));
    const String composite_reason = stats.get("stage_composite_reason", String());
    CHECK_MESSAGE(composite_reason.find("raster failed") != -1,
            vformat("Expected composite reason to mention raster failure, got '%s'", composite_reason));

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Stage results report streaming not-ready instead of painterly fallback") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    RenderingDevice *primary_rd = manager->get_primary_rendering_device();
    if (!primary_rd) {
        primary_rd = rs->create_local_rendering_device();
    }
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    const uint32_t total_gaussians = GaussianStreamingSystem::CHUNK_SIZE * 2;
    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, total_gaussians);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(total_gaussians);
    renderer->set_painterly_enabled(true);
    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        return;
    }

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
    scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = Ref<RenderSceneBuffersRD>();

    for (int i = 0; i < 2; i++) {
        renderer->render_scene_instance(&render_data);
    }

    Dictionary stats = renderer->get_render_stats();
    const String cull_route_uid = stats.get("cull_route_uid", String());
    CHECK_MESSAGE(cull_route_uid.begins_with(String(RenderRouteUID::COMMON_SKIP_STREAMING_NOT_READY)),
            vformat("Expected streaming not-ready route, got '%s'", cull_route_uid));
    const String cull_route_reason = stats.get("cull_route_reason", String());
    CHECK_MESSAGE(cull_route_reason.begins_with(String("streaming_not_ready_")),
            vformat("Expected streaming not-ready reason, got '%s'", cull_route_reason));
    const String cull_status = stats.get("stage_cull_status", String());
    CHECK_MESSAGE(cull_status == String("skipped"),
            vformat("Expected cull stage skipped, got '%s'", cull_status));
    const String sort_status = stats.get("stage_sort_status", String());
    CHECK_MESSAGE(sort_status == String("skipped"),
            vformat("Expected sort stage skipped, got '%s'", sort_status));
    const String raster_status = stats.get("stage_raster_status", String());
    CHECK_MESSAGE(raster_status == String("skipped"),
            vformat("Expected raster stage skipped, got '%s'", raster_status));
    CHECK_MESSAGE(!bool(stats.get("cull_route_uid_missing", true)),
            "Expected cull route UID to be present for streaming not-ready skips");

    renderer.unref();
}

static RID create_test_texture(RenderingDevice *p_rd, const Vector2i &p_size, RD::DataFormat p_format, RD::TextureUsageBits p_usage) {
    RD::TextureFormat format;
    format.format = p_format;
    format.width = p_size.x;
    format.height = p_size.y;
    format.depth = 1;
    format.array_layers = 1;
    format.mipmaps = 1;
    format.samples = RD::TEXTURE_SAMPLES_1;
    format.usage_bits = p_usage;
    return p_rd->texture_create(format, RD::TextureView());
}

static bool create_test_render_buffers(const Vector2i &p_resolution, RID &r_render_target, Ref<RenderSceneBuffersRD> &r_render_buffers) {
    RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
    if (texture_storage == nullptr) {
        return false;
    }

    r_render_target = texture_storage->render_target_create();
    if (!r_render_target.is_valid()) {
        return false;
    }

    texture_storage->render_target_set_use_hdr(r_render_target, true);
    texture_storage->render_target_set_transparent(r_render_target, false);
    texture_storage->render_target_set_size(r_render_target, p_resolution.x, p_resolution.y, 1);

    r_render_buffers.instantiate();
    Ref<RenderSceneBuffersConfiguration> rb_config;
    rb_config.instantiate();
    rb_config->set_render_target(r_render_target);
    rb_config->set_internal_size(p_resolution);
    rb_config->set_target_size(p_resolution);
    rb_config->set_view_count(1);
    r_render_buffers->configure(rb_config.ptr());
    return r_render_buffers->has_internal_texture();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Raster path eligibility follows viewport output format") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    RenderingDevice *primary_rd = manager->get_primary_rendering_device();
    if (!primary_rd) {
        primary_rd = rs->create_local_rendering_device();
    }
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }
    RenderingDevice *main_rd = RenderingDevice::get_singleton();
    if (main_rd == nullptr) {
        MESSAGE("Skipping test - Main rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();
    renderer->set_debug_compute_raster_policy(1); // ForceOn.

    const uint32_t total_gaussians = GaussianStreamingSystem::CHUNK_SIZE * 2;
    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, total_gaussians);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(total_gaussians);
    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        renderer.unref();
        return;
    }

    const Vector2i resolution(512, 288);
    RD::TextureUsageBits usage = static_cast<RD::TextureUsageBits>(
            RD::TEXTURE_USAGE_SAMPLING_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_TO_BIT |
            RD::TEXTURE_USAGE_STORAGE_BIT |
            RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT);

    RID ldr_target = create_test_texture(main_rd, resolution, RD::DATA_FORMAT_R8G8B8A8_UNORM, usage);
    RID hdr_target = create_test_texture(main_rd, resolution, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage);
    CHECK(ldr_target.is_valid());
    CHECK(hdr_target.is_valid());
    if (!ldr_target.is_valid() || !hdr_target.is_valid()) {
        if (ldr_target.is_valid()) {
            main_rd->free(ldr_target);
        }
        if (hdr_target.is_valid()) {
            main_rd->free(hdr_target);
        }
        renderer.unref();
        return;
    }

    Transform3D cam_a(Basis(), Vector3(0.0f, 0.0f, 6.0f));
    Projection projection;
    projection.set_perspective(65.0f, 16.0f / 9.0f, 0.1f, 200.0f);
    bool rendered_ldr = renderer->render_for_view(cam_a, projection, ldr_target, resolution);
    CHECK(rendered_ldr);

    Dictionary ldr_stats = renderer->get_render_stats();
    String ldr_raster_path = ldr_stats.get("raster_path", String());
    String ldr_raster_status = ldr_stats.get("stage_raster_status", String());
    CHECK_MESSAGE(ldr_raster_status == String("success"),
            vformat("Expected LDR raster stage success, got '%s'", ldr_raster_status));
    CHECK_MESSAGE(!ldr_raster_path.is_empty(), "Expected LDR raster path telemetry");

    Transform3D cam_b(Basis(), Vector3(0.0f, 0.0f, 7.0f)); // Prevent cache reuse masking path eligibility.
    bool rendered_hdr = renderer->render_for_view(cam_b, projection, hdr_target, resolution);
    CHECK(rendered_hdr);

    Dictionary hdr_stats = renderer->get_render_stats();
    String hdr_raster_path = hdr_stats.get("raster_path", String());
    String hdr_raster_status = hdr_stats.get("stage_raster_status", String());
    CHECK_MESSAGE(hdr_raster_status == String("success"),
            vformat("Expected HDR raster stage success, got '%s'", hdr_raster_status));
    CHECK_MESSAGE(hdr_raster_path != String("compute"),
            vformat("HDR target must not use compute raster path, got '%s'", hdr_raster_path));
    if (ldr_raster_path == String("compute")) {
        CHECK_MESSAGE(hdr_raster_path == String("fragment"),
                vformat("Expected HDR path to fall back to fragment when LDR used compute, got '%s'", hdr_raster_path));
    }

    if (ldr_target.is_valid()) {
        main_rd->free(ldr_target);
    }
    if (hdr_target.is_valid()) {
        main_rd->free(hdr_target);
    }
    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Final output copies between targets") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    RenderingDevice *primary_rd = manager->get_primary_rendering_device();
    if (!primary_rd) {
        primary_rd = rs->create_local_rendering_device();
    }
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }
    RenderingDevice *main_rd = RenderingDevice::get_singleton();
    if (main_rd == nullptr) {
        MESSAGE("Skipping test - Main rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    const uint32_t total_gaussians = GaussianStreamingSystem::CHUNK_SIZE * 2;
    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, total_gaussians);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(total_gaussians);
    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        return;
    }

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 6.0f));
    scene_data.cam_projection.set_perspective(65.0f, 16.0f / 9.0f, 0.1f, 200.0f);

    const Vector2i resolution(640, 360);
    bool rendered = renderer->render_for_view(scene_data.cam_transform, scene_data.cam_projection, RID(), resolution);
    CHECK(rendered);

    RID final_texture = renderer->get_final_texture();
    CHECK(final_texture.is_valid());

    RD::TextureUsageBits usage = static_cast<RD::TextureUsageBits>(RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT | RD::TEXTURE_USAGE_STORAGE_BIT);
    Vector2i copy_size = Vector2i(320, 180);
    RID color_target = create_test_texture(main_rd, copy_size, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage);
    CHECK(color_target.is_valid());
    if (!color_target.is_valid()) {
        renderer.unref();
        return;
    }

    bool copy_ok = renderer->copy_final_texture_to_target(color_target, copy_size);
    CHECK(copy_ok);
    CHECK(renderer->get_last_viewport_copy_source_size() == resolution);
    CHECK(renderer->get_last_viewport_copy_dest_size() == copy_size);
    CHECK_FALSE(renderer->was_last_viewport_copy_successful());

    if (color_target.is_valid()) {
        main_rd->free(color_target);
    }

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Tile renderer composites into viewport render buffers") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (project_settings == nullptr) {
        MESSAGE("Skipping test - ProjectSettings unavailable");
        return;
    }
    const String composite_depth_setting = "rendering/gaussian_splatting/composite/depth_test";
    ScopedProjectSetting composite_depth_guard(project_settings, composite_depth_setting);
    project_settings->set_setting(composite_depth_setting, true);

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    RenderingDevice *primary_rd = manager->get_primary_rendering_device();
    if (!primary_rd) {
        primary_rd = rs->create_local_rendering_device();
    }
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    const Vector2i resolution(320, 180);

    RID render_target;
    Ref<RenderSceneBuffersRD> render_buffers;
    const bool render_buffers_ok = create_test_render_buffers(resolution, render_target, render_buffers);
    CHECK(render_buffers_ok);
    if (!render_buffers_ok) {
        renderer.unref();
        return;
    }
    RID scene_depth = render_buffers->get_depth_texture();
    if (!scene_depth.is_valid()) {
        MESSAGE("Skipping test - Scene depth texture unavailable");
        renderer.unref();
        RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
        if (texture_storage != nullptr && render_target.is_valid()) {
            texture_storage->render_target_free(render_target);
        }
        return;
    }

    const uint32_t total_gaussians = GaussianStreamingSystem::CHUNK_SIZE;
    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, total_gaussians);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(total_gaussians);
    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        renderer.unref();
        RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
        if (texture_storage != nullptr && render_target.is_valid()) {
            texture_storage->render_target_free(render_target);
        }
        return;
    }

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 6.0f));
    scene_data.cam_projection.set_perspective(65.0f, 16.0f / 9.0f, 0.1f, 200.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = render_buffers;

    for (int frame = 0; frame < 2; frame++) {
        renderer->render_scene_instance(&render_data);
    }
    CHECK(renderer->test_has_output_compositor());
    if (!renderer->test_has_output_compositor()) {
        renderer.unref();
        RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
        if (texture_storage != nullptr && render_target.is_valid()) {
            texture_storage->render_target_free(render_target);
        }
        return;
    }

    const RID cached_depth = renderer->test_get_cached_render_depth();
    if (!cached_depth.is_valid()) {
        MESSAGE("Skipping test - Cached raster depth unavailable");
        renderer.unref();
        RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
        if (texture_storage != nullptr && render_target.is_valid()) {
            texture_storage->render_target_free(render_target);
        }
        return;
    }
    const uint32_t blit_variants_before_commit = renderer->test_get_output_blit_variant_count();

    renderer->commit_to_render_buffers(&render_data);

    CHECK(renderer->was_last_viewport_copy_successful());
    CHECK(renderer->get_last_viewport_copy_source_size() == resolution);
    CHECK(renderer->get_last_viewport_copy_dest_size() == resolution);
    CHECK(renderer->test_get_output_blit_variant_count() > blit_variants_before_commit);

    renderer.unref();

    RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
    if (texture_storage != nullptr && render_target.is_valid()) {
        texture_storage->render_target_free(render_target);
    }
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Scene composite keeps strict depth policy when cached depth is missing") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (project_settings == nullptr) {
        MESSAGE("Skipping test - ProjectSettings unavailable");
        return;
    }
    const String composite_depth_setting = "rendering/gaussian_splatting/composite/depth_test";
    ScopedProjectSetting composite_depth_guard(project_settings, composite_depth_setting);
    project_settings->set_setting(composite_depth_setting, true);

    ScopedGaussianManagerPipeline manager_scope;
    GaussianSplatManager *manager = manager_scope.get();
    if (manager == nullptr) {
        MESSAGE("Skipping test - GaussianSplatManager unavailable");
        return;
    }

    RenderingDevice *primary_rd = manager->get_primary_rendering_device();
    if (!primary_rd) {
        primary_rd = rs->create_local_rendering_device();
    }
    if (primary_rd == nullptr) {
        MESSAGE("Skipping test - Rendering device unavailable");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_rd);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }
    renderer->initialize();

    const Vector2i resolution(320, 180);
    RID render_target;
    Ref<RenderSceneBuffersRD> render_buffers;
    const bool render_buffers_ok = create_test_render_buffers(resolution, render_target, render_buffers);
    CHECK(render_buffers_ok);
    if (!render_buffers_ok) {
        renderer.unref();
        return;
    }

    RID scene_depth = render_buffers->get_depth_texture();
    if (!scene_depth.is_valid()) {
        MESSAGE("Skipping test - Scene depth texture unavailable");
        renderer.unref();
        RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
        if (texture_storage != nullptr && render_target.is_valid()) {
            texture_storage->render_target_free(render_target);
        }
        return;
    }

    const uint32_t total_gaussians = GaussianStreamingSystem::CHUNK_SIZE;
    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, total_gaussians);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(total_gaussians);
    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        renderer.unref();
        RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
        if (texture_storage != nullptr && render_target.is_valid()) {
            texture_storage->render_target_free(render_target);
        }
        return;
    }

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 6.0f));
    scene_data.cam_projection.set_perspective(65.0f, 16.0f / 9.0f, 0.1f, 200.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = render_buffers;

    for (int frame = 0; frame < 2; frame++) {
        renderer->render_scene_instance(&render_data);
    }

    CHECK(renderer->test_has_output_compositor());
    if (!renderer->test_has_output_compositor()) {
        renderer.unref();
        RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
        if (texture_storage != nullptr && render_target.is_valid()) {
            texture_storage->render_target_free(render_target);
        }
        return;
    }

    RID final_output = renderer->get_final_texture();
    RID cached_depth = renderer->test_get_cached_render_depth();
    if (!final_output.is_valid() || !cached_depth.is_valid()) {
        MESSAGE("Skipping test - Final output or cached depth unavailable");
        renderer.unref();
        RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
        if (texture_storage != nullptr && render_target.is_valid()) {
            texture_storage->render_target_free(render_target);
        }
        return;
    }

    renderer->test_clear_output_viewport_blit_resources();
    renderer->test_reset_output_viewport_copy_state();

    RID resolved_render_target;
    renderer->test_integrate_final_output(&render_data, render_buffers.ptr(), final_output,
            resolved_render_target, resolution, false, false, RID());

    CHECK_FALSE(renderer->was_last_viewport_copy_successful());
    CHECK(renderer->test_get_output_blit_variant_count() == 0);

    renderer.unref();

    RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
    if (texture_storage != nullptr && render_target.is_valid()) {
        texture_storage->render_target_free(render_target);
    }
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Cached render reuse requires cached depth when depth validation is requested") {
    REQUIRE_GPU_DEVICE();

    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - Rendering server unavailable");
        return;
    }

    RenderingDevice *local_rd = rs->create_local_rendering_device();
    if (local_rd == nullptr) {
        MESSAGE("Skipping test - Local rendering device unavailable");
        return;
    }

    Ref<OutputCompositor> compositor;
    compositor.instantiate();
    Error init_err = compositor->initialize(local_rd);
    CHECK(init_err == OK);
    if (init_err != OK) {
        memdelete(local_rd);
        return;
    }

    const Vector2i resolution(16, 16);
    const RD::TextureUsageBits color_usage = static_cast<RD::TextureUsageBits>(
            RD::TEXTURE_USAGE_SAMPLING_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_TO_BIT |
            RD::TEXTURE_USAGE_STORAGE_BIT);
    const RD::TextureUsageBits depth_usage = static_cast<RD::TextureUsageBits>(
            RD::TEXTURE_USAGE_SAMPLING_BIT |
            RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_TO_BIT);

    RID final_texture = create_test_texture(local_rd, resolution, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, color_usage);
    RID depth_texture = create_test_texture(local_rd, resolution, RD::DATA_FORMAT_D32_SFLOAT, depth_usage);
    if (!final_texture.is_valid() || !depth_texture.is_valid()) {
        MESSAGE("Skipping test - Unable to allocate color/depth textures for cache reuse validation");
        if (final_texture.is_valid()) {
            local_rd->free(final_texture);
        }
        if (depth_texture.is_valid()) {
            local_rd->free(depth_texture);
        }
        compositor->shutdown();
        memdelete(local_rd);
        return;
    }

    const Transform3D view_transform(Basis(), Vector3(0.0f, 0.0f, 6.0f));
    Projection projection;
    projection.set_perspective(65.0f, 1.0f, 0.1f, 200.0f);

    compositor->set_has_valid_render(true);
    compositor->update_render_cache_signature(view_transform, projection, resolution, false,
            depth_texture, resolution, final_texture, 11, 19, 13, 17, true);
    CHECK(compositor->can_reuse_cached_render(view_transform, projection, resolution, false,
            final_texture, 11, 19, 13, 17, true));
    CHECK_FALSE(compositor->can_reuse_cached_render(view_transform, projection, resolution, false,
            final_texture, 12, 19, 13, 17, true));
    CHECK_FALSE(compositor->can_reuse_cached_render(view_transform, projection, resolution, false,
            final_texture, 11, 23, 13, 17, true));

    compositor->update_render_cache_signature(view_transform, projection, resolution, false,
            RID(), resolution, final_texture, 11, 19, 13, 17, true);
    CHECK_FALSE(compositor->can_reuse_cached_render(view_transform, projection, resolution, false,
            final_texture, 11, 19, 13, 17, true));

    local_rd->free(final_texture);
    local_rd->free(depth_texture);
    compositor->shutdown();
    memdelete(local_rd);
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Render-thread blocking dispatch times out when callback never signals completion") {
    RenderingServer *rs = RenderingServer::get_singleton();
    OS *os = OS::get_singleton();
    if (rs == nullptr || os == nullptr) {
        MESSAGE("Skipping test - RenderingServer/OS unavailable");
        return;
    }
    if (rs->is_on_render_thread()) {
        MESSAGE("Skipping test - Test must run off the render thread");
        return;
    }
    if (!rs->is_render_loop_enabled()) {
        MESSAGE("Skipping test - Render loop disabled");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate();
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }

    const uint64_t original_timeout_usec = renderer->test_get_render_thread_dispatch_timeout_usec();
    renderer->test_set_render_thread_dispatch_timeout_usec(10000); // 10 ms timeout for test.

    const uint64_t start_usec = os->get_ticks_usec();
    const bool dispatched = renderer->test_dispatch_call_on_render_thread_blocking_without_completion();
    const uint64_t elapsed_usec = os->get_ticks_usec() - start_usec;

    renderer->test_set_render_thread_dispatch_timeout_usec(original_timeout_usec);

    CHECK_FALSE(dispatched);
    CHECK(elapsed_usec >= 10000);
    CHECK(elapsed_usec < 2000000);
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Render-thread blocking dispatch only advances completion state on success") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - RenderingServer unavailable");
        return;
    }
    if (rs->is_on_render_thread()) {
        MESSAGE("Skipping test - Test must run off the render thread");
        return;
    }
    if (!rs->is_render_loop_enabled()) {
        MESSAGE("Skipping test - Render loop disabled");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate();
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }

    const uint64_t original_timeout_usec = renderer->test_get_render_thread_dispatch_timeout_usec();
    const uint64_t completed_before = renderer->test_get_render_thread_dispatch_completed_request_id();
    renderer->test_set_render_thread_dispatch_timeout_usec(10000); // 10 ms timeout for test.

    const bool timed_out_dispatch = renderer->test_dispatch_call_on_render_thread_blocking_without_completion();
    CHECK_FALSE(timed_out_dispatch);
    CHECK(renderer->test_get_render_thread_dispatch_completed_request_id() == completed_before);

    const bool completed_dispatch = renderer->test_dispatch_call_on_render_thread_blocking_with_completion();
    CHECK(completed_dispatch);
    CHECK(renderer->test_get_render_thread_dispatch_completed_request_id() > completed_before);

    renderer->test_set_render_thread_dispatch_timeout_usec(original_timeout_usec);
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Render-thread blocking dispatch preserves forward progress after timeout escape") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping test - RenderingServer unavailable");
        return;
    }
    if (rs->is_on_render_thread()) {
        MESSAGE("Skipping test - Test must run off the render thread");
        return;
    }
    if (!rs->is_render_loop_enabled()) {
        MESSAGE("Skipping test - Render loop disabled");
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate();
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }

    const uint64_t original_timeout_usec = renderer->test_get_render_thread_dispatch_timeout_usec();
    renderer->test_set_render_thread_dispatch_timeout_usec(10000); // 10 ms timeout for test.

    const bool timed_out_dispatch = renderer->test_dispatch_call_on_render_thread_blocking_without_completion();
    CHECK_FALSE(timed_out_dispatch);

    const bool recovered_dispatch = renderer->test_dispatch_call_on_render_thread_blocking_with_completion();
    CHECK(recovered_dispatch);

    renderer->test_set_render_thread_dispatch_timeout_usec(original_timeout_usec);

    const uint64_t completed_before_stale = renderer->test_get_render_thread_dispatch_completed_request_id();
    if (completed_before_stale > 0) {
        renderer->test_notify_render_thread_dispatch_completed(completed_before_stale - 1);
        CHECK(renderer->test_get_render_thread_dispatch_completed_request_id() == completed_before_stale);
    }
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Render-thread dispatch teardown remains bounded after timeout recovery") {
    RenderingServer *rs = RenderingServer::get_singleton();
    OS *os = OS::get_singleton();
    if (rs == nullptr || os == nullptr) {
        MESSAGE("Skipping test - RenderingServer/OS unavailable");
        return;
    }
    if (rs->is_on_render_thread()) {
        MESSAGE("Skipping test - Test must run off the render thread");
        return;
    }
    if (!rs->is_render_loop_enabled()) {
        MESSAGE("Skipping test - Render loop disabled");
        return;
    }

    const uint64_t start_usec = os->get_ticks_usec();
    {
        Ref<GaussianSplatRenderer> renderer;
        renderer.instantiate();
        CHECK(renderer.is_valid());
        if (!renderer.is_valid()) {
            return;
        }

        const uint64_t original_timeout_usec = renderer->test_get_render_thread_dispatch_timeout_usec();
        renderer->test_set_render_thread_dispatch_timeout_usec(10000); // 10 ms timeout for test.

        const bool timed_out_dispatch = renderer->test_dispatch_call_on_render_thread_blocking_without_completion();
        CHECK_FALSE(timed_out_dispatch);
        const bool recovered_dispatch = renderer->test_dispatch_call_on_render_thread_blocking_with_completion();
        CHECK(recovered_dispatch);

        renderer->test_set_render_thread_dispatch_timeout_usec(original_timeout_usec);
    }
    const uint64_t elapsed_usec = os->get_ticks_usec() - start_usec;

    // Characterization guard: teardown after a dispatch cycle should remain bounded and not hang.
    CHECK(elapsed_usec < 5000000);
}

} // namespace TestGaussianSplatting
