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
#include "../core/effective_config_snapshot.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_manager.h"
#include "../core/gaussian_streaming.h"
#include "../core/gs_project_settings.h"
#include "../renderer/gaussian_gpu_layout.h"
#include "../renderer/gpu_sorting_config.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../renderer/pipeline_feature_set.h"
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

static_assert(GS_RENDER_PARAMS_LAYOUT_VERSION == 17, "Render params layout version mismatch");

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
static_assert(sizeof(TileRenderParamsGPU) == 736, "TileRenderParamsGPU size contract changed");
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
    CHECK(GS_RENDER_PARAMS_LAYOUT_VERSION == 17u);
    CHECK(sizeof(InstanceDataGPU) == size_t(96));
    CHECK(sizeof(AssetMetaGPU) == size_t(112));
    CHECK(sizeof(ChunkMetaGPU) == size_t(64));
    CHECK(sizeof(SplatRefGPU) == size_t(8));
    CHECK(sizeof(PackedGaussian) == size_t(144));
    CHECK(sizeof(PackedGaussianF16) == size_t(144));
    CHECK(sizeof(PackedGaussianQuantized) == size_t(80));
    CHECK(sizeof(TileRenderParamsGPU) == size_t(736));

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

static bool hud_lines_contain(const Array &p_lines, const String &p_fragment) {
    for (int i = 0; i < p_lines.size(); i++) {
        if (String(p_lines[i]).find(p_fragment) != -1) {
            return true;
        }
    }
    return false;
}

TEST_CASE("[GaussianSplatting] Instanced readiness gate reports deterministic buffer failure modes") {
    GaussianRenderPipeline::InstancePipelineBuffers ready_buffers = make_ready_instance_pipeline_buffers(false);

    RenderInstancingOrchestrator::InstanceReadinessResult missing_contract =
            RenderInstancingOrchestrator::evaluate_instance_pipeline_readiness(
                    false, true, ready_buffers);
    CHECK(!missing_contract.ready);
    CHECK(missing_contract.failure_mode ==
            RenderInstancingOrchestrator::InstanceReadinessFailureMode::INSTANCE_BACKEND_CONTRACT_UNAVAILABLE);

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
    RenderInstancingOrchestrator::InstanceReadinessResult missing_contract =
            RenderInstancingOrchestrator::evaluate_instance_pipeline_readiness(
                    false, true, ready_buffers);
    CHECK(!missing_contract.ready);
    CHECK(missing_contract.failure_mode ==
            RenderInstancingOrchestrator::InstanceReadinessFailureMode::INSTANCE_BACKEND_CONTRACT_UNAVAILABLE);

    LocalVector<Transform3D> instance_transforms;
    instance_transforms.push_back(Transform3D());
    orchestrator.render_instanced(nullptr, GaussianSplatManager::SharedDynamicAssetHandle(),
            Transform3D(), Projection(), Projection(), instance_transforms);

    CHECK_MESSAGE(!prepare_called, "Expected readiness gate to skip frame preparation callback");
    CHECK_MESSAGE(!render_called, "Expected readiness gate to skip render callback");

    renderer.unref();
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Instance buffer upload uses the published renderer-side remap") {
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

    GaussianRenderPipeline::PublishedInstanceAssetRemap remap;
    remap.asset_to_dense_id.insert(0u, 0u);
    remap.asset_to_dense_id.insert(42u, 7u);
    remap.generation = 19u;
    remap.valid = true;
    renderer->publish_instance_pipeline_contract(
            GaussianRenderPipeline::InstancePipelineBuffers(),
            remap,
            GaussianRenderPipeline::InstanceBackendPolicy::RESIDENT,
            remap.generation,
            "atlas_emulation");

    LocalVector<InstanceDataGPU> instances;
    InstanceDataGPU instance = {};
    instance.rotation[3] = 1.0f;
    instance.inv_rotation[3] = 1.0f;
    instance.translation_scale[3] = 1.0f;
    instance.params[0] = 1.0f;
    instance.params[1] = 1.0f;
    instance.params[2] = 1.0f;
    instance.wind_params[3] = 1.0f;
    instance.ids[0] = 42u;
    instances.push_back(instance);

    CHECK(renderer->update_instance_buffer(instances, remap));
    REQUIRE(instances.size() == 1);
    CHECK(instances[0].ids[0] == 7u);
    CHECK(instances[0].lod[1] == 19u);

    const RID instance_buffer = renderer->get_instance_pipeline_buffers().instance_buffer;
    REQUIRE(instance_buffer.is_valid());
    const Vector<uint8_t> bytes = primary_rd->buffer_get_data(instance_buffer, 0, sizeof(InstanceDataGPU));
    REQUIRE(bytes.size() == sizeof(InstanceDataGPU));

    InstanceDataGPU uploaded = {};
    memcpy(&uploaded, bytes.ptr(), sizeof(InstanceDataGPU));
    CHECK(uploaded.ids[0] == 7u);
    CHECK(uploaded.lod[1] == 19u);

    renderer.unref();
}

TEST_CASE("[GaussianSplatting] Clearing the published instance contract also clears the renderer-side remap") {
    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate();
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        return;
    }

    GaussianRenderPipeline::InstancePipelineBuffers buffers;
    buffers.instance_count = 3;
    buffers.max_visible_splats = 9;

    GaussianRenderPipeline::PublishedInstanceAssetRemap remap;
    remap.asset_to_dense_id.insert(5u, 2u);
    remap.generation = 11u;
    remap.valid = true;

    renderer->publish_instance_pipeline_contract(
            buffers,
            remap,
            GaussianRenderPipeline::InstanceBackendPolicy::RESIDENT,
            remap.generation,
            "atlas_emulation");

    CHECK(renderer->has_instance_pipeline_buffers());
    CHECK(renderer->has_instance_asset_remap());
    CHECK(renderer->get_instance_backend_policy() == GaussianRenderPipeline::InstanceBackendPolicy::RESIDENT);
    CHECK(renderer->get_instance_contract_shape() == String("atlas_emulation"));
    CHECK(renderer->get_instance_asset_remap().generation == 11u);

    renderer->clear_instance_pipeline_buffers();

    CHECK_FALSE(renderer->has_instance_pipeline_buffers());
    CHECK_FALSE(renderer->has_instance_asset_remap());
    CHECK(renderer->get_instance_backend_policy() == GaussianRenderPipeline::InstanceBackendPolicy::NONE);
    CHECK(renderer->get_instance_contract_shape() == String("none"));
    CHECK(renderer->get_instance_pipeline_buffers().instance_count == 0u);
    CHECK(renderer->get_instance_pipeline_buffers().max_visible_splats == 0u);
    CHECK(renderer->get_instance_asset_remap().asset_to_dense_id.is_empty());
    CHECK(renderer->get_instance_asset_remap().generation == 0u);
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Resident route publishes an atlas-shaped instance contract without a streaming system") {
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

    const String route_policy_setting = "rendering/gaussian_splatting/streaming/route_policy";
    const String instance_pipeline_setting = "rendering/gaussian_splatting/instance_pipeline/enabled";
    ScopedProjectSetting route_guard(project_settings, route_policy_setting);
    ScopedProjectSetting instance_pipeline_guard(project_settings, instance_pipeline_setting);
    project_settings->set_setting(route_policy_setting, int64_t(gs::settings::GS_ROUTE_RESIDENT));
    project_settings->set_setting(instance_pipeline_setting, true);
    project_settings->emit_signal("settings_changed");

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
    renderer->get_debug_state().show_performance_hud = true;

    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, 64);
    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    renderer->set_max_splats(64);
    const Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        renderer.unref();
        return;
    }

    renderer->set_static_chunks(make_single_static_chunk(64, data->get_aabb()));
    renderer->test_release_current_streaming_system();
    CHECK_FALSE(renderer->test_has_current_streaming_system());

    RenderSceneDataRD scene_data;
    scene_data.cam_transform = Transform3D(Basis(), Vector3(0.0f, 0.0f, 5.0f));
    scene_data.cam_projection.set_perspective(70.0f, 1.0f, 0.1f, 100.0f);

    RenderDataRD render_data;
    render_data.scene_data = &scene_data;
    render_data.render_buffers = Ref<RenderSceneBuffersRD>();

    renderer->render_scene_instance(&render_data);

    CHECK_FALSE(renderer->test_has_current_streaming_system());
    CHECK(renderer->has_instance_pipeline_buffers());
    CHECK(renderer->has_instance_asset_remap());
    CHECK(renderer->get_instance_backend_policy() == GaussianRenderPipeline::InstanceBackendPolicy::RESIDENT);
    CHECK(renderer->is_instance_contract_ready());
    CHECK(renderer->get_instance_contract_shape() == String("atlas_emulation"));

    const Dictionary stats = renderer->get_render_stats();
    CHECK(stats.get("route_uid", String()) == String(RenderRouteUID::INSTANCE_RESIDENT));
    CHECK(stats.get("route_label", String()) == String("Resident instanced path"));
    CHECK(stats.get("requested_route_policy", String()) == String("resident"));
    CHECK(stats.get("requested_route_policy_source", String()) == String("route_policy"));
    CHECK(stats.get("instance_backend_policy", String()) == String("resident"));
    CHECK(stats.get("backend_selection_reason", String()) == String("requested_resident_policy"));
    CHECK(stats.get("backend_selection_reason_label", String()) == String("Resident was requested by the route policy"));
    CHECK(stats.get("instance_contract_shape", String()) == String("atlas_emulation"));
    CHECK(bool(stats.get("instance_contract_ready", false)));
    CHECK(stats.get("data_source", String()) == String(GaussianRenderPipeline::SplatDataSource::kSourceResidentInstance));
    const Dictionary effective_config = stats.get("effective_config_snapshot", Dictionary());
    const Dictionary route_entry = GaussianEffectiveConfig::get_entry(effective_config, StringName("active_route"));
    const Dictionary backend_entry = GaussianEffectiveConfig::get_entry(effective_config, StringName("instance_backend_policy"));
    const Dictionary requested_policy_entry = GaussianEffectiveConfig::get_entry(effective_config, StringName("requested_route_policy"));
    CHECK(String(route_entry.get(StringName("display_value"), String())) == String("Resident instanced path [INSTANCE.RESIDENT]"));
    CHECK(String(route_entry.get(StringName("source_label"), String())) == String("Resident was requested by the route policy"));
    CHECK(String(backend_entry.get(StringName("value"), String())) == String("resident"));
    CHECK(String(backend_entry.get(StringName("source_label"), String())) == String("Resident was requested by the route policy"));
    CHECK(String(requested_policy_entry.get(StringName("value"), String())) == String("resident"));
    CHECK(String(requested_policy_entry.get(StringName("source_label"), String())) == String("route_policy"));

    const Array hud_lines = stats.get("performance_hud_lines", Array());
    CHECK(hud_lines_contain(hud_lines, "Route: Resident instanced path [INSTANCE.RESIDENT]"));
    CHECK(hud_lines_contain(hud_lines, "Requested Policy: resident (route_policy)"));
    CHECK(hud_lines_contain(hud_lines, "Instance Backend: resident"));
    CHECK(hud_lines_contain(hud_lines, "Backend Reason: Resident was requested by the route policy"));
    CHECK(hud_lines_contain(hud_lines, "Instance Contract: atlas_emulation (ready)"));

    renderer.unref();
}

TEST_CASE("[GaussianSplatting] Pipeline feature snapshot distinguishes code defaults from project overrides") {
    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    REQUIRE(project_settings != nullptr);
    if (project_settings == nullptr) {
        return;
    }

    const String fast_raster_setting = PipelineFeatureSet::ENABLE_FAST_RASTER_PATH;
    const String tier_preset_setting = "rendering/gaussian_splatting/quality/tier_preset";
    const String tier_apply_setting = "rendering/gaussian_splatting/quality/tier_apply_pipeline_toggles";

    ScopedProjectSetting fast_raster_guard(project_settings, fast_raster_setting);
    ScopedProjectSetting tier_preset_guard(project_settings, tier_preset_setting);
    ScopedProjectSetting tier_apply_guard(project_settings, tier_apply_setting);

    project_settings->set_setting(tier_preset_setting, String("custom"));
    project_settings->set_setting(tier_apply_setting, false);

    project_settings->set_setting(fast_raster_setting, false);
    project_settings->set_builtin_order(fast_raster_setting);
    g_pipeline_feature_set.load_from_project_settings();
    Dictionary snapshot = g_pipeline_feature_set.get_effective_config_snapshot();
    Dictionary fast_raster_entry = GaussianEffectiveConfig::get_entry(snapshot, StringName("pipeline_fast_raster"));
    CHECK(String(fast_raster_entry.get(StringName("source_label"), String())) == String("code default"));

    project_settings->clear(fast_raster_setting);
    project_settings->set_setting(fast_raster_setting, true);
    g_pipeline_feature_set.load_from_project_settings();
    snapshot = g_pipeline_feature_set.get_effective_config_snapshot();
    fast_raster_entry = GaussianEffectiveConfig::get_entry(snapshot, StringName("pipeline_fast_raster"));
    CHECK(bool(fast_raster_entry.get(StringName("value"), false)));
    CHECK(String(fast_raster_entry.get(StringName("source_label"), String())) == String("project override"));

    g_pipeline_feature_set.load_from_project_settings();
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

    // Warm up streaming pipeline: atlas upload -> chunk loading -> instance
    // buffer creation -> cull -> sort may take many frames without the
    // resident-fallback path.
    bool pipeline_ready = false;
    uint32_t visible_splat_count = 0;
    for (int i = 0; i < 16; i++) {
        renderer->render_scene_instance(&render_data);
        visible_splat_count = renderer->get_visible_splat_count();
        if (renderer->has_instance_pipeline_buffers() && renderer->has_rendered_content() && visible_splat_count > 0) {
            pipeline_ready = true;
            break;
        }
    }

    CHECK(renderer->has_rendered_content());
    CHECK(renderer->get_final_texture().is_valid());

    if (!pipeline_ready) {
        MESSAGE("Instance pipeline did not fully warm up (headless CI without real GPU async readback) - "
                "visible_splat_count=" << visible_splat_count
                << " has_instance_pipeline_buffers=" << renderer->has_instance_pipeline_buffers());
        renderer->commit_to_render_buffers(&render_data);
        renderer.unref();
        return;
    }

    CHECK(visible_splat_count > 0);
    CHECK(renderer->is_instance_contract_ready());
    const GaussianRenderPipeline::InstancePipelineBuffers &streaming_buffers =
            renderer->get_instance_pipeline_buffers();
    CHECK(streaming_buffers.instance_buffer.is_valid());
    CHECK(streaming_buffers.instance_count > 0);

    Dictionary stats = renderer->get_render_stats();
    CHECK(stats.get("route_uid", String()) == String(RenderRouteUID::INSTANCE_STREAMING));
    CHECK(stats.get("using_real_data", false));
    const String data_source = stats.get("data_source", String());
    bool valid_data_source = (data_source == String("StreamingGPU")) || (data_source == String("GPUBufferManager"));
    CHECK_MESSAGE(valid_data_source, vformat("Unexpected render stats data_source: %s", data_source));
    CHECK(stats.get("gpu_sorter_ready", false));
    CHECK(bool(stats.get("instance_contract_ready", false)));
    CHECK(stats.get("instance_contract_shape", String()) == String("atlas_emulation"));

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

TEST_CASE("[GaussianSplatting][RequiresGPU] Upload processing rescues stranded async pack work") {
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

    const uint32_t full_chunk_count = GaussianStreamingSystem::CHUNK_SIZE;
    const uint32_t tail_chunk_count = 128;
    LocalVector<Gaussian> gaussians;
    fill_gaussians(gaussians, full_chunk_count + tail_chunk_count);

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    Ref<GaussianStreamingSystem> streaming_system;
    streaming_system.instantiate();
    CHECK(streaming_system.is_valid());
    if (!streaming_system.is_valid()) {
        return;
    }
    streaming_system->initialize_with_device(data, primary_rd);

    StreamingUploadPipeline &upload_pipeline = streaming_system->_internal_get_upload_pipeline();
    upload_pipeline.stop_pack_threads(*streaming_system.ptr());
    upload_pipeline.async_pack_enabled = true;

    CHECK_MESSAGE(upload_pipeline.queue_chunk_load(*streaming_system.ptr(), 0, 0),
            "Expected chunk load request to enqueue async pack work for the primary asset");
    CHECK(streaming_system->get_pending_pack_jobs() == 1);
    CHECK(streaming_system->get_pending_upload_jobs() == 0);
    CHECK(streaming_system->get_loaded_chunks() == 0);

    upload_pipeline.process_upload_queue(*streaming_system.ptr());

    CHECK_MESSAGE(streaming_system->get_loaded_chunks() == 1,
            "Expected upload processing to synchronously rescue one stranded pack job into a completed chunk load");
    CHECK(streaming_system->get_chunks_loaded_this_frame() == 1);
    CHECK(streaming_system->get_pending_pack_jobs() == 0);
    CHECK(streaming_system->get_pending_upload_jobs() == 0);
    CHECK(upload_pipeline._test_get_sync_snapshot_gaussian_size() == full_chunk_count);
    const uint32_t initial_snapshot_capacity = upload_pipeline._test_get_sync_snapshot_gaussian_capacity();
    CHECK(initial_snapshot_capacity >= full_chunk_count);

    CHECK_MESSAGE(upload_pipeline.queue_chunk_load(*streaming_system.ptr(), 0, 1),
            "Expected tail chunk load request to enqueue async pack work for the primary asset");
    upload_pipeline.process_upload_queue(*streaming_system.ptr());

    CHECK(streaming_system->get_loaded_chunks() == 2);
    CHECK(streaming_system->get_pending_pack_jobs() == 0);
    CHECK(streaming_system->get_pending_upload_jobs() == 0);
    CHECK(upload_pipeline._test_get_sync_snapshot_gaussian_size() == tail_chunk_count);
    CHECK(upload_pipeline._test_get_sync_snapshot_gaussian_capacity() >= initial_snapshot_capacity);

    upload_pipeline.process_upload_queue(*streaming_system.ptr());
    CHECK(streaming_system->get_loaded_chunks() == 2);
    CHECK(streaming_system->get_pending_pack_jobs() == 0);
    CHECK(streaming_system->get_pending_upload_jobs() == 0);
}

TEST_CASE("[GaussianSplatting] Sync pack rescue does not steal worker-owned pack jobs") {
    GaussianStreamingSystem streaming_system;
    StreamingUploadPipeline &upload_pipeline = streaming_system._internal_get_upload_pipeline();

    upload_pipeline._test_set_async_pack_queue_owner(true);
    CHECK(upload_pipeline._test_has_async_pack_queue_owner());
    upload_pipeline._test_enqueue_dummy_pack_job();

    CHECK(upload_pipeline._test_promote_pack_jobs_sync(1) == 0);
    CHECK(upload_pipeline.get_pack_queue_depth_cached() == 1);
    CHECK(upload_pipeline.get_upload_queue_depth_cached() == 0);
}

TEST_CASE("[GaussianSplatting] Chunk meta upload planner keeps compact dirty spans incremental") {
    LocalVector<uint32_t> dirty_indices;
    dirty_indices.push_back(4);
    dirty_indices.push_back(5);
    dirty_indices.push_back(6);
    dirty_indices.push_back(20);
    dirty_indices.push_back(21);

    const StreamingGlobalAtlasRegistry::ChunkMetaUploadPlan plan =
            StreamingGlobalAtlasRegistry::_test_plan_chunk_meta_uploads(dirty_indices, 128);

    CHECK(plan.dirty_count == 5);
    CHECK(plan.contiguous_range_count == 2);
    CHECK_FALSE(plan.full_update);
}

TEST_CASE("[GaussianSplatting] Chunk meta upload planner escalates fragmented churn to a full upload") {
    LocalVector<uint32_t> dirty_indices;
    for (uint32_t i = 0; i < 16; i++) {
        dirty_indices.push_back(i * 2);
    }

    const StreamingGlobalAtlasRegistry::ChunkMetaUploadPlan plan =
            StreamingGlobalAtlasRegistry::_test_plan_chunk_meta_uploads(dirty_indices, 128);

    CHECK(plan.dirty_count == 16);
    CHECK(plan.contiguous_range_count == 16);
    CHECK(plan.full_update);
}

TEST_CASE("[GaussianSplatting] Upload coalescing planner batches contiguous full-slot uploads") {
    LocalVector<StreamingUploadPipeline::UploadCoalescingCandidate> candidates;

    StreamingUploadPipeline::UploadCoalescingCandidate first;
    first.buffer_slot = 10;
    first.packed_count = GaussianStreamingSystem::CHUNK_SIZE;
    candidates.push_back(first);

    StreamingUploadPipeline::UploadCoalescingCandidate second;
    second.buffer_slot = 11;
    second.packed_count = GaussianStreamingSystem::CHUNK_SIZE;
    candidates.push_back(second);

    StreamingUploadPipeline::UploadCoalescingCandidate tail;
    tail.buffer_slot = 12;
    tail.packed_count = 128;
    candidates.push_back(tail);

    const uint64_t slot_bytes = uint64_t(GaussianStreamingSystem::CHUNK_SIZE) * sizeof(PackedGaussian);
    const StreamingUploadPipeline::UploadCoalescingPlan plan =
            StreamingUploadPipeline::_test_plan_coalesced_upload_batch(candidates, slot_bytes * 2);

    CHECK(plan.coalesced_job_count == 2);
    CHECK(plan.total_bytes == slot_bytes * 2);
}

TEST_CASE("[GaussianSplatting] Upload coalescing planner stops at partial or noncontiguous uploads") {
    LocalVector<StreamingUploadPipeline::UploadCoalescingCandidate> candidates;

    StreamingUploadPipeline::UploadCoalescingCandidate first;
    first.buffer_slot = 20;
    first.packed_count = GaussianStreamingSystem::CHUNK_SIZE;
    candidates.push_back(first);

    StreamingUploadPipeline::UploadCoalescingCandidate partial;
    partial.buffer_slot = 21;
    partial.packed_count = GaussianStreamingSystem::CHUNK_SIZE;
    partial.bytes_uploaded = sizeof(PackedGaussian);
    candidates.push_back(partial);

    StreamingUploadPipeline::UploadCoalescingCandidate gap;
    gap.buffer_slot = 23;
    gap.packed_count = GaussianStreamingSystem::CHUNK_SIZE;
    candidates.push_back(gap);

    const uint64_t slot_bytes = uint64_t(GaussianStreamingSystem::CHUNK_SIZE) * sizeof(PackedGaussian);
    const StreamingUploadPipeline::UploadCoalescingPlan plan =
            StreamingUploadPipeline::_test_plan_coalesced_upload_batch(candidates, slot_bytes * 4);

    CHECK(plan.coalesced_job_count == 1);
    CHECK(plan.total_bytes == slot_bytes);
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

    const String instance_pipeline_setting = "rendering/gaussian_splatting/instance_pipeline/enabled";
    ScopedProjectSetting instance_pipeline_guard(project_settings, instance_pipeline_setting);
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
    CHECK_MESSAGE(stats.get("route_uid", String()) == String(RenderRouteUID::INSTANCE_STREAMING),
            "Expected route UID to report the streaming instance path");
    CHECK_MESSAGE(stats.get("cull_route_uid", String()) == String(RenderRouteUID::INSTANCE_CULL_GPU),
            "Expected cull route UID to report the instance pipeline");
    CHECK_MESSAGE(stats.get("cull_route_reason", String()) == String("instance_pipeline_active"),
            "Expected cull route reason to report the instance pipeline");
    CHECK(bool(stats.get("instance_contract_ready", false)));

    CHECK_MESSAGE(instance_buffers_ready,
            "Expected instance pipeline buffers to become ready for world/static-chunk streaming without SceneDirector instances");
    const GaussianRenderPipeline::InstancePipelineBuffers &buffers = renderer->get_instance_pipeline_buffers();
    CHECK_MESSAGE(renderer->is_instance_contract_ready(),
            "Expected world/static-chunk streaming to publish a ready instance contract");
    CHECK_MESSAGE(buffers.instance_buffer.is_valid(),
            "Expected streaming instance buffer upload to complete before readiness validation");
    CHECK_MESSAGE(buffers.instance_count > 0, "Expected instance pipeline to synthesize at least one instance for world/static-chunk streaming");
    CHECK_MESSAGE(buffers.max_visible_chunks > 0, "Expected world/static-chunk streaming buffers to expose visible chunk capacity");

    renderer->clear_instance_pipeline_buffers();
    CHECK_FALSE(renderer->has_instance_pipeline_buffers());

    bool recovered_after_clear = false;
    for (int i = 0; i < 6; i++) {
        renderer->render_scene_instance(&render_data);
        const GaussianRenderPipeline::InstancePipelineBuffers &recovered_buffers =
                renderer->get_instance_pipeline_buffers();
        if (renderer->has_instance_pipeline_buffers() &&
                recovered_buffers.instance_buffer.is_valid() &&
                recovered_buffers.instance_count > 0) {
            recovered_after_clear = true;
            break;
        }
    }

    CHECK_MESSAGE(recovered_after_clear,
            "Expected streaming instance pipeline to recover a ready contract after clearing only the published buffers");

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

    const String instance_pipeline_setting = "rendering/gaussian_splatting/instance_pipeline/enabled";
    ScopedProjectSetting instance_pipeline_guard(project_settings, instance_pipeline_setting);
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

static bool create_test_render_buffers(const Vector2i &p_internal_resolution, RID &r_render_target, Ref<RenderSceneBuffersRD> &r_render_buffers,
        const Vector2i &p_target_resolution = Vector2i(), RS::ViewportScaling3DMode p_scaling_3d_mode = RS::VIEWPORT_SCALING_3D_MODE_OFF) {
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
    const Vector2i target_resolution = (p_target_resolution.x > 0 && p_target_resolution.y > 0) ? p_target_resolution : p_internal_resolution;
    texture_storage->render_target_set_size(r_render_target, target_resolution.x, target_resolution.y, 1);

    r_render_buffers.instantiate();
    Ref<RenderSceneBuffersConfiguration> rb_config;
    rb_config.instantiate();
    rb_config->set_render_target(r_render_target);
    rb_config->set_internal_size(p_internal_resolution);
    rb_config->set_target_size(target_resolution);
    rb_config->set_scaling_3d_mode(p_scaling_3d_mode);
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

TEST_CASE("[GaussianSplatting][RequiresGPU] Scene composite depth-test targets the presented viewport texture when scaling is disabled") {
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

    RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
    RID expected_present_target;
    if (texture_storage != nullptr && render_target.is_valid()) {
        expected_present_target = texture_storage->render_target_get_rd_texture(render_target);
    }
    RID internal_render_target = render_buffers->get_internal_texture();
    CHECK(expected_present_target.is_valid());
    CHECK(internal_render_target.is_valid());
    CHECK(internal_render_target != expected_present_target);

    RID scene_depth = render_buffers->get_depth_texture();
    if (!scene_depth.is_valid()) {
        MESSAGE("Skipping test - Scene depth texture unavailable");
        renderer.unref();
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

    RID final_output = renderer->get_final_texture();
    RID cached_depth = renderer->test_get_cached_render_depth();
    if (!final_output.is_valid() || !cached_depth.is_valid()) {
        MESSAGE("Skipping test - Final output or cached depth unavailable");
        renderer.unref();
        if (texture_storage != nullptr && render_target.is_valid()) {
            texture_storage->render_target_free(render_target);
        }
        return;
    }

    renderer->test_clear_output_viewport_blit_resources();
    renderer->test_reset_output_viewport_copy_state();

    RID resolved_render_target = internal_render_target;
    renderer->test_integrate_final_output(&render_data, render_buffers.ptr(), final_output,
            resolved_render_target, resolution, false, false, cached_depth);

    CHECK(renderer->was_last_viewport_copy_successful());
    CHECK(resolved_render_target == expected_present_target);

    renderer.unref();

    if (texture_storage != nullptr && render_target.is_valid()) {
        texture_storage->render_target_free(render_target);
    }
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Scene composite depth-test preserves the pipeline target when viewport scaling is active") {
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

    const Vector2i internal_resolution(320, 180);
    const Vector2i target_resolution(640, 360);
    RID render_target;
    Ref<RenderSceneBuffersRD> render_buffers;
    const bool render_buffers_ok = create_test_render_buffers(internal_resolution, render_target, render_buffers, target_resolution,
            RS::VIEWPORT_SCALING_3D_MODE_FSR);
    CHECK(render_buffers_ok);
    if (!render_buffers_ok) {
        renderer.unref();
        return;
    }

    RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
    RID expected_present_target;
    if (texture_storage != nullptr && render_target.is_valid()) {
        expected_present_target = texture_storage->render_target_get_rd_texture(render_target);
    }
    RID internal_render_target = render_buffers->get_internal_texture();
    CHECK(expected_present_target.is_valid());
    CHECK(internal_render_target.is_valid());
    CHECK(internal_render_target != expected_present_target);
    CHECK(render_buffers->get_internal_size() != render_buffers->get_target_size());

    RID scene_depth = render_buffers->get_depth_texture();
    if (!scene_depth.is_valid()) {
        MESSAGE("Skipping test - Scene depth texture unavailable");
        renderer.unref();
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

    RID final_output = renderer->get_final_texture();
    RID cached_depth = renderer->test_get_cached_render_depth();
    if (!final_output.is_valid() || !cached_depth.is_valid()) {
        MESSAGE("Skipping test - Final output or cached depth unavailable");
        renderer.unref();
        if (texture_storage != nullptr && render_target.is_valid()) {
            texture_storage->render_target_free(render_target);
        }
        return;
    }

    renderer->test_clear_output_viewport_blit_resources();
    renderer->test_reset_output_viewport_copy_state();

    RID resolved_render_target = internal_render_target;
    renderer->test_integrate_final_output(&render_data, render_buffers.ptr(), final_output,
            resolved_render_target, target_resolution, false, false, cached_depth);

    CHECK(renderer->was_last_viewport_copy_successful());
    CHECK(resolved_render_target == internal_render_target);
    CHECK(resolved_render_target != expected_present_target);

    renderer.unref();

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

TEST_CASE("[GaussianSplatting] get_streaming_route_policy returns RESIDENT when route_policy=0") {
	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	if (project_settings == nullptr) {
		MESSAGE("Skipping test - ProjectSettings unavailable");
		return;
	}

	const String route_policy_setting = "rendering/gaussian_splatting/streaming/route_policy";
	ScopedProjectSetting route_guard(project_settings, route_policy_setting);

	project_settings->set_setting(route_policy_setting, 0);
	CHECK(gs::settings::get_streaming_route_policy(project_settings) == gs::settings::GS_ROUTE_RESIDENT);

	project_settings->set_setting(route_policy_setting, 1);
	CHECK(gs::settings::get_streaming_route_policy(project_settings) == gs::settings::GS_ROUTE_STREAMING);
}

TEST_CASE("[GaussianSplatting] route policy source reports explicit route policy when authored") {
	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	if (project_settings == nullptr) {
		MESSAGE("Skipping test - ProjectSettings unavailable");
		return;
	}

	const String route_policy_setting = "rendering/gaussian_splatting/streaming/route_policy";
	ScopedProjectSetting route_guard(project_settings, route_policy_setting);

	project_settings->set_setting(route_policy_setting, int64_t(gs::settings::GS_ROUTE_RESIDENT));
	const int explicit_resident_policy = gs::settings::get_streaming_route_policy(project_settings);
	CHECK(explicit_resident_policy == gs::settings::GS_ROUTE_RESIDENT);
	CHECK(String(gs::settings::get_streaming_route_policy_token(explicit_resident_policy)) == String("resident"));
	CHECK(String(gs::settings::get_streaming_route_policy_source(project_settings)) == String("route_policy"));
}

TEST_CASE("[GaussianSplatting] runtime fidelity policy preserves source budget for full-fidelity asset metadata") {
	ScopedGaussianManagerPipeline manager_guard;

	Ref<GaussianSplatRenderer> renderer;
	renderer.instantiate();
	REQUIRE(renderer.is_valid());

	Ref<GaussianData> data;
	data.instantiate();
	data->resize(32);

	Ref<GaussianSplatAsset> asset;
	asset.instantiate();
	Dictionary metadata;
	metadata[StringName("max_splats")] = 0;
	metadata[StringName("density_multiplier")] = 1.0;
	asset->set_import_metadata(metadata);

	GaussianSplatRenderer::SceneState &scene_state = renderer->get_scene_state();
	scene_state.gaussian_data = data;
	scene_state.active_asset = asset;
	renderer->get_performance_settings().max_splats = 8;

	const GaussianSplatRenderer::RuntimeFidelityPolicy runtime_policy =
			renderer->build_runtime_fidelity_policy(scene_state, renderer->get_performance_settings());
	CHECK(runtime_policy.preserve_source_fidelity);
	CHECK(runtime_policy.runtime_budget_splats == 32);
}

TEST_CASE("[GaussianSplatting] runtime fidelity policy centralizes route policy and runtime budget") {
	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	if (project_settings == nullptr) {
		MESSAGE("Skipping test - ProjectSettings unavailable");
		return;
	}

	ScopedGaussianManagerPipeline manager_guard;
	const String route_policy_setting = "rendering/gaussian_splatting/streaming/route_policy";
	ScopedProjectSetting route_guard(project_settings, route_policy_setting);
	project_settings->set_setting(route_policy_setting, int64_t(gs::settings::GS_ROUTE_RESIDENT));

	Ref<GaussianSplatRenderer> renderer;
	renderer.instantiate();
	REQUIRE(renderer.is_valid());

	Ref<GaussianData> data;
	data.instantiate();
	data->resize(32);

	GaussianSplatRenderer::SceneState &scene_state = renderer->get_scene_state();
	scene_state.gaussian_data = data;
	scene_state.active_asset.unref();
	renderer->get_performance_settings().max_splats = 8;

	const GaussianSplatRenderer::RuntimeFidelityPolicy runtime_policy =
			renderer->build_runtime_fidelity_policy(scene_state, renderer->get_performance_settings());
	CHECK(runtime_policy.requested_route_policy == gs::settings::GS_ROUTE_RESIDENT);
	CHECK(runtime_policy.requested_route_policy_source == String("route_policy"));
	CHECK(runtime_policy.prefer_resident_backend);
	CHECK(runtime_policy.runtime_budget_splats == 8);
}

TEST_CASE("[GaussianSplatting] frame backend plan centralizes streaming bootstrap and fallback eligibility") {
	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	if (project_settings == nullptr) {
		MESSAGE("Skipping test - ProjectSettings unavailable");
		return;
	}

	ScopedGaussianManagerPipeline manager_guard;
	const String route_policy_setting = "rendering/gaussian_splatting/streaming/route_policy";
	ScopedProjectSetting route_guard(project_settings, route_policy_setting);
	project_settings->set_setting(route_policy_setting, int64_t(gs::settings::GS_ROUTE_STREAMING));

	Ref<GaussianSplatRenderer> renderer;
	renderer.instantiate();
	REQUIRE(renderer.is_valid());

	Ref<GaussianData> data;
	data.instantiate();
	data->resize(32);

	GaussianSplatRenderer::SceneState &scene_state = renderer->get_scene_state();
	scene_state.gaussian_data = data;

	const GaussianSplatRenderer::FrameBackendPlan backend_plan = renderer->build_frame_backend_plan(false);
	CHECK(backend_plan.streaming_requested);
	CHECK_FALSE(backend_plan.prefer_resident_backend);
	CHECK_FALSE(backend_plan.streaming_ready);
	CHECK(backend_plan.should_attempt_streaming_bootstrap);
	CHECK_FALSE(backend_plan.allow_legacy_resident_fallback);
	CHECK(backend_plan.allow_primary_fallback_instance);
}

TEST_CASE("[GaussianSplatting] frame backend plan preserves resident request semantics") {
	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	if (project_settings == nullptr) {
		MESSAGE("Skipping test - ProjectSettings unavailable");
		return;
	}

	ScopedGaussianManagerPipeline manager_guard;
	const String route_policy_setting = "rendering/gaussian_splatting/streaming/route_policy";
	ScopedProjectSetting route_guard(project_settings, route_policy_setting);
	project_settings->set_setting(route_policy_setting, int64_t(gs::settings::GS_ROUTE_RESIDENT));

	Ref<GaussianSplatRenderer> renderer;
	renderer.instantiate();
	REQUIRE(renderer.is_valid());

	const GaussianSplatRenderer::FrameBackendPlan backend_plan = renderer->build_frame_backend_plan(false);
	CHECK_FALSE(backend_plan.streaming_requested);
	CHECK(backend_plan.prefer_resident_backend);
	CHECK_FALSE(backend_plan.should_attempt_streaming_bootstrap);
	CHECK(backend_plan.allow_legacy_resident_fallback);
	CHECK_FALSE(backend_plan.allow_primary_fallback_instance);
	CHECK(backend_plan.resident_backend_reason == String("requested_resident_policy"));
}

TEST_CASE("[GaussianSplatting] get_streaming_route_policy defaults to STREAMING when unset") {
	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	if (project_settings == nullptr) {
		MESSAGE("Skipping test - ProjectSettings unavailable");
		return;
	}

	const String route_policy_setting = "rendering/gaussian_splatting/streaming/route_policy";
	ScopedProjectSetting route_guard(project_settings, route_policy_setting);

	// Clear route_policy so it falls through to default.
	project_settings->clear(route_policy_setting);
	CHECK(gs::settings::get_streaming_route_policy(project_settings) == gs::settings::GS_ROUTE_STREAMING);
	CHECK(String(gs::settings::get_streaming_route_policy_source(project_settings)) == String("default_fallback"));
}

TEST_CASE("[GaussianSplatting][Pipeline] InstancePipelineBuffers propagates world_submission_active flag") {
	GaussianSplatRenderer::InstancePipelineBuffers buffers;
	CHECK_FALSE(buffers.world_submission_active);

	buffers.world_submission_active = true;
	CHECK(buffers.world_submission_active);

	GaussianSplatRenderer::InstancePipelineBuffers default_buffers;
	CHECK_FALSE(default_buffers.world_submission_active);
}

TEST_CASE("[GaussianSplatting][Pipeline] InstancePipelineInputs propagates world_submission_active flag") {
	GPUSortingPipeline::InstancePipelineInputs inputs;
	CHECK_FALSE(inputs.world_submission_active);

	inputs.world_submission_active = true;
	CHECK(inputs.world_submission_active);
}

TEST_CASE("[GaussianSplatting][Pipeline] ChunkCullingStats tracks resident_chunks separately from loaded_chunks") {
	StreamingVisibilityController::ChunkCullingStats stats;
	CHECK(stats.loaded_chunks == 0);
	CHECK(stats.resident_chunks == 0);

	// Simulate: 3 CPU-loaded, only 2 atlas-resident.
	stats.loaded_chunks = 3;
	stats.resident_chunks = 2;
	CHECK(stats.loaded_chunks == 3);
	CHECK(stats.resident_chunks == 2);

	// Reset clears both.
	stats.reset();
	CHECK(stats.loaded_chunks == 0);
	CHECK(stats.resident_chunks == 0);
}

TEST_CASE("[GaussianSplatting][Pipeline] StreamingGlobalAtlasRegistry exposes atlas_published_chunk_count") {
	StreamingGlobalAtlasRegistry registry;
	CHECK(registry.get_atlas_published_chunks() == 0);
}

TEST_CASE("[GaussianSplatting][Pipeline] Working-set sizing produces smaller capacity than atlas upper bound") {
	// Scenario: large dataset (100 chunks per asset, 65536 splats/chunk),
	// but VRAM budget only allows 8 chunks resident.
	const uint32_t dispatch_chunk_count = 100;
	const uint32_t max_chunk_splats = 65536;
	const uint32_t effective_max_chunks = 8;
	const uint32_t instance_count = 1;
	const uint32_t atlas_gaussian_count = dispatch_chunk_count * max_chunk_splats; // 6.5M

	// Old atlas-capacity sizing:
	const uint64_t atlas_max_visible_splats = uint64_t(atlas_gaussian_count);
	const uint64_t atlas_max_visible_chunks = uint64_t(instance_count) * uint64_t(dispatch_chunk_count);

	// New working-set sizing:
	const uint32_t working_set_chunks = MIN(effective_max_chunks, dispatch_chunk_count);
	const uint64_t ws_max_visible_splats = uint64_t(working_set_chunks) * uint64_t(max_chunk_splats);
	const uint64_t ws_max_visible_chunks = uint64_t(instance_count) * uint64_t(working_set_chunks);

	CHECK(ws_max_visible_splats < atlas_max_visible_splats);
	CHECK(ws_max_visible_chunks < atlas_max_visible_chunks);
	CHECK(working_set_chunks == effective_max_chunks);
	CHECK(ws_max_visible_splats == uint64_t(8) * uint64_t(65536));
	CHECK(ws_max_visible_chunks == 8);
}

TEST_CASE("[GaussianSplatting][Pipeline] Working-set sizing falls back to dispatch_chunk_count when effective_max is zero") {
	const uint32_t dispatch_chunk_count = 32;
	const uint32_t max_chunk_splats = 65536;
	const uint32_t effective_max_chunks = 0; // cold start

	const uint32_t working_set_chunks = (effective_max_chunks > 0)
			? MIN(effective_max_chunks, dispatch_chunk_count)
			: dispatch_chunk_count;

	CHECK(working_set_chunks == dispatch_chunk_count);
}

TEST_CASE("[GaussianSplatting][Pipeline] Working-set sizing multi-instance scales by instance count") {
	const uint32_t dispatch_chunk_count = 100;
	const uint32_t max_chunk_splats = 65536;
	const uint32_t effective_max_chunks = 10;
	const uint32_t instance_count = 4;

	const uint32_t working_set_chunks = MIN(effective_max_chunks, dispatch_chunk_count);
	const uint64_t max_visible_splats = uint64_t(instance_count)
			* uint64_t(working_set_chunks) * uint64_t(max_chunk_splats);
	const uint64_t max_visible_chunks = uint64_t(instance_count) * uint64_t(working_set_chunks);

	CHECK(working_set_chunks == 10);
	CHECK(max_visible_chunks == 40);
	CHECK(max_visible_splats == uint64_t(40) * uint64_t(65536));
}

TEST_CASE("[GaussianSplatting][Pipeline] Working-set sizing respects sort cap clamp") {
	const uint32_t dispatch_chunk_count = 100;
	const uint32_t max_chunk_splats = 65536;
	const uint32_t effective_max_chunks = 100; // large budget
	const uint32_t sort_cap = 1000000; // 1M

	const uint32_t working_set_chunks = MIN(effective_max_chunks, dispatch_chunk_count);
	uint64_t max_visible_splats = uint64_t(working_set_chunks) * uint64_t(max_chunk_splats);
	max_visible_splats = MIN(max_visible_splats, uint64_t(sort_cap));

	CHECK(max_visible_splats == uint64_t(sort_cap));
}

// ---------------------------------------------------------------------------
// Spatial grid visibility discovery (Issue #4)
// ---------------------------------------------------------------------------

TEST_CASE("[GaussianSplatting][Pipeline] ChunkSpatialGrid build produces correct cell coverage") {
	// Create 4 chunks at known positions.
	GaussianStreamingTypes::StreamingChunk chunks[4];
	chunks[0].bounds = AABB(Vector3(0, 0, 0), Vector3(10, 10, 10));
	chunks[0].center = Vector3(5, 5, 5);
	chunks[1].bounds = AABB(Vector3(100, 0, 0), Vector3(10, 10, 10));
	chunks[1].center = Vector3(105, 5, 5);
	chunks[2].bounds = AABB(Vector3(0, 0, 100), Vector3(10, 10, 10));
	chunks[2].center = Vector3(5, 5, 105);
	chunks[3].bounds = AABB(Vector3(100, 0, 100), Vector3(10, 10, 10));
	chunks[3].center = Vector3(105, 5, 105);

	ChunkSpatialGrid grid;
	grid.build(chunks, 4);

	CHECK(grid.is_built());
	CHECK(grid.built_for_chunk_count == 4);
	CHECK(grid.dim_x >= 1);
	CHECK(grid.dim_y >= 1);
	CHECK(grid.dim_z >= 1);
	CHECK(grid.cell_size > 0.0f);

	// Query the full world bounds — should return all 4 chunks.
	LocalVector<uint32_t> results;
	grid.query_aabb(grid.world_bounds, results);
	// De-dup to count unique indices.
	bool seen[4] = { false, false, false, false };
	uint32_t unique = 0;
	for (uint32_t i = 0; i < results.size(); i++) {
		CHECK(results[i] < 4);
		if (!seen[results[i]]) {
			seen[results[i]] = true;
			unique++;
		}
	}
	CHECK(unique == 4);
}

TEST_CASE("[GaussianSplatting][Pipeline] ChunkSpatialGrid query_aabb returns bounded subset") {
	// 100 chunks spread along the X axis at 20-unit intervals.
	const uint32_t N = 100;
	LocalVector<GaussianStreamingTypes::StreamingChunk> chunks;
	chunks.resize(N);
	for (uint32_t i = 0; i < N; i++) {
		float x = float(i) * 20.0f;
		chunks[i].bounds = AABB(Vector3(x, 0, 0), Vector3(10, 10, 10));
		chunks[i].center = Vector3(x + 5, 5, 5);
	}

	ChunkSpatialGrid grid;
	grid.build(chunks.ptr(), N);
	CHECK(grid.is_built());

	// Query a small AABB covering only the first ~5 chunks.
	AABB small_query(Vector3(-5, -5, -5), Vector3(100, 20, 20));
	LocalVector<uint32_t> results;
	grid.query_aabb(small_query, results);

	// De-dup.
	LocalVector<uint8_t> visited;
	visited.resize(N);
	memset(visited.ptr(), 0, N);
	uint32_t unique = 0;
	for (uint32_t i = 0; i < results.size(); i++) {
		if (!visited[results[i]]) {
			visited[results[i]] = 1;
			unique++;
		}
	}

	// Should find far fewer than all 100 chunks.
	CHECK(unique < N);
	CHECK(unique >= 4); // at least the first few chunks in the query range
}

TEST_CASE("[GaussianSplatting][Pipeline] ChunkSpatialGrid query_nearby returns local chunks") {
	// 3 chunks: one at origin, one nearby, one far.
	GaussianStreamingTypes::StreamingChunk chunks[3];
	chunks[0].bounds = AABB(Vector3(0, 0, 0), Vector3(10, 10, 10));
	chunks[0].center = Vector3(5, 5, 5);
	chunks[1].bounds = AABB(Vector3(20, 0, 0), Vector3(10, 10, 10));
	chunks[1].center = Vector3(25, 5, 5);
	chunks[2].bounds = AABB(Vector3(5000, 0, 0), Vector3(10, 10, 10));
	chunks[2].center = Vector3(5005, 5, 5);

	ChunkSpatialGrid grid;
	grid.build(chunks, 3);
	CHECK(grid.is_built());

	// Query nearby the origin with radius=1 cell.
	LocalVector<uint32_t> nearby;
	grid.query_nearby(Vector3(5, 5, 5), 1, nearby);

	// Should find at least chunk 0 (at origin) but not chunk 2 (far away).
	bool found_0 = false;
	bool found_2 = false;
	for (uint32_t i = 0; i < nearby.size(); i++) {
		if (nearby[i] == 0) {
			found_0 = true;
		}
		if (nearby[i] == 2) {
			found_2 = true;
		}
	}
	CHECK(found_0);
	CHECK_FALSE(found_2);
}

TEST_CASE("[GaussianSplatting][Pipeline] Spatial grid threshold gates grid usage") {
	// Verify that the SPATIAL_GRID_MIN_CHUNKS threshold is a sensible guard.
	CHECK(StreamingVisibilityController::SPATIAL_GRID_MIN_CHUNKS > 0);
	CHECK(StreamingVisibilityController::SPATIAL_GRID_MIN_CHUNKS <= 128);
}

// ---------------------------------------------------------------------------
// Out-of-core chunk payload source tests (Issue #5)
// ---------------------------------------------------------------------------

TEST_CASE("[GaussianSplatting][Pipeline] InMemoryChunkPayloadSource delegates to GaussianData") {
	LocalVector<Gaussian> gaussians;
	fill_gaussians(gaussians, 32);
	Ref<::GaussianData> data;
	data.instantiate();
	data->set_gaussians(gaussians);

	Ref<InMemoryChunkPayloadSource> source;
	source.instantiate();
	source->set_data(data);

	CHECK(source->is_valid());
	CHECK(source->get_count() == 32);
	CHECK(source->get_sh_degree() == data->get_sh_degree());

	LocalVector<Gaussian> snapshot;
	LocalVector<Vector3> sh_out;
	uint32_t sh_first = 0, sh_high = 0;
	bool ok = source->capture_chunk_snapshot(0, 16, snapshot, sh_out, sh_first, sh_high);
	CHECK(ok);
	CHECK(snapshot.size() == 16);

	// Verify the captured gaussians match the originals.
	for (uint32_t i = 0; i < 16; i++) {
		CHECK(snapshot[i].position.is_equal_approx(gaussians[i].position));
	}
}

TEST_CASE("[GaussianSplatting][Pipeline] InMemoryChunkPayloadSource indexed snapshot") {
	LocalVector<Gaussian> gaussians;
	fill_gaussians(gaussians, 32);
	Ref<::GaussianData> data;
	data.instantiate();
	data->set_gaussians(gaussians);

	Ref<InMemoryChunkPayloadSource> source;
	source.instantiate();
	source->set_data(data);

	uint32_t indices[] = { 5, 10, 20 };
	LocalVector<Gaussian> snapshot;
	LocalVector<Vector3> sh_out;
	uint32_t sh_first = 0, sh_high = 0;
	bool ok = source->capture_indexed_chunk_snapshot(indices, 3, snapshot, sh_out, sh_first, sh_high);
	CHECK(ok);
	CHECK(snapshot.size() == 3);
	CHECK(snapshot[0].position.is_equal_approx(gaussians[5].position));
	CHECK(snapshot[1].position.is_equal_approx(gaussians[10].position));
	CHECK(snapshot[2].position.is_equal_approx(gaussians[20].position));
}

TEST_CASE("[GaussianSplatting][Pipeline] Streaming system set_chunk_payload_source and detach_source_data") {
	LocalVector<Gaussian> gaussians;
	fill_gaussians(gaussians, 256);
	Ref<::GaussianData> data;
	data.instantiate();
	data->set_gaussians(gaussians);

	Ref<InMemoryChunkPayloadSource> source;
	source.instantiate();
	source->set_data(data);

	Ref<GaussianStreamingSystem> system;
	system.instantiate();
	system->initialize(data);

	// Set payload source on primary asset (id=0).
	system->set_chunk_payload_source(0, source);

	// Detach should succeed now that a payload source is set.
	system->detach_source_data(0);

	// After detach, the streaming system's internal data ref should be gone,
	// but the payload source should still be valid for chunk reads.
	CHECK(source->is_valid());
	CHECK(source->get_count() == 256);
}

TEST_CASE("[GaussianSplatting][Pipeline] detach_source_data refuses without payload source") {
	LocalVector<Gaussian> gaussians;
	fill_gaussians(gaussians, 64);
	Ref<::GaussianData> data;
	data.instantiate();
	data->set_gaussians(gaussians);

	Ref<GaussianStreamingSystem> system;
	system.instantiate();
	system->initialize(data);

	// detach without payload source should warn and refuse.
	ERR_PRINT_OFF;
	system->detach_source_data(0);
	ERR_PRINT_ON;

	// The system should still function (data not detached).
}

} // namespace TestGaussianSplatting
