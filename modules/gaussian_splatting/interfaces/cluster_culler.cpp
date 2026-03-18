#include "cluster_culler.h"
#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "../compute/cluster_cull.glsl.gen.h"
#include "../compute/compute_infrastructure.h"
#include "../core/gaussian_splat_manager.h"
#include "../logger/gs_logger.h"
#include "../renderer/pipeline_io_contracts.h"
#include <cstring>

namespace {

// GPU parameter struct - must match shader layout
struct ClusterCullParamsGPU {
    float view_matrix[16];
    float proj_matrix[16];
    float frustum_planes[6][4];
    float camera_position[3];
    float frustum_plane_slack;
    uint32_t total_clusters;
    uint32_t fine_cull_workgroup_size;
    uint32_t pad0;
    uint32_t pad1;
};

static_assert(sizeof(ClusterCullParamsGPU) == 256, "ClusterCullParamsGPU size must match shader");

static ClusterCullShaderRD &get_cluster_cull_shader_source() {
    static ClusterCullShaderRD cluster_cull_shader_source;
    static bool shader_initialized = false;

    if (!shader_initialized) {
        Vector<String> variants;
        variants.push_back("");
        cluster_cull_shader_source.initialize(variants);
        shader_initialized = true;
    }

    return cluster_cull_shader_source;
}

} // namespace

void ClusterCuller::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize", "device"), &ClusterCuller::initialize);
    ClassDB::bind_method(D_METHOD("shutdown"), &ClusterCuller::shutdown);
    ClassDB::bind_method(D_METHOD("is_ready"), &ClusterCuller::is_ready);
    ClassDB::bind_method(D_METHOD("invalidate"), &ClusterCuller::invalidate);
    ClassDB::bind_method(D_METHOD("get_cluster_count"), &ClusterCuller::get_cluster_count);
}

ClusterCuller::ClusterCuller() {
    cluster_builder = std::make_unique<GaussianSplatting::ClusterBuilder>();
}

ClusterCuller::~ClusterCuller() {
    shutdown();
}

Error ClusterCuller::initialize(RenderingDevice *p_device) {
    if (initialized) {
        return OK;
    }

    rd = p_device;
    if (!rd) {
        rd = RenderingDevice::get_singleton();
    }

    if (!rd) {
        GS_LOG_WARN_DEFAULT("[ClusterCuller] No RenderingDevice available");
        return ERR_UNAVAILABLE;
    }

    {
        const String stage_name = "ClusterCull.CapabilityGate";
        GaussianSplatting::ComputeInfrastructure::StageResult capability =
                GaussianSplatting::ComputeInfrastructure::check_compute_capabilities(rd, stage_name);
        if (!capability.ok()) {
            GaussianSplatting::ComputeInfrastructure::CapabilityGatePolicy fallback_policy;
            fallback_policy.allow_cpu_fallback = false;
            fallback_policy.allow_retry = false;
            GaussianSplatting::ComputeInfrastructure::FallbackDecision fallback =
                    GaussianSplatting::ComputeInfrastructure::resolve_fallback(capability, fallback_policy);
            last_compute_error = GaussianSplatting::ComputeInfrastructure::format_stage_error("ClusterCuller", capability) +
                    vformat(" fallback=%s",
                            String(GaussianSplatting::ComputeInfrastructure::fallback_route_name(fallback.route)));
            GS_LOG_WARN_DEFAULT(last_compute_error);
            return capability.to_error();
        }
    }

    _ensure_shader(rd);
    if (!cluster_cull_shader.is_valid() || !cluster_cull_pipeline.is_valid()) {
        if (last_compute_error.is_empty()) {
            last_compute_error = "[ClusterCuller] Failed to create shader/pipeline";
        }
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return ERR_CANT_CREATE;
    }

    // Load config from project settings
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        if (ps->has_setting("rendering/gaussian_splatting/culling/cluster_culling_enabled")) {
            config.enabled = ps->get_setting("rendering/gaussian_splatting/culling/cluster_culling_enabled");
        }
        if (ps->has_setting("rendering/gaussian_splatting/culling/cluster_target_size")) {
            config.target_cluster_size = ps->get_setting("rendering/gaussian_splatting/culling/cluster_target_size");
        }
    }

    last_compute_error = String();
    initialized = true;
    GS_LOG_RENDERER_INFO("[ClusterCuller] Initialized successfully");
    return OK;
}

void ClusterCuller::shutdown() {
    _release_resources();
    rd = nullptr;
    initialized = false;
}

bool ClusterCuller::is_ready() const {
    return initialized && cluster_cull_shader.is_valid() && cluster_cull_pipeline.is_valid();
}

void ClusterCuller::_ensure_shader(RenderingDevice *p_device) {
    if (!p_device || cluster_cull_shader.is_valid()) {
        return;
    }

    // Create shader from generated source.
    ClusterCullShaderRD &cluster_cull_shader_source = get_cluster_cull_shader_source();

    RID shader_version = cluster_cull_shader_source.version_create();
    if (!shader_version.is_valid()) {
        last_compute_error = "[ClusterCuller] Failed to create shader version";
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return;
    }
    cluster_cull_shader_version = shader_version;

    cluster_cull_shader = cluster_cull_shader_source.version_get_shader(shader_version, 0);
    if (!cluster_cull_shader.is_valid()) {
        cluster_cull_shader_source.version_free(cluster_cull_shader_version);
        cluster_cull_shader_version = RID();
        last_compute_error = "[ClusterCuller] Failed to get compiled shader";
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return;
    }

    GaussianSplatting::ComputeInfrastructure::StageResult pipeline_result =
            GaussianSplatting::ComputeInfrastructure::create_pipeline_checked(
                    p_device, cluster_cull_shader, "ClusterCull.PipelineCreate", cluster_cull_pipeline);
    if (!pipeline_result.ok()) {
        cluster_cull_shader_source.version_free(cluster_cull_shader_version);
        cluster_cull_shader_version = RID();
        last_compute_error = GaussianSplatting::ComputeInfrastructure::format_stage_error("ClusterCuller", pipeline_result);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        cluster_cull_shader = RID();
        return;
    }

    resource_device = p_device;
}

void ClusterCuller::_ensure_buffers(RenderingDevice *p_device, uint32_t p_cluster_count) {
    if (!p_device) {
        return;
    }

    // Check if we need to recreate buffers
    if (resource_device && resource_device != p_device) {
        _release_resources();
    }

    uint32_t required_cluster_capacity = MAX(p_cluster_count, 1u);

    if (buffer_cluster_capacity >= required_cluster_capacity &&
        cluster_buffer.is_valid() &&
        cluster_visibility_buffer.is_valid()) {
        return;
    }

    // Free existing buffers
    if (cluster_buffer.is_valid() && resource_device) {
        resource_device->free(cluster_buffer);
    }
    if (cluster_visibility_buffer.is_valid() && resource_device) {
        resource_device->free(cluster_visibility_buffer);
    }
    if (visible_cluster_buffer.is_valid() && resource_device) {
        resource_device->free(visible_cluster_buffer);
    }
    if (indirect_dispatch_buffer.is_valid() && resource_device) {
        resource_device->free(indirect_dispatch_buffer);
    }

    buffer_cluster_capacity = required_cluster_capacity;

    // Cluster AABB buffer (32 bytes per cluster)
    Vector<uint8_t> cluster_init;
    cluster_init.resize(buffer_cluster_capacity * 32);
    cluster_buffer = p_device->storage_buffer_create(cluster_init.size(), cluster_init);
    p_device->set_resource_name(cluster_buffer, "GS_ClusterCuller_ClusterAABBBuffer");

    // Visibility bitmask (1 bit per cluster, rounded up to uint32).
    // Cleared on the GPU each cull pass, so skip CPU-side zeroing here.
    uint32_t visibility_words = (buffer_cluster_capacity + 31) / 32;
    uint32_t visibility_bytes = visibility_words * sizeof(uint32_t);
    cluster_visibility_buffer = p_device->storage_buffer_create(visibility_bytes, Vector<uint8_t>());
    p_device->set_resource_name(cluster_visibility_buffer, "GS_ClusterCuller_VisibilityBitmask");

    // Visible cluster indices buffer
    Vector<uint8_t> visible_init;
    visible_init.resize(buffer_cluster_capacity * sizeof(uint32_t));
    visible_cluster_buffer = p_device->storage_buffer_create(visible_init.size(), visible_init);
    p_device->set_resource_name(visible_cluster_buffer, "GS_ClusterCuller_VisibleClusterIndices");

    // Indirect dispatch buffer (6 uint32)
    Vector<uint8_t> indirect_init;
    indirect_init.resize(sizeof(GaussianSplatting::ClusterCullIndirectDispatchLayout));
    memset(indirect_init.ptrw(), 0, indirect_init.size());
    indirect_dispatch_buffer = p_device->storage_buffer_create(indirect_init.size(), indirect_init);
    p_device->set_resource_name(indirect_dispatch_buffer, "GS_ClusterCuller_IndirectDispatch");

    // Param uniform buffer
    if (!param_buffer.is_valid()) {
        Vector<uint8_t> param_init;
        param_init.resize(sizeof(ClusterCullParamsGPU));
        param_buffer = p_device->uniform_buffer_create(param_init.size(), param_init);
        p_device->set_resource_name(param_buffer, "GS_ClusterCuller_ParamsUniform");
    }

    resource_device = p_device;
}

void ClusterCuller::_release_resources() {
    // Check if resource_device is still valid by comparing with the current shared device
    RenderingDevice *current_shared = nullptr;
    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        current_shared = manager->get_shared_submission_device();
    }
    bool device_still_valid = (resource_device != nullptr) && (resource_device == current_shared);

    if (device_still_valid) {
        // Device is still valid - properly free resources to avoid leaks
#define SAFE_FREE(buf) \
    do { \
        if (buf.is_valid()) { \
            resource_device->free(buf); \
            buf = RID(); \
        } \
    } while (0)

        // Free compute pipeline (must be freed before shader)
        if (cluster_cull_pipeline.is_valid()) {
            if (resource_device->compute_pipeline_is_valid(cluster_cull_pipeline)) {
                resource_device->free(cluster_cull_pipeline);
            }
            cluster_cull_pipeline = RID();
        }
        if (cluster_cull_shader_version.is_valid()) {
            ClusterCullShaderRD &cluster_cull_shader_source = get_cluster_cull_shader_source();
            cluster_cull_shader_source.version_free(cluster_cull_shader_version);
            cluster_cull_shader_version = RID();
        }
        // cluster_cull_shader originates from ClusterCullShaderRD::version_get_shader().
        // Lifetime is tied to the shader version and released via version_free() above.
        cluster_cull_shader = RID();
        // Free storage buffers
        SAFE_FREE(cluster_buffer);
        SAFE_FREE(cluster_visibility_buffer);
        SAFE_FREE(visible_cluster_buffer);
        SAFE_FREE(indirect_dispatch_buffer);
        SAFE_FREE(sorted_order_buffer);
        // Free uniform buffer
        SAFE_FREE(param_buffer);

#undef SAFE_FREE
    } else {
        // Device may be gone or different - just invalidate RIDs without freeing
        // Resources will be cleaned up when the owning device is destroyed
        cluster_cull_shader_version = RID();
        cluster_cull_shader = RID();
        cluster_cull_pipeline = RID();
        cluster_buffer = RID();
        cluster_visibility_buffer = RID();
        visible_cluster_buffer = RID();
        indirect_dispatch_buffer = RID();
        param_buffer = RID();
        sorted_order_buffer = RID();
    }

    buffer_cluster_capacity = 0;
    buffer_splat_capacity = 0;
    resource_device = nullptr;
}

bool ClusterCuller::build_clusters(const LocalVector<Gaussian> &p_gaussians, bool p_force_rebuild) {
    if (p_gaussians.is_empty()) {
        cluster_data = GaussianSplatting::ClusterBuildResult();
        cluster_count = 0;
        clusters_dirty = false;
        return false;
    }

    // Check if rebuild is needed
    bool needs_rebuild = p_force_rebuild || clusters_dirty || last_splat_count != p_gaussians.size();

    if (!needs_rebuild && !cluster_data.clusters.is_empty()) {
        return false; // No rebuild needed
    }

    // Build clusters
    GaussianSplatting::ClusterBuildParams params;
    params.target_cluster_size = config.target_cluster_size;
    params.min_cluster_size = config.min_cluster_size;
    params.max_cluster_size = config.max_cluster_size;
    params.use_morton_order = config.use_morton_order;
    params.compute_importance = true;

    cluster_data = cluster_builder->build_clusters(p_gaussians, params);
    cluster_count = cluster_data.clusters.size();
    last_splat_count = p_gaussians.size();
    clusters_dirty = false;

    return true;
}

bool ClusterCuller::upload_clusters_to_gpu() {
    if (!is_ready() || cluster_data.clusters.is_empty()) {
        return false;
    }

    _ensure_buffers(rd, cluster_data.clusters.size());

    if (!cluster_buffer.is_valid()) {
        return false;
    }

    // Pack cluster data for GPU
    Vector<uint8_t> packed = cluster_builder->pack_for_gpu(cluster_data.clusters);
    rd->buffer_update(cluster_buffer, 0, packed.size(), packed.ptr());

    // Upload sorted splat order
    if (buffer_splat_capacity < cluster_data.sorted_splat_order.size()) {
        if (sorted_order_buffer.is_valid() && resource_device) {
            resource_device->free(sorted_order_buffer);
        }

        buffer_splat_capacity = cluster_data.sorted_splat_order.size();
        Vector<uint8_t> order_init;
        order_init.resize(buffer_splat_capacity * sizeof(uint32_t));
        sorted_order_buffer = rd->storage_buffer_create(order_init.size(), order_init);
        rd->set_resource_name(sorted_order_buffer, "GS_ClusterCuller_SortedSplatOrder");
    }

    if (sorted_order_buffer.is_valid() && !cluster_data.sorted_splat_order.is_empty()) {
        rd->buffer_update(sorted_order_buffer, 0,
            cluster_data.sorted_splat_order.size() * sizeof(uint32_t),
            cluster_data.sorted_splat_order.ptr());
    }

    return true;
}

bool ClusterCuller::cull_clusters(const CullParams &p_params) {
    if (!is_ready() || cluster_count == 0) {
        return false;
    }

    if (!config.enabled) {
        // Cluster culling disabled - all clusters visible
        last_stats = ClusterCullStats();
        last_stats.total_clusters = cluster_count;
        last_stats.visible_clusters = cluster_count;
        return true;
    }

    uint64_t start_time = OS::get_singleton()->get_ticks_usec();

    // Build GPU parameters
    ClusterCullParamsGPU params = {};

    // View matrix
    Transform3D view_matrix = p_params.world_to_camera_transform;
    for (int col = 0; col < 3; col++) {
        Vector3 column_vec = view_matrix.basis.get_column(col);
        params.view_matrix[col * 4 + 0] = column_vec.x;
        params.view_matrix[col * 4 + 1] = column_vec.y;
        params.view_matrix[col * 4 + 2] = column_vec.z;
        params.view_matrix[col * 4 + 3] = 0.0f;
    }
    params.view_matrix[12] = view_matrix.origin.x;
    params.view_matrix[13] = view_matrix.origin.y;
    params.view_matrix[14] = view_matrix.origin.z;
    params.view_matrix[15] = 1.0f;

    // Projection matrix
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            params.proj_matrix[col * 4 + row] = p_params.projection.columns[col][row];
        }
    }

    // Frustum planes
    for (uint32_t i = 0; i < 6 && i < (uint32_t)p_params.frustum_planes.size(); i++) {
        const Plane &plane = p_params.frustum_planes[i];
        params.frustum_planes[i][0] = plane.normal.x;
        params.frustum_planes[i][1] = plane.normal.y;
        params.frustum_planes[i][2] = plane.normal.z;
        params.frustum_planes[i][3] = plane.d;
    }

    // Other parameters
    params.camera_position[0] = p_params.camera_position.x;
    params.camera_position[1] = p_params.camera_position.y;
    params.camera_position[2] = p_params.camera_position.z;
    params.frustum_plane_slack = config.frustum_slack;
    params.total_clusters = cluster_count;
    params.fine_cull_workgroup_size = 256; // Standard workgroup size

    // Upload parameters
    Vector<uint8_t> param_bytes;
    param_bytes.resize(sizeof(ClusterCullParamsGPU));
    memcpy(param_bytes.ptrw(), &params, sizeof(ClusterCullParamsGPU));
    rd->buffer_update(param_buffer, 0, param_bytes.size(), param_bytes.ptr());

    // Clear indirect dispatch buffer
    GaussianSplatting::ClusterCullIndirectDispatchLayout clear_args = {};
    clear_args.dispatch_y = 1;
    clear_args.dispatch_z = 0; // Used as counter during dispatch
    rd->buffer_update(indirect_dispatch_buffer, 0, sizeof(GaussianSplatting::ClusterCullIndirectDispatchLayout), &clear_args);

    // Clear visibility buffer
    uint32_t visibility_words = (cluster_count + 31) / 32;
    uint32_t visibility_bytes = visibility_words * sizeof(uint32_t);
    rd->buffer_clear(cluster_visibility_buffer, 0, visibility_bytes);

    uint32_t workgroup_size = 64;
    uint32_t group_count = (cluster_count + workgroup_size - 1) / workgroup_size;

    Vector<GaussianSplatting::ComputeInfrastructure::UniformBindingContract> bindings;
    auto append_binding = [&](RenderingDevice::UniformType p_type, uint32_t p_binding,
                                  const RID &p_resource, const char *p_label) {
        GaussianSplatting::ComputeInfrastructure::UniformBindingContract contract;
        contract.type = p_type;
        contract.binding = p_binding;
        contract.resource = p_resource;
        contract.label = p_label;
        bindings.push_back(contract);
    };
    append_binding(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, cluster_buffer, "cluster_buffer");
    append_binding(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, cluster_visibility_buffer, "cluster_visibility_buffer");
    append_binding(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, indirect_dispatch_buffer, "indirect_dispatch_buffer");
    append_binding(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, visible_cluster_buffer, "visible_cluster_buffer");
    append_binding(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 4, param_buffer, "param_buffer");

    GaussianSplatting::ComputeInfrastructure::StageValidationHarness validation_harness;
    GaussianSplatting::ComputeInfrastructure::StageValidationInput validation_input;
    validation_input.stage_name = "ClusterCull.Dispatch";
    validation_input.bindings = bindings;
    validation_input.validate_dispatch = true;
    validation_input.dispatch_x = group_count;
    validation_input.dispatch_y = 1;
    validation_input.dispatch_z = 1;
    GaussianSplatting::ComputeInfrastructure::StageResult contract_result = validation_harness.validate(validation_input);
    if (!contract_result.ok()) {
        last_compute_error = GaussianSplatting::ComputeInfrastructure::format_stage_error("ClusterCuller", contract_result);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }

    RID uniform_set;
    GaussianSplatting::ComputeInfrastructure::StageResult uniform_result =
            GaussianSplatting::ComputeInfrastructure::create_uniform_set_checked(
                    rd, cluster_cull_shader, 0, bindings, "ClusterCull.UniformSet",
                    "GS_ClusterCuller_CullUniformSet", uniform_set);
    if (!uniform_result.ok()) {
        last_compute_error = GaussianSplatting::ComputeInfrastructure::format_stage_error("ClusterCuller", uniform_result);
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }

    RD::ComputeListID compute_list = rd->compute_list_begin();
    if (compute_list == RD::INVALID_ID) {
        if (uniform_set.is_valid() && rd->uniform_set_is_valid(uniform_set)) {
            rd->free(uniform_set);
        }
        last_compute_error = "[ClusterCuller] compute_list_begin failed";
        GS_LOG_WARN_DEFAULT(last_compute_error);
        return false;
    }
    rd->compute_list_bind_compute_pipeline(compute_list, cluster_cull_pipeline);
    rd->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
    rd->compute_list_dispatch(compute_list, group_count, 1, 1);
    rd->compute_list_add_barrier(compute_list);
    rd->compute_list_end();

    // Free the temporary uniform set we just created (safe - same device, same frame)
    if (uniform_set.is_valid() && rd->uniform_set_is_valid(uniform_set)) {
        rd->free(uniform_set);
    }

	// Request async readback of stats
	if (!async_stats.pending) {
		Callable stats_cb = callable_mp(this, &ClusterCuller::_on_stats_readback);
		Error enqueue_err = rd->buffer_get_data_async(indirect_dispatch_buffer, stats_cb, 0,
                sizeof(GaussianSplatting::ClusterCullIndirectDispatchLayout));
		if (enqueue_err == OK) {
			async_stats.pending = true;
			async_stats.request_id++;
			async_stats.generation++;
		} else {
			GS_LOG_WARN_DEFAULT(vformat("[ClusterCuller] Failed to enqueue async stats readback (err=%d)", int(enqueue_err)));
		}
	}

    last_stats.cluster_cull_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;
    last_stats.total_clusters = cluster_count;
    last_compute_error = String();

    return true;
}

void ClusterCuller::_on_stats_readback(const Vector<uint8_t> &p_data) {
    async_stats.pending = false;

    if (p_data.size() < (int)sizeof(GaussianSplatting::ClusterCullIndirectDispatchLayout)) {
        return;
    }

    const auto *args = reinterpret_cast<const GaussianSplatting::ClusterCullIndirectDispatchLayout *>(p_data.ptr());
    last_stats.visible_clusters = args->visible_cluster_count;
    last_stats.visible_splats = args->visible_splat_count;
    last_stats.culled_clusters = args->clusters_culled;

    // Compute culled splat count (requires knowing total splats)
    if (last_stats.total_clusters > 0) {
        // Approximate based on average cluster size
        uint32_t avg_cluster_size = last_splat_count / MAX(last_stats.total_clusters, 1u);
        last_stats.culled_splats = last_stats.culled_clusters * avg_cluster_size;
    }

    // Compute cull ratio
    if (last_splat_count > 0) {
        last_stats.cluster_cull_ratio = float(last_stats.culled_splats) / float(last_splat_count) * 100.0f;
    }
}

void ClusterCuller::invalidate() {
    clusters_dirty = true;
}
