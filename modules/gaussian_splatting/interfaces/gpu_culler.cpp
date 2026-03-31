#include "gpu_culler.h"
#include "../core/gs_project_settings.h"
#include "../lod/lod_config.h"
#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/os/mutex.h"
#include "core/os/os.h"
#include "core/templates/hash_set.h"
#include "core/variant/variant.h"
#include "servers/rendering/renderer_scene_cull.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_manager.h"
#include "../renderer/gaussian_gpu_layout.h"
#include "../renderer/gpu_debug_utils.h"
#include "../interfaces/overflow_auto_tuner.h"
#include "../compute/frustum_cull.glsl.gen.h"
#include "../logger/gs_logger.h"
#include "../logger/gs_debug_trace.h"
#include <cstdint>
#include <cstring>

using GaussianSplatting::PassColors;
using GaussianSplatting::ScopedGpuMarker;
using GaussianSplatting::ScopedGpuMarkerEx;

// GPU parameter struct - must match shader layout
namespace {
constexpr uint32_t kFrustumPlaneCount = FrustumCullParamsGPU::kFrustumPlaneCount;

struct FrustumCullCounters {
    uint32_t visible_count = 0;
    uint32_t frustum_culled = 0;
    uint32_t distance_culled = 0;
    uint32_t screen_culled = 0;
    uint32_t importance_culled = 0;
    uint32_t clipped_count = 0;
    uint32_t near_clamped = 0;
    uint32_t behind_culled = 0;
};

static_assert(sizeof(FrustumCullCounters) == sizeof(uint32_t) * 8, "FrustumCullCounters must match shader layout");

struct FrustumShaderState {
    FrustumCullShaderRD source;
    bool initialized = false;
    int enabled_group = GPUCuller::SHADER_GROUP_STANDARD;
    bool last_debug_counters_enabled = false;
    Mutex mutex;
};

static FrustumShaderState &_get_frustum_shader_state() {
    static FrustumShaderState state;
    return state;
}

// Project settings helpers provided by gs_project_settings.h (gs::settings namespace).
static int _get_int_setting(ProjectSettings *ps, const StringName &name, int fallback) {
    return static_cast<int>(gs::settings::get_uint(ps, name, static_cast<uint32_t>(fallback)));
}

static float _get_float_setting(ProjectSettings *ps, const StringName &name, float fallback) {
    return gs::settings::get_float(ps, name, fallback);
}

static bool _get_bool_setting(ProjectSettings *ps, const StringName &name, bool fallback) {
    return gs::settings::get_bool(ps, name, fallback);
}

static bool _is_cull_debug_counters_enabled() {
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        const bool all_debug = gs::settings::is_all_debug_enabled(ps);
        const bool cull_debug = gs::settings::get_bool(ps, "rendering/gaussian_splatting/debug/enable_cull_counters", false);
        return all_debug || cull_debug;
    }
    return false;
}

} // namespace

void GPUCuller::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_readback_enabled", "enabled"), &GPUCuller::set_readback_enabled);
    ClassDB::bind_method(D_METHOD("is_readback_enabled"), &GPUCuller::is_readback_enabled);
}

GPUCuller::GPUCuller() {
}

void GPUCuller::set_instance_pipeline_inputs(const InstancePipelineInputs &p_inputs) {
    instance_inputs = p_inputs;
    instance_inputs_valid = true;
}

void GPUCuller::clear_instance_pipeline_inputs() {
    instance_inputs = InstancePipelineInputs();
    instance_inputs_valid = false;
    last_instance_visible_chunk_count = 0;
    instance_readback_state.pending = false;
    instance_readback_state.generation++;
    instance_readback_state.pending_request_id = 0;
    instance_readback_state.next_request_id = 1;
    instance_readback_state.last_applied_request_id = 0;
    instance_readback_state.last_frame_chunk_limit = UINT32_MAX;
}

GPUCuller::~GPUCuller() {
    shutdown();
}

Error GPUCuller::initialize(RenderingDevice *p_device) {
    if (initialized) {
        return OK;
    }

    rd = p_device;
    if (!rd) {
        rd = RenderingDevice::get_singleton();
    }

    if (!rd) {
        GS_LOG_WARN_DEFAULT("[GPUCuller] No RenderingDevice available");
        return ERR_UNAVAILABLE;
    }

    _ensure_shader(rd);
    if (!shader.is_valid() || !pipeline.is_valid()) {
        GS_LOG_WARN_DEFAULT("[GPUCuller] Failed to create shader/pipeline");
        return ERR_CANT_CREATE;
    }

    // PERF (#634): Initialize batched async readback to reduce CPU/GPU sync points
    batched_readback.instantiate();
    // PERF: Staging buffer sized for: counters + indices + distances + importance (with alignment)
    // For 5M splats: indices (20MB) + distances (20MB) + importance (20MB) + counters (~1KB) = ~60MB
    // Using 64MB to allow headroom for larger scenes without falling back to individual readbacks.
    Error batch_err = batched_readback->initialize(rd, 64 * 1024 * 1024);
    if (batch_err != OK) {
        GS_LOG_WARN_DEFAULT("[GPUCuller] Failed to initialize batched async readback; falling back to individual readbacks");
        batched_readback.unref();
    }

    initialized = true;
    return OK;
}

void GPUCuller::shutdown() {
    // PERF (#634): Clean up batched async readback
    if (batched_readback.is_valid()) {
        batched_readback->shutdown();
        batched_readback.unref();
    }
    _release_resources();
    rd = nullptr;
    initialized = false;
}

bool GPUCuller::is_ready() const {
    return initialized && shader.is_valid() && pipeline.is_valid();
}

void GPUCuller::invalidate_lod_cache() {
    culling_config.lod_cache_dirty = true;
}

void GPUCuller::update_lod_cache() {
    if (!culling_config.lod_cache_dirty) {
        return;
    }

    if (!culling_config.lod_enabled) {
        culling_config.lod_cached_min_screen_threshold = 0.0f;
        culling_config.lod_cached_max_distance = 0.0f;
        culling_config.lod_cached_max_distance_sq = 0.0f;
        culling_config.lod_cache_dirty = false;
        return;
    }

    float effective_bias = MAX(culling_config.lod_bias, 0.0001f);
    culling_config.lod_cached_min_screen_threshold = culling_config.lod_min_screen_size > 0.0f
            ? culling_config.lod_min_screen_size * effective_bias
            : 0.0f;

    if (culling_config.lod_max_distance > 0.0f) {
        float adjusted = culling_config.lod_max_distance / effective_bias;
        culling_config.lod_cached_max_distance = adjusted;
        culling_config.lod_cached_max_distance_sq = adjusted * adjusted;
    } else {
        culling_config.lod_cached_max_distance = 0.0f;
        culling_config.lod_cached_max_distance_sq = 0.0f;
    }

    culling_config.lod_cache_dirty = false;
}

void GPUCuller::update_culling_settings() {
    ProjectSettings *ps = ProjectSettings::get_singleton();

    float min_screen_setting = culling_config.lod_min_screen_size;
    float max_distance_setting = culling_config.lod_max_distance;
    float project_bias_setting = culling_config.lod_bias;
    float importance_setting = culling_config.importance_cull_threshold;
    float frustum_slack_setting = culling_config.cull_frustum_plane_slack;

    if (!ps) {
        culling_state.culling_octree_max_depth = 8;
        culling_state.culling_min_gaussians = 32;
    } else {
        int octree_depth = _get_int_setting(ps, "rendering/gaussian_splatting/culling/octree_max_depth", 8);
        culling_state.culling_octree_max_depth = CLAMP(octree_depth, 1, 16);

        int min_gaussians = _get_int_setting(ps, "rendering/gaussian_splatting/culling/min_gaussians_per_leaf", 32);
        if (min_gaussians < 1) {
            min_gaussians = 1;
        }
        culling_state.culling_min_gaussians = (uint32_t)min_gaussians;

        min_screen_setting = _get_float_setting(ps, "rendering/gaussian_splatting/lod/min_screen_size_pixels", min_screen_setting);
        // Read from g_lod_config which resolves sentinel defaults via tier config,
        // instead of ProjectSettings directly where the sentinel -1.0f would be clamped to 0.
        max_distance_setting = g_lod_config.max_distance;
        project_bias_setting = _get_float_setting(ps, "rendering/gaussian_splatting/lod/bias", project_bias_setting);
        importance_setting = _get_float_setting(ps, "rendering/gaussian_splatting/lod/importance_threshold", importance_setting);
        frustum_slack_setting = _get_float_setting(ps, "rendering/gaussian_splatting/cull/frustum_plane_slack", frustum_slack_setting);
    }

    min_screen_setting = MAX(0.0f, min_screen_setting);
    max_distance_setting = MAX(0.0f, max_distance_setting);
    culling_config.lod_project_bias = MAX(0.001f, project_bias_setting);
    if (!culling_config.importance_cull_override) {
        culling_config.importance_cull_threshold = MAX(0.0f, importance_setting);
        culling_config.importance_cull_baseline = culling_config.importance_cull_threshold;
    }

    if (!culling_config.lod_min_screen_size_override) {
        culling_config.lod_min_screen_size = min_screen_setting;
    }
    if (!culling_config.lod_max_distance_override) {
        culling_config.lod_max_distance = max_distance_setting;
    }
    culling_config.cull_frustum_plane_slack = CLAMP(frustum_slack_setting, 1.0f, 8.0f);
}

void GPUCuller::ensure_hierarchical_structure(const Ref<GaussianData> &p_data) {
    if (p_data.is_null()) {
        culling_state.hierarchical_structure.reset();
        culling_state.hierarchical_structure_dirty = true;
        return;
    }

    if (!culling_state.hierarchical_structure) {
        culling_state.hierarchical_structure = std::make_unique<GaussianSplatting::HierarchicalSplatStructure>();
        culling_state.hierarchical_structure_dirty = true;
    }

    if (!culling_state.hierarchical_structure_dirty) {
        return;
    }

    const LocalVector<Gaussian> &gaussians = p_data->get_gaussian_storage();
    if (gaussians.is_empty()) {
        culling_state.hierarchical_structure.reset();
        culling_state.hierarchical_structure_dirty = true;
        return;
    }

    Vector<GaussianSplatting::GaussianData> build_data;
    build_data.resize(gaussians.size());

    const uint32_t gaussian_count = static_cast<uint32_t>(gaussians.size());
    for (uint32_t i = 0; i < gaussian_count; i++) {
        const Gaussian &src = gaussians[i];
        GaussianSplatting::GaussianData &dst = build_data.write[i];

        dst.position = src.position;
        dst.color = Color(src.sh_dc.r, src.sh_dc.g, src.sh_dc.b, src.opacity);
        dst.rotation = src.rotation;
        dst.scale = src.scale;
        dst.normal = src.normal;
        dst.area = src.area;
        dst.index = i;

        for (int j = 0; j < 6; j++) {
            dst.covariance[j] = 0.0f;
        }

        float opacity = CLAMP(src.opacity, 0.0f, 1.0f);
        float size_factor = MAX(MAX(src.scale.x, src.scale.y), src.scale.z);
        dst.importance = MAX(0.0001f, opacity * (size_factor + 0.0001f));
    }

    GaussianSplatting::HierarchicalSplatStructure::BuildParams params;
    params.max_depth = culling_state.culling_octree_max_depth;
    params.min_splats_per_node = culling_state.culling_min_gaussians;
    params.compute_importance = true;
    params.parallel_build = gaussians.size() > 16384;

    culling_state.hierarchical_structure->build_hierarchy(build_data, params);
    culling_state.hierarchical_structure_dirty = false;
}

void GPUCuller::_ensure_shader(RenderingDevice *p_device) {
    if (!p_device) {
        return;
    }

    const bool debug_counters_enabled = _is_cull_debug_counters_enabled();
    FrustumShaderState &state = _get_frustum_shader_state();
    {
        MutexLock lock(state.mutex);
        if (shader.is_valid() && debug_counters_enabled == state.last_debug_counters_enabled) {
            return;
        }

        if (!state.initialized || debug_counters_enabled != state.last_debug_counters_enabled) {
            state.initialized = false;
            shader = RID();
            pipeline = RID();

            if (frustum_shader_version.is_valid()) {
                state.source.version_free(frustum_shader_version);
                frustum_shader_version = RID();
            }

            state.source.~FrustumCullShaderRD();
            new (&state.source) FrustumCullShaderRD();

            // Build common defines - disable debug counters in production for ~10% GPU savings
            String debug_counter_define = debug_counters_enabled ? "" : "#define GS_DEBUG_COUNTERS_DISABLED 1\n";

            // Define shader variant groups:
            // Group 0 (SHADER_GROUP_STANDARD): Fallback using atomicAdd
            // Group 1 (SHADER_GROUP_SUBGROUPS): Optimized using subgroup operations
            Vector<ShaderRD::VariantDefine> variants;
            variants.push_back(ShaderRD::VariantDefine(
                    SHADER_GROUP_STANDARD,
                    debug_counter_define, // Debug counter control
                    false // Disabled by default, will enable the appropriate one
            ));
            variants.push_back(ShaderRD::VariantDefine(
                    SHADER_GROUP_SUBGROUPS,
                    debug_counter_define + "#define GS_ENABLE_SUBGROUPS 1\n", // Subgroups + debug counter control
                    false // Disabled by default
            ));

            state.source.initialize(variants);

            // Detect subgroup support at runtime
            // We need: basic (for gl_SubgroupInvocationID), ballot (for subgroupBallot),
            // and shuffle (for subgroupShuffle with dynamic id)
            uint64_t subgroup_ops = p_device->limit_get(RenderingDevice::LIMIT_SUBGROUP_OPERATIONS);
            uint64_t subgroup_stages = p_device->limit_get(RenderingDevice::LIMIT_SUBGROUP_IN_SHADERS);

            bool has_basic = (subgroup_ops & RenderingDevice::SUBGROUP_BASIC_BIT) != 0;
            bool has_ballot = (subgroup_ops & RenderingDevice::SUBGROUP_BALLOT_BIT) != 0;
            bool has_shuffle = (subgroup_ops & RenderingDevice::SUBGROUP_SHUFFLE_BIT) != 0;
            bool has_compute = (subgroup_stages & RenderingDevice::SHADER_STAGE_COMPUTE_BIT) != 0;

            if (has_basic && has_ballot && has_shuffle && has_compute) {
                // Enable subgroup-optimized variant
                state.source.enable_group(SHADER_GROUP_SUBGROUPS);
                state.enabled_group = SHADER_GROUP_SUBGROUPS;
                GS_LOG_WARN_DEFAULT("[GPUCuller] Subgroup operations available - using optimized ballot/shuffle path");
            } else {
                // Enable standard variant with atomicAdd fallback
                state.source.enable_group(SHADER_GROUP_STANDARD);
                state.enabled_group = SHADER_GROUP_STANDARD;
                GS_LOG_WARN_DEFAULT(vformat("[GPUCuller] Subgroup operations unavailable (basic=%d, ballot=%d, shuffle=%d, compute=%d) - using atomicAdd fallback",
                        has_basic, has_ballot, has_shuffle, has_compute));
                if (debug_counters_enabled) {
                    GS_LOG_WARN_DEFAULT("[GPUCuller] Cull debug counters enabled without subgroups; expect heavy atomic contention.");
                }
            }

            state.initialized = true;
            state.last_debug_counters_enabled = debug_counters_enabled;
        }
    }

    // Track which shader group is active
    subgroups_available = (state.enabled_group == SHADER_GROUP_SUBGROUPS);
    active_shader_group = state.enabled_group;

    // Create shader version
    frustum_shader_version = state.source.version_create();
    if (!frustum_shader_version.is_valid()) {
        GS_LOG_WARN_DEFAULT("[GPUCuller] Failed to create frustum cull shader version");
        return;
    }

    // Get the compiled shader for the active variant
    shader = state.source.version_get_shader(frustum_shader_version, active_shader_group);
    if (!shader.is_valid()) {
        GS_LOG_WARN_DEFAULT("[GPUCuller] Failed to get shader for variant group");
        state.source.version_free(frustum_shader_version);
        frustum_shader_version = RID();
        return;
    }

    pipeline = p_device->compute_pipeline_create(shader);
    if (!pipeline.is_valid()) {
        GS_LOG_WARN_DEFAULT("[GPUCuller] Failed to create pipeline");
        shader = RID();
        state.source.version_free(frustum_shader_version);
        frustum_shader_version = RID();
        return;
    }

    resource_device = p_device;
    GS_LOG_WARN_DEFAULT(vformat("[GPUCuller] Shader initialized successfully (subgroups=%s)",
            subgroups_available ? "enabled" : "disabled"));
}

void GPUCuller::_ensure_buffers(RenderingDevice *p_device, uint32_t p_required_capacity) {
    if (!p_device) {
        return;
    }

    uint32_t required_capacity = MAX(p_required_capacity, 1u);

    // Check if we need to recreate buffers
    if (resource_device && resource_device != p_device) {
        _release_resources();
    }

    if (buffer_capacity >= required_capacity &&
            visible_buffer.is_valid() &&
            distance_buffer.is_valid() &&
            importance_buffer.is_valid()) {
        // Existing buffers are sufficient
        if (!param_buffer.is_valid()) {
            Vector<uint8_t> param_bytes_init;
            param_bytes_init.resize(sizeof(FrustumCullParamsGPU));
            param_buffer = p_device->uniform_buffer_create(param_bytes_init.size(), param_bytes_init);
            if (param_buffer.is_valid()) {
                p_device->set_resource_name(param_buffer, "GS_CullParamsBuffer");
            }
        }
        if (!counter_buffer.is_valid()) {
            Vector<uint8_t> counter_bytes_init;
            counter_bytes_init.resize(sizeof(FrustumCullCounters));
            counter_buffer = p_device->storage_buffer_create(counter_bytes_init.size(), counter_bytes_init);
            if (counter_buffer.is_valid()) {
                p_device->set_resource_name(counter_buffer, "GS_CullCounterBuffer");
            }
        }
        return;
    }

    _reset_async_readback_state();

    // Free existing buffers
    if (visible_buffer.is_valid() && resource_device) {
        resource_device->free(visible_buffer);
        visible_buffer = RID();
    }
    if (distance_buffer.is_valid() && resource_device) {
        resource_device->free(distance_buffer);
        distance_buffer = RID();
    }
    if (importance_buffer.is_valid() && resource_device) {
        resource_device->free(importance_buffer);
        importance_buffer = RID();
    }

    buffer_capacity = required_capacity;

    Vector<uint8_t> zero_indices;
    zero_indices.resize((size_t)buffer_capacity * sizeof(uint32_t));
    Vector<uint8_t> zero_floats;
    zero_floats.resize((size_t)buffer_capacity * sizeof(float));

    visible_buffer = p_device->storage_buffer_create(zero_indices.size(), zero_indices);
    distance_buffer = p_device->storage_buffer_create(zero_floats.size(), zero_floats);
    importance_buffer = p_device->storage_buffer_create(zero_floats.size(), zero_floats);

    // Name the buffers for RenderDoc visibility
    if (visible_buffer.is_valid()) {
        p_device->set_resource_name(visible_buffer, "GS_CullVisibleIndices");
    }
    if (distance_buffer.is_valid()) {
        p_device->set_resource_name(distance_buffer, "GS_CullDistanceBuffer");
    }
    if (importance_buffer.is_valid()) {
        p_device->set_resource_name(importance_buffer, "GS_CullImportanceBuffer");
    }

    // Consolidated readback buffer - packs indices/distances/importance for single GPU->CPU transfer
    // Size: 3 * buffer_capacity * sizeof(uint32_t) for [indices][distances][importance]
    Vector<uint8_t> zero_consolidated;
    zero_consolidated.resize((size_t)buffer_capacity * 3 * sizeof(uint32_t));
    consolidated_buffer = p_device->storage_buffer_create(zero_consolidated.size(), zero_consolidated);
    if (consolidated_buffer.is_valid()) {
        p_device->set_resource_name(consolidated_buffer, "GS_CullConsolidatedBuffer");
    }

    if (!visible_buffer.is_valid() || !distance_buffer.is_valid() || !importance_buffer.is_valid() || !consolidated_buffer.is_valid()) {
        GS_LOG_WARN_DEFAULT("[GPUCuller] Failed to allocate culling result buffers");
        if (visible_buffer.is_valid()) {
            p_device->free(visible_buffer);
        }
        if (distance_buffer.is_valid()) {
            p_device->free(distance_buffer);
        }
        if (importance_buffer.is_valid()) {
            p_device->free(importance_buffer);
        }
        if (consolidated_buffer.is_valid()) {
            p_device->free(consolidated_buffer);
        }
        visible_buffer = RID();
        distance_buffer = RID();
        importance_buffer = RID();
        consolidated_buffer = RID();
        buffer_capacity = 0;
        return;
    }

    // Create/update uniform and counter buffers
    if (!param_buffer.is_valid()) {
        Vector<uint8_t> param_bytes_init;
        param_bytes_init.resize(sizeof(FrustumCullParamsGPU));
        param_buffer = p_device->uniform_buffer_create(param_bytes_init.size(), param_bytes_init);
        if (param_buffer.is_valid()) {
            p_device->set_resource_name(param_buffer, "GS_CullParamsBuffer");
        }
    }

    if (!counter_buffer.is_valid()) {
        Vector<uint8_t> counter_bytes_init;
        counter_bytes_init.resize(sizeof(FrustumCullCounters));
        counter_buffer = p_device->storage_buffer_create(counter_bytes_init.size(), counter_bytes_init);
        if (counter_buffer.is_valid()) {
            p_device->set_resource_name(counter_buffer, "GS_CullCounterBuffer");
        }
    }

    resource_device = p_device;

    if (param_bytes.size() != sizeof(FrustumCullParamsGPU)) {
        param_bytes.resize(sizeof(FrustumCullParamsGPU));
    }
    if (counter_bytes.size() != sizeof(FrustumCullCounters)) {
        counter_bytes.resize(sizeof(FrustumCullCounters));
    }
}

void GPUCuller::_ensure_instance_param_buffer(RenderingDevice *p_device) {
    if (!p_device) {
        return;
    }

    if (instance_resource_device && instance_resource_device != p_device) {
        _invalidate_instance_uniform_set_cache();
        if (instance_param_buffer.is_valid()) {
            instance_resource_device->free(instance_param_buffer);
        }
        instance_param_buffer = RID();
        instance_resource_device = nullptr;
    }

    if (!instance_param_buffer.is_valid()) {
        Vector<uint8_t> param_bytes_init;
        param_bytes_init.resize(sizeof(InstanceCullParamsGPU));
        instance_param_buffer = p_device->uniform_buffer_create(param_bytes_init.size(), param_bytes_init);
        if (instance_param_buffer.is_valid()) {
            p_device->set_resource_name(instance_param_buffer, "GS_InstanceCullParamsBuffer");
        }
    }

    instance_resource_device = p_device;
    if (instance_param_bytes.size() != sizeof(InstanceCullParamsGPU)) {
        instance_param_bytes.resize(sizeof(InstanceCullParamsGPU));
    }
}

void GPUCuller::_invalidate_instance_uniform_set_cache() {
    if (instance_uniform_set_cache.uniform_set.is_valid() &&
            instance_uniform_set_cache.device &&
            instance_uniform_set_cache.device->uniform_set_is_valid(instance_uniform_set_cache.uniform_set)) {
        instance_uniform_set_cache.device->free(instance_uniform_set_cache.uniform_set);
    }
    instance_uniform_set_cache = InstanceUniformSetCache();
}

RID GPUCuller::_get_instance_cull_uniform_set(RenderingDevice *p_device, const InstancePipelineInputs &p_inputs) {
    if (!p_device || !shader.is_valid() || !instance_param_buffer.is_valid()) {
        return RID();
    }

    const bool cache_matches =
            instance_uniform_set_cache.uniform_set.is_valid() &&
            instance_uniform_set_cache.device == p_device &&
            instance_uniform_set_cache.shader_rid == shader &&
            instance_uniform_set_cache.instance_buffer == p_inputs.instance_buffer &&
            instance_uniform_set_cache.asset_meta_buffer == p_inputs.asset_meta_buffer &&
            instance_uniform_set_cache.asset_chunk_index_buffer == p_inputs.asset_chunk_index_buffer &&
            instance_uniform_set_cache.chunk_meta_buffer == p_inputs.chunk_meta_buffer &&
            instance_uniform_set_cache.visible_chunk_buffer == p_inputs.visible_chunk_buffer &&
            instance_uniform_set_cache.counter_buffer == p_inputs.counter_buffer &&
            instance_uniform_set_cache.param_buffer == instance_param_buffer;
    if (cache_matches && p_device->uniform_set_is_valid(instance_uniform_set_cache.uniform_set)) {
        return instance_uniform_set_cache.uniform_set;
    }

    _invalidate_instance_uniform_set_cache();

    Vector<RD::Uniform> uniforms;
    RD::Uniform instance_uniform;
    instance_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    instance_uniform.binding = 0;
    instance_uniform.append_id(p_inputs.instance_buffer);
    uniforms.push_back(instance_uniform);

    RD::Uniform asset_meta_uniform;
    asset_meta_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    asset_meta_uniform.binding = 1;
    asset_meta_uniform.append_id(p_inputs.asset_meta_buffer);
    uniforms.push_back(asset_meta_uniform);

    RD::Uniform chunk_index_uniform;
    chunk_index_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    chunk_index_uniform.binding = 2;
    chunk_index_uniform.append_id(p_inputs.asset_chunk_index_buffer);
    uniforms.push_back(chunk_index_uniform);

    RD::Uniform chunk_meta_uniform;
    chunk_meta_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    chunk_meta_uniform.binding = 3;
    chunk_meta_uniform.append_id(p_inputs.chunk_meta_buffer);
    uniforms.push_back(chunk_meta_uniform);

    RD::Uniform visible_uniform;
    visible_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    visible_uniform.binding = 4;
    visible_uniform.append_id(p_inputs.visible_chunk_buffer);
    uniforms.push_back(visible_uniform);

    RD::Uniform counter_uniform;
    counter_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    counter_uniform.binding = 5;
    counter_uniform.append_id(p_inputs.counter_buffer);
    uniforms.push_back(counter_uniform);

    RD::Uniform params_uniform;
    params_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
    params_uniform.binding = 6;
    params_uniform.append_id(instance_param_buffer);
    uniforms.push_back(params_uniform);

    RID cull_uniform_set = p_device->uniform_set_create(uniforms, shader, 0);
    if (!cull_uniform_set.is_valid()) {
        return RID();
    }
    p_device->set_resource_name(cull_uniform_set, "GS_GPUCuller_InstanceCullSet");

    instance_uniform_set_cache.uniform_set = cull_uniform_set;
    instance_uniform_set_cache.instance_buffer = p_inputs.instance_buffer;
    instance_uniform_set_cache.asset_meta_buffer = p_inputs.asset_meta_buffer;
    instance_uniform_set_cache.asset_chunk_index_buffer = p_inputs.asset_chunk_index_buffer;
    instance_uniform_set_cache.chunk_meta_buffer = p_inputs.chunk_meta_buffer;
    instance_uniform_set_cache.visible_chunk_buffer = p_inputs.visible_chunk_buffer;
    instance_uniform_set_cache.counter_buffer = p_inputs.counter_buffer;
    instance_uniform_set_cache.param_buffer = instance_param_buffer;
    instance_uniform_set_cache.shader_rid = shader;
    instance_uniform_set_cache.device = p_device;
    return cull_uniform_set;
}

void GPUCuller::_release_resources() {
    _reset_async_readback_state();
    instance_readback_state.pending = false;
    instance_readback_state.generation++;
    instance_readback_state.pending_request_id = 0;
    instance_readback_state.next_request_id = 1;
    instance_readback_state.last_applied_request_id = 0;
    instance_readback_state.last_frame_chunk_limit = UINT32_MAX;
    _invalidate_instance_uniform_set_cache();

    if (frustum_shader_version.is_valid()) {
        _get_frustum_shader_state().source.version_free(frustum_shader_version);
        frustum_shader_version = RID();
    }

    // Check if device is still valid by comparing with current shared device
    bool device_still_valid = resource_device != nullptr;

    if (device_still_valid) {
        // Device is still valid - properly free GPU resources
        if (pipeline.is_valid()) {
            if (resource_device->compute_pipeline_is_valid(pipeline)) {
                resource_device->free(pipeline);
            }
        }
        // shader originates from FrustumCullShaderRD::version_get_shader().
        // It is released by version_free(frustum_shader_version) above.
        // Avoid explicit free here to prevent teardown-order invalid RID noise.
        if (param_buffer.is_valid()) {
            resource_device->free(param_buffer);
        }
        if (instance_param_buffer.is_valid() && instance_resource_device) {
            instance_resource_device->free(instance_param_buffer);
        }
        if (counter_buffer.is_valid()) {
            resource_device->free(counter_buffer);
        }
        if (visible_buffer.is_valid()) {
            resource_device->free(visible_buffer);
        }
        if (distance_buffer.is_valid()) {
            resource_device->free(distance_buffer);
        }
        if (importance_buffer.is_valid()) {
            resource_device->free(importance_buffer);
        }
        if (consolidated_buffer.is_valid()) {
            resource_device->free(consolidated_buffer);
        }
    }

    // Invalidate all RIDs
    shader = RID();
    pipeline = RID();
    param_buffer = RID();
    instance_param_buffer = RID();
    counter_buffer = RID();
    visible_buffer = RID();
    distance_buffer = RID();
    importance_buffer = RID();
    consolidated_buffer = RID();

    buffer_capacity = 0;
    resource_device = nullptr;
    instance_resource_device = nullptr;
}

RenderingDevice *GPUCuller::_acquire_submission_device(RenderingDevice *p_device) {
    if (!p_device) {
        return nullptr;
    }

    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        GaussianSplatManager::ScopedSubmissionLock lock;
        RenderingDevice *acquired = manager->acquire_submission_device(p_device, lock);
        if (acquired) {
            return acquired;
        }
    }

    return p_device;
}

void GPUCuller::_on_cull_readback(const Vector<uint8_t> &p_data, int p_type, int64_t p_request_id, int64_t p_generation) {
    const uint64_t request_id = static_cast<uint64_t>(p_request_id);
    const uint64_t generation = static_cast<uint64_t>(p_generation);
    if (generation != async_readback_state.generation) {
        return;
    }

    const int pending_count = async_readback_state.pending.size();
    if (pending_count == 0) {
        return;
    }

    int request_index = -1;
    for (int i = 0; i < pending_count; i++) {
        const AsyncReadbackRequest &request = async_readback_state.pending[i];
        if (request.request_id == request_id && request.generation == generation) {
            request_index = i;
            break;
        }
    }

    if (request_index < 0) {
        return;
    }

    AsyncReadbackRequest *requests = async_readback_state.pending.ptrw();
    AsyncReadbackRequest &request = requests[request_index];

    switch (p_type) {
        case ASYNC_READBACK_COUNTERS: {
            if (p_data.size() < (int)sizeof(FrustumCullCounters)) {
                GS_LOG_WARN_DEFAULT("[GPUCuller] Async counter readback failed");
                async_readback_state.pending.remove_at(request_index);
                return;
            }
            const FrustumCullCounters *counters = reinterpret_cast<const FrustumCullCounters *>(p_data.ptr());
            request.counters.visible_count = counters->visible_count;
            request.counters.frustum_culled = counters->frustum_culled;
            request.counters.distance_culled = counters->distance_culled;
            request.counters.screen_culled = counters->screen_culled;
            request.counters.importance_culled = counters->importance_culled;
            request.counters.clipped_count = counters->clipped_count;
            request.counters.near_clamped = counters->near_clamped;
            request.counters.behind_culled = counters->behind_culled;
            request.counters_ready = true;
        } break;
        case ASYNC_READBACK_INDICES: {
            request.indices_bytes = p_data;
            request.indices_ready = true;
        } break;
        case ASYNC_READBACK_DISTANCES: {
            request.distance_bytes = p_data;
            request.distances_ready = true;
        } break;
        case ASYNC_READBACK_IMPORTANCE: {
            request.importance_bytes = p_data;
            request.importance_ready = true;
        } break;
        default:
            return;
    }

    const bool ready = request.counters_ready &&
            (!request.readback_indices || request.indices_ready) &&
            (!request.readback_distances || request.distances_ready) &&
            (!request.readback_importance || request.importance_ready);
    if (!ready) {
        return;
    }

    const size_t expected_indices = request.readback_indices ? (size_t(request.max_visible) * sizeof(uint32_t)) : 0;
    const size_t expected_distances = request.readback_distances ? (size_t(request.max_visible) * sizeof(float)) : 0;
    const size_t expected_importance = request.readback_importance ? (size_t(request.max_visible) * sizeof(float)) : 0;
    const size_t indices_size = size_t(request.indices_bytes.size());
    const size_t distance_size = size_t(request.distance_bytes.size());
    const size_t importance_size = size_t(request.importance_bytes.size());
    if ((request.readback_indices && indices_size < expected_indices) ||
            (request.readback_distances && distance_size < expected_distances) ||
            (request.readback_importance && importance_size < expected_importance)) {
        GS_LOG_WARN_DEFAULT("[GPUCuller] Async readback data size mismatch");
        async_readback_state.pending.remove_at(request_index);
        return;
    }

    const uint32_t visible = MIN(request.counters.visible_count, request.max_visible);
    if (request.counters.visible_count > request.max_visible) {
        static uint32_t overflow_log_throttle = 0;
        if (++overflow_log_throttle % 300 == 1) {
            WARN_PRINT(vformat("[GPUCuller] Visible count %d exceeds buffer capacity %d; %d splats dropped. "
                    "Consider increasing max_sort_elements.",
                    request.counters.visible_count, request.max_visible,
                    request.counters.visible_count - request.max_visible));
        }
    }

    CullResult result;
    result.success = true;
    result.counters = request.counters;
    result.counters.visible_count = visible;
    result.counters.total_input = request.total_splats;
    result.counters.culling_time_ms = request.dispatch_time_ms;

    if (request.readback_indices) {
        result.visible_indices.resize(visible);
    }
    if (request.readback_distances) {
        result.distances_sq.resize(visible);
    }
    if (request.readback_importance) {
        result.importance_weights.resize(visible);
    }

    if (visible > 0) {
        if (request.readback_indices) {
            const size_t visible_index_bytes = size_t(visible) * sizeof(uint32_t);
            std::memcpy(result.visible_indices.ptr(), request.indices_bytes.ptr(), visible_index_bytes);
        }
        if (request.readback_distances) {
            const size_t visible_distance_bytes = size_t(visible) * sizeof(float);
            std::memcpy(result.distances_sq.ptr(), request.distance_bytes.ptr(), visible_distance_bytes);
        }
        if (request.readback_importance) {
            const size_t visible_importance_bytes = size_t(visible) * sizeof(float);
            std::memcpy(result.importance_weights.ptr(), request.importance_bytes.ptr(), visible_importance_bytes);
        }
    }

    if (!async_readback_state.last_result_valid || request.request_id >= async_readback_state.last_result_id) {
        async_readback_state.last_result = result;
        async_readback_state.last_result_id = request.request_id;
        async_readback_state.last_result_valid = true;
    }

    async_readback_state.pending.remove_at(request_index);
}

void GPUCuller::_on_instance_counter_readback(
        const Vector<uint8_t> &p_data,
        int64_t p_generation,
        int64_t p_max_visible_chunks,
        int64_t p_request_id) {
    const uint64_t generation = static_cast<uint64_t>(p_generation);
    const uint64_t request_id = p_request_id > 0 ? static_cast<uint64_t>(p_request_id) : 0;
    if (!instance_readback_state.pending || generation != instance_readback_state.generation) {
        return;
    }
    if (request_id != 0 && request_id != instance_readback_state.pending_request_id) {
        return;
    }
    instance_readback_state.pending = false;
    instance_readback_state.pending_request_id = 0;

    if (request_id != 0 && request_id < instance_readback_state.last_applied_request_id) {
        return;
    }

    if (p_data.size() < static_cast<int>(sizeof(uint32_t) * 2)) {
        return;
    }

    const uint32_t max_visible_chunks = p_max_visible_chunks > 0 ? static_cast<uint32_t>(p_max_visible_chunks) : UINT32_MAX;
    const uint32_t callback_chunk_limit = MIN(max_visible_chunks, instance_readback_state.last_frame_chunk_limit);
    const uint32_t *counter_values = reinterpret_cast<const uint32_t *>(p_data.ptr());
    last_instance_visible_chunk_count = MIN(counter_values[0], callback_chunk_limit);
    if (request_id > instance_readback_state.last_applied_request_id) {
        instance_readback_state.last_applied_request_id = request_id;
    }
    culling_state.gpu_visible_indices_count = last_instance_visible_chunk_count;
}

void GPUCuller::_reset_async_readback_state() {
    async_readback_state.pending.clear();
    async_readback_state.last_result = CullResult();
    async_readback_state.last_result_id = 0;
    async_readback_state.last_result_valid = false;
    async_readback_state.next_request_id = 1;
    async_readback_state.generation++;
    async_readback_state.last_submitted_id = 0;
}

// PERF (#634): Batched readback callback - receives individual buffer slices from BatchedAsyncReadback
// user_data encodes: request_id (bits 0-31), type (bits 32-35), generation (bits 36-51), request_index (bits 52-63)
void GPUCuller::_on_batched_cull_readback(const Vector<uint8_t> &p_data, int64_t p_user_data) {
    const uint64_t user = static_cast<uint64_t>(p_user_data);
    const uint64_t request_id = user & 0xFFFFFFFF;
    const int type = static_cast<int>((user >> 32) & 0xF);
    const uint64_t generation = (user >> 36) & 0xFFFF;
    const int request_index = static_cast<int>((user >> 52) & 0xFFF);

    if (generation != async_readback_state.generation) {
        return;
    }

    const int pending_count = async_readback_state.pending.size();
    if (request_index >= 0 && request_index < pending_count) {
        const AsyncReadbackRequest &request = async_readback_state.pending[request_index];
        if (request.request_id == request_id && request.generation == generation) {
            _on_cull_readback(p_data, type, int64_t(request_id), int64_t(generation));
            return;
        }
    }

    // Dispatch to the appropriate handler based on type - reuse existing callback logic
    _on_cull_readback(p_data, type, int64_t(request_id), int64_t(generation));
}

CullResult GPUCuller::cull(const CullParams &p_params, const CullInputBuffers &p_input) {
    CullResult result;
    result.success = false;

    uint64_t start_time = OS::get_singleton()->get_ticks_usec();

    bool readback_indices = p_params.readback_indices;
    bool readback_distances = p_params.readback_distances;
    bool readback_importance = p_params.readback_importance;
    if (!readback_indices) {
        readback_distances = false;
        readback_importance = false;
    }

    if (!is_ready() || !readback_enabled) {
        if (async_readback_state.last_result_valid || !async_readback_state.pending.is_empty()) {
            _reset_async_readback_state();
        }
        return result;
    }

    if (!p_input.gaussian_buffer.is_valid()) {
        _reset_async_readback_state();
        return result;
    }

    if (p_input.total_splat_count == 0) {
        _reset_async_readback_state();
        result.success = true;
        result.counters.total_input = 0;
        result.counters.visible_count = 0;
        result.counters.culling_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;
        return result;
    }

    RenderingDevice *dispatch_device = p_input.buffer_device ? p_input.buffer_device : rd;
    GaussianSplatManager::ScopedSubmissionLock submission_lock;
    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        dispatch_device = manager->acquire_submission_device(dispatch_device, submission_lock);
    }

    if (!dispatch_device) {
        GS_LOG_WARN_DEFAULT("[GPUCuller] Submission device unavailable");
        return result;
    }

    // Ensure shader is created on this device
    if (!shader.is_valid() || resource_device != dispatch_device) {
        _ensure_shader(dispatch_device);
    }

    uint32_t max_visible = p_params.max_visible;
    if (max_visible == UINT32_MAX || max_visible == 0) {
        max_visible = p_input.total_splat_count;
    }
    max_visible = MIN(max_visible, p_input.total_splat_count);
    max_visible = MAX(max_visible, 1u);

    _ensure_buffers(dispatch_device, max_visible);
    if (!pipeline.is_valid() || !visible_buffer.is_valid() || !param_buffer.is_valid() || !counter_buffer.is_valid()) {
        GS_LOG_WARN_DEFAULT("[GPUCuller] Failed to prepare GPU culling buffers");
        return result;
    }

    // Build GPU parameters
    FrustumCullParamsGPU params = {};

    // View matrix
    Transform3D view_matrix = p_params.world_to_camera_transform;
    for (int column = 0; column < 3; column++) {
        Vector3 column_vec = view_matrix.basis.get_column(column);
        params.view_matrix[column * 4 + 0] = column_vec.x;
        params.view_matrix[column * 4 + 1] = column_vec.y;
        params.view_matrix[column * 4 + 2] = column_vec.z;
        params.view_matrix[column * 4 + 3] = 0.0f;
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
    for (uint32_t i = 0; i < kFrustumPlaneCount; i++) {
        if (i < (uint32_t)p_params.frustum_planes.size()) {
            const Plane &plane = p_params.frustum_planes[i];
            params.frustum_planes[i][0] = plane.normal.x;
            params.frustum_planes[i][1] = plane.normal.y;
            params.frustum_planes[i][2] = plane.normal.z;
            params.frustum_planes[i][3] = plane.d;
        } else {
            params.frustum_planes[i][0] = 0.0f;
            params.frustum_planes[i][1] = 0.0f;
            params.frustum_planes[i][2] = 0.0f;
            params.frustum_planes[i][3] = 0.0f;
        }
    }

    // Other parameters
    params.camera_position[0] = p_params.camera_position.x;
    params.camera_position[1] = p_params.camera_position.y;
    params.camera_position[2] = p_params.camera_position.z;
    params.pixel_scale_y = p_params.pixel_scale_y;
    params.min_screen_size = p_params.min_screen_size;
    params.max_distance_sq = p_params.max_distance_sq;
    params.importance_threshold = p_params.importance_threshold;
    params.near_tolerance = p_params.near_tolerance;
    params.far_tolerance = p_params.far_tolerance;
    params.tiny_splat_screen_radius = p_params.tiny_splat_screen_radius;
    params.frustum_plane_slack = p_params.frustum_plane_slack;
    params.radius_multiplier = p_params.radius_multiplier;
    params.total_splats = p_input.total_splat_count;
    params.max_visible = max_visible;
    params.enable_frustum = p_params.frustum_culling_enabled ? 1 : 0;
    params.orthographic = p_params.orthographic ? 1 : 0;

    // Upload parameters
    if (param_bytes.size() != sizeof(FrustumCullParamsGPU)) {
        param_bytes.resize(sizeof(FrustumCullParamsGPU));
    }
    std::memcpy(param_bytes.ptrw(), &params, sizeof(FrustumCullParamsGPU));
    dispatch_device->buffer_update(param_buffer, 0, param_bytes.size(), param_bytes.ptr());

    // Reset counters with an explicit host write so zero-visibility frames are
    // deterministic even when command submission is deferred on the main device.
    static const FrustumCullCounters zero_counters = {};
    dispatch_device->buffer_update(counter_buffer, 0, sizeof(zero_counters), &zero_counters);

    // Build uniform set
    Vector<RD::Uniform> uniforms;

    RD::Uniform gaussian_uniform;
    gaussian_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    gaussian_uniform.binding = 0;
    gaussian_uniform.append_id(p_input.gaussian_buffer);
    uniforms.push_back(gaussian_uniform);

    RD::Uniform visible_uniform;
    visible_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    visible_uniform.binding = 1;
    visible_uniform.append_id(visible_buffer);
    uniforms.push_back(visible_uniform);

    RD::Uniform distance_uniform;
    distance_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    distance_uniform.binding = 2;
    distance_uniform.append_id(distance_buffer);
    uniforms.push_back(distance_uniform);

    RD::Uniform importance_uniform;
    importance_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    importance_uniform.binding = 3;
    importance_uniform.append_id(importance_buffer);
    uniforms.push_back(importance_uniform);

    RD::Uniform counter_uniform;
    counter_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    counter_uniform.binding = 4;
    counter_uniform.append_id(counter_buffer);
    uniforms.push_back(counter_uniform);

    RD::Uniform param_uniform;
    param_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
    param_uniform.binding = 5;
    param_uniform.append_id(param_buffer);
    uniforms.push_back(param_uniform);

    RD::Uniform consolidated_uniform;
    consolidated_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    consolidated_uniform.binding = 6;
    consolidated_uniform.append_id(consolidated_buffer);
    uniforms.push_back(consolidated_uniform);

    RID cull_uniform_set = dispatch_device->uniform_set_create(uniforms, shader, 0);
    if (!cull_uniform_set.is_valid()) {
        GS_LOG_WARN_DEFAULT("[GPUCuller] Failed to create uniform set");
        return result;
    }

    // Dispatch compute shader
    uint32_t group_count = (p_input.total_splat_count + 255) / 256;
    if (group_count == 0) {
        group_count = 1;
    }

    // GPU Debug: Frustum culling pass
    ScopedGpuMarkerEx cull_marker(dispatch_device, "GS_FrustumCull", PassColors::CULLING);

    RD::ComputeListID compute_list = dispatch_device->compute_list_begin();
    if (compute_list == RD::INVALID_ID) {
        if (dispatch_device->uniform_set_is_valid(cull_uniform_set)) {
            dispatch_device->free(cull_uniform_set);
        }
        return result;
    }

    dispatch_device->compute_list_bind_compute_pipeline(compute_list, pipeline);
    dispatch_device->compute_list_bind_uniform_set(compute_list, cull_uniform_set, 0);
    dispatch_device->compute_list_dispatch(compute_list, group_count, 1, 1);
    dispatch_device->compute_list_add_barrier(compute_list);
    dispatch_device->compute_list_end();

    if (dispatch_device->uniform_set_is_valid(cull_uniform_set)) {
        dispatch_device->free(cull_uniform_set);
    }

    const uint64_t request_id = async_readback_state.next_request_id++;
    async_readback_state.last_submitted_id = request_id;
    const uint64_t generation = async_readback_state.generation;
    const float dispatch_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;

    const size_t counter_size = sizeof(FrustumCullCounters);
    const size_t index_bytes = readback_indices ? size_t(max_visible) * sizeof(uint32_t) : 0;
    const size_t distance_bytes = readback_distances ? size_t(max_visible) * sizeof(float) : 0;
    const size_t importance_bytes = readback_importance ? size_t(max_visible) * sizeof(float) : 0;

    if (counter_size <= UINT32_MAX && index_bytes <= UINT32_MAX &&
            distance_bytes <= UINT32_MAX && importance_bytes <= UINT32_MAX) {
        AsyncReadbackRequest request;
        request.request_id = request_id;
        request.generation = generation;
        request.max_visible = max_visible;
        request.total_splats = p_input.total_splat_count;
        request.dispatch_time_ms = dispatch_time_ms;
        request.readback_indices = readback_indices;
        request.readback_distances = readback_distances;
        request.readback_importance = readback_importance;
        async_readback_state.pending.push_back(request);

        const int request_index = async_readback_state.pending.size() - 1;
        bool readback_ok = false;

        // PERF (#634): Use batched async readback to consolidate 4 separate GPU->CPU transfers into 1
        if (batched_readback.is_valid() && batched_readback->is_initialized() &&
                batched_readback->get_state() == BatchedAsyncReadback::BATCH_IDLE) {
            // Calculate total size needed (with 16-byte alignment)
            const uint32_t counter_aligned = ((uint32_t)counter_size + 15) & ~15;
            const uint32_t index_aligned = readback_indices ? ((uint32_t)index_bytes + 15) & ~15 : 0;
            const uint32_t distance_aligned = readback_distances ? ((uint32_t)distance_bytes + 15) & ~15 : 0;
            const uint32_t importance_aligned = readback_importance ? ((uint32_t)importance_bytes + 15) & ~15 : 0;
            const uint32_t total_needed = counter_aligned + index_aligned + distance_aligned + importance_aligned;

            if (total_needed <= batched_readback->get_staging_buffer_capacity()) {
                // Encode request info per-buffer: request_id (32 bits) | type (4 bits) | generation (16 bits) | request_index (12 bits)
                auto encode_user_data = [&](int type) -> int64_t {
                    return int64_t(request_id & 0xFFFFFFFF) |
                            (int64_t(type & 0xF) << 32) |
                            (int64_t(generation & 0xFFFF) << 36) |
                            (int64_t(request_index & 0xFFF) << 52);
                };

                Callable batch_cb = callable_mp(this, &GPUCuller::_on_batched_cull_readback);

                // Add requested buffers to the batch with type-specific user_data.
                bool add_ok = batched_readback->add_request(counter_buffer, 0, (uint32_t)counter_size,
                        batch_cb, encode_user_data(ASYNC_READBACK_COUNTERS));
                if (add_ok && readback_indices) {
                    add_ok = batched_readback->add_request(visible_buffer, 0, (uint32_t)index_bytes,
                            batch_cb, encode_user_data(ASYNC_READBACK_INDICES));
                }
                if (add_ok && readback_distances) {
                    add_ok = batched_readback->add_request(distance_buffer, 0, (uint32_t)distance_bytes,
                            batch_cb, encode_user_data(ASYNC_READBACK_DISTANCES));
                }
                if (add_ok && readback_importance) {
                    add_ok = batched_readback->add_request(importance_buffer, 0, (uint32_t)importance_bytes,
                            batch_cb, encode_user_data(ASYNC_READBACK_IMPORTANCE));
                }

                if (add_ok) {
                    readback_ok = batched_readback->submit_batch();
                } else {
                    batched_readback->cancel_batch();
                }
            }
        }

        // Fallback: Use individual async readbacks if batched readback failed or unavailable
        if (!readback_ok) {
            readback_ok = true;
            Callable counters_cb = callable_mp(this, &GPUCuller::_on_cull_readback)
                    .bind(ASYNC_READBACK_COUNTERS, int64_t(request_id), int64_t(generation));
            Error err = dispatch_device->buffer_get_data_async(counter_buffer, counters_cb, 0, (uint32_t)counter_size);
            if (err != OK) {
                readback_ok = false;
            }

            if (readback_ok && readback_indices) {
                Callable indices_cb = callable_mp(this, &GPUCuller::_on_cull_readback)
                        .bind(ASYNC_READBACK_INDICES, int64_t(request_id), int64_t(generation));
                err = dispatch_device->buffer_get_data_async(visible_buffer, indices_cb, 0, (uint32_t)index_bytes);
                if (err != OK) {
                    readback_ok = false;
                }
            }

            if (readback_ok && readback_distances) {
                Callable distances_cb = callable_mp(this, &GPUCuller::_on_cull_readback)
                        .bind(ASYNC_READBACK_DISTANCES, int64_t(request_id), int64_t(generation));
                err = dispatch_device->buffer_get_data_async(distance_buffer, distances_cb, 0, (uint32_t)distance_bytes);
                if (err != OK) {
                    readback_ok = false;
                }
            }

            if (readback_ok && readback_importance) {
                Callable importance_cb = callable_mp(this, &GPUCuller::_on_cull_readback)
                        .bind(ASYNC_READBACK_IMPORTANCE, int64_t(request_id), int64_t(generation));
                err = dispatch_device->buffer_get_data_async(importance_buffer, importance_cb, 0, (uint32_t)importance_bytes);
                if (err != OK) {
                    readback_ok = false;
                }
            }
        }

        if (!readback_ok) {
            GS_LOG_WARN_DEFAULT("[GPUCuller] Async readback request failed");
            async_readback_state.pending.remove_at(request_index);
        }
    } else {
        GS_LOG_WARN_DEFAULT("[GPUCuller] Async readback size exceeds RD limits");
    }

    // Submit if not main device to flush async readbacks.
    if (!dispatch_device->is_main_rendering_device()) {
        dispatch_device->submit();
    }

    if (async_readback_state.last_result_valid) {
        result = async_readback_state.last_result;
        result.success = true;
    }

    return result;
}

// PERF (#659): Pass pre-computed inverse to avoid redundant affine_inverse() calls
bool GPUCuller::_gpu_frustum_cull(const Transform3D &p_cam_transform, const Transform3D &p_cam_to_world,
        const Projection &p_projection, const Size2i &p_viewport_size,
        const Vector<Plane> &p_planes, float p_pixel_scale_y, bool p_orthographic, uint64_t p_start_time_usec,
        const GpuCullInput &p_input, uint32_t p_max_splats, bool p_readback_indices, bool p_readback_distances,
        bool p_readback_importance) {
    (void)p_cam_transform;
    (void)p_cam_to_world;
    (void)p_projection;
    (void)p_viewport_size;
    (void)p_planes;
    (void)p_pixel_scale_y;
    (void)p_orthographic;
    (void)p_start_time_usec;
    (void)p_input;
    (void)p_max_splats;
    (void)p_readback_indices;
    (void)p_readback_distances;
    (void)p_readback_importance;
    if (!culling_config.gpu_culling_enabled || !culling_config.gpu_culling_readback_enabled) {
        return false;
    }
    // Legacy GPU cull shader path was retired with the instance-only compute shaders.
    return false;
}


bool GPUCuller::_gpu_frustum_cull_instance(const CullParams &p_params, const InstancePipelineInputs &p_inputs,
        uint64_t p_start_time_usec, CullingSummary &r_summary) {
    if (!p_inputs.device || !p_inputs.instance_buffer.is_valid() || !p_inputs.asset_meta_buffer.is_valid() ||
            !p_inputs.asset_chunk_index_buffer.is_valid() || !p_inputs.chunk_meta_buffer.is_valid() ||
            !p_inputs.visible_chunk_buffer.is_valid() || !p_inputs.counter_buffer.is_valid()) {
        return false;
    }

    if (!is_ready()) {
        Error init_err = initialize(p_inputs.device);
        if (init_err != OK) {
            GS_LOG_WARN_DEFAULT(vformat("[GPU Cull] GPUCuller initialization failed: %d", init_err));
        }
    }

    _ensure_shader(p_inputs.device);
    _ensure_instance_param_buffer(p_inputs.device);
    if (!shader.is_valid() || !pipeline.is_valid() || !instance_param_buffer.is_valid()) {
        return false;
    }

    const uint32_t instance_count = p_inputs.instance_count;
    const uint32_t chunk_dispatch = p_inputs.dispatch_chunk_count;
    if (instance_count == 0 || chunk_dispatch == 0) {
        last_instance_visible_chunk_count = 0;
        r_summary.visible_after_culling = 0;
        r_summary.culling_candidate_count = 0;
        r_summary.used_instance_pipeline = true;
        r_summary.culling_time_ms = (OS::get_singleton()->get_ticks_usec() - p_start_time_usec) / 1000.0f;
        culling_state.cull_time_ms = r_summary.culling_time_ms;
        return true;
    }

    InstanceCullParamsGPU params = {};

    Transform3D view_matrix = p_params.world_to_camera_transform;
    for (int column = 0; column < 3; column++) {
        Vector3 column_vec = view_matrix.basis.get_column(column);
        params.view_matrix[column * 4 + 0] = column_vec.x;
        params.view_matrix[column * 4 + 1] = column_vec.y;
        params.view_matrix[column * 4 + 2] = column_vec.z;
        params.view_matrix[column * 4 + 3] = 0.0f;
    }
    params.view_matrix[12] = view_matrix.origin.x;
    params.view_matrix[13] = view_matrix.origin.y;
    params.view_matrix[14] = view_matrix.origin.z;
    params.view_matrix[15] = 1.0f;

    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            params.proj_matrix[col * 4 + row] = p_params.projection.columns[col][row];
        }
    }

    for (uint32_t i = 0; i < kFrustumPlaneCount; i++) {
        if (i < (uint32_t)p_params.frustum_planes.size()) {
            const Plane &plane = p_params.frustum_planes[i];
            params.frustum_planes[i][0] = plane.normal.x;
            params.frustum_planes[i][1] = plane.normal.y;
            params.frustum_planes[i][2] = plane.normal.z;
            params.frustum_planes[i][3] = plane.d;
        } else {
            params.frustum_planes[i][0] = 0.0f;
            params.frustum_planes[i][1] = 0.0f;
            params.frustum_planes[i][2] = 0.0f;
            params.frustum_planes[i][3] = 0.0f;
        }
    }

    params.frustum_plane_slack = p_params.frustum_plane_slack;
    params.instance_count = instance_count;
    params.max_visible_chunks = p_inputs.max_visible_chunks > 0 ? p_inputs.max_visible_chunks : instance_count;
    params.enable_frustum = p_params.frustum_culling_enabled ? 1u : 0u;

    if (instance_param_bytes.size() != sizeof(InstanceCullParamsGPU)) {
        instance_param_bytes.resize(sizeof(InstanceCullParamsGPU));
    }
    std::memcpy(instance_param_bytes.ptrw(), &params, sizeof(InstanceCullParamsGPU));
    p_inputs.device->buffer_update(instance_param_buffer, 0, instance_param_bytes.size(), instance_param_bytes.ptr());

    // Reset counters with an explicit host write before dispatch. The visible
    // count is consumed via async-latched readback callbacks.
    static const uint32_t zero_instance_counters[2] = { 0u, 0u };
    p_inputs.device->buffer_update(p_inputs.counter_buffer, 0, sizeof(zero_instance_counters), zero_instance_counters);

    RID cull_uniform_set = _get_instance_cull_uniform_set(p_inputs.device, p_inputs);
    if (!cull_uniform_set.is_valid()) {
        GS_LOG_WARN_DEFAULT("[GPUCuller] Failed to create instance cull uniform set");
        return false;
    }

    uint32_t group_x = (chunk_dispatch + 255) / 256;
    if (group_x == 0) {
        group_x = 1;
    }

    ScopedGpuMarkerEx cull_marker(p_inputs.device, "GS_InstanceFrustumCull", PassColors::CULLING);
    RD::ComputeListID compute_list = p_inputs.device->compute_list_begin();
    if (compute_list == RD::INVALID_ID) {
        return false;
    }

    p_inputs.device->compute_list_bind_compute_pipeline(compute_list, pipeline);
    p_inputs.device->compute_list_bind_uniform_set(compute_list, cull_uniform_set, 0);

    p_inputs.device->compute_list_dispatch(compute_list, group_x, instance_count, 1);
    p_inputs.device->compute_list_add_barrier(compute_list);
    p_inputs.device->compute_list_end();

    const uint32_t max_visible_chunks = params.max_visible_chunks > 0 ? params.max_visible_chunks : instance_count;
    // Keep published visible-chunk counts aligned to this frame's dispatch domain.
    // Async callbacks can still deliver prior-frame counts; never allow those to
    // exceed the chunks dispatched in the current frame.
    const uint64_t total_dispatched_chunks_u64 = uint64_t(instance_count) * uint64_t(chunk_dispatch);
    const uint32_t total_dispatched_chunks = total_dispatched_chunks_u64 > uint64_t(UINT32_MAX)
            ? UINT32_MAX
            : uint32_t(total_dispatched_chunks_u64);
    const uint32_t current_frame_chunk_limit = MIN(total_dispatched_chunks, max_visible_chunks);
    instance_readback_state.last_frame_chunk_limit = current_frame_chunk_limit;
    // Keep this hot path non-blocking: visible count publication uses the most
    // recent async sample, clamped to the current frame dispatch domain.
    uint32_t visible_chunk_count = MIN(last_instance_visible_chunk_count, current_frame_chunk_limit);
    uint64_t enqueued_request_id = 0;
    if (p_inputs.counter_buffer.is_valid() && !instance_readback_state.pending) {
        enqueued_request_id = instance_readback_state.next_request_id++;
        instance_readback_state.pending = true;
        instance_readback_state.generation++;
        instance_readback_state.pending_request_id = enqueued_request_id;
        const int64_t readback_generation = static_cast<int64_t>(instance_readback_state.generation);
        Callable counter_cb = callable_mp(this, &GPUCuller::_on_instance_counter_readback)
                .bind(readback_generation, int64_t(max_visible_chunks), int64_t(enqueued_request_id));
        Error readback_err = p_inputs.device->buffer_get_data_async(p_inputs.counter_buffer, counter_cb, 0, sizeof(uint32_t) * 2);
        if (readback_err != OK) {
            instance_readback_state.pending = false;
            instance_readback_state.pending_request_id = 0;
            GS_LOG_WARN_DEFAULT(vformat("[GPUCuller] Instance counter async readback enqueue failed (%d)", int(readback_err)));
        }
    }
    if (!p_inputs.device->is_main_rendering_device()) {
        p_inputs.device->submit();
    }

    last_instance_visible_chunk_count = visible_chunk_count;

    r_summary.visible_after_culling = visible_chunk_count;
    uint64_t candidate_count = uint64_t(instance_count) * uint64_t(chunk_dispatch);
    r_summary.culling_candidate_count = candidate_count > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(candidate_count);
    r_summary.used_instance_pipeline = true;
    r_summary.culling_time_ms = (OS::get_singleton()->get_ticks_usec() - p_start_time_usec) / 1000.0f;
    culling_state.cull_time_ms = r_summary.culling_time_ms;
    culling_state.gpu_visible_indices_buffer = p_inputs.visible_chunk_buffer;
    culling_state.gpu_visible_indices_device = p_inputs.device;
    culling_state.gpu_visible_indices_count = visible_chunk_count;
    culling_state.total_splats_pre_cull = r_summary.culling_candidate_count;
    return true;
}

GPUCuller::CullingSummary GPUCuller::cull_for_view(const Transform3D &p_cam_transform, const Projection &p_projection,
        const Size2i &p_viewport_size, const CullingInputs &p_inputs) {
    CullingSummary summary;
    uint64_t start_time = OS::get_singleton()->get_ticks_usec();

    update_culling_settings();
    update_lod_cache();

    const bool using_preculled = p_inputs.preculled_indices != nullptr;
    if (!using_preculled || p_inputs.preculled_generation != culling_state.preculled_generation) {
        culling_state.culled_indices.clear();
    }
    culling_state.culled_distances_sq.clear();
    culling_state.culled_importance_weights.clear();
    culling_state.gpu_visible_indices_buffer = RID();
    culling_state.gpu_visible_indices_device = nullptr;
    culling_state.gpu_visible_indices_count = 0;
    culling_state.culled_by_frustum = 0;
    culling_state.culled_by_distance = 0;
    culling_state.culled_by_screen = 0;
    culling_state.culled_by_importance = 0;
    culling_state.culled_by_limit = 0;
    culling_state.total_splats_pre_cull = 0;
    culling_state.visible_static_chunk_indices.clear();

    if (using_preculled) {
        if (p_inputs.preculled_generation != culling_state.preculled_generation) {
            culling_state.culled_indices = *p_inputs.preculled_indices;
            culling_state.preculled_generation = p_inputs.preculled_generation;
        }
        summary.visible_after_culling = static_cast<uint32_t>(culling_state.culled_indices.size());
        summary.culling_candidate_count = summary.visible_after_culling;
        summary.used_hierarchical_culling = false;
        summary.culling_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;
        culling_state.cull_time_ms = summary.culling_time_ms;
        culling_state.total_splats_pre_cull = p_inputs.preculled_total_splats > 0
                ? p_inputs.preculled_total_splats
                : summary.visible_after_culling;
        return summary;
    }

    Transform3D view_transform = p_cam_transform;
    Transform3D camera_transform = view_transform.affine_inverse();

    Vector<Plane> planes;
    if (culling_config.frustum_culling) {
        planes = p_projection.get_projection_planes(camera_transform);
    }

    // Instance pipeline takes priority - check it first BEFORE gaussian_data check
    if (instance_inputs_valid) {
        CullParams instance_params;
        instance_params.world_to_camera_transform = view_transform;
        instance_params.projection = p_projection;
        instance_params.frustum_planes = planes;
        instance_params.frustum_plane_slack = culling_config.cull_frustum_plane_slack;
        instance_params.frustum_culling_enabled = culling_config.frustum_culling;
        if (GaussianSplatting::debug_trace_is_enabled()) {
            GaussianSplatting::debug_trace_record_event("cull",
                    vformat("Instance pipeline path: instance_count=%d", instance_inputs.instance_count),
                    false);
        }
        if (_gpu_frustum_cull_instance(instance_params, instance_inputs, start_time, summary)) {
            summary.used_hierarchical_culling = false;
            return summary;
        }
    }

    // Legacy path: requires gaussian_data or test positions
    if (!p_inputs.gaussian_data.is_valid() &&
            (!p_inputs.test_positions || p_inputs.test_positions->is_empty())) {
        summary.culling_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;
        culling_state.cull_time_ms = summary.culling_time_ms;
        return summary;
    }

    Vector3 camera_pos = camera_transform.origin;
    bool orthographic = p_projection.is_orthogonal();

    float viewport_height = p_viewport_size.y > 0 ? (float)p_viewport_size.y : 720.0f;
    float vertical_scale = Math::abs(p_projection[1][1]);
    if (vertical_scale <= 0.0f) {
        vertical_scale = 1.0f;
    }
    float pixel_scale_y = vertical_scale * (viewport_height * 0.5f);

    bool gpu_cull_result = false;
    if (p_inputs.gpu_cull_attempted) {
        // PERF (#659): Pass pre-computed inverse to avoid redundant affine_inverse() call
        gpu_cull_result = _gpu_frustum_cull(view_transform, camera_transform, p_projection, p_viewport_size, planes, pixel_scale_y, orthographic,
                start_time, p_inputs.gpu_input, p_inputs.max_splats, p_inputs.readback_indices, p_inputs.readback_distances,
                p_inputs.readback_importance);
    }
    if (p_inputs.gpu_cull_attempted && gpu_cull_result) {
        uint32_t visible_after = static_cast<uint32_t>(culling_state.culled_indices.size());
        if (visible_after == 0 && culling_state.gpu_visible_indices_count > 0) {
            visible_after = culling_state.gpu_visible_indices_count;
        }
        summary.visible_after_culling = visible_after;
        summary.culling_candidate_count = culling_state.total_splats_pre_cull;
        summary.culled_frustum_count = culling_state.culled_by_frustum;
        summary.culled_distance_count = culling_state.culled_by_distance;
        summary.culled_screen_count = culling_state.culled_by_screen;
        summary.culled_importance_count = culling_state.culled_by_importance;
        summary.used_hierarchical_culling = false;
        summary.culling_time_ms = culling_state.cull_time_ms;
        return summary;
    }

    auto sphere_intersects_planes = [](const Vector3 &p_center, float p_radius, const Vector<Plane> &p_planes) {
        if (p_planes.is_empty()) {
            return true;
        }
        const int plane_count = p_planes.size();
        for (int i = 0; i < plane_count; i++) {
            if (p_planes[i].distance_to(p_center) > p_radius) {
                return false;
            }
        }
        return true;
    };

    // FIX (#660): Accept pre-computed view_pos to avoid duplicate view_transform.xform() calls
    auto compute_screen_size = [&](const Vector3 &p_view_pos, float p_radius) {
        if (orthographic) {
            return (p_radius * pixel_scale_y) * 2.0f;
        }

        float depth = -p_view_pos.z;
        if (depth + p_radius <= 0.0f) {
            return 0.0f;
        }
        float screen_depth = MAX(depth, 0.0001f);
        float screen_radius = (p_radius * pixel_scale_y) / screen_depth;
        return screen_radius * 2.0f;
    };

    bool used_hierarchical = false;
    uint32_t candidate_count = 0;

    if (p_inputs.gaussian_data.is_valid()) {
        const LocalVector<Gaussian> &gaussians = p_inputs.gaussian_data->get_gaussian_storage();
        culling_state.total_splats_pre_cull = gaussians.size();

        LocalVector<uint32_t> candidate_indices;
        LocalVector<float> candidate_weights;

        bool using_static_chunks = !culling_state.static_chunks.is_empty();

        if (using_static_chunks) {
            used_hierarchical = false;
            HashSet<uint32_t> unique_indices;
            unique_indices.reserve(culling_state.total_splats_pre_cull);

            const int chunk_count = static_cast<int>(culling_state.static_chunks.size());
            for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
                const StaticChunk &chunk = culling_state.static_chunks[chunk_idx];
                const int index_count = chunk.indices.size();

                float chunk_radius = chunk.radius;
                if (chunk_radius <= 0.0f) {
                    const Vector3 half_extents = chunk.bounds.size * 0.5f;
                    chunk_radius = half_extents.length();
                }

                if (culling_config.frustum_culling &&
                        !sphere_intersects_planes(chunk.center, chunk_radius, planes)) {
                    culling_state.culled_by_frustum += static_cast<uint32_t>(MAX(index_count, 0));
                    continue;
                }

                culling_state.visible_static_chunk_indices.push_back(chunk_idx);

                for (int index_idx = 0; index_idx < index_count; index_idx++) {
                    uint32_t idx = chunk.indices[index_idx];
                    if (idx >= (uint32_t)gaussians.size()) {
                        continue;
                    }
                    if (unique_indices.has(idx)) {
                        continue;
                    }
                    unique_indices.insert(idx);
                    candidate_indices.push_back(idx);
                }
            }

            candidate_count = static_cast<uint32_t>(candidate_indices.size());
        } else {
            ensure_hierarchical_structure(p_inputs.gaussian_data);

            if (culling_state.hierarchical_structure && !culling_state.hierarchical_structure_dirty) {
                RendererSceneCull::Frustum frustum_obj(planes);
                uint32_t max_query = p_inputs.max_splats > 0
                        ? p_inputs.max_splats
                        : static_cast<uint32_t>(gaussians.size());
                GaussianSplatting::HierarchicalSplatStructure::QueryResult query =
                        culling_state.hierarchical_structure->query_visible_splats(frustum_obj, camera_pos, culling_config.lod_bias, max_query);

                candidate_indices = query.visible_indices;
                candidate_weights = query.lod_weights;
                used_hierarchical = true;
                candidate_count = static_cast<uint32_t>(candidate_indices.size());
            } else {
                used_hierarchical = false;
                candidate_count = culling_state.total_splats_pre_cull;
                candidate_indices.resize(culling_state.total_splats_pre_cull);
                for (uint32_t i = 0; i < culling_state.total_splats_pre_cull; i++) {
                    candidate_indices[i] = i;
                }
            }
        }

        culling_state.culled_indices.reserve(candidate_indices.size());
        culling_state.culled_distances_sq.reserve(candidate_indices.size());
        culling_state.culled_importance_weights.reserve(candidate_indices.size());

        const uint32_t candidate_list_count = static_cast<uint32_t>(candidate_indices.size());
        const uint32_t candidate_weight_count = static_cast<uint32_t>(candidate_weights.size());
        for (uint32_t i = 0; i < candidate_list_count; i++) {
            uint32_t idx = candidate_indices[i];
            if (idx >= (uint32_t)gaussians.size()) {
                continue;
            }

            const Gaussian &g = gaussians[idx];
            Vector3 position = g.position;
            Vector3 scale = g.scale;

            float radius = MAX(MAX(scale.x, scale.y), scale.z);
            if (radius <= 0.0f) {
                radius = 1.0f;
            }
            radius *= culling_config.cull_radius_multiplier;
            float frustum_radius = radius * MAX(culling_config.cull_frustum_plane_slack, 1.0f);

            if (culling_config.frustum_culling && !sphere_intersects_planes(position, frustum_radius, planes)) {
                culling_state.culled_by_frustum++;
                continue;
            }

            Vector3 view_pos = view_transform.xform(position);
            if (!orthographic) {
                float depth = -view_pos.z;
                if (depth + radius <= 0.0f) {
                    culling_state.culled_by_frustum++;
                    continue;
                }
            }

            // FIX (#660): Reuse pre-computed view_pos instead of recomputing in lambda
            float screen_size = compute_screen_size(view_pos, radius);
            if (culling_state.tiny_splat_screen_radius_px > 0.0f &&
                    screen_size < culling_state.tiny_splat_screen_radius_px * 2.0f) {
                culling_state.culled_by_screen++;
                continue;
            }
            if (culling_config.lod_cached_min_screen_threshold > 0.0f && screen_size < culling_config.lod_cached_min_screen_threshold) {
                culling_state.culled_by_screen++;
                continue;
            }

            float distance_sq = (position - camera_pos).length_squared();
            if (culling_config.lod_cached_max_distance_sq > 0.0f && distance_sq > culling_config.lod_cached_max_distance_sq) {
                culling_state.culled_by_distance++;
                continue;
            }

            float importance_weight = 1.0f;
            if (!candidate_weights.is_empty() && i < candidate_weight_count) {
                importance_weight = candidate_weights[i];
            } else {
                float opacity = CLAMP(g.opacity, 0.0f, 1.0f);
                float scale_max = MAX(MAX(scale.x, scale.y), scale.z);
                importance_weight = opacity * scale_max;
            }

            if (importance_weight < culling_config.importance_cull_threshold) {
                culling_state.culled_by_importance++;
                continue;
            }

            culling_state.culled_indices.push_back(idx);
            culling_state.culled_distances_sq.push_back(distance_sq);
            culling_state.culled_importance_weights.push_back(importance_weight);
        }
    } else {
        const LocalVector<Vector3> &positions = *p_inputs.test_positions;
        const LocalVector<Vector3> *scales = p_inputs.test_scales;
        culling_state.total_splats_pre_cull = positions.size();
        used_hierarchical = false;
        candidate_count = culling_state.total_splats_pre_cull;

        culling_state.culled_indices.reserve(culling_state.total_splats_pre_cull);
        culling_state.culled_distances_sq.reserve(culling_state.total_splats_pre_cull);
        culling_state.culled_importance_weights.reserve(culling_state.total_splats_pre_cull);

        const uint32_t test_splat_count = static_cast<uint32_t>(positions.size());
        const uint32_t test_scale_count = scales ? static_cast<uint32_t>(scales->size()) : 0;
        for (uint32_t i = 0; i < test_splat_count; i++) {
            const Vector3 &pos = positions[i];
            Vector3 scale = (scales && i < test_scale_count) ? (*scales)[i] : Vector3(1, 1, 1);

            float radius = MAX(MAX(scale.x, scale.y), scale.z);
            if (radius <= 0.0f) {
                radius = 1.0f;
            }
            radius *= culling_config.cull_radius_multiplier;
            float frustum_radius = radius * MAX(culling_config.cull_frustum_plane_slack, 1.0f);

            if (culling_config.frustum_culling && !sphere_intersects_planes(pos, frustum_radius, planes)) {
                culling_state.culled_by_frustum++;
                continue;
            }

            Vector3 view_pos = view_transform.xform(pos);
            if (!orthographic) {
                float depth = -view_pos.z;
                if (depth + radius <= 0.0f) {
                    culling_state.culled_by_frustum++;
                    continue;
                }
            }

            // FIX (#660): Reuse pre-computed view_pos instead of recomputing in lambda
            float screen_size = compute_screen_size(view_pos, radius);
            if (culling_state.tiny_splat_screen_radius_px > 0.0f &&
                    screen_size < culling_state.tiny_splat_screen_radius_px * 2.0f) {
                culling_state.culled_by_screen++;
                continue;
            }
            if (culling_config.lod_cached_min_screen_threshold > 0.0f && screen_size < culling_config.lod_cached_min_screen_threshold) {
                culling_state.culled_by_screen++;
                continue;
            }

            float distance_sq = (pos - camera_pos).length_squared();
            if (culling_config.lod_cached_max_distance_sq > 0.0f && distance_sq > culling_config.lod_cached_max_distance_sq) {
                culling_state.culled_by_distance++;
                continue;
            }

            float importance_weight = 1.0f;
            if (importance_weight < culling_config.importance_cull_threshold) {
                culling_state.culled_by_importance++;
                continue;
            }

            culling_state.culled_indices.push_back(i);
            culling_state.culled_distances_sq.push_back(distance_sq);
            culling_state.culled_importance_weights.push_back(importance_weight);
        }
    }

    summary.visible_after_culling = static_cast<uint32_t>(culling_state.culled_indices.size());
    summary.culling_candidate_count = candidate_count;
    summary.culled_frustum_count = culling_state.culled_by_frustum;
    summary.culled_distance_count = culling_state.culled_by_distance;
    summary.culled_screen_count = culling_state.culled_by_screen;
    summary.culled_importance_count = culling_state.culled_by_importance;
    summary.used_hierarchical_culling = used_hierarchical;
    summary.culling_time_ms = culling_state.cull_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;

    return summary;
}

void GPUCuller::apply_overflow_feedback(const RasterOverflowStats &p_stats, uint32_t p_splat_count, uint32_t p_tile_count,
        IOverflowAutoTuner *p_auto_tuner, float p_avg_splats_per_tile) {
    if (!culling_state.overflow_autotune_enabled || p_splat_count == 0) {
        return;
    }

    if (!p_auto_tuner) {
        return;
    }

    OverflowAutoTuneConfig config;
    config.enabled = culling_state.overflow_autotune_enabled;
    config.trigger_ratio = culling_state.overflow_autotune_trigger_ratio;
    config.importance_step = culling_state.overflow_autotune_importance_step;
    config.importance_max = culling_state.overflow_autotune_importance_max;
    config.importance_decay = culling_state.overflow_autotune_importance_decay;
    config.tiny_step = culling_state.overflow_autotune_tiny_step;
    config.tiny_max = culling_state.overflow_autotune_tiny_max;
    config.tiny_decay = culling_state.overflow_autotune_tiny_decay;
    p_auto_tuner->set_config(config);
    p_auto_tuner->set_baselines(culling_config.importance_cull_baseline, culling_state.tiny_splat_screen_radius_baseline);

    // Calculate screen coverage estimate for close-up detection
    // Higher average splats per tile indicates splats are covering more screen area (close-up view)
    // Normalize by a typical "far view" baseline of ~10 splats/tile
    float screen_coverage_estimate = 0.0f;
    if (p_avg_splats_per_tile > 0.0f) {
        constexpr float kFarViewBaseline = 10.0f;  // Typical splats/tile when viewing from distance
        screen_coverage_estimate = p_avg_splats_per_tile / kFarViewBaseline;
    }

    AutoTuneResult result = p_auto_tuner->apply_feedback(p_stats, p_splat_count, p_tile_count, screen_coverage_estimate);
    if (result.parameters_changed) {
        culling_config.importance_cull_threshold = result.new_importance_threshold;
        culling_state.tiny_splat_screen_radius_px = result.new_tiny_splat_radius;
        culling_config.cull_params_dirty = true;
    }
}
