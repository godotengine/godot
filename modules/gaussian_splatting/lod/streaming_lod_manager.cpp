#include "streaming_lod_manager.h"
#include "core/os/os.h"
#include "core/math/math_funcs.h"
#include "core/templates/local_vector.h"
#include "core/error/error_macros.h"
#include "../logger/gs_logger.h"
#include "../core/gaussian_splat_manager.h"
#include <algorithm>
#include <cstring>
#include <cstdint>

namespace {

static constexpr float CONTENT_DISTANCE_INVALID = -1.0f;

static inline bool _is_valid_content_distance(float p_distance) {
    return p_distance >= 0.0f && Math::is_finite(p_distance);
}

static inline bool _frustum_intersects_aabb(const GaussianSplatting::Frustum &p_frustum, const AABB &p_bounds) {
    if (p_frustum.plane_count == 0 && p_frustum.planes.is_empty()) {
        return true;
    }

    const Plane *planes_ptr = p_frustum.planes_ptr;
    uint32_t plane_count = p_frustum.plane_count;
    if (!planes_ptr) {
        planes_ptr = p_frustum.planes.ptr();
        plane_count = p_frustum.planes.size();
    }

    const Vector3 min_bound = p_bounds.position;
    const Vector3 max_bound = p_bounds.position + p_bounds.size;

    for (uint32_t i = 0; i < plane_count; i++) {
        const Plane &plane = planes_ptr[i];
        Vector3 negative_vertex;
        negative_vertex.x = plane.normal.x >= 0.0f ? min_bound.x : max_bound.x;
        negative_vertex.y = plane.normal.y >= 0.0f ? min_bound.y : max_bound.y;
        negative_vertex.z = plane.normal.z >= 0.0f ? min_bound.z : max_bound.z;

        // `distance_to >= 0` is outside, matching point-vs-frustum semantics in LOD paths.
        if (plane.distance_to(negative_vertex) >= 0.0f) {
            return false;
        }
    }

    return true;
}

} // namespace

namespace GaussianSplatting {

StreamingLODManager::StreamingLODManager()
    : has_content_bounds(false), rd(nullptr), last_frame_delta(1.0f / 60.0f), should_stop_loading(false) {

    spatial_structure = std::make_unique<HierarchicalSplatStructure>();
    adaptive_lod = std::make_unique<AdaptiveLODSystem>();
    clusterer = std::make_unique<SplatClusterer>();
    gpu_stream = std::make_unique<GaussianMemoryStream>();
    painterly_manager = std::make_unique<PainterlyManager>();

    stats.reset();
}

StreamingLODManager::~StreamingLODManager() {
    stop_async_loading();

    // Release GPU resources
    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        for (auto &lod : lod_levels) {
            if (lod.gpu_buffer.is_valid()) {
                release_gpu_buffer(lod);
            }
        }
    }
}

void StreamingLODManager::_initialize_gpu_streaming(const Vector<GaussianData>& base_splats) {
    // Initialize GPU memory streaming
    uint32_t base_gaussians = base_splats.size();
    if (base_gaussians == 0) {
        base_gaussians = 1;
    }

    // Derive a reasonable streaming capacity from the configured GPU memory budget.
    // GaussianMemoryStream internally triple-buffers GPU allocations, so take that into account
    // when converting the byte budget to a gaussian capacity.
    constexpr uint64_t stream_buffer_count = 3; // Matches GaussianMemoryStream triple buffering
    const uint64_t bytes_per_gaussian = sizeof(PackedGaussian) + sizeof(uint32_t); // GPU + sort key

    uint64_t gpu_budget_bytes = config.target_gpu_memory > 0 ? config.target_gpu_memory : config.max_gpu_memory;
    if (config.max_gpu_memory > 0) {
        gpu_budget_bytes = std::min<uint64_t>(gpu_budget_bytes, config.max_gpu_memory);
    }

    // Prevent division by zero and ensure we always have capacity for the base dataset.
    uint64_t budget_capacity = 0;
    if (gpu_budget_bytes > 0) {
        uint64_t denom = bytes_per_gaussian * stream_buffer_count;
        if (denom > 0) {
            budget_capacity = gpu_budget_bytes / denom;
        }
    }
    if (budget_capacity == 0) {
        budget_capacity = base_gaussians;
    } else if (budget_capacity < base_gaussians) {
        GS_LOG_STREAMING_WARN(vformat("Streaming GPU budget (%d MB) is insufficient for %d base splats. Clamping to base capacity.",
                static_cast<int>(gpu_budget_bytes / (1024 * 1024)), base_gaussians));
        budget_capacity = base_gaussians;
    }

    uint64_t desired_capacity = static_cast<uint64_t>(base_gaussians) * std::max<uint32_t>(config.num_lod_levels, 1u);
    desired_capacity = std::min<uint64_t>(desired_capacity, static_cast<uint64_t>(UINT32_MAX));

    uint64_t max_gaussians_u64 = std::min<uint64_t>(desired_capacity, budget_capacity);
    if (max_gaussians_u64 < base_gaussians) {
        max_gaussians_u64 = base_gaussians;
    }

    uint32_t max_gaussians = static_cast<uint32_t>(max_gaussians_u64);
    gpu_stream->initialize(rd, max_gaussians);
}

void StreamingLODManager::start_async_loading() {
    if (!config.enable_async_loading || loading_thread.joinable()) {
        return;
    }

    should_stop_loading = false;
    loading_thread = std::thread(&StreamingLODManager::async_load_worker, this);
}

void StreamingLODManager::stop_async_loading() {
    should_stop_loading = true;
    load_cv.notify_all();
    if (loading_thread.joinable()) {
        loading_thread.join();
    }
}

void StreamingLODManager::clear_async_load_queues() {
    std::lock_guard<std::mutex> lock(load_queue_mutex);
    while (!load_queue.empty()) {
        load_queue.pop();
    }
    while (!ready_queue.empty()) {
        ready_queue.pop();
    }
    pending_async_loads.clear();
}

void StreamingLODManager::initialize(
    const Vector<GaussianData>& base_splats,
    const StreamingConfig& p_config) {

    stop_async_loading();
    clear_async_load_queues();

    config = p_config;
    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        stats.reset();
    }
    has_content_bounds = false;

    if (!base_splats.is_empty()) {
        Vector3 min_bound = base_splats[0].position;
        Vector3 max_bound = min_bound;

        for (uint32_t i = 1; i < base_splats.size(); i++) {
            const Vector3 &pos = base_splats[i].position;
            min_bound.x = MIN(min_bound.x, pos.x);
            min_bound.y = MIN(min_bound.y, pos.y);
            min_bound.z = MIN(min_bound.z, pos.z);
            max_bound.x = MAX(max_bound.x, pos.x);
            max_bound.y = MAX(max_bound.y, pos.y);
            max_bound.z = MAX(max_bound.z, pos.z);
        }

        content_bounds = AABB(min_bound, max_bound - min_bound);
        has_content_bounds = true;
    }

    // Use the primary rendering device from GaussianSplatManager
    // This respects RenderDoc compatibility mode (no local devices)
    GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
    if (manager) {
        rd = manager->get_primary_rendering_device();
    }
    if (!rd) {
        // Fallback: try to get main rendering device directly
        rd = RenderingServer::get_singleton()->get_rendering_device();
    }

    _initialize_gpu_streaming(base_splats);

    // Build spatial structure
    HierarchicalSplatStructure::BuildParams build_params;
    build_params.max_depth = 8;
    build_params.min_splats_per_node = 16;
    build_params.compute_importance = true;
    spatial_structure->build_hierarchy(base_splats, build_params);

    // Initialize adaptive LOD system
    AdaptiveLODSystem::LODConfig lod_config;
    lod_config.max_splats_per_frame = 500000;
    lod_config.lod_bias = 1.0f;
    lod_config.enable_temporal_coherence = true;
    lod_config.enable_painterly_mode = config.enable_painterly_mode;
    adaptive_lod->initialize(lod_config);

    if (config.enable_painterly_mode) {
        PainterlyManager::Settings painterly_settings;
        painterly_settings.base_seed = config.painterly_seed;
        painterly_settings.blend_rate = config.painterly_transition_rate;
        painterly_settings.hold_strength = config.painterly_hold_strength;
        painterly_manager->configure(painterly_settings);
    }

    // Generate LOD levels
    generate_lod_levels(base_splats);
    if (config.enable_async_loading) {
        start_async_loading();
    }

    GS_LOG_STREAMING_INFO(vformat("StreamingLODManager initialized with %d LOD levels, %d base splats",
            lod_levels.size(), base_splats.size()));
}

void StreamingLODManager::set_config(const StreamingConfig& p_config) {
    const bool was_async = config.enable_async_loading;
    const bool will_be_async = p_config.enable_async_loading;
    config = p_config;

    if (was_async && !will_be_async) {
        stop_async_loading();
        clear_async_load_queues();
        std::lock_guard<std::mutex> lock(lod_mutex);
        for (auto &lod : lod_levels) {
            if (!lod.is_loaded) {
                lod.is_loading = false;
            }
        }
    } else if (!was_async && will_be_async) {
        start_async_loading();
    }

    if (was_async != will_be_async) {
        std::lock_guard<std::mutex> lock(lod_mutex);
        stats.async_prepare_jobs_completed = 0;
        stats.async_apply_jobs_completed = 0;
        stats.async_prepare_observed_off_main_thread = false;
        stats.async_apply_observed_on_main_thread = false;
        stats.async_prepare_main_thread_violations = 0;
        stats.async_apply_off_main_thread_violations = 0;
        stats.performance.avg_load_time_ms = 0.0f;
        stats.performance.avg_async_prepare_time_ms = 0.0f;
        stats.performance.avg_async_apply_time_ms = 0.0f;
    }
}

void StreamingLODManager::generate_lod_levels(const Vector<GaussianData>& base_splats) {
    std::lock_guard<std::mutex> lock(lod_mutex);
    uint32_t num_levels = config.num_lod_levels;
    lod_levels.resize(num_levels);
    lod_data.resize(num_levels);

    float base_distance = 10.0f;

    for (uint32_t i = 0; i < num_levels; i++) {
        LODLevel& lod = lod_levels.write[i];
        lod.level_index = i;
        lod.min_distance = (i == 0) ? 0.0f : base_distance * pow(config.lod_distance_multiplier, i - 1);
        lod.max_distance = base_distance * pow(config.lod_distance_multiplier, i);

        // Generate LOD data
        generate_single_lod(i, base_splats);

        // Update statistics
        lod.splat_count = lod_data[i].size();
        lod.cpu_memory_bytes = lod.splat_count * sizeof(GaussianData);
        lod.is_loaded = false;
        lod.is_loading = false;

        GS_LOG_STREAMING_INFO(vformat("LOD %d: %d splats, distance [%.1f, %.1f]",
                i, lod.splat_count, lod.min_distance, lod.max_distance));
    }
}

void StreamingLODManager::generate_single_lod(
    uint32_t level,
    const Vector<GaussianData>& source_splats) {

    Vector<GaussianData>& lod_splats = lod_data.write[level];

    if (level == 0) {
        // LOD 0 is full detail
        lod_splats = source_splats;
    } else {
        // Generate clustered LOD
        auto clustering_result = clusterer->generate_lod_clusters(source_splats, level);

        // Convert clusters to GaussianData
        lod_splats.resize(clustering_result.clusters.size());

        for (uint32_t i = 0; i < clustering_result.clusters.size(); i++) {
            lod_splats.write[i] = clustering_result.clusters[i].to_gaussian_data();
        }

        GS_LOG_STREAMING_INFO(vformat("Generated LOD %d: %d splats (%.1f%% reduction)",
                level, lod_splats.size(),
                clustering_result.reduction_ratio * 100.0f));
    }

    if (config.enable_painterly_mode) {
        painterly_manager->ensure_metadata_for_level(lod_splats, level);
    }
}

void StreamingLODManager::update(const Camera3D* camera, float delta_time) {
    if (!camera) return;

    struct AsyncCandidate {
        uint32_t level = UINT32_MAX;
        float priority = 0.0f;
        bool needs_enqueue = false;
    };

    last_frame_delta = delta_time;

    uint64_t frame_start = OS::get_singleton()->get_ticks_usec();
    streaming_time_budget = config.stream_budget_ms * 1000;  // Convert to microseconds

    const Vector3 camera_pos = camera->get_global_transform().origin;
    const Frustum camera_frustum = camera->get_frustum();
    const float content_distance = compute_distance_to_content(camera_pos);
    const bool content_distance_valid = _is_valid_content_distance(content_distance);
    const bool content_in_frustum = is_content_in_frustum(camera_frustum);

    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        stats.load_requests_last_frame = 0;
        stats.frustum_rejected_lods_last_frame = 0;
        stats.content_visible_last_frame = content_in_frustum;
        stats.content_distance_valid_last_frame = content_distance_valid;
    }

    // Update camera history for prediction
    camera_history.update(camera_pos, delta_time);

    // Unload distant LODs
    unload_distant_lods(camera_pos, camera_frustum);

    // Load nearby LODs
    LocalVector<AsyncCandidate> async_candidates;
    LocalVector<uint32_t> sync_loads;
    uint32_t frustum_rejected_lods = 0;
    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        for (uint32_t i = 0; i < lod_levels.size(); i++) {
            const LODLevel &lod = lod_levels[i];
            if (!lod.is_loaded) {
                if (!content_in_frustum) {
                    frustum_rejected_lods++;
                } else if (content_distance_valid &&
                        content_distance >= lod.min_distance - config.load_ahead_distance &&
                        content_distance <= lod.max_distance + config.load_ahead_distance) {
                    if (config.enable_async_loading) {
                        AsyncCandidate candidate;
                        candidate.level = i;
                        candidate.priority = compute_lod_priority(lod, camera_pos, &camera_frustum);
                        candidate.needs_enqueue = !lod.is_loading;
                        async_candidates.push_back(candidate);
                    } else if (!lod.is_loading) {
                        sync_loads.push_back(i);
                    }
                }
            }

            // Check streaming time budget
            uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - frame_start;
            if (elapsed > streaming_time_budget) {
                break;  // Exceeded budget for this frame
            }
        }

        stats.frustum_rejected_lods_last_frame = frustum_rejected_lods;
    }

    std::sort(
        async_candidates.ptr(),
        async_candidates.ptr() + async_candidates.size(),
        [](const AsyncCandidate& a, const AsyncCandidate& b) {
            return a.priority > b.priority;
        });

    LocalVector<uint32_t> desired_async_levels;
    LocalVector<uint32_t> new_async_requests;
    HashSet<uint32_t> seen_desired_async_levels;
    HashSet<uint32_t> seen_new_async_requests;
    seen_desired_async_levels.reserve(async_candidates.size());
    seen_new_async_requests.reserve(async_candidates.size());

    for (uint32_t i = 0; i < async_candidates.size(); i++) {
        const AsyncCandidate &candidate = async_candidates[i];
        if (!seen_desired_async_levels.has(candidate.level)) {
            seen_desired_async_levels.insert(candidate.level);
            desired_async_levels.push_back(candidate.level);
        }
        if (candidate.needs_enqueue) {
            seen_new_async_requests.insert(candidate.level);
        }
    }

    LocalVector<uint32_t> predicted_desired_levels;
    LocalVector<uint32_t> predicted_new_levels;
    if (config.enable_predictive_loading && content_distance_valid && content_in_frustum) {
        PredictedMovement prediction = predict_camera_movement(camera_pos, delta_time);
        collect_predicted_lods(prediction, camera_frustum, predicted_desired_levels, predicted_new_levels);
    }

    if (config.enable_async_loading) {
        for (uint32_t i = 0; i < predicted_desired_levels.size(); i++) {
            const uint32_t level = predicted_desired_levels[i];
            if (!seen_desired_async_levels.has(level)) {
                seen_desired_async_levels.insert(level);
                desired_async_levels.push_back(level);
            }
        }
        for (uint32_t i = 0; i < predicted_new_levels.size(); i++) {
            const uint32_t level = predicted_new_levels[i];
            seen_new_async_requests.insert(level);
        }

        const uint32_t max_loads = config.max_concurrent_loads == 0
                ? UINT32_MAX
                : config.max_concurrent_loads;
        if (desired_async_levels.size() > max_loads) {
            desired_async_levels.resize(max_loads);
        }

        new_async_requests.clear();
        for (uint32_t i = 0; i < desired_async_levels.size(); i++) {
            const uint32_t level = desired_async_levels[i];
            if (seen_new_async_requests.has(level)) {
                new_async_requests.push_back(level);
            }
        }

        reprioritize_async_loads(desired_async_levels);
        for (uint32_t i = 0; i < new_async_requests.size(); i++) {
            enqueue_load_request(new_async_requests[i], desired_async_levels);
        }
    }
    for (uint32_t i = 0; i < sync_loads.size(); i++) {
        stream_lod_level(sync_loads[i]);
    }

    // Predictive loading
    if (!config.enable_async_loading) {
        for (uint32_t i = 0; i < predicted_new_levels.size(); i++) {
            stream_lod_level(predicted_new_levels[i]);
        }
    }

    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        stats.load_requests_last_frame = config.enable_async_loading
                ? new_async_requests.size()
                : sync_loads.size() + predicted_new_levels.size();
    }

    apply_completed_async_loads(frame_start);

    // Update adaptive quality
    if (config.enable_adaptive_quality) {
        float current_fps = 1.0f / delta_time;
        adaptive_lod->update_adaptive_quality(current_fps);
    }

    // Enforce memory limits
    enforce_memory_limits();

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        stats.loaded_lod_levels = 0;
        stats.loading_lod_levels = 0;
        stats.total_gpu_memory = 0;
        stats.total_cpu_memory = 0;

        for (const auto &lod : lod_levels) {
            if (lod.is_loaded) {
                stats.loaded_lod_levels++;
                stats.total_gpu_memory += lod.gpu_memory_bytes;
            }
            if (lod.is_loading) {
                stats.loading_lod_levels++;
            }
            stats.total_cpu_memory += lod.cpu_memory_bytes;
        }
    }
}

StreamingLODManager::VisibleSplats StreamingLODManager::get_visible_splats(
    const Camera3D* camera,
    uint32_t max_splats) {

    VisibleSplats result;
    result.total_count = 0;

    if (!camera || !spatial_structure->get_root()) {
        return result;
    }

    // Hierarchical hybrid selection does not require a per-call aggregated splat vector.
    // Passing an empty vector avoids O(N) append churn on every frame.
    Vector<GaussianData> selection_input;

    auto selection = adaptive_lod->select_lod_splats(
        selection_input,
        camera,
        spatial_structure.get(),
        AdaptiveLODSystem::HYBRID
    );

    if (config.enable_painterly_mode) {
        painterly_manager->apply_temporal_smoothing(selection, last_frame_delta);
    }

    // Convert to result format
    result.indices = selection.visible_indices;
    result.lod_weights = selection.lod_weights;
    result.lod_levels = selection.lod_levels;
    if (config.enable_painterly_mode) {
        result.painterly_metadata = selection.painterly_metadata;
        result.painterly_seeds = selection.painterly_seeds;
        result.painterly_prev_seeds = selection.painterly_prev_seeds;
        result.painterly_blend_weights = selection.painterly_blend_weights;
    }
    result.total_count = result.indices.size();

    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        stats.total_visible_splats = result.total_count;
    }

    return result;
}

void StreamingLODManager::stream_lod_level(uint32_t level) {
    Vector<GaussianData> source_data;
    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        if (level >= lod_levels.size()) {
            return;
        }

        LODLevel &lod = lod_levels.write[level];
        if (lod.is_loaded || lod.is_loading) {
            return;
        }

        if (level >= lod_data.size() || lod_data[level].is_empty()) {
            return;
        }

        lod.is_loading = true;
        source_data = lod_data[level];
    }

    uint64_t start_time = OS::get_singleton()->get_ticks_usec();
    LocalVector<::Gaussian> gpu_gaussians;
    prepare_gpu_upload_data(source_data, gpu_gaussians);

    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        if (level >= lod_levels.size()) {
            return;
        }

        LODLevel &lod = lod_levels.write[level];
        if (!lod.is_loading) {
            return;
        }

        upload_to_gpu(lod, gpu_gaussians, source_data.size());
        if (lod.gpu_buffer.is_valid()) {
            lod.is_loaded = true;
            lod.last_access_time = OS::get_singleton()->get_ticks_msec();
            lod.access_count++;
        }
        lod.is_loading = false;
    }

    float load_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;
    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        stats.performance.avg_load_time_ms =
                stats.performance.avg_load_time_ms * 0.9f + load_time_ms * 0.1f;
    }

    GS_LOG_STREAMING_DEBUG(vformat("Loaded LOD %d in %.2f ms", level, load_time_ms));
}

void StreamingLODManager::prepare_gpu_upload_data(const Vector<GaussianData>& data, LocalVector<::Gaussian>& gpu_gaussians) const {
    if (data.is_empty()) {
        return;
    }

    gpu_gaussians.resize(data.size());

    for (uint32_t i = 0; i < data.size(); i++) {
        const GaussianData &src = data[i];
        ::Gaussian &dst = gpu_gaussians[i];
        dst = ::Gaussian{};

        dst.position = src.position;
        dst.opacity = src.color.a;
        dst.scale = src.scale;
        dst.area = src.area;
        dst.rotation = src.rotation;
        dst.sh_dc = src.color;
        dst.normal = src.normal;
        dst.stroke_age = 0.0f;
        dst.brush_axes = src.painterly.jitter;
        dst.painterly_meta = gaussian_pack_painterly_meta(0);
    }
}

void StreamingLODManager::unload_distant_lods(const Vector3& camera_pos, const Frustum& frustum) {
    const float content_distance = compute_distance_to_content(camera_pos);
    const bool content_distance_valid = _is_valid_content_distance(content_distance);
    const bool content_in_frustum = is_content_in_frustum(frustum);
    LocalVector<uint32_t> to_unload;
    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        for (uint32_t i = 0; i < lod_levels.size(); i++) {
            const LODLevel &lod = lod_levels[i];
            if (lod.is_loaded && !lod.is_loading) {
                if (!content_distance_valid || !content_in_frustum) {
                    to_unload.push_back(i);
                    continue;
                }

                // Check if LOD should be unloaded
                if (content_distance < lod.min_distance - config.unload_distance ||
                        content_distance > lod.max_distance + config.unload_distance) {
                    to_unload.push_back(i);
                }
            }
        }
    }

    for (uint32_t i = 0; i < to_unload.size(); i++) {
        uint64_t start_time = OS::get_singleton()->get_ticks_usec();
        unload_lod_level(to_unload[i]);
        float unload_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;
        {
            std::lock_guard<std::mutex> lock(lod_mutex);
            stats.performance.avg_unload_time_ms =
                    stats.performance.avg_unload_time_ms * 0.9f + unload_time_ms * 0.1f;
        }
    }
}

void StreamingLODManager::unload_lod_level(uint32_t level) {
    std::lock_guard<std::mutex> lock(lod_mutex);
    if (level >= lod_levels.size()) {
        return;
    }

    LODLevel &lod = lod_levels.write[level];
    if (!lod.is_loaded) {
        return;
    }

    const uint64_t unloaded_gpu_bytes = lod.gpu_memory_bytes;

    // Release GPU buffer
    release_gpu_buffer(lod);

    lod.is_loaded = false;
    if (stats.total_gpu_memory >= unloaded_gpu_bytes) {
        stats.total_gpu_memory -= unloaded_gpu_bytes;
    } else {
        stats.total_gpu_memory = 0;
    }
    if (stats.loaded_lod_levels > 0) {
        stats.loaded_lod_levels--;
    }
    GS_LOG_STREAMING_DEBUG(vformat("Unloaded LOD %d", level));
}

void StreamingLODManager::upload_to_gpu(LODLevel& lod, const LocalVector<::Gaussian>& gpu_gaussians, uint32_t splat_count) {
    if (gpu_gaussians.is_empty()) {
        return;
    }

    Error stream_err = gpu_stream->stream_gaussians_immediate(gpu_gaussians);
    if (stream_err != OK) {
        static bool warned_stream_failure = false;
        if (!warned_stream_failure) {
            GS_LOG_STREAMING_WARN("Failed to stream clustered LOD data to the GPU.");
            warned_stream_failure = true;
        }
        return;
    }

    lod.gpu_buffer = gpu_stream->get_current_gpu_buffer();
    lod.gpu_memory_bytes = splat_count * sizeof(GaussianData);
    lod.buffer_capacity = gpu_stream->get_max_gaussians();
}

void StreamingLODManager::release_gpu_buffer(LODLevel& lod) {
    // GPU buffers returned by GaussianMemoryStream are stream-owned.
    // Only clear LOD-local references here; the stream frees its own buffers.
    lod.gpu_buffer = RID();
    lod.gpu_memory_bytes = 0;
    lod.buffer_capacity = 0;
}

void StreamingLODManager::reprioritize_async_loads(const LocalVector<uint32_t>& desired_levels) {
    if (!config.enable_async_loading) {
        return;
    }

    HashSet<uint32_t> desired_set;
    desired_set.reserve(desired_levels.size());
    for (uint32_t i = 0; i < desired_levels.size(); i++) {
        desired_set.insert(desired_levels[i]);
    }

    HashSet<uint32_t> cancelled_levels;
    {
        std::lock_guard<std::mutex> lock(load_queue_mutex);

        LocalVector<AsyncLoadRequest> retained_load_requests;
        while (!load_queue.empty()) {
            AsyncLoadRequest request = std::move(load_queue.front());
            load_queue.pop();
            if (desired_set.has(request.level)) {
                retained_load_requests.push_back(std::move(request));
            } else {
                cancelled_levels.insert(request.level);
            }
        }
        std::queue<AsyncLoadRequest> reordered_load_queue;
        for (uint32_t i = 0; i < desired_levels.size(); i++) {
            for (uint32_t j = 0; j < retained_load_requests.size(); j++) {
                if (retained_load_requests[j].level == desired_levels[i]) {
                    reordered_load_queue.push(std::move(retained_load_requests.write[j]));
                    retained_load_requests.write[j].level = UINT32_MAX;
                    break;
                }
            }
        }
        load_queue = std::move(reordered_load_queue);

        LocalVector<AsyncLoadJob> retained_ready_jobs;
        while (!ready_queue.empty()) {
            AsyncLoadJob job = std::move(ready_queue.front());
            ready_queue.pop();
            if (desired_set.has(job.level)) {
                retained_ready_jobs.push_back(std::move(job));
            } else {
                cancelled_levels.insert(job.level);
            }
        }
        std::queue<AsyncLoadJob> reordered_ready_queue;
        for (uint32_t i = 0; i < desired_levels.size(); i++) {
            for (uint32_t j = 0; j < retained_ready_jobs.size(); j++) {
                if (retained_ready_jobs[j].level == desired_levels[i]) {
                    reordered_ready_queue.push(std::move(retained_ready_jobs.write[j]));
                    retained_ready_jobs.write[j].level = UINT32_MAX;
                    break;
                }
            }
        }
        ready_queue = std::move(reordered_ready_queue);

        HashSet<uint32_t> filtered_pending;
        for (const uint32_t level : pending_async_loads) {
            if (desired_set.has(level)) {
                filtered_pending.insert(level);
            } else {
                cancelled_levels.insert(level);
            }
        }
        pending_async_loads = std::move(filtered_pending);
    }

    if (cancelled_levels.is_empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(lod_mutex);
    for (const uint32_t level : cancelled_levels) {
        if (level < lod_levels.size() && !lod_levels[level].is_loaded) {
            lod_levels.write[level].is_loading = false;
        }
    }
}

StreamingLODManager::PredictedMovement StreamingLODManager::predict_camera_movement(
    const Vector3& current_pos,
    float delta_time) {

    PredictedMovement prediction;
    prediction.predicted_position = camera_history.predict_position(config.prediction_time);
    prediction.predicted_direction = (prediction.predicted_position - current_pos).normalized();
    prediction.confidence = MIN(1.0f, camera_history.avg_speed * delta_time);

    return prediction;
}

void StreamingLODManager::collect_predicted_lods(
    const PredictedMovement& prediction,
    const Frustum& frustum,
    LocalVector<uint32_t>& r_desired_levels,
    LocalVector<uint32_t>& r_new_levels) {
    if (prediction.confidence < 0.5f) {
        return;  // Low confidence, don't prefetch
    }
    if (!is_content_in_frustum(frustum)) {
        return;
    }

    const float predicted_distance = compute_distance_to_content(prediction.predicted_position);
    if (!_is_valid_content_distance(predicted_distance)) {
        return;
    }

    // Check which LODs might be needed at predicted position
    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        for (uint32_t i = 0; i < lod_levels.size(); i++) {
            const LODLevel &lod = lod_levels[i];

            if (!lod.is_loaded) {
                if (predicted_distance >= lod.min_distance - config.load_ahead_distance * 0.5f &&
                        predicted_distance <= lod.max_distance + config.load_ahead_distance * 0.5f) {
                    r_desired_levels.push_back(i);
                    if (!lod.is_loading) {
                        r_new_levels.push_back(i);
                    }
                }
            }
        }
    }
}

void StreamingLODManager::enforce_memory_limits() {
    // Recompute live usage in-loop so pressure decisions track current unload state.
    while (true) {
        uint64_t live_gpu_memory = 0;
        {
            std::lock_guard<std::mutex> lock(lod_mutex);
            for (const LODLevel &lod : lod_levels) {
                if (lod.is_loaded) {
                    live_gpu_memory += lod.gpu_memory_bytes;
                }
            }
            stats.total_gpu_memory = live_gpu_memory;
        }

        if (live_gpu_memory <= config.target_gpu_memory) {
            break;
        }

        uint32_t to_unload = select_lod_to_unload();
        if (to_unload == UINT32_MAX) {
            break;  // No more LODs to unload
        }
        unload_lod_level(to_unload);
    }
}

uint32_t StreamingLODManager::select_lod_to_unload() {
    // Select least recently used LOD
    uint32_t oldest_idx = UINT32_MAX;
    uint64_t oldest_time = UINT64_MAX;

    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        for (uint32_t i = 0; i < lod_levels.size(); i++) {
            if (lod_levels[i].is_loaded && !lod_levels[i].is_loading) {
                if (lod_levels[i].last_access_time < oldest_time) {
                    oldest_time = lod_levels[i].last_access_time;
                    oldest_idx = i;
                }
            }
        }
    }

    return oldest_idx;
}

float StreamingLODManager::compute_distance_to_content(const Vector3& camera_pos) const {
    if (!has_content_bounds) {
        static bool warned_missing_bounds = false;
        if (!warned_missing_bounds) {
            GS_LOG_STREAMING_WARN("StreamingLODManager content bounds missing; skipping distance-driven LOD decisions until initialized with valid bounds.");
            warned_missing_bounds = true;
        }
        return CONTENT_DISTANCE_INVALID;
    }

    const Vector3 min_bound = content_bounds.position;
    const Vector3 max_bound = content_bounds.position + content_bounds.size;
    const Vector3 closest(
        std::clamp(camera_pos.x, min_bound.x, max_bound.x),
        std::clamp(camera_pos.y, min_bound.y, max_bound.y),
        std::clamp(camera_pos.z, min_bound.z, max_bound.z));

    const float surface_distance = (camera_pos - closest).length();
    if (surface_distance > 0.0001f) {
        return surface_distance;
    }

    // Treat inside-bounds camera positions as zero distance to content.
    return 0.0f;
}

bool StreamingLODManager::is_content_in_frustum(const Frustum& frustum) const {
    if (!has_content_bounds) {
        return false;
    }
    return _frustum_intersects_aabb(frustum, content_bounds);
}

bool StreamingLODManager::is_lod_visible(
    const LODLevel& lod,
    const Vector3& camera_pos,
    const Frustum* frustum) const {

    if (frustum && !is_content_in_frustum(*frustum)) {
        return false;
    }

    const float distance = compute_distance_to_content(camera_pos);
    if (!_is_valid_content_distance(distance)) {
        return false;
    }

    return distance >= lod.min_distance && distance <= lod.max_distance;
}

float StreamingLODManager::compute_lod_priority(
    const LODLevel& lod,
    const Vector3& camera_pos,
    const Frustum* frustum) const {

    if (!is_lod_visible(lod, camera_pos, frustum)) {
        return 0.0f;
    }

    const float distance = compute_distance_to_content(camera_pos);
    if (!_is_valid_content_distance(distance)) {
        return 0.0f;
    }

    const float distance_priority = 1.0f / (1.0f + abs(distance - lod.min_distance));
    const float access_priority = float(lod.access_count) / (OS::get_singleton()->get_ticks_msec() - lod.last_access_time + 1);

    return distance_priority * 0.7f + access_priority * 0.3f;
}

void StreamingLODManager::async_load_worker() {
    while (!should_stop_loading) {
        AsyncLoadRequest request;

        {
            std::unique_lock<std::mutex> lock(load_queue_mutex);
            load_cv.wait(lock, [this] {
                return !load_queue.empty() || should_stop_loading;
            });

            if (should_stop_loading) {
                break;
            }

            if (!load_queue.empty()) {
                request = std::move(load_queue.front());
                load_queue.pop();
            }
        }

        if (request.level == UINT32_MAX) {
            continue;
        }

        AsyncLoadJob job;
        uint64_t prepare_start_usec = OS::get_singleton()->get_ticks_usec();
        job.level = request.level;
        job.splat_count = request.source_data.size();
        job.enqueue_time_usec = request.enqueue_time_usec;
        prepare_gpu_upload_data(request.source_data, job.gpu_gaussians);
        job.prepare_time_ms = (OS::get_singleton()->get_ticks_usec() - prepare_start_usec) / 1000.0f;

        {
            std::lock_guard<std::mutex> lock(lod_mutex);
            stats.async_prepare_jobs_completed++;
            const bool prepare_on_main_thread = Thread::is_main_thread();
            if (prepare_on_main_thread) {
                stats.async_prepare_main_thread_violations++;
            } else {
                stats.async_prepare_observed_off_main_thread = true;
            }
            stats.performance.avg_async_prepare_time_ms =
                    stats.performance.avg_async_prepare_time_ms * 0.9f + job.prepare_time_ms * 0.1f;
        }

        {
            std::lock_guard<std::mutex> lock(load_queue_mutex);
            if (pending_async_loads.has(request.level)) {
                ready_queue.push(std::move(job));
            }
        }
    }
}

void StreamingLODManager::enqueue_load_request(uint32_t level, const LocalVector<uint32_t>& desired_levels) {
    if (!config.enable_async_loading) {
        return;
    }

    const uint32_t max_loads = config.max_concurrent_loads == 0
            ? UINT32_MAX
            : config.max_concurrent_loads;

    {
        std::lock_guard<std::mutex> lock(load_queue_mutex);
        if (pending_async_loads.has(level) || pending_async_loads.size() >= max_loads) {
            return;
        }
    }

    Vector<GaussianData> source_data;
    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        if (level >= lod_levels.size() || level >= lod_data.size()) {
            return;
        }

        LODLevel &lod = lod_levels.write[level];
        if (lod.is_loaded || lod.is_loading || lod_data[level].is_empty()) {
            return;
        }

        lod.is_loading = true;
        source_data = lod_data[level];
    }

    {
        std::lock_guard<std::mutex> lock(load_queue_mutex);
        pending_async_loads.insert(level);

        LocalVector<AsyncLoadRequest> retained_load_requests;
        while (!load_queue.empty()) {
            retained_load_requests.push_back(std::move(load_queue.front()));
            load_queue.pop();
        }
        retained_load_requests.push_back(AsyncLoadRequest{
                level,
                std::move(source_data),
                OS::get_singleton()->get_ticks_usec(),
        });

        std::queue<AsyncLoadRequest> reordered_load_queue;
        for (uint32_t i = 0; i < desired_levels.size(); i++) {
            for (uint32_t j = 0; j < retained_load_requests.size(); j++) {
                if (retained_load_requests[j].level == desired_levels[i]) {
                    reordered_load_queue.push(std::move(retained_load_requests.write[j]));
                    retained_load_requests.write[j].level = UINT32_MAX;
                    break;
                }
            }
        }
        load_queue = std::move(reordered_load_queue);
    }

    start_async_loading();
    load_cv.notify_one();
}

void StreamingLODManager::apply_completed_async_loads(uint64_t frame_start_usec) {
    if (!config.enable_async_loading) {
        return;
    }

    const uint32_t max_loads = config.max_concurrent_loads == 0
            ? UINT32_MAX
            : config.max_concurrent_loads;
    uint32_t processed = 0;

    while (processed < max_loads) {
        if (streaming_time_budget > 0) {
            uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - frame_start_usec;
            if (elapsed > streaming_time_budget) {
                break;
            }
        }

        AsyncLoadJob job;
        {
            std::lock_guard<std::mutex> lock(load_queue_mutex);
            if (ready_queue.empty()) {
                break;
            }
            job = std::move(ready_queue.front());
            ready_queue.pop();
            if (!pending_async_loads.has(job.level)) {
                continue;
            }
        }

        if (job.level == UINT32_MAX) {
            break;
        }

        const bool apply_on_main_thread = Thread::is_main_thread();
        uint64_t apply_start_usec = OS::get_singleton()->get_ticks_usec();
        bool apply_succeeded = false;
        {
            std::lock_guard<std::mutex> lock(lod_mutex);
            if (!apply_on_main_thread) {
                stats.async_apply_off_main_thread_violations++;
            } else {
                stats.async_apply_observed_on_main_thread = true;
            }
            if (job.level < lod_levels.size()) {
                LODLevel &lod = lod_levels.write[job.level];
                if (lod.is_loading && !lod.is_loaded) {
                    upload_to_gpu(lod, job.gpu_gaussians, job.splat_count);
                    if (lod.gpu_buffer.is_valid()) {
                        lod.is_loaded = true;
                        lod.last_access_time = OS::get_singleton()->get_ticks_msec();
                        lod.access_count++;
                        apply_succeeded = true;
                    }
                    lod.is_loading = false;
                }
            }
            if (apply_succeeded) {
                const uint64_t complete_time_usec = OS::get_singleton()->get_ticks_usec();
                const float apply_time_ms = (complete_time_usec - apply_start_usec) / 1000.0f;
                const float end_to_end_load_time_ms = job.enqueue_time_usec == 0
                        ? (job.prepare_time_ms + apply_time_ms)
                        : (complete_time_usec - job.enqueue_time_usec) / 1000.0f;
                stats.async_apply_jobs_completed++;
                stats.performance.avg_async_apply_time_ms =
                        stats.performance.avg_async_apply_time_ms * 0.9f + apply_time_ms * 0.1f;
                stats.performance.avg_load_time_ms =
                        stats.performance.avg_load_time_ms * 0.9f + end_to_end_load_time_ms * 0.1f;
            }
        }

        {
            std::lock_guard<std::mutex> lock(load_queue_mutex);
            pending_async_loads.erase(job.level);
        }

        processed++;
    }
}

void StreamingLODManager::CameraHistory::update(const Vector3& pos, float delta_time) {
    if (positions.is_empty()) {
        last_position = pos;
    }

    Vector3 velocity = (pos - last_position) / MAX(0.001f, delta_time);

    if (positions.size() >= HISTORY_SIZE) {
        positions.remove_at(0);
        velocities.remove_at(0);
    }

    positions.push_back(pos);
    velocities.push_back(velocity);
    last_position = pos;

    // Update average speed
    avg_speed = 0.0f;
    for (uint32_t i = 0; i < velocities.size(); i++) {
        avg_speed += velocities[i].length();
    }
    avg_speed /= MAX(1, static_cast<int>(velocities.size()));
}

Vector3 StreamingLODManager::CameraHistory::predict_position(float time_ahead) const {
    if (velocities.is_empty()) {
        return last_position;
    }

    // Simple linear prediction
    Vector3 avg_velocity;
    for (uint32_t i = 0; i < velocities.size(); i++) {
        avg_velocity += velocities[i];
    }
    avg_velocity /= static_cast<float>(velocities.size());

    return last_position + avg_velocity * time_ahead;
}

void StreamingLODManager::compact_memory() {
    // Compact GPU memory by re-allocating buffers
    std::lock_guard<std::mutex> lock(lod_mutex);
    for (auto &lod : lod_levels) {
        if (lod.is_loaded && lod.splat_count < lod.buffer_capacity * 0.5f) {
            // Re-allocate with smaller buffer
            // Implementation would go here
        }
    }
}

void StreamingLODManager::clear_cache() {
    // Clear CPU cache
    std::lock_guard<std::mutex> lock(lod_mutex);
    for (auto &data : lod_data) {
        data.clear();
    }

    stats.performance.cache_hits = 0;
    stats.performance.cache_misses = 0;
}

uint64_t StreamingLODManager::get_memory_usage() const {
    std::lock_guard<std::mutex> lock(lod_mutex);
    return stats.total_gpu_memory + stats.total_cpu_memory;
}

} // namespace GaussianSplatting
