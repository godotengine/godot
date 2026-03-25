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
    // Stop async loading thread
    should_stop_loading = true;
    if (loading_thread.joinable()) {
        load_cv.notify_all();
        loading_thread.join();
    }

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

void StreamingLODManager::initialize(
    const Vector<GaussianData>& base_splats,
    const StreamingConfig& p_config) {

    config = p_config;
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

    // Async loads run synchronously on the main thread via process_async_load_queue().
    // The background thread is intentionally disabled -- lod_mutex coverage is incomplete
    // for concurrent lod_data access. Do not re-enable without a full lock audit.

    GS_LOG_STREAMING_INFO(vformat("StreamingLODManager initialized with %d LOD levels, %d base splats",
            lod_levels.size(), base_splats.size()));
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

    // Update LOD priorities
    update_lod_priorities(camera_pos, camera_frustum);

    // Unload distant LODs
    unload_distant_lods(camera_pos, camera_frustum);

    // Load nearby LODs
    LocalVector<uint32_t> async_loads;
    LocalVector<uint32_t> sync_loads;
    uint32_t frustum_rejected_lods = 0;
    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        for (uint32_t i = 0; i < lod_levels.size(); i++) {
            const LODLevel &lod = lod_levels[i];
            if (!lod.is_loaded && !lod.is_loading) {
                if (!content_in_frustum) {
                    frustum_rejected_lods++;
                } else if (content_distance_valid &&
                        content_distance >= lod.min_distance - config.load_ahead_distance &&
                        content_distance <= lod.max_distance + config.load_ahead_distance) {
                    if (config.enable_async_loading) {
                        async_loads.push_back(i);
                    } else {
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

        stats.load_requests_last_frame = async_loads.size() + sync_loads.size();
        stats.frustum_rejected_lods_last_frame = frustum_rejected_lods;
    }

    for (uint32_t i = 0; i < async_loads.size(); i++) {
        enqueue_load_request(async_loads[i]);
    }
    for (uint32_t i = 0; i < sync_loads.size(); i++) {
        stream_lod_level(sync_loads[i]);
    }

    // Predictive loading
    if (config.enable_predictive_loading && content_distance_valid && content_in_frustum) {
        PredictedMovement prediction = predict_camera_movement(camera_pos, delta_time);
        prefetch_predicted_lods(prediction, camera_frustum);
    }

    process_async_load_queue(frame_start);

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
        stats.total_gpu_memory = 0;
        stats.total_cpu_memory = 0;

        for (const auto &lod : lod_levels) {
            if (lod.is_loaded) {
                stats.loaded_lod_levels++;
                stats.total_gpu_memory += lod.gpu_memory_bytes;
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
    std::lock_guard<std::mutex> lock(lod_mutex);
    if (level >= lod_levels.size()) {
        return;
    }

    LODLevel &lod = lod_levels.write[level];
    if (lod.is_loaded || lod.is_loading) {
        return;
    }

    lod.is_loading = true;
    uint64_t start_time = OS::get_singleton()->get_ticks_usec();

    // Upload to GPU
    if (level < lod_data.size() && !lod_data[level].is_empty()) {
        upload_to_gpu(lod, lod_data[level]);
        if (lod.gpu_buffer.is_valid()) {
            lod.is_loaded = true;
            lod.last_access_time = OS::get_singleton()->get_ticks_msec();
            lod.access_count++;
        }
    }

    lod.is_loading = false;

    float load_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;
    stats.performance.avg_load_time_ms =
            stats.performance.avg_load_time_ms * 0.9f + load_time_ms * 0.1f;

    GS_LOG_STREAMING_DEBUG(vformat("Loaded LOD %d in %.2f ms", level, load_time_ms));
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

void StreamingLODManager::upload_to_gpu(LODLevel& lod, const Vector<GaussianData>& data) {
    if (data.is_empty()) return;

    // Convert the clustered splats to GPU-aligned gaussians.
    LocalVector<::Gaussian> gpu_gaussians;
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
    lod.gpu_memory_bytes = data.size() * sizeof(GaussianData);
    lod.buffer_capacity = gpu_stream->get_max_gaussians();
}

void StreamingLODManager::release_gpu_buffer(LODLevel& lod) {
    // GPU buffers returned by GaussianMemoryStream are stream-owned.
    // Only clear LOD-local references here; the stream frees its own buffers.
    lod.gpu_buffer = RID();
    lod.gpu_memory_bytes = 0;
    lod.buffer_capacity = 0;
}

void StreamingLODManager::update_lod_priorities(const Vector3& camera_pos, const Frustum& frustum) {
    // Sort LODs by priority
    struct LODPriority {
        uint32_t index;
        float priority;
        bool is_loaded;
        bool is_loading;
    };

    LocalVector<LODPriority> priorities;
    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        priorities.resize(lod_levels.size());
        for (uint32_t i = 0; i < lod_levels.size(); i++) {
            priorities[i].index = i;
            priorities[i].priority = compute_lod_priority(lod_levels[i], camera_pos, &frustum);
            priorities[i].is_loaded = lod_levels[i].is_loaded;
            priorities[i].is_loading = lod_levels[i].is_loading;
        }
    }

    std::sort(
        priorities.ptr(),
        priorities.ptr() + priorities.size(),
        [](const auto& a, const auto& b) { return a.priority > b.priority; }
    );

    // Update load queue based on priorities
    if (config.enable_async_loading) {
        std::lock_guard<std::mutex> lock(load_queue_mutex);

        // Clear and rebuild queue
        while (!load_queue.empty()) {
            load_queue.pop();
        }
        pending_async_loads.clear();

        for (const auto &p : priorities) {
            if (p.priority > 0.0f && !p.is_loaded && !p.is_loading) {
                if (pending_async_loads.has(p.index)) {
                    continue;
                }
                pending_async_loads.insert(p.index);
                load_queue.push(p.index);
            }
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

void StreamingLODManager::prefetch_predicted_lods(const PredictedMovement& prediction, const Frustum& frustum) {
    if (prediction.confidence < 0.5f) {
        return;  // Low confidence, don't prefetch
    }
    if (!is_content_in_frustum(frustum)) {
        return;
    }

    // Check which LODs might be needed at predicted position
    LocalVector<uint32_t> to_prefetch;
    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        for (uint32_t i = 0; i < lod_levels.size(); i++) {
            const LODLevel &lod = lod_levels[i];

            if (!lod.is_loaded && !lod.is_loading) {
                const float predicted_distance = compute_distance_to_content(prediction.predicted_position);
                if (!_is_valid_content_distance(predicted_distance)) {
                    continue;
                }

                if (predicted_distance >= lod.min_distance - config.load_ahead_distance * 0.5f &&
                        predicted_distance <= lod.max_distance + config.load_ahead_distance * 0.5f) {
                    if (config.enable_async_loading) {
                        to_prefetch.push_back(i);
                    }
                }
            }
        }
    }

    for (uint32_t i = 0; i < to_prefetch.size(); i++) {
        enqueue_load_request(to_prefetch[i]);
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
        uint32_t level_to_load = UINT32_MAX;

        {
            std::unique_lock<std::mutex> lock(load_queue_mutex);
            load_cv.wait(lock, [this] {
                return !load_queue.empty() || should_stop_loading;
            });

            if (should_stop_loading) {
                break;
            }

            if (!load_queue.empty()) {
                level_to_load = load_queue.front();
                load_queue.pop();
                pending_async_loads.erase(level_to_load);
            }
        }

        if (level_to_load != UINT32_MAX) {
            stream_lod_level(level_to_load);
        }
    }
}

void StreamingLODManager::enqueue_load_request(uint32_t level) {
    {
        std::lock_guard<std::mutex> lock(lod_mutex);
        if (level >= lod_levels.size()) {
            return;
        }
    }

    {
        std::lock_guard<std::mutex> lock(load_queue_mutex);
        if (pending_async_loads.has(level)) {
            return;
        }
        pending_async_loads.insert(level);
        load_queue.push(level);
    }
    load_cv.notify_one();
}

void StreamingLODManager::process_async_load_queue(uint64_t frame_start_usec) {
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

        uint32_t level = UINT32_MAX;
        {
            std::lock_guard<std::mutex> lock(load_queue_mutex);
            if (load_queue.empty()) {
                break;
            }
            level = load_queue.front();
            load_queue.pop();
            pending_async_loads.erase(level);
        }

        if (level == UINT32_MAX) {
            break;
        }

        stream_lod_level(level);
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
