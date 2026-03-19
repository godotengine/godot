#include "test_macros.h"
#include "../renderer/gpu_memory_stream.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../renderer/resource_owner_mismatch_contract.h"
#include "../renderer/quantization_config.h"
#include "../core/gaussian_data.h"
#include "../core/residency_budget_controller.h"
#include "../core/gaussian_streaming.h"
#include "../core/gaussian_splat_manager.h"
#include "../core/streaming_queue_pressure_controller.h"
#include "core/config/project_settings.h"
#include "servers/rendering/rendering_device.h"
#include "core/os/os.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <random>

// Helper to create test gaussian data
LocalVector<Gaussian> create_test_gaussians(uint32_t count) {
    LocalVector<Gaussian> gaussians;
    gaussians.resize(count);

    for (uint32_t i = 0; i < count; i++) {
        Gaussian &g = gaussians[i];
        g.position = Vector3(i * 0.1f, i * 0.2f, i * 0.3f);
        g.scale = Vector3(0.5f, 0.5f, 0.5f);
        g.rotation = Quaternion();
        g.opacity = 0.9f;
        g.sh_dc = Color(1.0f, 0.5f, 0.2f, 0.9f);
        g.normal = Vector3(0, 1, 0);
        g.area = 0.25f;
    }

    return gaussians;
}

Ref<::GaussianData> create_test_gaussian_data(uint32_t count) {
    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(create_test_gaussians(count));
    return data;
}

Ref<::GaussianData> create_clustered_test_gaussian_data(uint32_t count, const Vector3 &center) {
    LocalVector<Gaussian> gaussians;
    gaussians.resize(count);

    for (uint32_t i = 0; i < count; i++) {
        Gaussian &g = gaussians[i];
        const float x = float(i % 16) * 0.01f;
        const float y = float((i / 16) % 16) * 0.01f;
        const float z = float(i / 256) * 0.01f;
        g.position = center + Vector3(x, y, z);
        g.scale = Vector3(0.05f, 0.05f, 0.05f);
        g.rotation = Quaternion();
        g.opacity = 0.95f;
        g.sh_dc = Color(0.8f, 0.7f, 0.6f, 0.95f);
        g.normal = Vector3(0, 1, 0);
        g.area = 0.01f;
    }

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);
    return data;
}

namespace {

bool _is_equal_approx_vec3(const Vector3 &p_a, const Vector3 &p_b) {
    return Math::is_equal_approx(p_a.x, p_b.x) &&
            Math::is_equal_approx(p_a.y, p_b.y) &&
            Math::is_equal_approx(p_a.z, p_b.z);
}

bool _is_equal_approx_color(const Color &p_a, const Color &p_b) {
    return Math::is_equal_approx(p_a.r, p_b.r) &&
            Math::is_equal_approx(p_a.g, p_b.g) &&
            Math::is_equal_approx(p_a.b, p_b.b) &&
            Math::is_equal_approx(p_a.a, p_b.a);
}

class ScopedProjectSettingRestore {
    ProjectSettings *settings = nullptr;
    String setting_path;
    Variant previous_value;
    bool had_previous_value = false;

public:
    ScopedProjectSettingRestore(ProjectSettings *p_settings, const String &p_setting_path) :
            settings(p_settings),
            setting_path(p_setting_path) {
        if (settings && settings->has_setting(setting_path)) {
            previous_value = settings->get_setting(setting_path);
            had_previous_value = true;
        }
    }

    ~ScopedProjectSettingRestore() {
        if (!settings) {
            return;
        }

        if (had_previous_value) {
            settings->set_setting(setting_path, previous_value);
        } else {
            settings->clear(setting_path);
        }
    }
};

struct SnapshotPositionStressContext {
    Ref<::GaussianData> data;
    const PackedVector3Array *pattern_a = nullptr;
    const PackedVector3Array *pattern_b = nullptr;
    uint32_t splat_count = 0;

    std::atomic<bool> stop{false};
    std::atomic<bool> writer_failed{false};
    std::atomic<bool> reader_failed{false};

    Semaphore writer_begin;
    Semaphore writer_done;
    Semaphore reader_begin;
    Semaphore reader_done;
};

bool _snapshot_matches_position_pattern(const LocalVector<Gaussian> &p_snapshot, const PackedVector3Array &p_pattern) {
    if (p_snapshot.size() != p_pattern.size()) {
        return false;
    }
    for (uint32_t i = 0; i < p_snapshot.size(); i++) {
        if (!_is_equal_approx_vec3(p_snapshot[i].position, p_pattern[i])) {
            return false;
        }
    }
    return true;
}

void _snapshot_position_writer_thread(void *p_userdata) {
    SnapshotPositionStressContext *ctx = static_cast<SnapshotPositionStressContext *>(p_userdata);
    if (!ctx || !ctx->data.is_valid() || !ctx->pattern_a || !ctx->pattern_b) {
        return;
    }

    uint32_t iteration = 0;
    while (true) {
        ctx->writer_begin.wait();
        if (ctx->stop.load(std::memory_order_acquire)) {
            break;
        }

        const PackedVector3Array &pattern = (iteration % 2 == 0) ? *ctx->pattern_a : *ctx->pattern_b;
        if (pattern.size() != int(ctx->splat_count)) {
            ctx->writer_failed.store(true, std::memory_order_release);
            ctx->writer_done.post();
            continue;
        }

        ctx->data->set_positions(pattern);
        iteration++;
        ctx->writer_done.post();
    }
}

void _snapshot_position_reader_thread(void *p_userdata) {
    SnapshotPositionStressContext *ctx = static_cast<SnapshotPositionStressContext *>(p_userdata);
    if (!ctx || !ctx->data.is_valid() || !ctx->pattern_a || !ctx->pattern_b) {
        return;
    }

    while (true) {
        ctx->reader_begin.wait();
        if (ctx->stop.load(std::memory_order_acquire)) {
            break;
        }

        LocalVector<Gaussian> gaussians_snapshot;
        LocalVector<Vector3> sh_snapshot;
        uint32_t sh_first = 0;
        uint32_t sh_high = 0;
        const bool capture_ok = ctx->data->capture_chunk_snapshot(0, ctx->splat_count,
                gaussians_snapshot, sh_snapshot, sh_first, sh_high);
        if (!capture_ok) {
            ctx->reader_failed.store(true, std::memory_order_release);
            ctx->reader_done.post();
            continue;
        }

        const bool matches_a = _snapshot_matches_position_pattern(gaussians_snapshot, *ctx->pattern_a);
        const bool matches_b = _snapshot_matches_position_pattern(gaussians_snapshot, *ctx->pattern_b);
        if (!matches_a && !matches_b) {
            ctx->reader_failed.store(true, std::memory_order_release);
        }

        ctx->reader_done.post();
    }
}

struct SnapshotSHStressContext {
    Ref<::GaussianData> data;
    const PackedFloat32Array *dc_only_data = nullptr;
    const PackedFloat32Array *full_data = nullptr;
    uint32_t splat_count = 0;

    std::atomic<bool> stop{false};
    std::atomic<bool> writer_failed{false};
    std::atomic<bool> reader_failed{false};

    Semaphore writer_begin;
    Semaphore writer_done;
    Semaphore reader_begin;
    Semaphore reader_done;
};

struct StreamingPipelineHammerContext {
    Ref<StreamingPipeline> pipeline;
    uint32_t total_splats = 0;

    std::atomic<bool> stop{false};
    std::atomic<uint32_t> iteration{0};

    Semaphore lod_begin;
    Semaphore lod_done;
    Semaphore range_begin;
    Semaphore range_done;
};

Vector<GaussianStreamingSystem::ChunkLayoutHint> _build_partitioned_hints(uint32_t p_total_splats, uint32_t p_chunk_cap, std::mt19937 &p_rng) {
    Vector<GaussianStreamingSystem::ChunkLayoutHint> hints;
    uint32_t remaining = p_total_splats;
    uint32_t cursor = 0;

    while (remaining > 0) {
        const uint32_t step_cap = MIN(p_chunk_cap, remaining);
        std::uniform_int_distribution<uint32_t> count_dist(1u, step_cap);
        const uint32_t count = count_dist(p_rng);

        GaussianStreamingSystem::ChunkLayoutHint hint;
        hint.start_idx = cursor;
        hint.count = count;
        hint.bounds = AABB(Vector3(-1.0f, -1.0f, -1.0f), Vector3(2.0f, 2.0f, 2.0f));
        hint.center = Vector3();
        hint.radius = 1.0f;
        hints.push_back(hint);

        cursor += count;
        remaining -= count;
    }

    return hints;
}

String _layout_hint_last_reason(const Ref<GaussianStreamingSystem> &p_system) {
    Dictionary stats = p_system->get_chunk_culling_stats();
    Dictionary validation = stats.get("layout_hint_validation", Dictionary());
    return validation.get("last_reason", String("none"));
}

int64_t _layout_hint_reason_count(const Ref<GaussianStreamingSystem> &p_system, const String &p_reason) {
    Dictionary stats = p_system->get_chunk_culling_stats();
    Dictionary validation = stats.get("layout_hint_validation", Dictionary());
    Dictionary reason_counts = validation.get("reason_counts", Dictionary());
    return reason_counts.get(p_reason, int64_t(0));
}

struct ConcurrentStreamingContext {
    Ref<GaussianMemoryStream> stream;
    uint32_t uploads_per_thread = 0;
    uint32_t gaussians_per_upload = 0;
    std::atomic<int> successful_uploads{0};
    std::atomic<int> failed_uploads{0};
    Semaphore begin;
    Semaphore done;
};

void _concurrent_streaming_worker(void *p_userdata) {
    ConcurrentStreamingContext *ctx = static_cast<ConcurrentStreamingContext *>(p_userdata);
    if (!ctx || !ctx->stream.is_valid()) {
        return;
    }

    ctx->begin.wait();
    for (uint32_t i = 0; i < ctx->uploads_per_thread; i++) {
        LocalVector<Gaussian> gaussians = create_test_gaussians(ctx->gaussians_per_upload);
        Error upload_err = ctx->stream->stream_gaussians_async(gaussians);
        if (upload_err == OK) {
            ctx->successful_uploads.fetch_add(1, std::memory_order_relaxed);
        } else {
            ctx->failed_uploads.fetch_add(1, std::memory_order_relaxed);
        }
    }
    ctx->done.post();
}

void _compute_streaming_pipeline_hammer_values(uint32_t p_iteration, uint32_t p_total_splats,
        uint32_t &r_lod, uint32_t &r_start, uint32_t &r_count) {
    r_lod = p_iteration % 6;

    r_count = 128 + ((p_iteration % 5) * 64);
    if (r_count > p_total_splats) {
        r_count = p_total_splats;
    }

    if (p_total_splats > r_count) {
        const uint32_t max_start = p_total_splats - r_count;
        r_start = (p_iteration * 131u + 17u) % (max_start + 1);
    } else {
        r_start = 0;
    }
}

void _streaming_pipeline_lod_hammer_thread(void *p_userdata) {
    StreamingPipelineHammerContext *ctx = static_cast<StreamingPipelineHammerContext *>(p_userdata);
    if (!ctx || !ctx->pipeline.is_valid()) {
        return;
    }

    while (true) {
        ctx->lod_begin.wait();
        if (ctx->stop.load(std::memory_order_acquire)) {
            break;
        }

        const uint32_t iteration = ctx->iteration.load(std::memory_order_acquire);
        ctx->pipeline->set_lod_level(iteration % 6);
        ctx->lod_done.post();
    }
}

void _streaming_pipeline_visible_range_hammer_thread(void *p_userdata) {
    StreamingPipelineHammerContext *ctx = static_cast<StreamingPipelineHammerContext *>(p_userdata);
    if (!ctx || !ctx->pipeline.is_valid()) {
        return;
    }

    while (true) {
        ctx->range_begin.wait();
        if (ctx->stop.load(std::memory_order_acquire)) {
            break;
        }

        uint32_t start = 0;
        uint32_t count = 0;
        const uint32_t iteration = ctx->iteration.load(std::memory_order_acquire);
        count = 128 + ((iteration % 5) * 64);
        if (count > ctx->total_splats) {
            count = ctx->total_splats;
        }
        if (ctx->total_splats > count) {
            const uint32_t max_start = ctx->total_splats - count;
            start = (iteration * 131u + 17u) % (max_start + 1);
        }
        ctx->pipeline->update_visible_range(start, count);
        ctx->range_done.post();
    }
}

Color _dc_only_color_for_index(uint32_t p_index) {
    const float base = float(p_index) * 0.0001f;
    return Color(0.1f + base, 0.2f + base, 0.3f + base, 1.0f);
}

Color _full_color_for_index(uint32_t p_index) {
    const float base = float(p_index) * 0.0001f;
    return Color(0.4f + base, 0.5f + base, 0.6f + base, 1.0f);
}

Vector3 _full_first_order_for_index(uint32_t p_index, uint32_t p_term) {
    const float base = float(p_index) * 0.0002f;
    return Vector3(1.0f + float(p_term) + base,
            2.0f + float(p_term) + base,
            3.0f + float(p_term) + base);
}

Vector3 _full_high_order_for_index(uint32_t p_index, uint32_t p_term) {
    const float base = float(p_index) * 0.0003f;
    if (p_term == 0) {
        return Vector3(7.0f + base, 8.0f + base, 9.0f + base);
    }
    return Vector3(10.0f + base, 11.0f + base, 12.0f + base);
}

void _snapshot_sh_writer_thread(void *p_userdata) {
    SnapshotSHStressContext *ctx = static_cast<SnapshotSHStressContext *>(p_userdata);
    if (!ctx || !ctx->data.is_valid() || !ctx->dc_only_data || !ctx->full_data) {
        return;
    }

    uint32_t iteration = 0;
    while (true) {
        ctx->writer_begin.wait();
        if (ctx->stop.load(std::memory_order_acquire)) {
            break;
        }

        const PackedFloat32Array &sh_data = (iteration % 2 == 0) ? *ctx->dc_only_data : *ctx->full_data;
        ctx->data->set_spherical_harmonics(sh_data);
        iteration++;
        ctx->writer_done.post();
    }
}

void _snapshot_sh_reader_thread(void *p_userdata) {
    SnapshotSHStressContext *ctx = static_cast<SnapshotSHStressContext *>(p_userdata);
    if (!ctx || !ctx->data.is_valid()) {
        return;
    }

    while (true) {
        ctx->reader_begin.wait();
        if (ctx->stop.load(std::memory_order_acquire)) {
            break;
        }

        LocalVector<Gaussian> gaussians_snapshot;
        LocalVector<Vector3> sh_snapshot;
        uint32_t sh_first = 0;
        uint32_t sh_high = 0;
        const bool capture_ok = ctx->data->capture_chunk_snapshot(0, ctx->splat_count,
                gaussians_snapshot, sh_snapshot, sh_first, sh_high);
        if (!capture_ok || gaussians_snapshot.size() != int(ctx->splat_count)) {
            ctx->reader_failed.store(true, std::memory_order_release);
            ctx->reader_done.post();
            continue;
        }

        bool snapshot_valid = true;
        if (sh_high == 0) {
            snapshot_valid = sh_first == 0 && sh_snapshot.is_empty();
            for (uint32_t i = 0; snapshot_valid && i < ctx->splat_count; i++) {
                const Gaussian &g = gaussians_snapshot[i];
                if (!_is_equal_approx_color(g.sh_dc, _dc_only_color_for_index(i))) {
                    snapshot_valid = false;
                    break;
                }
                for (uint32_t j = 0; j < 3; j++) {
                    if (!_is_equal_approx_vec3(g.sh_1[j], Vector3())) {
                        snapshot_valid = false;
                        break;
                    }
                }
            }
        } else if (sh_high == 2) {
            snapshot_valid = sh_first == 3 && sh_snapshot.size() == int(ctx->splat_count * 2);
            for (uint32_t i = 0; snapshot_valid && i < ctx->splat_count; i++) {
                const Gaussian &g = gaussians_snapshot[i];
                if (!_is_equal_approx_color(g.sh_dc, _full_color_for_index(i))) {
                    snapshot_valid = false;
                    break;
                }
                for (uint32_t j = 0; j < 3; j++) {
                    if (!_is_equal_approx_vec3(g.sh_1[j], _full_first_order_for_index(i, j))) {
                        snapshot_valid = false;
                        break;
                    }
                }
                const uint32_t base = i * 2;
                if (!_is_equal_approx_vec3(sh_snapshot[base + 0], _full_high_order_for_index(i, 0)) ||
                        !_is_equal_approx_vec3(sh_snapshot[base + 1], _full_high_order_for_index(i, 1))) {
                    snapshot_valid = false;
                    break;
                }
            }
        } else {
            snapshot_valid = false;
        }

        if (!snapshot_valid) {
            ctx->reader_failed.store(true, std::memory_order_release);
        }

        ctx->reader_done.post();
    }
}

} // namespace

TEST_CASE("[GPU Memory Stream] Initialization") {
    Ref<GaussianMemoryStream> stream;
    stream.instantiate();

    // Test initialization without RenderingDevice (should fail)
    Error err = stream->initialize(nullptr, 100000, 256);
    CHECK(err == ERR_INVALID_PARAMETER);

    // Get or create RenderingDevice
    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        // Create local rendering device for testing
        RenderingServer *rs = RenderingServer::get_singleton();
        if (rs) {
            rd = rs->create_local_rendering_device();
        }
    }

    if (rd) {
        // Test valid initialization
        err = stream->initialize(rd, 100000, 256);
        CHECK(err == OK);

        // Verify initialization
        CHECK(stream->get_max_gaussians() == 100000);
        CHECK(stream->get_allocated_memory_mb() > 0);

        stream->shutdown();
    }
}

TEST_CASE("[GPU Memory Stream] Triple Buffering") {
    Ref<GaussianMemoryStream> stream;
    stream.instantiate();

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        RenderingServer *rs = RenderingServer::get_singleton();
        if (rs) {
            rd = rs->create_local_rendering_device();
        }
    }

    if (rd) {
        Error err = stream->initialize(rd, 10000, 64);
        CHECK(err == OK);
        if (err != OK) {
            return;
        }

        // Create test data
        LocalVector<Gaussian> gaussians = create_test_gaussians(1000);

        // Test triple buffer rotation
        for (int i = 0; i < 5; i++) {
            err = stream->stream_gaussians_async(gaussians);
            CHECK(err == OK);

            // Simulate frame
            stream->begin_frame(i);
            stream->swap_buffers();
            stream->end_frame();
        }

        // Check no stalls occurred
        StreamingStats stats = stream->get_stats();
        CHECK(stats.stalls == 0);
        CHECK(stats.buffer_switches >= 5);

        stream->shutdown();
    }
}

TEST_CASE("[GPU Memory Stream] Memory Pool Allocation") {
    // Test memory pool functionality
    GaussianMemoryStream::MemoryPool pool;
    pool.total_size = 1024 * 1024; // 1MB
    pool.blocks.push_back({0, pool.total_size, true, 0});

    // Test allocations
    uint32_t offset1 = pool.allocate(1024, 16);
    CHECK(offset1 != UINT32_MAX);
    CHECK(offset1 == 0);

    uint32_t offset2 = pool.allocate(2048, 16);
    CHECK(offset2 != UINT32_MAX);
    CHECK(offset2 >= 1024);

    // Test deallocation
    pool.deallocate(offset1);

    // Allocate in freed space
    uint32_t offset3 = pool.allocate(512, 16);
    CHECK(offset3 == 0); // Should reuse freed space

    // Test fragmentation calculation
    float frag = pool.get_fragmentation_ratio();
    CHECK(frag >= 0.0f);
    CHECK(frag <= 1.0f);
}

TEST_CASE("[GPU Memory Stream] Streaming Performance") {
    Ref<GaussianMemoryStream> stream;
    stream.instantiate();

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        RenderingServer *rs = RenderingServer::get_singleton();
        if (rs) {
            rd = rs->create_local_rendering_device();
        }
    }

    if (rd) {
        // Initialize for 100K gaussians
        Error err = stream->initialize(rd, 100000, 256);
        CHECK(err == OK);
        if (err != OK) {
            return;
        }

        // Create 100K test gaussians
        LocalVector<Gaussian> gaussians = create_test_gaussians(100000);

        // Measure streaming time
        uint64_t start = OS::get_singleton()->get_ticks_usec();

        err = stream->stream_gaussians_async(gaussians);
        CHECK(err == OK);

        stream->wait_for_all_uploads();

        uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - start;
        float ms = elapsed / 1000.0f;

        // Should complete in reasonable time (< 100ms for 100K splats)
        CHECK_MESSAGE(ms < 100.0f,
            vformat("Streaming 100K gaussians took %.2f ms (target < 100ms)", ms));

        // Check memory efficiency
        float efficiency = stream->get_memory_efficiency();
        CHECK(efficiency > 0.0f);

        stream->shutdown();
    }
}

TEST_CASE("[GPU Memory Stream] Memory Defragmentation") {
    GaussianMemoryStream::MemoryPool pool;
    pool.total_size = 1024 * 1024; // 1MB
    pool.blocks.clear();

    // Create fragmented memory layout
    pool.blocks.push_back({0, 1024, false, 0}); // Allocated
    pool.blocks.push_back({1024, 512, true, 0}); // Free
    pool.blocks.push_back({1536, 2048, false, 0}); // Allocated
    pool.blocks.push_back({3584, 256, true, 0}); // Free
    pool.blocks.push_back({3840, 1024, false, 0}); // Allocated
    pool.used_size = 1024 + 2048 + 1024;

    // Check fragmentation before
    float frag_before = pool.get_fragmentation_ratio();
    CHECK(frag_before > 0.5f); // Should be fragmented

    // Defragment
    pool.defragment();

    // Check fragmentation after
    float frag_after = pool.get_fragmentation_ratio();
    CHECK(frag_after < frag_before); // Should be less fragmented

    // Verify blocks are compacted
    bool found_large_free = false;
    for (const auto &block : pool.blocks) {
        if (block.free && block.size >= (pool.total_size - pool.used_size)) {
            found_large_free = true;
        }
    }
    CHECK(found_large_free);
}

TEST_CASE("[GPU Memory Stream] Concurrent Streaming") {
    Ref<GaussianMemoryStream> stream;
    stream.instantiate();

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        RenderingServer *rs = RenderingServer::get_singleton();
        if (rs) {
            rd = rs->create_local_rendering_device();
        }
    }

    if (rd) {
        Error err = stream->initialize(rd, 50000, 128);
        CHECK(err == OK);
        if (err != OK) {
            return;
        }

        ConcurrentStreamingContext ctx;
        ctx.stream = stream;
        ctx.uploads_per_thread = 10;
        ctx.gaussians_per_upload = 1000;

        Thread worker_a;
        Thread worker_b;
        const Thread::ID worker_a_id = worker_a.start(_concurrent_streaming_worker, &ctx);
        const Thread::ID worker_b_id = worker_b.start(_concurrent_streaming_worker, &ctx);
        const bool worker_a_started = worker_a_id != Thread::UNASSIGNED_ID;
        const bool worker_b_started = worker_b_id != Thread::UNASSIGNED_ID;
        CHECK(worker_a_started);
        CHECK(worker_b_started);
        if (!worker_a_started || !worker_b_started) {
            if (worker_a_started) {
                ctx.begin.post();
                worker_a.wait_to_finish();
            }
            if (worker_b_started) {
                ctx.begin.post();
                worker_b.wait_to_finish();
            }
            stream->shutdown();
            return;
        }

        ctx.begin.post(2);
        ctx.done.wait();
        ctx.done.wait();
        worker_a.wait_to_finish();
        worker_b.wait_to_finish();

        const int successful_uploads = ctx.successful_uploads.load(std::memory_order_relaxed);
        const int failed_uploads = ctx.failed_uploads.load(std::memory_order_relaxed);
        const int total_uploads = int(ctx.uploads_per_thread * 2);
        CHECK(successful_uploads > 0);
        CHECK_MESSAGE(failed_uploads < total_uploads / 2,
                "Too many failed uploads in concurrent streaming test");

        stream->shutdown();
    }
}

TEST_CASE("[Streaming Pipeline] Basic Operations") {
    Ref<StreamingPipeline> pipeline;
    pipeline.instantiate();

    Ref<GaussianMemoryStream> stream;
    stream.instantiate();

    Ref<::GaussianData> data;
    data.instantiate();
    data->resize(10000);

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        RenderingServer *rs = RenderingServer::get_singleton();
        if (rs) {
            rd = rs->create_local_rendering_device();
        }
    }

    if (rd) {
        // Initialize stream
        Error err = stream->initialize(rd, 10000, 64);
        CHECK(err == OK);
        if (err != OK) {
            return;
        }

        // Initialize pipeline
        err = pipeline->initialize(stream, data);
        CHECK(err == OK);

        // Start streaming
        pipeline->start_streaming();

        // Update visible range
        pipeline->update_visible_range(0, 1000);

        // Wait a bit for streaming
        OS::get_singleton()->delay_usec(10 * 1000);

        // Get stats
        Dictionary stats = pipeline->get_streaming_stats();
        CHECK(stats.has("visible_count"));
        CHECK(int(stats["visible_count"]) == 1000);

        // Stop streaming
        pipeline->stop_streaming();

        pipeline->shutdown();
    }
}

TEST_CASE("[Streaming Pipeline] Concurrent LOD and visibility updates remain coherent while worker is active") {
#ifndef THREADS_ENABLED
    MESSAGE("Skipping - THREADS_ENABLED is not enabled in this build");
    return;
#endif

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        RenderingServer *rs = RenderingServer::get_singleton();
        if (rs) {
            rd = rs->create_local_rendering_device();
        }
    }

    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    const uint32_t total_splats = 32768;

    Ref<GaussianMemoryStream> stream;
    stream.instantiate();
    Error err = stream->initialize(rd, total_splats, 64);
    CHECK(err == OK);
    if (err != OK) {
        return;
    }

    Ref<StreamingPipeline> pipeline;
    pipeline.instantiate();
    Ref<::GaussianData> data = create_test_gaussian_data(total_splats);
    err = pipeline->initialize(stream, data);
    CHECK(err == OK);
    if (err != OK) {
        stream->shutdown();
        return;
    }

    pipeline->start_streaming();

    StreamingPipelineHammerContext ctx;
    ctx.pipeline = pipeline;
    ctx.total_splats = total_splats;

    Thread lod_thread;
    Thread range_thread;
    const Thread::ID lod_thread_id = lod_thread.start(_streaming_pipeline_lod_hammer_thread, &ctx);
    const Thread::ID range_thread_id = range_thread.start(_streaming_pipeline_visible_range_hammer_thread, &ctx);
    const bool lod_thread_started = lod_thread_id != Thread::UNASSIGNED_ID;
    const bool range_thread_started = range_thread_id != Thread::UNASSIGNED_ID;
    CHECK(lod_thread_started);
    CHECK(range_thread_started);
    if (!lod_thread_started || !range_thread_started) {
        ctx.stop.store(true, std::memory_order_release);
        if (lod_thread_started) {
            ctx.lod_begin.post();
            lod_thread.wait_to_finish();
        }
        if (range_thread_started) {
            ctx.range_begin.post();
            range_thread.wait_to_finish();
        }
        pipeline->stop_streaming();
        pipeline->shutdown();
        return;
    }

    constexpr uint32_t iterations = 192;
    for (uint32_t i = 0; i < iterations; i++) {
        ctx.iteration.store(i, std::memory_order_release);
        ctx.lod_begin.post();
        ctx.range_begin.post();
        Thread::yield();
        ctx.lod_done.wait();
        ctx.range_done.wait();
    }

    ctx.stop.store(true, std::memory_order_release);
    ctx.lod_begin.post();
    ctx.range_begin.post();
    lod_thread.wait_to_finish();
    range_thread.wait_to_finish();

    uint32_t expected_lod = 0;
    uint32_t expected_start = 0;
    uint32_t expected_count = 0;
    _compute_streaming_pipeline_hammer_values(iterations - 1, total_splats, expected_lod, expected_start, expected_count);

    bool drained = false;
    for (int i = 0; i < 300; i++) {
        Dictionary stats = pipeline->get_streaming_stats();
        if (!bool(stats["is_streaming"])) {
            drained = true;
            break;
        }
        OS::get_singleton()->delay_usec(1000);
    }
    CHECK(drained);

    stream->wait_for_all_uploads();

    Dictionary stats = pipeline->get_streaming_stats();
    CHECK(uint32_t(int(stats["current_lod"])) == expected_lod);
    CHECK(uint32_t(int(stats["visible_start"])) == expected_start);
    CHECK(uint32_t(int(stats["visible_count"])) == expected_count);
    CHECK(stream->get_stats().buffer_switches > 0);

    pipeline->stop_streaming();
    pipeline->shutdown();
}

TEST_CASE("[GPU Memory Stream] Memory Leak Detection") {
    // Track initial memory usage
    uint64_t initial_static_memory = OS::get_singleton()->get_static_memory_usage();

    {
        Ref<GaussianMemoryStream> stream;
        stream.instantiate();

        RenderingDevice *rd = RenderingDevice::get_singleton();
        if (!rd) {
            RenderingServer *rs = RenderingServer::get_singleton();
            if (rs) {
                rd = rs->create_local_rendering_device();
            }
        }

        if (rd) {
            // Perform multiple init/shutdown cycles
            for (int i = 0; i < 5; i++) {
                Error err = stream->initialize(rd, 10000, 32);
                CHECK(err == OK);

                // Stream some data
                LocalVector<Gaussian> gaussians = create_test_gaussians(1000);
                stream->stream_gaussians_async(gaussians);
                stream->wait_for_all_uploads();

                stream->shutdown();
            }
        }

        // Stream goes out of scope
    }

    // Check memory wasn't leaked
    uint64_t final_static_memory = OS::get_singleton()->get_static_memory_usage();
    int64_t memory_diff = final_static_memory - initial_static_memory;

    // Allow small variation (< 1MB)
    CHECK_MESSAGE(Math::abs(memory_diff) < 1024 * 1024,
                 vformat("Memory leak detected: %d bytes difference", memory_diff));
}

TEST_CASE("[Streaming Pipeline] Chunk snapshot stays coherent under concurrent position mutations") {
#ifndef THREADS_ENABLED
    MESSAGE("Skipping - THREADS_ENABLED is not enabled in this build");
    return;
#endif

    const uint32_t splat_count = GaussianStreamingSystem::CHUNK_SIZE;
    Ref<::GaussianData> data = create_test_gaussian_data(splat_count);

    PackedVector3Array pattern_a;
    PackedVector3Array pattern_b;
    pattern_a.resize(splat_count);
    pattern_b.resize(splat_count);
    Vector3 *pattern_a_write = pattern_a.ptrw();
    Vector3 *pattern_b_write = pattern_b.ptrw();
    for (uint32_t i = 0; i < splat_count; i++) {
        pattern_a_write[i] = Vector3(11.0f, float(i), -3.0f);
        pattern_b_write[i] = Vector3(-7.0f, float(i) + 0.25f, 5.0f);
    }
    data->set_positions(pattern_a);

    SnapshotPositionStressContext ctx;
    ctx.data = data;
    ctx.pattern_a = &pattern_a;
    ctx.pattern_b = &pattern_b;
    ctx.splat_count = splat_count;

    Thread writer_thread;
    Thread reader_thread;
    writer_thread.start(_snapshot_position_writer_thread, &ctx);
    reader_thread.start(_snapshot_position_reader_thread, &ctx);

    constexpr uint32_t iterations = 96;
    for (uint32_t i = 0; i < iterations; i++) {
        if ((i % 2) == 0) {
            ctx.writer_begin.post();
            Thread::yield();
            ctx.reader_begin.post();
        } else {
            ctx.reader_begin.post();
            Thread::yield();
            ctx.writer_begin.post();
        }
        ctx.writer_done.wait();
        ctx.reader_done.wait();
    }

    ctx.stop.store(true, std::memory_order_release);
    ctx.writer_begin.post();
    ctx.reader_begin.post();
    writer_thread.wait_to_finish();
    reader_thread.wait_to_finish();

    CHECK_FALSE(ctx.writer_failed.load(std::memory_order_acquire));
    CHECK_FALSE(ctx.reader_failed.load(std::memory_order_acquire));
}

TEST_CASE("[Streaming Pipeline] Chunk snapshot stays coherent under concurrent SH mutations") {
#ifndef THREADS_ENABLED
    MESSAGE("Skipping - THREADS_ENABLED is not enabled in this build");
    return;
#endif

    const uint32_t splat_count = 8192;
    Ref<::GaussianData> data = create_test_gaussian_data(splat_count);

    PackedFloat32Array dc_only_data;
    dc_only_data.resize(splat_count * 3);
    float *dc_write = dc_only_data.ptrw();

    constexpr uint32_t full_sh_high_order = 2;
    constexpr uint32_t full_sh_first_order = 3;
    constexpr uint32_t full_floats_per_gaussian = (1 + full_sh_first_order + full_sh_high_order) * 3;
    PackedFloat32Array full_data;
    full_data.resize(splat_count * full_floats_per_gaussian);
    float *full_write = full_data.ptrw();

    for (uint32_t i = 0; i < splat_count; i++) {
        const Color dc_only_color = _dc_only_color_for_index(i);
        const uint32_t dc_base = i * 3;
        dc_write[dc_base + 0] = dc_only_color.r;
        dc_write[dc_base + 1] = dc_only_color.g;
        dc_write[dc_base + 2] = dc_only_color.b;

        const uint32_t full_base = i * full_floats_per_gaussian;
        const Color full_color = _full_color_for_index(i);
        full_write[full_base + 0] = full_color.r;
        full_write[full_base + 1] = full_color.g;
        full_write[full_base + 2] = full_color.b;
        for (uint32_t j = 0; j < full_sh_first_order; j++) {
            const Vector3 coeff = _full_first_order_for_index(i, j);
            const uint32_t coeff_base = full_base + 3 + (j * 3);
            full_write[coeff_base + 0] = coeff.x;
            full_write[coeff_base + 1] = coeff.y;
            full_write[coeff_base + 2] = coeff.z;
        }
        for (uint32_t j = 0; j < full_sh_high_order; j++) {
            const Vector3 coeff = _full_high_order_for_index(i, j);
            const uint32_t coeff_base = full_base + 3 + (full_sh_first_order * 3) + (j * 3);
            full_write[coeff_base + 0] = coeff.x;
            full_write[coeff_base + 1] = coeff.y;
            full_write[coeff_base + 2] = coeff.z;
        }
    }

    data->set_spherical_harmonics(dc_only_data);

    SnapshotSHStressContext ctx;
    ctx.data = data;
    ctx.dc_only_data = &dc_only_data;
    ctx.full_data = &full_data;
    ctx.splat_count = splat_count;

    Thread writer_thread;
    Thread reader_thread;
    writer_thread.start(_snapshot_sh_writer_thread, &ctx);
    reader_thread.start(_snapshot_sh_reader_thread, &ctx);

    constexpr uint32_t iterations = 72;
    for (uint32_t i = 0; i < iterations; i++) {
        if ((i % 2) == 0) {
            ctx.writer_begin.post();
            Thread::yield();
            ctx.reader_begin.post();
        } else {
            ctx.reader_begin.post();
            Thread::yield();
            ctx.writer_begin.post();
        }
        ctx.writer_done.wait();
        ctx.reader_done.wait();
    }

    ctx.stop.store(true, std::memory_order_release);
    ctx.writer_begin.post();
    ctx.reader_begin.post();
    writer_thread.wait_to_finish();
    reader_thread.wait_to_finish();

    CHECK_FALSE(ctx.writer_failed.load(std::memory_order_acquire));
    CHECK_FALSE(ctx.reader_failed.load(std::memory_order_acquire));
}

TEST_CASE("[Streaming Pipeline] Stale generation upload jobs are dropped") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        rd = rs->create_local_rendering_device();
    }
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(rd);

    const uint32_t asset_id = 7;
    system->register_asset(asset_id, create_test_gaussian_data(GaussianStreamingSystem::CHUNK_SIZE));

    Transform3D camera_transform;
    camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);

    system->begin_residency_requests();
    system->request_chunk_residency(asset_id, 0, 0);
    system->finalize_residency_requests();
    system->update_streaming(camera_transform, projection);

    bool saw_async_queue = false;
    for (int i = 0; i < 32; i++) {
        if (system->get_pending_pack_jobs() > 0 || system->get_pending_upload_jobs() > 0) {
            saw_async_queue = true;
            break;
        }
        system->update_streaming(camera_transform, projection);
        OS::get_singleton()->delay_usec(1000);
    }
    if (!saw_async_queue) {
        MESSAGE("Skipping - Async pack/upload queue not observed in test environment");
        return;
    }

    system->unregister_asset(asset_id);
    system->register_asset(asset_id, create_test_gaussian_data(1024));

    for (int i = 0; i < 96; i++) {
        system->update_streaming(camera_transform, projection);
        if (system->get_pending_pack_jobs() == 0 && system->get_pending_upload_jobs() == 0) {
            break;
        }
        OS::get_singleton()->delay_usec(500);
    }

    CHECK(system->get_pending_pack_jobs() == 0);
    CHECK(system->get_pending_upload_jobs() == 0);
    CHECK(system->get_loaded_chunks() == 0);
    for (int i = 0; i < 8; i++) {
        system->update_streaming(camera_transform, projection);
        CHECK(system->get_pending_pack_jobs() == 0);
        CHECK(system->get_pending_upload_jobs() == 0);
        CHECK(system->get_loaded_chunks() == 0);
    }

    system->begin_residency_requests();
    system->request_chunk_residency(asset_id, 0, 0);
    system->finalize_residency_requests();
    for (int i = 0; i < 96; i++) {
        system->update_streaming(camera_transform, projection);
        if (system->get_loaded_chunks() > 0) {
            break;
        }
        OS::get_singleton()->delay_usec(500);
    }

    CHECK(system->get_loaded_chunks() > 0);
}

TEST_CASE("[Streaming Pipeline] Dense-id generation rejects stale remapped instance mappings") {
    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(nullptr);

    const uint32_t asset_a = 101;
    const uint32_t asset_b = 202;
    const uint32_t asset_c = 303;
    system->register_asset(asset_a, create_test_gaussian_data(1024));
    system->register_asset(asset_b, create_test_gaussian_data(1024));

    LocalVector<InstanceDataGPU> mapped_a;
    mapped_a.resize(1);
    mapped_a[0] = {};
    mapped_a[0].ids[0] = asset_a;
    CHECK(system->remap_instance_asset_ids(mapped_a, false));

    const uint32_t dense_a = mapped_a[0].ids[0];
    const uint32_t dense_a_generation = mapped_a[0].lod[1];
    CHECK(dense_a != 0);
    CHECK(dense_a_generation != 0);

    system->unregister_asset(asset_a);
    system->register_asset(asset_c, create_test_gaussian_data(1024));

    LocalVector<InstanceDataGPU> mapped_c;
    mapped_c.resize(1);
    mapped_c[0] = {};
    mapped_c[0].ids[0] = asset_c;
    CHECK(system->remap_instance_asset_ids(mapped_c, false));

    const uint32_t dense_c = mapped_c[0].ids[0];
    const uint32_t dense_c_generation = mapped_c[0].lod[1];
    CHECK(dense_c == dense_a); // Dense slot can be reused.
    CHECK(dense_c_generation != dense_a_generation); // Generation must advance on reuse.

    LocalVector<InstanceDataGPU> stale_dense_mapping;
    stale_dense_mapping.resize(1);
    stale_dense_mapping[0] = {};
    stale_dense_mapping[0].ids[0] = dense_a;
    stale_dense_mapping[0].lod[1] = dense_a_generation;

    CHECK_FALSE(system->remap_instance_asset_ids(stale_dense_mapping, false));
    CHECK(stale_dense_mapping[0].ids[0] == 0u);
    CHECK(stale_dense_mapping[0].lod[1] != dense_a_generation);
}

TEST_CASE("[Streaming Pipeline] Upload abort clears pending chunk state") {
    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(nullptr);

    if (system->get_frame_buffer().is_valid()) {
        MESSAGE("Skipping - Streaming buffer is valid, missing-buffer abort path unavailable in this environment");
        return;
    }

    const uint32_t asset_id = 21;
    system->register_asset(asset_id, create_test_gaussian_data(GaussianStreamingSystem::CHUNK_SIZE));

    Transform3D camera_transform;
    camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);

    bool saw_pending_work = false;
    bool drained_after_pending = false;
    for (int i = 0; i < 160; i++) {
        system->begin_residency_requests();
        system->request_chunk_residency(asset_id, 0, 0);
        system->finalize_residency_requests();
        system->update_streaming(camera_transform, projection);

        const uint32_t pending_pack = system->get_pending_pack_jobs();
        const uint32_t pending_upload = system->get_pending_upload_jobs();
        if (pending_pack > 0 || pending_upload > 0) {
            saw_pending_work = true;
        }

        if (saw_pending_work && pending_pack == 0 && pending_upload == 0 && system->get_loaded_chunks() == 0) {
            drained_after_pending = true;
            break;
        }
        OS::get_singleton()->delay_usec(1000);
    }

    if (!saw_pending_work) {
        MESSAGE("Skipping - Async queue activity not observed");
        return;
    }

    CHECK(drained_after_pending);
    CHECK(system->get_pending_pack_jobs() == 0);
    CHECK(system->get_pending_upload_jobs() == 0);
    CHECK(system->get_loaded_chunks() == 0);
    for (int i = 0; i < 8; i++) {
        system->update_streaming(camera_transform, projection);
        CHECK(system->get_pending_pack_jobs() == 0);
        CHECK(system->get_pending_upload_jobs() == 0);
        CHECK(system->get_loaded_chunks() == 0);
    }
}

TEST_CASE("[Streaming Pipeline] Repeated upload aborts do not leave pending jobs stuck") {
    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(nullptr);

    if (system->get_frame_buffer().is_valid()) {
        MESSAGE("Skipping - Streaming buffer is valid, missing-buffer abort path unavailable in this environment");
        return;
    }

    const uint32_t asset_id = 22;
    system->register_asset(asset_id, create_test_gaussian_data(GaussianStreamingSystem::CHUNK_SIZE));

    Transform3D camera_transform;
    camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);

    int observed_cycles = 0;
    for (int cycle = 0; cycle < 5; cycle++) {
        bool saw_pending_work = false;
        bool drained = false;
        for (int i = 0; i < 80; i++) {
            system->begin_residency_requests();
            system->request_chunk_residency(asset_id, 0, 0);
            system->finalize_residency_requests();
            system->update_streaming(camera_transform, projection);

            const uint32_t pending_pack = system->get_pending_pack_jobs();
            const uint32_t pending_upload = system->get_pending_upload_jobs();
            if (pending_pack > 0 || pending_upload > 0) {
                saw_pending_work = true;
            }
            if (saw_pending_work && pending_pack == 0 && pending_upload == 0) {
                drained = true;
                break;
            }
            OS::get_singleton()->delay_usec(1000);
        }

        if (!saw_pending_work) {
            break;
        }
        observed_cycles++;
        CHECK(drained);
        CHECK(system->get_pending_pack_jobs() == 0);
        CHECK(system->get_pending_upload_jobs() == 0);
        CHECK(system->get_loaded_chunks() == 0);
    }

    if (observed_cycles == 0) {
        MESSAGE("Skipping - Async queue activity not observed");
        return;
    }

    CHECK(observed_cycles >= 1);
    CHECK(system->get_pending_pack_jobs() == 0);
    CHECK(system->get_pending_upload_jobs() == 0);
    CHECK(system->get_loaded_chunks() == 0);
    for (int i = 0; i < 8; i++) {
        system->update_streaming(camera_transform, projection);
        CHECK(system->get_pending_pack_jobs() == 0);
        CHECK(system->get_pending_upload_jobs() == 0);
        CHECK(system->get_loaded_chunks() == 0);
    }
}

TEST_CASE("[Streaming Pipeline] Atlas generation bumps on quantization buffer resize") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        rd = rs->create_local_rendering_device();
    }
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    const QuantizationConfig saved_quantization_config = g_quantization_config;
    g_quantization_config.per_chunk_quantization = true;
    g_quantization_config.position_bits = 16;
    g_quantization_config.scale_bits = 12;
    g_quantization_config.quantize_scales = false;

    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(rd);

    Transform3D camera_transform;
    camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);

    const uint32_t asset_id = 11;
    system->register_asset(asset_id, create_test_gaussian_data(GaussianStreamingSystem::CHUNK_SIZE));
    system->update_streaming(camera_transform, projection);

    const uint64_t generation_before_resize = system->get_atlas_generation();
    const RID quant_buffer_before_resize = system->get_atlas_quantization_buffer();
    CHECK(quant_buffer_before_resize.is_valid());

    system->unregister_asset(asset_id);
    system->register_asset(asset_id, create_test_gaussian_data(GaussianStreamingSystem::CHUNK_SIZE + 1));
    system->update_streaming(camera_transform, projection);

    const uint64_t generation_after_resize = system->get_atlas_generation();
    const RID quant_buffer_after_resize = system->get_atlas_quantization_buffer();
    CHECK(quant_buffer_after_resize.is_valid());
    CHECK(generation_after_resize > generation_before_resize);

    g_quantization_config = saved_quantization_config;
}

TEST_CASE("[Streaming Pipeline] IO layout hints clamp chunk size to CHUNK_SIZE") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        rd = rs->create_local_rendering_device();
    }
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(rd);

    GaussianStreamingSystem::ConfigOverrides overrides;
    overrides.override_io_source = true;
    system->set_config_overrides(overrides);

    const uint32_t asset_id = 4242;
    const uint32_t oversized_count = GaussianStreamingSystem::CHUNK_SIZE + 1024;
    Vector<GaussianStreamingSystem::ChunkLayoutHint> hints;
    hints.resize(1);
    hints.write[0].start_idx = 0;
    hints.write[0].count = oversized_count;
    hints.write[0].bounds = AABB(Vector3(-1.0f, -1.0f, -1.0f), Vector3(2.0f, 2.0f, 2.0f));
    hints.write[0].center = Vector3();
    hints.write[0].radius = 1.0f;
    system->set_io_chunk_layout_hints(hints, asset_id);

    system->register_asset(asset_id, create_test_gaussian_data(oversized_count));

    Transform3D camera_transform;
    camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);
    system->update_streaming(camera_transform, projection);

    CHECK(system->get_max_chunk_splats() == GaussianStreamingSystem::CHUNK_SIZE);
    CHECK(system->get_max_chunk_count_per_asset() >= 2u);
}

TEST_CASE("[Streaming Pipeline] Randomized IO layout hint cases keep stable fallback reasons and chunk counts") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        rd = rs->create_local_rendering_device();
    }
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    const uint32_t ordered_total = 2048;
    const uint32_t fallback_total = GaussianStreamingSystem::CHUNK_SIZE * 2 + 17;
    const uint32_t fallback_chunk_count = (fallback_total + GaussianStreamingSystem::CHUNK_SIZE - 1) / GaussianStreamingSystem::CHUNK_SIZE;
    const uint32_t oversize_total = GaussianStreamingSystem::CHUNK_SIZE * 3 + 50;
    const uint32_t oversize_chunk_count = 5;
    const uint32_t seeds[] = { 7u, 42u, 99u, 1337u, 9001u };

    for (uint32_t seed : seeds) {
        std::mt19937 rng(seed);

        {
            Ref<GaussianStreamingSystem> system;
            system.instantiate();
            system->initialize_empty(rd);

            GaussianStreamingSystem::ConfigOverrides overrides;
            overrides.override_io_source = true;
            system->set_config_overrides(overrides);

            const uint32_t asset_id = 1000u + seed;
            system->register_asset(asset_id, create_test_gaussian_data(ordered_total));

            Vector<GaussianStreamingSystem::ChunkLayoutHint> ordered_hints;
            ordered_hints.resize(4);
            const uint32_t segment = ordered_total / 4;
            for (int i = 0; i < 4; i++) {
                ordered_hints.write[i].start_idx = uint32_t(i) * segment;
                ordered_hints.write[i].count = (i == 3) ? (ordered_total - uint32_t(i) * segment) : segment;
                ordered_hints.write[i].bounds = AABB(Vector3(-1.0f, -1.0f, -1.0f), Vector3(2.0f, 2.0f, 2.0f));
                ordered_hints.write[i].center = Vector3();
                ordered_hints.write[i].radius = 1.0f;
            }
            std::shuffle(ordered_hints.ptrw(), ordered_hints.ptrw() + ordered_hints.size(), rng);
            system->set_io_chunk_layout_hints(ordered_hints, asset_id);

            Transform3D camera_transform;
            camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
            Projection projection;
            projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);
            system->update_streaming(camera_transform, projection);

            CHECK(_layout_hint_last_reason(system) == "none");
            CHECK(system->get_max_chunk_count_per_asset() == 4u);
        }

        {
            Ref<GaussianStreamingSystem> system;
            system.instantiate();
            system->initialize_empty(rd);

            GaussianStreamingSystem::ConfigOverrides overrides;
            overrides.override_io_source = true;
            system->set_config_overrides(overrides);

            const uint32_t asset_id = 2000u + seed;
            system->register_asset(asset_id, create_test_gaussian_data(fallback_total));

            Vector<GaussianStreamingSystem::ChunkLayoutHint> overlap_hints = _build_partitioned_hints(fallback_total, 128, rng);
            REQUIRE(overlap_hints.size() > 1);
            const int mutate_index = int(seed % uint32_t(overlap_hints.size() - 1)) + 1;
            overlap_hints.write[mutate_index].start_idx = overlap_hints[mutate_index - 1].start_idx;
            system->set_io_chunk_layout_hints(overlap_hints, asset_id);

            Transform3D camera_transform;
            camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
            Projection projection;
            projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);
            system->update_streaming(camera_transform, projection);

            CHECK(_layout_hint_last_reason(system) == "hint_overlapping_ranges");
            CHECK(_layout_hint_reason_count(system, "hint_overlapping_ranges") == 1);
            CHECK(system->get_max_chunk_count_per_asset() == fallback_chunk_count);
        }

        {
            Ref<GaussianStreamingSystem> system;
            system.instantiate();
            system->initialize_empty(rd);

            GaussianStreamingSystem::ConfigOverrides overrides;
            overrides.override_io_source = true;
            system->set_config_overrides(overrides);

            const uint32_t asset_id = 3000u + seed;
            system->register_asset(asset_id, create_test_gaussian_data(fallback_total));

            Vector<GaussianStreamingSystem::ChunkLayoutHint> remap_hints = _build_partitioned_hints(fallback_total, 192, rng);
            const int remap_index = int(seed % uint32_t(remap_hints.size()));
            remap_hints.write[remap_index].source_indices_remapped = true;
            remap_hints.write[remap_index].source_index_offset = remap_hints[remap_index].start_idx;
            system->set_io_chunk_layout_hints(remap_hints, asset_id);

            Transform3D camera_transform;
            camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
            Projection projection;
            projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);
            system->update_streaming(camera_transform, projection);

            CHECK(_layout_hint_last_reason(system) == "remap_flag_unexpected");
            CHECK(_layout_hint_reason_count(system, "remap_flag_unexpected") == 1);
            CHECK(system->get_max_chunk_count_per_asset() == fallback_chunk_count);
        }

        {
            Ref<GaussianStreamingSystem> system;
            system.instantiate();
            system->initialize_empty(rd);

            GaussianStreamingSystem::ConfigOverrides overrides;
            overrides.override_io_source = true;
            system->set_config_overrides(overrides);

            const uint32_t asset_id = 4000u + seed;
            system->register_asset(asset_id, create_test_gaussian_data(oversize_total));

            std::uniform_int_distribution<uint32_t> first_count_dist(
                    GaussianStreamingSystem::CHUNK_SIZE + 1,
                    GaussianStreamingSystem::CHUNK_SIZE + 32);
            const uint32_t first_count = first_count_dist(rng);
            const uint32_t second_count = oversize_total - first_count;

            Vector<GaussianStreamingSystem::ChunkLayoutHint> oversize_hints;
            oversize_hints.resize(2);
            oversize_hints.write[0].start_idx = 0;
            oversize_hints.write[0].count = first_count;
            oversize_hints.write[0].bounds = AABB(Vector3(-1.0f, -1.0f, -1.0f), Vector3(2.0f, 2.0f, 2.0f));
            oversize_hints.write[0].center = Vector3();
            oversize_hints.write[0].radius = 1.0f;

            oversize_hints.write[1].start_idx = first_count;
            oversize_hints.write[1].count = second_count;
            oversize_hints.write[1].bounds = AABB(Vector3(-1.0f, -1.0f, -1.0f), Vector3(2.0f, 2.0f, 2.0f));
            oversize_hints.write[1].center = Vector3();
            oversize_hints.write[1].radius = 1.0f;
            system->set_io_chunk_layout_hints(oversize_hints, asset_id);

            Transform3D camera_transform;
            camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
            Projection projection;
            projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);
            system->update_streaming(camera_transform, projection);

            CHECK(_layout_hint_last_reason(system) == "none");
            CHECK(system->get_max_chunk_count_per_asset() == oversize_chunk_count);
        }
    }
}

TEST_CASE("[Streaming Pipeline] VRAM accounting includes auxiliary atlas overhead") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        rd = rs->create_local_rendering_device();
    }
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(rd);

    const uint32_t asset_id = 77;
    system->register_asset(asset_id, create_test_gaussian_data(1024));

    Transform3D camera_transform;
    camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);

    // One update is enough to materialize atlas metadata buffers.
    system->update_streaming(camera_transform, projection);
    Dictionary pre_residency_stats = system->get_chunk_culling_stats();
    const double payload_before_mb = pre_residency_stats.get("vram_payload_mb", 0.0);
    const double overhead_before_mb = pre_residency_stats.get("vram_overhead_mb", 0.0);
    const uint64_t total_before = system->get_vram_usage();
    CHECK(overhead_before_mb > 0.0);
    CHECK(total_before > 0);

    system->begin_residency_requests();
    system->request_chunk_residency(asset_id, 0, 0);
    system->finalize_residency_requests();

    for (int i = 0; i < 96; i++) {
        system->update_streaming(camera_transform, projection);
        if (system->get_loaded_chunks() > 0) {
            break;
        }
        OS::get_singleton()->delay_usec(500);
    }
    if (system->get_loaded_chunks() == 0) {
        MESSAGE("Skipping - Residency chunk failed to load in current test environment");
        return;
    }

    Dictionary post_residency_stats = system->get_chunk_culling_stats();
    const double payload_after_mb = post_residency_stats.get("vram_payload_mb", 0.0);
    const double overhead_after_mb = post_residency_stats.get("vram_overhead_mb", 0.0);
    const uint64_t total_after = system->get_vram_usage();

    CHECK(payload_after_mb >= payload_before_mb);
    CHECK(overhead_after_mb >= overhead_before_mb);
    CHECK(total_after > total_before);
}

TEST_CASE("[Streaming Pipeline] Invalid camera/projection input is rejected safely") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        rd = rs->create_local_rendering_device();
    }
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(rd);

    Transform3D valid_transform;
    valid_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
    Projection valid_projection;
    valid_projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);

    Dictionary before_stats = system->get_chunk_culling_stats();
    const int invalid_before = int(before_stats.get("invalid_camera_input_events", 0));

    Transform3D invalid_transform = valid_transform;
    invalid_transform.origin.x = std::numeric_limits<float>::quiet_NaN();
    system->update_streaming(invalid_transform, valid_projection);

    Dictionary after_transform_stats = system->get_chunk_culling_stats();
    const int invalid_after_transform = int(after_transform_stats.get("invalid_camera_input_events", 0));
    CHECK(invalid_after_transform == invalid_before + 1);

    Projection invalid_projection = valid_projection;
    invalid_projection.columns[0][0] = std::numeric_limits<float>::infinity();
    system->update_streaming(valid_transform, invalid_projection);

    Dictionary after_projection_stats = system->get_chunk_culling_stats();
    const int invalid_after_projection = int(after_projection_stats.get("invalid_camera_input_events", 0));
    CHECK(invalid_after_projection == invalid_before + 2);
}

TEST_CASE("[Streaming Pipeline] VRAM debug stats expose reported usage vs capacity semantics") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        rd = rs->create_local_rendering_device();
    }
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(rd);

    Dictionary vram_stats = system->get_vram_debug_stats();
    CHECK(vram_stats.has("device_reported_usage_bytes"));
    CHECK(vram_stats.has("device_capacity_bytes"));
    CHECK(vram_stats.has("device_capacity_known"));
    CHECK(vram_stats.has("device_total_semantics"));
    CHECK(vram_stats.has("device_total_bytes"));

    const int64_t reported_usage_bytes = int64_t(vram_stats.get("device_reported_usage_bytes", int64_t(-1)));
    const int64_t capacity_bytes = int64_t(vram_stats.get("device_capacity_bytes", int64_t(-1)));
    const bool capacity_known = bool(vram_stats.get("device_capacity_known", false));

    CHECK(reported_usage_bytes >= 0);
    CHECK(capacity_bytes >= 0);
    CHECK(String(vram_stats.get("device_total_semantics", String())) == String("reported_usage"));
    CHECK(int64_t(vram_stats.get("device_total_bytes", int64_t(-1))) == reported_usage_bytes);
    if (!capacity_known) {
        CHECK(capacity_bytes == 0);
    }
}

TEST_CASE("[Streaming Pipeline] Effective max chunks are clamped to runtime buffer capacity") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        rd = rs->create_local_rendering_device();
    }
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    Ref<GaussianStreamingSystem> system;
    system.instantiate();

    GaussianStreamingSystem::ConfigOverrides low_capacity_overrides;
    low_capacity_overrides.override_vram_budget = true;
    low_capacity_overrides.vram_budget_config.auto_regulate_enabled = false;
    low_capacity_overrides.vram_budget_config.budget_mb = 1024;
    low_capacity_overrides.vram_budget_config.min_chunks = 4;
    low_capacity_overrides.vram_budget_config.max_chunks = 4;
    system->set_config_overrides(low_capacity_overrides);
    system->initialize_empty(rd);

    const uint64_t chunk_bytes = uint64_t(GaussianStreamingSystem::CHUNK_SIZE) * sizeof(PackedGaussian);
    const uint32_t runtime_capacity_chunks = chunk_bytes > 0
            ? static_cast<uint32_t>(uint64_t(system->get_buffer_capacity_splats()) / uint64_t(GaussianStreamingSystem::CHUNK_SIZE))
            : 0;
    if (runtime_capacity_chunks == 0) {
        MESSAGE("Skipping - Runtime streaming buffer capacity is zero");
        return;
    }

    GaussianStreamingSystem::ConfigOverrides high_capacity_overrides = low_capacity_overrides;
    high_capacity_overrides.vram_budget_config.min_chunks = runtime_capacity_chunks;
    high_capacity_overrides.vram_budget_config.max_chunks = runtime_capacity_chunks + 32;
    system->set_config_overrides(high_capacity_overrides);

    Transform3D camera_transform;
    camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);
    system->update_streaming(camera_transform, projection, 1.0f / 30.0f);

    const uint32_t effective_max_chunks = system->get_effective_max_chunks();
    CHECK(high_capacity_overrides.vram_budget_config.max_chunks > runtime_capacity_chunks);
    CHECK(effective_max_chunks == runtime_capacity_chunks);
    CHECK(effective_max_chunks < high_capacity_overrides.vram_budget_config.max_chunks);
}

TEST_CASE("[Streaming Pipeline] Budget eviction prioritizes non-primary chunks under regulator pressure") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        rd = rs->create_local_rendering_device();
    }
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (!project_settings) {
        MESSAGE("Skipping - ProjectSettings unavailable");
        return;
    }

    const String async_pack_setting = "rendering/gaussian_splatting/streaming/async_pack_enabled";
    ScopedProjectSettingRestore async_pack_guard(project_settings, async_pack_setting);
    project_settings->set_setting(async_pack_setting, false);

    Ref<GaussianStreamingSystem> system;
    system.instantiate();

    GaussianStreamingSystem::ConfigOverrides relaxed_overrides;
    relaxed_overrides.override_vram_budget = true;
    relaxed_overrides.vram_budget_config.auto_regulate_enabled = false;
    relaxed_overrides.vram_budget_config.budget_mb = 1024;
    relaxed_overrides.vram_budget_config.min_chunks = 1;
    relaxed_overrides.vram_budget_config.max_chunks = 64;
    system->set_config_overrides(relaxed_overrides);
    system->initialize_empty(rd);
    if (!system->is_runtime_ready()) {
        MESSAGE("Skipping - Streaming runtime not ready");
        return;
    }

    const uint32_t asset_id = 909;
    system->register_asset(asset_id, create_test_gaussian_data(GaussianStreamingSystem::CHUNK_SIZE));

    Transform3D camera_transform;
    camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);

    system->begin_residency_requests();
    system->request_chunk_residency(asset_id, 0, 0);
    system->finalize_residency_requests();
    for (int i = 0; i < 32; i++) {
        system->update_streaming(camera_transform, projection);
        if (system->get_loaded_chunks() > 0) {
            break;
        }
    }
    if (system->get_loaded_chunks() == 0) {
        MESSAGE("Skipping - Non-primary residency chunk failed to load in current test environment");
        return;
    }

    const uint32_t loaded_before = system->get_loaded_chunks();
    CHECK(loaded_before > 0);

    GaussianStreamingSystem::ConfigOverrides constrained_overrides = relaxed_overrides;
    constrained_overrides.vram_budget_config.budget_mb = 1;
    system->set_config_overrides(constrained_overrides);

    bool observed_budget_eviction = false;
    for (int i = 0; i < 8; i++) {
        system->update_streaming(camera_transform, projection);
        if (system->get_loaded_chunks() < loaded_before) {
            observed_budget_eviction = true;
            break;
        }
    }

    CHECK(observed_budget_eviction);
    CHECK(system->get_loaded_chunks() == 0);
    CHECK(system->get_visible_chunks_evicted_this_frame() == 0);

    Dictionary analytics = system->get_streaming_analytics();
    CHECK(int64_t(analytics.get("scheduler_non_primary_scan_chunks", int64_t(0))) > 0);
}

TEST_CASE("[Streaming Pipeline] Tier presets apply streaming caps while project overrides remain traceable") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        rd = rs->create_local_rendering_device();
    }
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (!project_settings) {
        MESSAGE("Skipping - ProjectSettings unavailable");
        return;
    }

    const String tier_preset_setting = "rendering/gaussian_splatting/quality/tier_preset";
    const String tier_apply_setting = "rendering/gaussian_splatting/quality/tier_apply_streaming_budgets";
    const String upload_frame_setting = "rendering/gaussian_splatting/streaming/max_upload_mb_per_frame";
    const String upload_slice_setting = "rendering/gaussian_splatting/streaming/max_upload_mb_per_slice";
    const String upload_bandwidth_setting = "rendering/gaussian_splatting/streaming/max_upload_mb_per_second";
    const String vram_budget_setting = "rendering/gaussian_splatting/streaming/vram_budget_mb";
    const String min_chunks_setting = "rendering/gaussian_splatting/streaming/min_chunks_in_vram";
    const String max_chunks_setting = "rendering/gaussian_splatting/streaming/max_chunks_in_vram";

    ScopedProjectSettingRestore tier_preset_guard(project_settings, tier_preset_setting);
    ScopedProjectSettingRestore tier_apply_guard(project_settings, tier_apply_setting);
    ScopedProjectSettingRestore upload_frame_guard(project_settings, upload_frame_setting);
    ScopedProjectSettingRestore upload_slice_guard(project_settings, upload_slice_setting);
    ScopedProjectSettingRestore upload_bandwidth_guard(project_settings, upload_bandwidth_setting);
    ScopedProjectSettingRestore vram_budget_guard(project_settings, vram_budget_setting);
    ScopedProjectSettingRestore min_chunks_guard(project_settings, min_chunks_setting);
    ScopedProjectSettingRestore max_chunks_guard(project_settings, max_chunks_setting);

    project_settings->set_setting(tier_apply_setting, true);
    project_settings->set_setting(tier_preset_setting, "low");
    project_settings->set_setting(upload_frame_setting, 128);
    project_settings->set_setting(upload_slice_setting, 16);
    project_settings->set_setting(upload_bandwidth_setting, 0);
    project_settings->set_setting(vram_budget_setting, 12288);
    project_settings->set_setting(min_chunks_setting, 4);
    project_settings->set_setting(max_chunks_setting, 128);

    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(rd);

    if (!system->is_runtime_ready()) {
        MESSAGE("Skipping - Streaming runtime not ready");
        return;
    }

    Transform3D camera_transform;
    camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);

    for (int i = 0; i < 2; i++) {
        system->begin_frame();
        system->update_streaming(camera_transform, projection, 1.0f / 60.0f);
        system->end_frame();
    }

    Dictionary analytics = system->get_streaming_analytics();
    CHECK(analytics.has("cap_tier_preset"));
    CHECK(analytics.has("cap_tier_active"));
    CHECK(String(analytics.get("cap_tier_preset", String())) == String("low"));
    CHECK(bool(analytics.get("cap_tier_active", false)));
    CHECK(int64_t(analytics.get("effective_upload_cap_mb_per_frame", int64_t(-1))) == 32);
    CHECK(int64_t(analytics.get("effective_upload_cap_mb_per_slice", int64_t(-1))) == 8);
    CHECK(int64_t(analytics.get("effective_upload_cap_mb_per_second", int64_t(-1))) == 256);
    CHECK(int64_t(analytics.get("effective_vram_budget_mb", int64_t(-1))) == 2048);
    CHECK(int64_t(analytics.get("effective_vram_min_chunks", int64_t(-1))) == 2);
    CHECK(int64_t(analytics.get("effective_vram_max_chunks", int64_t(-1))) == 32);
    CHECK(String(analytics.get("cap_source_upload_mb_per_frame", String())) == String("tier_preset"));
    CHECK(String(analytics.get("cap_source_upload_mb_per_slice", String())) == String("tier_preset"));
    CHECK(String(analytics.get("cap_source_upload_mb_per_second", String())) == String("tier_preset"));
    CHECK(String(analytics.get("cap_source_vram_budget_mb", String())) == String("tier_preset"));
    CHECK(String(analytics.get("cap_source_vram_min_chunks", String())) == String("tier_preset"));
    CHECK(String(analytics.get("cap_source_vram_max_chunks", String())) == String("tier_preset"));

    project_settings->set_setting(upload_frame_setting, 77);
    project_settings->set_setting(vram_budget_setting, 3333);

    for (int i = 0; i < 2; i++) {
        system->begin_frame();
        system->update_streaming(camera_transform, projection, 1.0f / 60.0f);
        system->end_frame();
    }

    analytics = system->get_streaming_analytics();
    CHECK(int64_t(analytics.get("effective_upload_cap_mb_per_frame", int64_t(-1))) == 77);
    CHECK(int64_t(analytics.get("effective_upload_cap_mb_per_slice", int64_t(-1))) == 8);
    CHECK(int64_t(analytics.get("effective_vram_budget_mb", int64_t(-1))) == 3333);
    CHECK(String(analytics.get("cap_source_upload_mb_per_frame", String())) == String("project_override"));
    CHECK(String(analytics.get("cap_source_upload_mb_per_slice", String())) == String("tier_preset"));
    CHECK(String(analytics.get("cap_source_vram_budget_mb", String())) == String("project_override"));
    CHECK(String(analytics.get("cap_source_vram_max_chunks", String())) == String("tier_preset"));
}

TEST_CASE("[Streaming Pipeline] Queue pressure summary and latch invariants are deterministic") {
    StreamingQueuePressureController::PressureSample idle_sample;
    const StreamingQueuePressureController::PressureSummary idle_summary =
            StreamingQueuePressureController::summarize(idle_sample);
    CHECK_FALSE(idle_summary.active);
    CHECK(idle_summary.source == String(StreamingQueuePressureController::SOURCE_NONE));
    CHECK(idle_summary.reason == String(StreamingQueuePressureController::REASON_NONE));
    CHECK(StreamingQueuePressureController::validate_summary_invariants(idle_summary, idle_sample));

    StreamingQueuePressureController::PressureSample pack_backlog_sample;
    pack_backlog_sample.pack_queue_depth = 5;
    const StreamingQueuePressureController::PressureSummary pack_backlog_summary =
            StreamingQueuePressureController::summarize(pack_backlog_sample);
    CHECK(pack_backlog_summary.active);
    CHECK(pack_backlog_summary.source == String(StreamingQueuePressureController::SOURCE_PACK));
    CHECK(pack_backlog_summary.reason == String(StreamingQueuePressureController::REASON_PACK_QUEUE_BACKLOG));
    CHECK(pack_backlog_summary.backlog_depth == 5);
    CHECK(StreamingQueuePressureController::validate_summary_invariants(pack_backlog_summary, pack_backlog_sample));

    StreamingQueuePressureController::PressureSample queue_and_caps_sample;
    queue_and_caps_sample.pack_queue_depth = 2;
    queue_and_caps_sample.upload_frame_cap_hit = true;
    const StreamingQueuePressureController::PressureSummary queue_and_caps_summary =
            StreamingQueuePressureController::summarize(queue_and_caps_sample);
    CHECK(queue_and_caps_summary.active);
    CHECK(queue_and_caps_summary.source == String(StreamingQueuePressureController::SOURCE_COMBINED));
    CHECK(queue_and_caps_summary.reason == String(StreamingQueuePressureController::REASON_QUEUE_AND_CAPS));
    CHECK(StreamingQueuePressureController::validate_summary_invariants(queue_and_caps_summary, queue_and_caps_sample));

    bool latch_active = false;
    String latch_source = "bad_source";
    String latch_reason = "bad_reason";
    StreamingQueuePressureController::reset_latched_state(latch_active, latch_source, latch_reason);
    CHECK_FALSE(latch_active);
    CHECK(latch_source == String(StreamingQueuePressureController::SOURCE_NONE));
    CHECK(latch_reason == String(StreamingQueuePressureController::REASON_NONE));

    StreamingQueuePressureController::latch_summary(pack_backlog_summary, latch_active, latch_source, latch_reason);
    CHECK(latch_active);
    CHECK(latch_source == String(StreamingQueuePressureController::SOURCE_PACK));
    CHECK(latch_reason == String(StreamingQueuePressureController::REASON_PACK_QUEUE_BACKLOG));
    CHECK(StreamingQueuePressureController::validate_latched_state_invariants(latch_active, latch_source, latch_reason));

    // A later idle summary in the same frame should not clear the latch.
    StreamingQueuePressureController::latch_summary(idle_summary, latch_active, latch_source, latch_reason);
    CHECK(latch_active);
    CHECK(latch_source == String(StreamingQueuePressureController::SOURCE_PACK));
    CHECK(latch_reason == String(StreamingQueuePressureController::REASON_PACK_QUEUE_BACKLOG));
}

TEST_CASE("[Streaming Pipeline] Queue pressure scan-budget throttle transitions are deterministic") {
    StreamingQueuePressureController::ScanBudgetInput baseline_input;
    baseline_input.base_scan_budget = 10;
    baseline_input.throttle_enabled = false;
    const StreamingQueuePressureController::ScanBudgetResult baseline_result =
            StreamingQueuePressureController::compute_candidate_scan_budget(baseline_input);
    CHECK(baseline_result.scan_budget == 10);
    CHECK_FALSE(baseline_result.throttle_active);

    StreamingQueuePressureController::ScanBudgetInput throttled_input;
    throttled_input.base_scan_budget = 10;
    throttled_input.throttle_enabled = true;
    throttled_input.throttle_min_queue_depth = 4;
    throttled_input.observed_queue_depth = 4;
    throttled_input.throttle_scan_cap = 8;
    throttled_input.scanned_this_frame = 2;
    throttled_input.enqueue_headroom = UINT32_MAX;
    const StreamingQueuePressureController::ScanBudgetResult throttled_result =
            StreamingQueuePressureController::compute_candidate_scan_budget(throttled_input);
    CHECK(throttled_result.throttle_active);
    CHECK(throttled_result.scan_budget == 6);

    throttled_input.observed_queue_depth = 6;
    const StreamingQueuePressureController::ScanBudgetResult deep_pressure_result =
            StreamingQueuePressureController::compute_candidate_scan_budget(throttled_input);
    CHECK(deep_pressure_result.throttle_active);
    CHECK(deep_pressure_result.scan_budget == 2);

    throttled_input.enqueue_headroom = 0;
    const StreamingQueuePressureController::ScanBudgetResult zero_headroom_result =
            StreamingQueuePressureController::compute_candidate_scan_budget(throttled_input);
    CHECK(zero_headroom_result.throttle_active);
    CHECK(zero_headroom_result.scan_budget == 1);
}

TEST_CASE("[Streaming Pipeline] Residency admission controller enforces deterministic invariants") {
    ResidencyBudgetController::AdmissionPolicy policy;
    policy.can_replace_without_eviction = false;
    policy.enforce_vram_regulator_gate = false;
    policy.vram_regulator_allows_load = true;

    ResidencyBudgetController::AdmissionFrameBudget frame_budget =
            ResidencyBudgetController::make_frame_budget(8, 2, false);
    ResidencyBudgetController::AdmissionGate below_capacity_gate =
            ResidencyBudgetController::compute_admission_gate(4, frame_budget, policy);
    CHECK(below_capacity_gate.decision == ResidencyBudgetController::AdmissionDecision::LoadDirect);
    CHECK_FALSE(ResidencyBudgetController::should_attempt_visible_evict_fallback(below_capacity_gate));

    ResidencyBudgetController::AdmissionGate at_capacity_gate =
            ResidencyBudgetController::compute_admission_gate(8, frame_budget, policy);
    CHECK(at_capacity_gate.decision == ResidencyBudgetController::AdmissionDecision::EvictThenLoad);
    CHECK(ResidencyBudgetController::should_attempt_visible_evict_fallback(at_capacity_gate));

    ResidencyBudgetController::note_successful_eviction(frame_budget);
    CHECK(frame_budget.evictions_left == 1);

    ResidencyBudgetController::note_blocked_eviction(frame_budget);
    CHECK(frame_budget.eviction_blocked);
    ResidencyBudgetController::AdmissionGate blocked_eviction_gate =
            ResidencyBudgetController::compute_admission_gate(8, frame_budget, policy);
    CHECK(blocked_eviction_gate.decision == ResidencyBudgetController::AdmissionDecision::Skip);
    CHECK_FALSE(ResidencyBudgetController::should_attempt_visible_evict_fallback(blocked_eviction_gate));

    ResidencyBudgetController::AdmissionPolicy replacement_policy;
    replacement_policy.can_replace_without_eviction = true;
    replacement_policy.enforce_vram_regulator_gate = true;
    replacement_policy.vram_regulator_allows_load = true;
    ResidencyBudgetController::AdmissionGate replacement_gate =
            ResidencyBudgetController::compute_admission_gate(8, frame_budget, replacement_policy);
    CHECK(replacement_gate.decision == ResidencyBudgetController::AdmissionDecision::LoadDirect);
    CHECK_FALSE(ResidencyBudgetController::should_attempt_visible_evict_fallback(replacement_gate));

    ResidencyBudgetController::AdmissionPolicy regulator_gate_policy = policy;
    regulator_gate_policy.enforce_vram_regulator_gate = true;
    regulator_gate_policy.vram_regulator_allows_load = false;
    ResidencyBudgetController::AdmissionFrameBudget regulator_frame_budget =
            ResidencyBudgetController::make_frame_budget(8, 2, false);
    ResidencyBudgetController::AdmissionGate regulator_gate =
            ResidencyBudgetController::compute_admission_gate(4, regulator_frame_budget, regulator_gate_policy);
    CHECK(regulator_gate.decision == ResidencyBudgetController::AdmissionDecision::EvictThenLoad);
    CHECK(ResidencyBudgetController::should_attempt_visible_evict_fallback(regulator_gate));

    ResidencyBudgetController::AdmissionFrameBudget regulator_blocked_budget =
            ResidencyBudgetController::make_frame_budget(8, 0, false);
    ResidencyBudgetController::AdmissionGate regulator_blocked_gate =
            ResidencyBudgetController::compute_admission_gate(4, regulator_blocked_budget, regulator_gate_policy);
    CHECK(regulator_blocked_gate.decision == ResidencyBudgetController::AdmissionDecision::Skip);
    CHECK_FALSE(ResidencyBudgetController::should_attempt_visible_evict_fallback(regulator_blocked_gate));
}

TEST_CASE("[Streaming Pipeline] Owner mismatch contract encodes remediation paths deterministically") {
    ResourceOwnerMismatchContract::Inputs invalid_rid_inputs;
    invalid_rid_inputs.rid_valid = false;
    const ResourceOwnerMismatchContract::Decision invalid_rid_decision =
            ResourceOwnerMismatchContract::evaluate(invalid_rid_inputs);
    CHECK_FALSE(invalid_rid_decision.mismatch_detected);
    CHECK(ResourceOwnerMismatchContract::validate(invalid_rid_inputs, invalid_rid_decision));

    ResourceOwnerMismatchContract::Inputs matched_owner_inputs;
    matched_owner_inputs.rid_valid = true;
    matched_owner_inputs.has_owner = true;
    matched_owner_inputs.owner_instance_id = 42;
    matched_owner_inputs.active_instance_id = 42;
    const ResourceOwnerMismatchContract::Decision matched_owner_decision =
            ResourceOwnerMismatchContract::evaluate(matched_owner_inputs);
    CHECK_FALSE(matched_owner_decision.mismatch_detected);
    CHECK(ResourceOwnerMismatchContract::validate(matched_owner_inputs, matched_owner_decision));

    ResourceOwnerMismatchContract::Inputs foreign_owner_inputs;
    foreign_owner_inputs.rid_valid = true;
    foreign_owner_inputs.has_owner = true;
    foreign_owner_inputs.owner_instance_id = 100;
    foreign_owner_inputs.active_instance_id = 200;
    const ResourceOwnerMismatchContract::Decision foreign_owner_decision =
            ResourceOwnerMismatchContract::evaluate(foreign_owner_inputs);
    CHECK(foreign_owner_decision.mismatch_detected);
    CHECK(foreign_owner_decision.should_attempt_release);
    CHECK_FALSE(foreign_owner_decision.should_force_invalidate_after_release);
    CHECK(ResourceOwnerMismatchContract::validate(foreign_owner_inputs, foreign_owner_decision));

    ResourceOwnerMismatchContract::Inputs missing_owner_inputs;
    missing_owner_inputs.rid_valid = true;
    missing_owner_inputs.has_owner = false;
    missing_owner_inputs.owner_instance_id = 0;
    missing_owner_inputs.active_instance_id = 300;
    const ResourceOwnerMismatchContract::Decision missing_owner_decision =
            ResourceOwnerMismatchContract::evaluate(missing_owner_inputs);
    CHECK(missing_owner_decision.mismatch_detected);
    CHECK(missing_owner_decision.should_attempt_release);
    CHECK(missing_owner_decision.should_force_invalidate_after_release);
    CHECK(ResourceOwnerMismatchContract::validate(missing_owner_inputs, missing_owner_decision));
}

TEST_CASE("[Streaming Pipeline] Instance content generation tracks instance pipeline budget changes") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    GaussianSplatManager *manager_owner = nullptr;
    GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
    if (!manager) {
        manager_owner = memnew(GaussianSplatManager);
        manager = manager_owner;
    }
    CHECK(manager != nullptr);
    if (manager == nullptr) {
        return;
    }

    RenderingDevice *primary_device = manager->get_primary_rendering_device();
    if (primary_device == nullptr) {
        MESSAGE("Skipping - Primary rendering device unavailable");
        if (manager_owner) {
            memdelete(manager_owner);
        }
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_device);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        if (manager_owner) {
            memdelete(manager_owner);
        }
        return;
    }

    const uint32_t total_gaussians = GaussianStreamingSystem::CHUNK_SIZE * 2;
    Error set_data_err = renderer->set_gaussian_data(create_test_gaussian_data(total_gaussians));
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        if (manager_owner) {
            memdelete(manager_owner);
        }
        return;
    }

    Transform3D cam_transform;
    cam_transform.origin = Vector3(1000.0f, 0.0f, 10.0f);
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 5000.0f);

    uint64_t stable_generation_prev = 0;
    uint64_t stable_generation_curr = 0;
    for (int i = 0; i < 180; i++) {
        const bool rendered = renderer->render_for_view(cam_transform, projection, RID(), Size2i(512, 512));
        CHECK(rendered);
        if (!rendered) {
            break;
        }
        if (renderer->has_rendered_content()) {
            stable_generation_prev = stable_generation_curr;
            stable_generation_curr = renderer->get_instance_pipeline_content_generation();
            if (stable_generation_prev != 0 && stable_generation_prev == stable_generation_curr) {
                break;
            }
        }
        OS::get_singleton()->delay_usec(500);
    }

    if (stable_generation_prev == 0 || stable_generation_prev != stable_generation_curr) {
        MESSAGE("Skipping - Instance pipeline content generation did not stabilize in test environment");
        renderer.unref();
        if (manager_owner) {
            memdelete(manager_owner);
        }
        return;
    }

    const uint64_t generation_before_budget_change = stable_generation_curr;
    const int previous_max_splats = renderer->get_max_splats();
    renderer->set_max_splats(MAX(1024, previous_max_splats / 2));
    renderer->clear_instance_pipeline_buffers();

    uint64_t generation_after_budget_change = generation_before_budget_change;
    for (int i = 0; i < 90; i++) {
        const bool rendered = renderer->render_for_view(cam_transform, projection, RID(), Size2i(512, 512));
        CHECK(rendered);
        if (!rendered) {
            break;
        }
        generation_after_budget_change = renderer->get_instance_pipeline_content_generation();
        if (generation_after_budget_change != generation_before_budget_change) {
            break;
        }
        OS::get_singleton()->delay_usec(500);
    }

    CHECK(generation_after_budget_change != generation_before_budget_change);

    renderer.unref();
    if (manager_owner) {
        memdelete(manager_owner);
    }
}

TEST_CASE("[Streaming Pipeline] LOD debug stats track transitions_this_frame across camera moves") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        rd = rs->create_local_rendering_device();
    }
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    ProjectSettings *project_settings = ProjectSettings::get_singleton();
    if (!project_settings) {
        MESSAGE("Skipping - ProjectSettings unavailable");
        return;
    }

    const String lod_enabled_setting = "rendering/gaussian_splatting/lod/enabled";
    const String lod_levels_setting = "rendering/gaussian_splatting/lod/num_levels";
    const String lod_base_threshold_setting = "rendering/gaussian_splatting/lod/base_threshold";
    const String lod_max_distance_setting = "rendering/gaussian_splatting/lod/max_distance";

    ScopedProjectSettingRestore lod_enabled_guard(project_settings, lod_enabled_setting);
    ScopedProjectSettingRestore lod_levels_guard(project_settings, lod_levels_setting);
    ScopedProjectSettingRestore lod_base_threshold_guard(project_settings, lod_base_threshold_setting);
    ScopedProjectSettingRestore lod_max_distance_guard(project_settings, lod_max_distance_setting);

    project_settings->set_setting(lod_enabled_setting, true);
    project_settings->set_setting(lod_levels_setting, 8);
    project_settings->set_setting(lod_base_threshold_setting, 4.0f);
    project_settings->set_setting(lod_max_distance_setting, 80.0f);

    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(rd);
    if (!system->is_runtime_ready()) {
        MESSAGE("Skipping - Streaming runtime not ready");
        return;
    }

    const uint32_t asset_id = 404;
    system->register_asset(asset_id, create_clustered_test_gaussian_data(1024, Vector3(0.0f, 0.0f, 0.0f)));

    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);

    Transform3D near_camera_transform;
    near_camera_transform.origin = Vector3(0.0f, 0.0f, 4.0f);

    Transform3D far_camera_transform;
    far_camera_transform.origin = Vector3(0.0f, 0.0f, 60.0f);

    system->update_streaming(near_camera_transform, projection);
    Dictionary near_stats = system->get_lod_debug_stats();
    CHECK(near_stats.has("transitions_this_frame"));

    system->update_streaming(far_camera_transform, projection);
    Dictionary transition_stats = system->get_lod_debug_stats();
    const int transitions_after_move = int(transition_stats.get("transitions_this_frame", 0));
    CHECK(transitions_after_move > 0);

    system->update_streaming(far_camera_transform, projection);
    Dictionary stable_stats = system->get_lod_debug_stats();
    CHECK(int(stable_stats.get("transitions_this_frame", -1)) == 0);
}

TEST_CASE("[Streaming Pipeline] Renderer renders streamed non-zero chunk") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    GaussianSplatManager *manager_owner = nullptr;
    GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
    if (!manager) {
        manager_owner = memnew(GaussianSplatManager);
        manager = manager_owner;
    }
    CHECK(manager != nullptr);
    if (manager == nullptr) {
        return;
    }

    RenderingDevice *primary_device = manager->get_primary_rendering_device();
    if (primary_device == nullptr) {
        MESSAGE("Skipping - Primary rendering device unavailable");
        if (manager_owner) {
            memdelete(manager_owner);
        }
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_device);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        if (manager_owner) {
            memdelete(manager_owner);
        }
        return;
    }

    const uint32_t chunk_size = GaussianStreamingSystem::CHUNK_SIZE;
    const uint32_t total_gaussians = chunk_size * 2; // Ensure at least one non-zero chunk

    LocalVector<Gaussian> gaussians;
    gaussians.resize(total_gaussians);

    for (uint32_t i = 0; i < total_gaussians; i++) {
        Gaussian &g = gaussians[i];
        g = Gaussian{};
        const bool in_first_chunk = i < chunk_size;
        const float base_x = in_first_chunk ? 0.0f : 1000.0f;
        const uint32_t local_index = in_first_chunk ? i : (i - chunk_size);
        g.position = Vector3(base_x + float(local_index % 16) * 0.01f, 0.0f, float(local_index / 16) * 0.01f);
        g.scale = Vector3(0.1f, 0.1f, 0.1f);
        g.rotation = Quaternion();
        g.opacity = 1.0f;
        g.sh_dc = Color(1.0f, 1.0f, 1.0f, 1.0f);
        g.normal = Vector3(0, 1, 0);
        g.area = 0.01f;
        g.brush_axes = Vector2(1.0f, 0.0f);
        g.painterly_meta = gaussian_pack_painterly_meta(0);
    }

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        if (manager_owner) {
            memdelete(manager_owner);
        }
        return;
    }

    // Keep this regression deterministic by disabling Stage-B filters for the baseline.
    renderer->set_frustum_culling(false);
    renderer->set_lod_enabled(false);
    renderer->set_lod_min_screen_size(0.0f);
    renderer->set_lod_max_distance(0.0f);
    renderer->set_tiny_splat_screen_radius(0.0f);

    Transform3D cam_transform;
    cam_transform.origin = Vector3(1000.0f, 0.0f, 10.0f); // Position near non-zero chunk center
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 5000.0f);

    CHECK_FALSE(renderer->has_rendered_content());

    bool rendered = renderer->render_for_view(cam_transform, projection, RID(), Size2i(512, 512));
    CHECK(rendered);

    if (!renderer->has_rendered_content()) {
        rendered = renderer->render_for_view(cam_transform, projection, RID(), Size2i(512, 512));
        CHECK(rendered);
    }

    CHECK(renderer->has_rendered_content());
    CHECK(renderer->get_visible_splat_count() == chunk_size);

    renderer.unref();
    if (manager_owner) {
        memdelete(manager_owner);
    }
}

TEST_CASE("[Streaming Pipeline] Instance depth Stage-B applies frustum/screen/distance culling") {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs == nullptr) {
        MESSAGE("Skipping - Rendering server unavailable");
        return;
    }

    GaussianSplatManager *manager_owner = nullptr;
    GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
    if (!manager) {
        manager_owner = memnew(GaussianSplatManager);
        manager = manager_owner;
    }
    CHECK(manager != nullptr);
    if (manager == nullptr) {
        return;
    }

    RenderingDevice *primary_device = manager->get_primary_rendering_device();
    if (primary_device == nullptr) {
        MESSAGE("Skipping - Primary rendering device unavailable");
        if (manager_owner) {
            memdelete(manager_owner);
        }
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(primary_device);
    CHECK(renderer.is_valid());
    if (!renderer.is_valid()) {
        if (manager_owner) {
            memdelete(manager_owner);
        }
        return;
    }

    const uint32_t total_gaussians = 8192;
    LocalVector<Gaussian> gaussians;
    gaussians.resize(total_gaussians);
    const uint32_t inside_count = total_gaussians / 2;
    for (uint32_t i = 0; i < total_gaussians; i++) {
        Gaussian &g = gaussians[i];
        g = Gaussian{};
        const bool inside_frustum_band = i < inside_count;
        const float band_x = inside_frustum_band ? -0.8f : 30.0f;
        const float local_x = float(i % 64) * 0.025f;
        const float local_y = (float((i / 64) % 64) - 32.0f) * 0.03f;
        g.position = Vector3(band_x + local_x, local_y, -10.0f);
        g.scale = Vector3(0.06f, 0.06f, 0.06f);
        g.rotation = Quaternion();
        g.opacity = 1.0f;
        g.sh_dc = Color(1.0f, 1.0f, 1.0f, 1.0f);
        g.normal = Vector3(0, 1, 0);
        g.area = 0.01f;
        g.brush_axes = Vector2(1.0f, 0.0f);
        g.painterly_meta = gaussian_pack_painterly_meta(0);
    }

    Ref<::GaussianData> data;
    data.instantiate();
    data->set_gaussians(gaussians);

    Error set_data_err = renderer->set_gaussian_data(data);
    CHECK(set_data_err == OK);
    if (set_data_err != OK) {
        if (manager_owner) {
            memdelete(manager_owner);
        }
        return;
    }

    renderer->set_lod_enabled(true);
    renderer->set_lod_bias(1.0f);
    renderer->set_lod_min_screen_size(0.0f);
    renderer->set_lod_max_distance(0.0f);
    renderer->set_tiny_splat_screen_radius(0.0f);
    renderer->set_frustum_culling(false);

    Transform3D cam_transform;
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 200.0f);

    auto render_sample = [&](int p_frames) {
        uint32_t visible = 0;
        for (int i = 0; i < p_frames; i++) {
            bool rendered = renderer->render_for_view(cam_transform, projection, RID(), Size2i(512, 512));
            CHECK(rendered);
            if (!rendered) {
                break;
            }
            visible = renderer->get_visible_splat_count();
            OS::get_singleton()->delay_usec(500);
        }
        return visible;
    };

    uint32_t baseline_visible = 0;
    bool instance_pipeline_ready = false;
    for (int i = 0; i < 180; i++) {
        const uint32_t visible = render_sample(1);
        if (renderer->has_instance_pipeline_buffers() && renderer->has_rendered_content() && visible > 0) {
            baseline_visible = visible;
            instance_pipeline_ready = true;
            break;
        }
    }

    if (!instance_pipeline_ready || baseline_visible == 0) {
        MESSAGE("Skipping - Instance pipeline did not become ready in Stage-B culling regression test");
        renderer.unref();
        if (manager_owner) {
            memdelete(manager_owner);
        }
        return;
    }

    renderer->set_frustum_culling(true);
    const uint32_t frustum_visible = render_sample(6);

    renderer->set_frustum_culling(false);
    renderer->set_tiny_splat_screen_radius(64.0f);
    const uint32_t tiny_visible = render_sample(6);

    renderer->set_tiny_splat_screen_radius(0.0f);
    renderer->set_lod_max_distance(15.0f);
    const uint32_t distance_visible = render_sample(6);

    CHECK(frustum_visible < baseline_visible);
    CHECK(tiny_visible < baseline_visible);
    CHECK(distance_visible < baseline_visible);
    CHECK(distance_visible > 0);

    renderer.unref();
    if (manager_owner) {
        memdelete(manager_owner);
    }
}
