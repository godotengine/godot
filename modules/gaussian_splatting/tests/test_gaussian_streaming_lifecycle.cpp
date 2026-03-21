#include "../core/gaussian_streaming.h"

#include "test_macros.h"

#include "servers/rendering_server.h"

namespace {

Ref<GaussianData> _create_streaming_phase_order_test_data(uint32_t p_count = 1024) {
    Ref<GaussianData> data;
    data.instantiate();

    LocalVector<Gaussian> gaussians;
    gaussians.resize(p_count);

    const uint32_t grid_width = 32;
    for (uint32_t i = 0; i < p_count; i++) {
        Gaussian &g = gaussians[i];
        const float x = float(i % grid_width) * 0.05f;
        const float y = float((i / grid_width) % grid_width) * 0.05f;
        const float z = -2.0f - float(i / (grid_width * grid_width)) * 0.05f;
        g.position = Vector3(x, y, z);
        g.scale = Vector3(0.05f, 0.05f, 0.05f);
        g.rotation = Quaternion();
        g.opacity = 1.0f;
        g.sh_dc = Color(1.0f, 0.85f, 0.7f, 1.0f);
        g.normal = Vector3(0.0f, 1.0f, 0.0f);
        g.area = 0.01f;
    }

    data->set_gaussians(gaussians);
    return data;
}

struct TestRenderingDeviceHandle {
    RenderingDevice *rd = nullptr;
    bool owns_rd = false;

    ~TestRenderingDeviceHandle() {
        if (owns_rd && rd) {
            memdelete(rd);
        }
    }
};

TestRenderingDeviceHandle _get_test_rendering_device() {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        return {};
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd) {
        rd = rs->create_local_rendering_device();
        return { rd, rd != nullptr };
    }
    return { rd, false };
}

} // namespace

TEST_CASE("[Streaming Pipeline] stop_pack_threads clears partial lifecycle state") {
    GaussianStreamingSystem system;
    auto &uploads = system._internal_get_upload_pipeline();

    uploads.pack_thread_running.store(false, std::memory_order_release);
    uploads.pack_thread_exit.store(true, std::memory_order_release);
    uploads.pack_threads.resize(2);
    uploads.pack_thread_contexts.resize(2);
    uploads.pack_threads[0] = nullptr;
    uploads.pack_threads[1] = nullptr;

    uploads.stop_pack_threads(system);

    CHECK(uploads.pack_threads.is_empty());
    CHECK(uploads.pack_thread_contexts.is_empty());
    CHECK_FALSE(uploads.pack_thread_running.load(std::memory_order_acquire));
    CHECK_FALSE(uploads.pack_thread_exit.load(std::memory_order_acquire));
}

TEST_CASE("[Streaming Pipeline] update_streaming publishes phase timings before atlas sync and keeps atlas generation stable when idle") {
    const TestRenderingDeviceHandle rd_handle = _get_test_rendering_device();
    RenderingDevice *rd = rd_handle.rd;
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    {
        Ref<GaussianStreamingSystem> system;
        system.instantiate();
        system->initialize_empty(rd);
        if (!system->is_runtime_ready()) {
            MESSAGE("Skipping - Streaming runtime not ready");
            return;
        }

        const uint32_t asset_id = 31415;
        system->register_asset(asset_id, _create_streaming_phase_order_test_data());

        Transform3D camera_transform;
        camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
        Projection projection;
        projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);

        const RID asset_meta_before = system->get_asset_meta_buffer();
        const RID chunk_meta_before = system->get_chunk_meta_buffer();
        const RID asset_chunk_index_before = system->get_asset_chunk_index_buffer();
        const uint64_t generation_before = system->get_atlas_generation();

        system->begin_frame();
        system->update_streaming(camera_transform, projection);
        const uint64_t generation_after_update = system->get_atlas_generation();
        CHECK(generation_after_update > generation_before);
        CHECK(generation_after_update > 0);
        system->end_frame();

        Dictionary analytics = system->get_streaming_analytics();
        CHECK(analytics.has("scheduler_visibility_cpu_ms"));
        CHECK(analytics.has("scheduler_load_cpu_ms"));
        CHECK(analytics.has("scheduler_build_visible_cpu_ms"));
        CHECK(analytics.has("scheduler_prefetch_cpu_ms"));
        CHECK(analytics.has("scheduler_update_cpu_ms"));
        CHECK(analytics.has("scheduler_cpu_total_attributed_ms"));
        CHECK(analytics.has("scheduler_cpu_unattributed_ms"));
        CHECK(analytics.has("atlas_generation"));

        const double visibility_cpu_ms = double(analytics.get("scheduler_visibility_cpu_ms", 0.0));
        const double load_cpu_ms = double(analytics.get("scheduler_load_cpu_ms", 0.0));
        const double build_visible_cpu_ms = double(analytics.get("scheduler_build_visible_cpu_ms", 0.0));
        const double prefetch_cpu_ms = double(analytics.get("scheduler_prefetch_cpu_ms", 0.0));
        const double update_cpu_ms = double(analytics.get("scheduler_update_cpu_ms", 0.0));
        const double attributed_cpu_ms = double(analytics.get("scheduler_cpu_total_attributed_ms", 0.0));
        const double unattributed_cpu_ms = double(analytics.get("scheduler_cpu_unattributed_ms", 0.0));

        CHECK(visibility_cpu_ms >= 0.0);
        CHECK(load_cpu_ms >= 0.0);
        CHECK(build_visible_cpu_ms >= 0.0);
        CHECK(prefetch_cpu_ms >= 0.0);
        CHECK(update_cpu_ms >= 0.0);
        CHECK(attributed_cpu_ms >= 0.0);
        CHECK(unattributed_cpu_ms >= 0.0);
        CHECK(update_cpu_ms + 0.0001 >= attributed_cpu_ms);
        CHECK(attributed_cpu_ms + 0.0001 >= visibility_cpu_ms);
        CHECK(attributed_cpu_ms + 0.0001 >= load_cpu_ms);
        CHECK(attributed_cpu_ms + 0.0001 >= build_visible_cpu_ms);
        CHECK(attributed_cpu_ms + 0.0001 >= prefetch_cpu_ms);

        CHECK(int64_t(analytics.get("atlas_generation", int64_t(-1))) == int64_t(generation_after_update));
        CHECK(system->get_atlas_generation() == generation_after_update);
        CHECK(system->get_asset_meta_buffer().is_valid());
        CHECK(system->get_chunk_meta_buffer().is_valid());
        CHECK(system->get_asset_chunk_index_buffer().is_valid());

        const uint64_t generation_after_first_update = generation_after_update;
        const RID asset_meta_after_first_update = system->get_asset_meta_buffer();
        const RID chunk_meta_after_first_update = system->get_chunk_meta_buffer();
        const RID asset_chunk_index_after_first_update = system->get_asset_chunk_index_buffer();

        system->begin_frame();
        system->update_streaming(camera_transform, projection);
        system->end_frame();

        CHECK(system->get_atlas_generation() == generation_after_first_update);
        CHECK(system->get_asset_meta_buffer().get_id() == asset_meta_after_first_update.get_id());
        CHECK(system->get_chunk_meta_buffer().get_id() == chunk_meta_after_first_update.get_id());
        CHECK(system->get_asset_chunk_index_buffer().get_id() == asset_chunk_index_after_first_update.get_id());
        const bool asset_meta_stable = (system->get_asset_meta_buffer().get_id() == asset_meta_before.get_id()) || (asset_meta_before.get_id() == 0);
        CHECK(asset_meta_stable);
        const bool chunk_meta_stable = (system->get_chunk_meta_buffer().get_id() == chunk_meta_before.get_id()) || (chunk_meta_before.get_id() == 0);
        CHECK(chunk_meta_stable);
        const bool chunk_index_stable = (system->get_asset_chunk_index_buffer().get_id() == asset_chunk_index_before.get_id()) || (asset_chunk_index_before.get_id() == 0);
        CHECK(chunk_index_stable);
    }
}
