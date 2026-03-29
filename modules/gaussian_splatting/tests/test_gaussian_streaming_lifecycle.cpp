#include "../core/gaussian_streaming.h"

#include "test_macros.h"

#include "core/os/os.h"
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

TEST_CASE("[Streaming Pipeline] async chunk upload rejects tampered payload checksums") {
    const TestRenderingDeviceHandle rd_handle = _get_test_rendering_device();
    RenderingDevice *rd = rd_handle.rd;
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(rd);
    if (!system->is_runtime_ready()) {
        MESSAGE("Skipping - Streaming runtime not ready");
        return;
    }

    GaussianStreamingSystem &system_ref = *system.ptr();
    auto &uploads = system->_internal_get_upload_pipeline();
    if (!uploads.async_pack_enabled || !uploads.pack_thread_running.load(std::memory_order_acquire)) {
        MESSAGE("Skipping - Async pack threads unavailable");
        return;
    }

    const uint32_t asset_id = 4242;
    system->register_asset(asset_id, _create_streaming_phase_order_test_data());

    const bool queued_upload = uploads.queue_chunk_load(system_ref, asset_id, 0);
    REQUIRE(queued_upload);

    StreamingUploadPipeline::PendingChunkUpload *prepared_job = nullptr;
    for (int i = 0; i < 500; i++) {
        {
            MutexLock lock(uploads.pack_mutex);
            if (uploads.upload_queue_read_idx < uploads.upload_queue.size()) {
                prepared_job = uploads.upload_queue[uploads.upload_queue_read_idx];
                if (prepared_job && !prepared_job->packed_data.is_empty()) {
                    break;
                }
                prepared_job = nullptr;
            }
        }
        OS::get_singleton()->delay_usec(1000);
    }

    REQUIRE(prepared_job != nullptr);

    {
        MutexLock lock(uploads.pack_mutex);
        prepared_job = uploads.upload_queue[uploads.upload_queue_read_idx];
        REQUIRE(prepared_job != nullptr);
        REQUIRE(!prepared_job->packed_data.is_empty());
        PackedGaussian *packed_data = prepared_job->packed_data.ptrw();
        REQUIRE(packed_data != nullptr);
        uint8_t *payload_bytes = reinterpret_cast<uint8_t *>(packed_data);
        payload_bytes[0] ^= 0x01;
    }

    uploads.process_upload_queue(system_ref);
    system->begin_frame();
    system->end_frame();

    CHECK(system->get_pending_pack_jobs() == 0);
    CHECK(system->get_pending_upload_jobs() == 0);
    CHECK(system->get_loaded_chunks() == 0);

    Dictionary analytics = system->get_streaming_analytics();
    Dictionary diagnostics = analytics.get("diagnostics", Dictionary());
    CHECK(String(analytics.get("diagnostics_category", String())) == "integrity_mismatch");
    CHECK(String(analytics.get("diagnostics_reason", String())).contains("checksum mismatch"));
    CHECK(bool(analytics.get("diagnostics_has_failure", false)));
    CHECK(String(diagnostics.get("category", String())) == "integrity_mismatch");
    CHECK(String(diagnostics.get("reason", String())).contains("checksum mismatch"));
    CHECK(int64_t(diagnostics.get("invariant_upload_lifecycle_violations", int64_t(0))) == 1);
    CHECK(String(diagnostics.get("last_invariant_context", String())) == "process_upload_queue.payload_checksum");
    CHECK(String(diagnostics.get("last_invariant_message", String())).contains("checksum mismatch"));
    CHECK(int64_t(diagnostics.get("integrity_mismatch_count", int64_t(0))) == 1);
    CHECK(String(diagnostics.get("last_integrity_mismatch_message", String())).contains("checksum mismatch"));

    system->initialize_empty(rd);
    system->begin_frame();
    system->end_frame();

    Dictionary reset_analytics = system->get_streaming_analytics();
    Dictionary reset_diagnostics = reset_analytics.get("diagnostics", Dictionary());
    CHECK(String(reset_analytics.get("diagnostics_category", String("ok"))) == "ok");
    CHECK(String(reset_analytics.get("diagnostics_reason", String("healthy"))) == "healthy");
    CHECK_FALSE(bool(reset_analytics.get("diagnostics_has_failure", true)));
    CHECK(int64_t(reset_diagnostics.get("integrity_mismatch_count", int64_t(-1))) == 0);
    CHECK(String(reset_diagnostics.get("last_integrity_mismatch_message", String())).is_empty());
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

TEST_CASE("[Streaming Pipeline] initialize_empty republishes atlas state after registry cleanup") {
    const TestRenderingDeviceHandle rd_handle = _get_test_rendering_device();
    RenderingDevice *rd = rd_handle.rd;
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(rd);
    if (!system->is_runtime_ready()) {
        MESSAGE("Skipping - Streaming runtime not ready");
        return;
    }

    const uint32_t asset_id = 27182;
    system->register_asset(asset_id, _create_streaming_phase_order_test_data());

    Transform3D camera_transform;
    camera_transform.origin = Vector3(0.0f, 0.0f, 5.0f);
    Projection projection;
    projection.set_perspective(60.0f, 1.0f, 0.1f, 2000.0f);

    system->begin_frame();
    system->update_streaming(camera_transform, projection);
    system->end_frame();

    const uint64_t generation_before_reinit = system->get_atlas_generation();
    CHECK(generation_before_reinit > 0);
    CHECK(system->get_asset_meta_buffer().is_valid());
    CHECK(system->get_chunk_meta_buffer().is_valid());
    CHECK(system->get_asset_chunk_index_buffer().is_valid());

    system->initialize_empty(rd);
    CHECK(system->is_runtime_ready());
    CHECK(system->get_atlas_generation() > generation_before_reinit);
    CHECK(system->get_asset_meta_buffer().is_valid());
    CHECK(system->get_chunk_meta_buffer().is_valid());
    CHECK(system->get_asset_chunk_index_buffer().is_valid());
    CHECK_FALSE(system->has_asset(asset_id));
}

TEST_CASE("[Streaming Pipeline] initialize_empty keeps atlas metadata buffers valid with zero chunks") {
    const TestRenderingDeviceHandle rd_handle = _get_test_rendering_device();
    RenderingDevice *rd = rd_handle.rd;
    if (!rd) {
        MESSAGE("Skipping - Rendering device unavailable");
        return;
    }

    Ref<GaussianStreamingSystem> system;
    system.instantiate();
    system->initialize_empty(rd);
    if (!system->is_runtime_ready()) {
        MESSAGE("Skipping - Streaming runtime not ready");
        return;
    }

    const uint64_t first_generation = system->get_atlas_generation();
    CHECK(first_generation > 0);
    CHECK(system->get_asset_meta_buffer().is_valid());
    CHECK(system->get_chunk_meta_buffer().is_valid());
    CHECK(system->get_asset_chunk_index_buffer().is_valid());

    system->initialize_empty(rd);

    CHECK(system->is_runtime_ready());
    CHECK(system->get_atlas_generation() > first_generation);
    CHECK(system->get_asset_meta_buffer().is_valid());
    CHECK(system->get_chunk_meta_buffer().is_valid());
    CHECK(system->get_asset_chunk_index_buffer().is_valid());
}
