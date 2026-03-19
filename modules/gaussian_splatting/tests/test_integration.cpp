// Comprehensive integration tests for Gaussian Splatting modular renderer
#include "test_macros.h"
#include "test_utils.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../renderer/gpu_memory_stream.h"
#include "../renderer/gpu_sorter.h"
#include "../renderer/tile_renderer.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_manager.h"
#include "../interfaces/sync_policy.h"
#include "core/error/error_macros.h"
#include "../lod/streaming_lod_manager.h"
#include "../lod/hierarchical_splat_structure.h"
#include "../lod/adaptive_lod_system.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "core/config/engine.h"
#include "servers/rendering_server.h"
#include "servers/rendering/rendering_device.h"
#include <chrono>
#include <thread>

// Compile-time validation of structure sizes and alignments
static_assert(sizeof(Gaussian) == 144, "Gaussian struct size should be 144 bytes for GPU alignment");
static_assert(sizeof(GaussianMemoryStream::StreamBuffer) > 0, "StreamBuffer should be defined");
// Note: Gaussian struct size is 16-byte aligned (144 bytes) but alignof may differ
// depending on largest member alignment. GPU layout uses PackedGaussian with explicit alignas(16).
static_assert(sizeof(Gaussian) % 16 == 0, "Gaussian struct size should be 16-byte aligned for GPU");

// Verify class hierarchy
static_assert(std::is_base_of<RefCounted, GaussianSplatRenderer>::value, "GaussianSplatRenderer should inherit from RefCounted");
static_assert(std::is_base_of<RefCounted, GaussianMemoryStream>::value, "GaussianMemoryStream should inherit from RefCounted");
static_assert(std::is_base_of<RefCounted, StreamingPipeline>::value, "StreamingPipeline should inherit from RefCounted");
static_assert(std::is_base_of<RefCounted, IGPUSorter>::value, "IGPUSorter should inherit from RefCounted");

TEST_SUITE("[Gaussian Splatting Integration]") {

    // Helper to check if rendering device is available
    bool is_rendering_device_available() {
        RenderingServer *rs = RenderingServer::get_singleton();
        if (!rs) return false;
        RenderingDevice *rd = rs->create_local_rendering_device();
        bool available = rd != nullptr;
        if (rd) memdelete(rd);
        return available;
    }

    // Helper to generate test Gaussian data
    Ref<::GaussianData> generate_test_gaussians(uint32_t count, LocalVector<Gaussian> *out_gaussians = nullptr) {
        Ref<::GaussianData> data;
        data.instantiate();
        data->resize(count);

        if (out_gaussians) {
            out_gaussians->resize(count);
        }

        for (uint32_t i = 0; i < count; i++) {
            Gaussian g = Gaussian();
            // Generate procedural positions in a sphere
            float theta = Math::randf() * static_cast<float>(Math::TAU);
            float phi = acos(2.0f * Math::randf() - 1.0f);
            float radius = Math::randf() * 10.0f;

            g.position.x = radius * sin(phi) * cos(theta);
            g.position.y = radius * sin(phi) * sin(theta);
            g.position.z = radius * cos(phi);

            // Random scales and rotations
            g.scale.x = Math::randf() * 0.5f + 0.1f;
            g.scale.y = Math::randf() * 0.5f + 0.1f;
            g.scale.z = Math::randf() * 0.5f + 0.1f;

            g.rotation.x = Math::randf();
            g.rotation.y = Math::randf();
            g.rotation.z = Math::randf();
            g.rotation.w = Math::randf();
            g.rotation.normalize();

            // Random colors with opacity
            g.sh_dc = Color(Math::randf(), Math::randf(), Math::randf(), 1.0f);
            for (int sh = 0; sh < 3; sh++) {
                g.sh_1[sh] = Vector3(Math::randf(), Math::randf(), Math::randf());
            }
            g.opacity = Math::randf() * 0.8f + 0.2f;

            data->set_gaussian(i, g);
            if (out_gaussians) {
                (*out_gaussians)[i] = g;
            }
        }

        return data;
    }

    struct ScopedErrorCapture : public ErrorHandlerList {
        Vector<String> messages;

        static void _error_handler(void *p_userdata, const char *, const char *, int, const char *p_error, const char *p_message,
                bool, ErrorHandlerType) {
            ScopedErrorCapture *self = static_cast<ScopedErrorCapture *>(p_userdata);
            String message;
            if (p_message && p_message[0]) {
                message = String::utf8(p_message);
            } else if (p_error) {
                message = String::utf8(p_error);
            }
            if (!message.is_empty()) {
                self->messages.push_back(message);
            }
        }

        ScopedErrorCapture() {
            errfunc = _error_handler;
            userdata = this;
            add_error_handler(this);
        }

        ~ScopedErrorCapture() {
            remove_error_handler(this);
        }

        bool has_message_containing(const String &p_text) const {
            for (int i = 0; i < messages.size(); i++) {
                if (messages[i].find(p_text) != -1) {
                    return true;
                }
            }
            return false;
        }

        bool has_errors() const { return !messages.is_empty(); }
    };

    TEST_CASE("[Integration] Sync policy rejects null device") {
        Ref<CoarseSyncPolicy> sync_policy;
        sync_policy.instantiate();
        CHECK(sync_policy.is_valid());
        if (!sync_policy.is_valid()) {
            return;
        }

        CHECK_FALSE(gs_device_utils::is_local_device(nullptr));
        CHECK_FALSE(sync_policy->sync(nullptr, "null_device"));

        // Should be harmless no-ops.
        gs_device_utils::safe_submit(nullptr);
        gs_device_utils::safe_sync(nullptr);
        gs_device_utils::safe_submit_and_sync(nullptr);
    }

    TEST_CASE("[Integration] Sync policy enforces main-device no-submit/no-sync contract") {
        RenderingServer *rs = RenderingServer::get_singleton();
        CHECK(rs != nullptr);
        if (rs == nullptr) {
            return;
        }

        RenderingDevice *main_rd = rs->get_rendering_device();
        if (main_rd == nullptr) {
            MESSAGE("Skipping test - Main RenderingDevice unavailable");
            return;
        }

        CHECK(main_rd->is_main_rendering_device());
        CHECK_FALSE(gs_device_utils::is_local_device(main_rd));

        ScopedErrorCapture error_capture;

        Ref<CoarseSyncPolicy> sync_policy;
        sync_policy.instantiate();
        CHECK(sync_policy.is_valid());
        if (!sync_policy.is_valid()) {
            return;
        }

        CHECK(sync_policy->sync(main_rd, "main_device"));
        gs_device_utils::safe_submit(main_rd);
        gs_device_utils::safe_sync(main_rd);
        gs_device_utils::safe_submit_and_sync(main_rd);

        CHECK_FALSE(error_capture.has_message_containing("Only local devices can submit and sync"));
        CHECK_FALSE(error_capture.has_message_containing("sync can only be called after submit"));
    }

    TEST_CASE("[Integration] Sync policy treats local devices as submission-capable") {
        RenderingServer *rs = RenderingServer::get_singleton();
        CHECK(rs != nullptr);
        if (rs == nullptr) {
            return;
        }

        RenderingDevice *local_rd = rs->create_local_rendering_device();
        if (local_rd == nullptr) {
            MESSAGE("Skipping test - Local RenderingDevice unavailable");
            return;
        }

        CHECK_FALSE(local_rd->is_main_rendering_device());
        CHECK(gs_device_utils::is_local_device(local_rd));

        ScopedErrorCapture error_capture;

        Ref<CoarseSyncPolicy> sync_policy;
        sync_policy.instantiate();
        CHECK(sync_policy.is_valid());
        if (!sync_policy.is_valid()) {
            memdelete(local_rd);
            return;
        }

        gs_device_utils::safe_submit(local_rd);
        gs_device_utils::safe_sync(local_rd);
        CHECK(sync_policy->sync(local_rd, "local_device"));

        CHECK_FALSE(error_capture.has_message_containing("Only local devices can submit and sync"));
        CHECK_FALSE(error_capture.has_message_containing("sync can only be called after submit"));

        memdelete(local_rd);
    }

    TEST_CASE("[Integration] Component instantiation and basic linkage") {
        // Verify all components can be instantiated
        Ref<::GaussianData> data;
        data.instantiate();
        CHECK(data.is_valid());

        Ref<GaussianMemoryStream> stream;
        stream.instantiate();
        CHECK(stream.is_valid());

        Ref<StreamingPipeline> pipeline;
        pipeline.instantiate();
        CHECK(pipeline.is_valid());

        GaussianSplatManager *manager = memnew(GaussianSplatManager);
        CHECK(manager != nullptr);

        Ref<GaussianSplatRenderer> renderer;
        renderer.instantiate(manager->get_primary_rendering_device());
        CHECK(renderer.is_valid());

        // Verify linkage
        CHECK(renderer->set_gaussian_data(data) == OK);
        CHECK(renderer->get_gaussian_data() == data);

        renderer->set_max_splats(100000);
        CHECK(renderer->get_max_splats() == 100000);

        renderer.unref();
        memdelete(manager);

        print_line("[Integration] Basic component linkage verified");
    }

    TEST_CASE("[Integration] Renderer debug overlay toggles reflected in stats") {
        GaussianSplatManager *manager = memnew(GaussianSplatManager);
        CHECK(manager != nullptr);
        if (manager == nullptr) {
            return;
        }

        Ref<GaussianSplatRenderer> renderer;
        renderer.instantiate(manager->get_primary_rendering_device());
        CHECK(renderer.is_valid());
        if (!renderer.is_valid()) {
            memdelete(manager);
            return;
        }

        Dictionary stats = renderer->get_render_stats();
        CHECK(stats.has("debug_show_tile_grid"));
        CHECK(stats.has("debug_show_density_heatmap"));
        CHECK(stats.has("debug_show_performance_hud"));
        CHECK(stats.has("debug_show_residency_hud"));
        CHECK(stats.has("tile_assignment_ms"));
        CHECK(stats.has("tile_rasterization_ms"));
        CHECK(stats.has("debug_overlay_version"));
        CHECK(stats.has("debug_hud_version"));
        CHECK(stats.has("performance_hud_lines"));
        CHECK(stats.has("debug_tile_density_peak"));
        CHECK(stats.has("debug_tile_density_size"));
        CHECK(!bool(stats["debug_show_tile_grid"]));
        CHECK(!bool(stats["debug_show_density_heatmap"]));
        CHECK(!bool(stats["debug_show_performance_hud"]));
        CHECK(!bool(stats["debug_show_residency_hud"]));

        int overlay_version = int(stats["debug_overlay_version"]);
        int hud_version = int(stats["debug_hud_version"]);
        Array hud_lines = stats["performance_hud_lines"];
        CHECK(hud_lines.is_empty());

        renderer->set_debug_show_tile_grid(true);
        renderer->set_debug_show_density_heatmap(true);
        renderer->set_debug_show_performance_hud(true);

        stats = renderer->get_render_stats();
        CHECK(bool(stats["debug_show_tile_grid"]));
        CHECK(bool(stats["debug_show_density_heatmap"]));
        CHECK(bool(stats["debug_show_performance_hud"]));
        CHECK(int(stats["debug_overlay_version"]) >= overlay_version + 2);
        CHECK(int(stats["debug_hud_version"]) > hud_version);
        Array hud_enabled_lines = stats["performance_hud_lines"];
        CHECK(hud_enabled_lines.size() > 0);
        int overlay_version_enabled = int(stats["debug_overlay_version"]);

        renderer->set_debug_show_performance_hud(false);
        renderer->set_debug_show_residency_hud(true);

        stats = renderer->get_render_stats();
        CHECK(!bool(stats["debug_show_performance_hud"]));
        CHECK(bool(stats["debug_show_residency_hud"]));
        Array residency_lines = stats["performance_hud_lines"];
        CHECK(residency_lines.size() > 0);
        bool has_residency_heading = false;
        for (int i = 0; i < residency_lines.size(); i++) {
            if (String(residency_lines[i]) == String("Residency")) {
                has_residency_heading = true;
                break;
            }
        }
        CHECK(has_residency_heading);

        renderer->set_debug_show_tile_grid(false);
        renderer->set_debug_show_density_heatmap(false);
        renderer->set_debug_show_performance_hud(false);
        renderer->set_debug_show_residency_hud(false);

        stats = renderer->get_render_stats();
        CHECK(!bool(stats["debug_show_tile_grid"]));
        CHECK(!bool(stats["debug_show_density_heatmap"]));
        CHECK(!bool(stats["debug_show_performance_hud"]));
        CHECK(!bool(stats["debug_show_residency_hud"]));
        CHECK(int(stats["debug_overlay_version"]) >= overlay_version_enabled + 2);
        Array hud_disabled_lines = stats["performance_hud_lines"];
        CHECK(hud_disabled_lines.is_empty());

        renderer.unref();
        memdelete(manager);
    }

    // NOTE: Test case "[Integration] Runtime wiring and reference counting" was removed.
    // GaussianSplatManager inherits from Object (not RefCounted) and does not expose
    // set_memory_stream/get_memory_stream, set_gpu_sorter/get_gpu_sorter,
    // set_gaussian_data/get_gaussian_data, or clear() methods.
    // The manager uses register_gaussian_buffer/unregister_gaussian_buffer and
    // acquire_dynamic_asset/release_dynamic_asset for resource management instead.

    TEST_CASE("[Integration] Full pipeline execution with 100K splats") {
        if (!is_rendering_device_available()) {
            MESSAGE("Skipping test - Rendering device not available");
            return;
        }

        RenderingServer *rs = RenderingServer::get_singleton();
        RenderingDevice *rd = rs->create_local_rendering_device();
        CHECK(rd != nullptr);
        if (rd == nullptr) {
            return;
        }

        // Create and initialize all components
        LocalVector<Gaussian> gaussian_batch;
        Ref<::GaussianData> data = generate_test_gaussians(100000, &gaussian_batch);
        CHECK(data->get_count() == 100000);

        Ref<GaussianMemoryStream> stream;
        stream.instantiate();
        Error err = stream->initialize(rd, 100000);
        CHECK(err == OK);

        Ref<BitonicSort> sorter;
        sorter.instantiate();
        err = sorter->initialize(rd, 131072); // Next power of 2
        CHECK(err == OK);

        Ref<TileRenderer> tile_renderer;
        tile_renderer.instantiate();
        err = tile_renderer->initialize(rd);
        CHECK(err == OK);

        print_line("[Integration] Using synchronous compute queue");

        // Begin frame and upload data
        stream->begin_frame(0);
        err = stream->stream_gaussians_async(gaussian_batch, 0, gaussian_batch.size());
        CHECK(err == OK);
        stream->wait_for_all_uploads();
        RID gaussian_buffer = stream->get_current_gpu_buffer();
        CHECK(gaussian_buffer.is_valid());

        // Allocate sort buffers using RenderingDevice directly since the memory stream
        // exposes Gaussian data only.
        uint32_t splat_count = data->get_count();
        Vector<float> depth_keys;
        depth_keys.resize(splat_count);
        Vector<uint32_t> splat_indices;
        splat_indices.resize(splat_count);

        float *depth_ptr = depth_keys.ptrw();
        uint32_t *index_ptr = splat_indices.ptrw();
        for (uint32_t i = 0; i < splat_count; i++) {
            const Gaussian &g = gaussian_batch[i];
            depth_ptr[i] = g.position.length();
            index_ptr[i] = i;
        }

        Vector<uint8_t> keys_bytes;
        keys_bytes.resize(splat_count * sizeof(float));
        memcpy(keys_bytes.ptrw(), depth_ptr, keys_bytes.size());
        RID keys_buffer = rd->storage_buffer_create(keys_bytes.size(), keys_bytes, RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
        rd->set_resource_name(keys_buffer, "GS_Test_FullPipeline_SortKeys");

        Vector<uint8_t> values_bytes;
        values_bytes.resize(splat_count * sizeof(uint32_t));
        memcpy(values_bytes.ptrw(), index_ptr, values_bytes.size());
        RID values_buffer = rd->storage_buffer_create(values_bytes.size(), values_bytes, RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
        rd->set_resource_name(values_buffer, "GS_Test_FullPipeline_SortValues");
        CHECK(keys_buffer.is_valid());
        CHECK(values_buffer.is_valid());

        // Execute sorting
        auto sort_start = std::chrono::high_resolution_clock::now();
        err = sorter->sort(keys_buffer, values_buffer, splat_count);
        CHECK(err == OK);
        auto sort_end = std::chrono::high_resolution_clock::now();
        float sort_time_ms = std::chrono::duration<float, std::milli>(sort_end - sort_start).count();

        print_line(vformat("[Integration] Sorted 100K splats in %.2f ms", sort_time_ms));
        CHECK(sort_time_ms < 100.0f); // Should be well under 100ms

        // Get sorting metrics
        SortingMetrics sort_metrics = sorter->get_metrics();
        CHECK(sort_metrics.total_sorts >= 1);
        CHECK(sort_metrics.total_elements_sorted >= splat_count);

        // Execute tile rendering
        TileRenderer::RenderParams render_params;
        render_params.gaussian_buffer = gaussian_buffer;
        render_params.sorted_indices = values_buffer;
        render_params.splat_count = splat_count;
        render_params.total_gaussians = splat_count;
        render_params.viewport_size = Vector2i(1920, 1080);
        render_params.tile_size = 16;

        auto render_start = std::chrono::high_resolution_clock::now();
        RID output_texture = tile_renderer->render(rd, render_params);
        auto render_end = std::chrono::high_resolution_clock::now();
        float render_time_ms = std::chrono::duration<float, std::milli>(render_end - render_start).count();

        CHECK(output_texture.is_valid());
        print_line(vformat("[Integration] Rendered 100K splats in %.2f ms", render_time_ms));

        // End frame and get statistics
        stream->end_frame();
        StreamingStats stream_stats = stream->get_stats();
        CHECK(stream_stats.total_bytes_uploaded > 0);
        CHECK(stream_stats.stalls == 0); // Should have no stalls on first frame

        // Cleanup
        tile_renderer->cleanup();
        sorter->shutdown();
        stream->shutdown();
        if (keys_buffer.is_valid()) {
            rd->free(keys_buffer);
        }
        if (values_buffer.is_valid()) {
            rd->free(values_buffer);
        }

        memdelete(rd);

        print_line("[Integration] Full pipeline execution completed successfully");
    }

    TEST_CASE("[Integration] Shared submission device prevents main queue submission errors") {
        if (!is_rendering_device_available()) {
            MESSAGE("Skipping test - Rendering device not available");
            return;
        }

        RenderingServer *rs = RenderingServer::get_singleton();
        CHECK(rs != nullptr);
        if (rs == nullptr) {
            return;
        }

        RenderingDevice *rd = rs->create_local_rendering_device();
        CHECK(rd != nullptr);
        if (rd == nullptr) {
            return;
        }

        ScopedErrorCapture error_capture;

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

        const uint32_t splat_count = 4096;
        LocalVector<Gaussian> gaussian_batch;
        Ref<::GaussianData> data = generate_test_gaussians(splat_count, &gaussian_batch);
        CHECK(data.is_valid());

        Ref<GaussianMemoryStream> stream;
        stream.instantiate();
        CHECK(stream.is_valid());
        CHECK(stream->initialize(rd, splat_count, 8) == OK);

        Ref<BitonicSort> sorter;
        sorter.instantiate();
        CHECK(sorter.is_valid());
        CHECK(sorter->initialize(rd, splat_count) == OK);

        Ref<TileRenderer> tile_renderer;
        tile_renderer.instantiate();
        CHECK(tile_renderer.is_valid());
        CHECK(tile_renderer->initialize(rd, Vector2i(640, 360)) == OK);

        stream->begin_frame(0);
        CHECK(stream->stream_gaussians_async(gaussian_batch, 0, splat_count) == OK);
        stream->wait_for_all_uploads();
        RID gaussian_buffer = stream->get_current_gpu_buffer();
        CHECK(gaussian_buffer.is_valid());

        Vector<float> depth_keys;
        depth_keys.resize(splat_count);
        Vector<uint32_t> splat_indices;
        splat_indices.resize(splat_count);

        float *depth_ptr = depth_keys.ptrw();
        uint32_t *index_ptr = splat_indices.ptrw();
        for (uint32_t i = 0; i < splat_count; i++) {
            const Gaussian &g = gaussian_batch[i];
            depth_ptr[i] = g.position.length();
            index_ptr[i] = i;
        }

        Vector<uint8_t> keys_bytes;
        keys_bytes.resize(splat_count * sizeof(float));
        memcpy(keys_bytes.ptrw(), depth_ptr, keys_bytes.size());
        RID keys_buffer = rd->storage_buffer_create(keys_bytes.size(), keys_bytes, RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
        rd->set_resource_name(keys_buffer, "GS_Test_SharedSubmission_SortKeys");

        Vector<uint8_t> values_bytes;
        values_bytes.resize(splat_count * sizeof(uint32_t));
        memcpy(values_bytes.ptrw(), index_ptr, values_bytes.size());
        RID values_buffer = rd->storage_buffer_create(values_bytes.size(), values_bytes, RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
        rd->set_resource_name(values_buffer, "GS_Test_SharedSubmission_SortValues");
        CHECK(keys_buffer.is_valid());
        CHECK(values_buffer.is_valid());

        CHECK(sorter->sort(keys_buffer, values_buffer, splat_count) == OK);

        TileRenderer::RenderParams render_params;
        render_params.gaussian_buffer = gaussian_buffer;
        render_params.sorted_indices = values_buffer;
        render_params.splat_count = splat_count;
        render_params.total_gaussians = splat_count;
        render_params.viewport_size = Vector2i(640, 360);
        render_params.tile_size = 16;

        Projection projection;
        projection.set_perspective(60.0f, 640.0f / 360.0f, 0.1f, 50.0f);
        render_params.projection = projection;
        render_params.render_projection = render_params.projection;
        render_params.world_to_camera_transform = Transform3D();

        RID output_texture = tile_renderer->render(rd, render_params);
        CHECK(output_texture.is_valid());

        stream->end_frame();
        stream->shutdown();
        sorter->shutdown();
        tile_renderer->cleanup();

        if (keys_buffer.is_valid()) {
            rd->free(keys_buffer);
        }
        if (values_buffer.is_valid()) {
            rd->free(values_buffer);
        }

        CHECK_FALSE(error_capture.has_message_containing("Only local devices can submit and sync"));
        CHECK_FALSE(error_capture.has_errors());

        if (manager_owner) {
            memdelete(manager_owner);
        }
        memdelete(rd);
    }

    TEST_CASE("[Integration] Memory stream buffer cycling") {
        if (!is_rendering_device_available()) {
            MESSAGE("Skipping test - Rendering device not available");
            return;
        }

        RenderingServer *rs = RenderingServer::get_singleton();
        RenderingDevice *rd = rs->create_local_rendering_device();
        CHECK(rd != nullptr);
        if (rd == nullptr) {
            return;
        }

        Ref<GaussianMemoryStream> stream;
        stream.instantiate();
        stream->initialize(rd, 10000);

        // Test triple buffering behavior
        for (int frame = 0; frame < 10; frame++) {
            stream->begin_frame(frame);

            LocalVector<Gaussian> frame_gaussians;
            Ref<::GaussianData> data = generate_test_gaussians(5000 + frame * 100, &frame_gaussians);
            Error upload_err = stream->stream_gaussians_async(frame_gaussians, 0, frame_gaussians.size());
            CHECK(upload_err == OK);
            stream->wait_for_all_uploads();
            RID buffer = stream->get_current_gpu_buffer();
            CHECK(buffer.is_valid());

            // Simulate rendering
            std::this_thread::sleep_for(std::chrono::milliseconds(5));

            stream->end_frame();

            // Check for stalls after frame 3 (when triple buffering should be active)
            if (frame >= 3) {
                StreamingStats stats = stream->get_stats();
                float stall_percentage = (float)stats.stalls / (float)stats.total_frames * 100.0f;
                CHECK(stall_percentage < 10.0f); // Less than 10% stalls
            }
        }

        StreamingStats final_stats = stream->get_stats();
        print_line(vformat("[Integration] Buffer cycling: %d switches, %d stalls (%.1f%%)",
                          final_stats.buffer_switches,
                          final_stats.stalls,
                          (float)final_stats.stalls / (float)final_stats.total_frames * 100.0f));

        stream->shutdown();
        memdelete(rd);
    }

    TEST_CASE("[Integration] API consistency validation") {
        // Test that all major APIs follow consistent patterns

        // Test RefCounted pattern
        Ref<::GaussianData> data;
        data.instantiate();
        CHECK(data.is_valid());
        CHECK(data->get_reference_count() == 1);

        Ref<::GaussianData> data2 = data;
        CHECK(data->get_reference_count() == 2);
        data2.unref();
        CHECK(data->get_reference_count() == 1);

        // Test initialize/cleanup pattern
        if (is_rendering_device_available()) {
            RenderingServer *rs = RenderingServer::get_singleton();
            RenderingDevice *rd = rs->create_local_rendering_device();

            // All GPU components should follow initialize/shutdown pattern
            Ref<GaussianMemoryStream> stream;
            stream.instantiate();
            CHECK(stream->initialize(rd, 1000) == OK);
            stream->shutdown(); // Should not crash

            Ref<BitonicSort> sorter;
            sorter.instantiate();
            CHECK(sorter->initialize(rd, 1024) == OK);
            sorter->shutdown(); // Should not crash

            memdelete(rd);
        }

        print_line("[Integration] API consistency validated");
    }

    TEST_CASE("[Integration] Error handling and recovery") {
        // Test graceful handling of error conditions

        // Test null data handling
        GaussianSplatManager *manager = memnew(GaussianSplatManager);
        CHECK(manager != nullptr);
        if (manager == nullptr) {
            return;
        }

        Ref<GaussianSplatRenderer> renderer;
        renderer.instantiate(manager->get_primary_rendering_device());
        CHECK(renderer->set_gaussian_data(Ref<::GaussianData>()) == OK); // Null ref
        CHECK(renderer->get_gaussian_data().is_null());

        // Test invalid size handling
        Ref<::GaussianData> data;
        data.instantiate();
        data->resize(0); // Empty data
        CHECK(renderer->set_gaussian_data(data) == OK);
        // Should not crash when rendering with empty data

        if (is_rendering_device_available()) {
            RenderingServer *rs = RenderingServer::get_singleton();
            RenderingDevice *rd = rs->create_local_rendering_device();

            // Test initialization with invalid parameters
            Ref<GaussianMemoryStream> stream;
            stream.instantiate();
            CHECK(stream->initialize(rd, 0) == ERR_INVALID_PARAMETER);
            CHECK(stream->initialize(rd, UINT32_MAX) == ERR_INVALID_PARAMETER);

            // Test operations on uninitialized components
            Ref<BitonicSort> sorter;
            sorter.instantiate();
            RID invalid_buffer;
            CHECK(sorter->sort(invalid_buffer, invalid_buffer, 100) == ERR_UNCONFIGURED);

            memdelete(rd);
        }

        renderer.unref();
        memdelete(manager);

        print_line("[Integration] Error handling validated");
    }
}
