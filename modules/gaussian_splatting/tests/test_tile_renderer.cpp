#include "test_macros.h"

#include "../renderer/tile_prefix_scan_utils.h"
#include "../renderer/tile_renderer.h"
#include "servers/rendering_server.h"

TEST_CASE("[TileRenderer] Shader compilation on local device") {
    RenderingServer *rs = RenderingServer::get_singleton();
    CHECK_MESSAGE(rs != nullptr, "RenderingServer singleton must exist for TileRenderer tests");
    if (rs == nullptr) {
        return;
    }

    RenderingDevice *local_device = rs->create_local_rendering_device();
    CHECK_MESSAGE(local_device != nullptr, "Failed to create local RenderingDevice for TileRenderer test");
    if (local_device == nullptr) {
        return;
    }

    Ref<TileRenderer> renderer;
    renderer.instantiate();
    Error err = renderer->initialize(local_device, Vector2i(1920, 1080), TileRenderer::DEFAULT_TILE_SIZE);
    CHECK_MESSAGE(err == OK, "TileRenderer initialization should succeed on a local RenderingDevice");
    CHECK(renderer->is_initialized());
    CHECK(renderer->get_tile_binning_pipeline().is_valid());
    CHECK(renderer->get_tile_raster_pipeline().is_valid());

    renderer->cleanup();
    memdelete(local_device);
}

TEST_CASE("[TileRenderer] Output format coercion keeps deterministic defaults") {
    RenderingServer *rs = RenderingServer::get_singleton();
    CHECK_MESSAGE(rs != nullptr, "RenderingServer singleton must exist for TileRenderer tests");
    if (rs == nullptr) {
        return;
    }

    RenderingDevice *local_device = rs->create_local_rendering_device();
    CHECK_MESSAGE(local_device != nullptr, "Failed to create local RenderingDevice for TileRenderer test");
    if (local_device == nullptr) {
        return;
    }

    Ref<TileRenderer> renderer;
    renderer.instantiate();
    Error err = renderer->initialize(local_device, Vector2i(512, 320), TileRenderer::DEFAULT_TILE_SIZE,
            RD::DATA_FORMAT_R8G8B8A8_SRGB);
    CHECK_MESSAGE(err == OK, "TileRenderer initialization should succeed with explicit SRGB output format");
    if (err != OK) {
        memdelete(local_device);
        return;
    }

    CHECK(renderer->get_output_format() == RD::DATA_FORMAT_R8G8B8A8_SRGB);

    renderer->set_output_format(RD::DATA_FORMAT_MAX);
    CHECK(renderer->get_output_format() == RD::DATA_FORMAT_R8G8B8A8_UNORM);

    err = renderer->resize(Vector2i(256, 160), RD::DATA_FORMAT_MAX);
    CHECK_MESSAGE(err == OK, "TileRenderer resize should accept DATA_FORMAT_MAX and preserve fallback format");
    CHECK(renderer->get_output_format() == RD::DATA_FORMAT_R8G8B8A8_UNORM);

    renderer->cleanup();
    memdelete(local_device);
}

TEST_CASE("[TileRenderer] Prefix emergency fallback only triggers at device-dispatch limits") {
    const uint32_t total_workgroups = 8193u;
    const GaussianSplatting::TilePrefixDispatchCounts dispatch_counts =
            GaussianSplatting::tile_prefix_compute_dispatch_counts(total_workgroups);

    CHECK(dispatch_counts.pass2_dispatch_x > 1u);
    CHECK(!GaussianSplatting::tile_prefix_any_pass_requires_cpu_fallback(total_workgroups, total_workgroups));
    CHECK(GaussianSplatting::tile_prefix_any_pass_requires_cpu_fallback(total_workgroups, total_workgroups - 1u));
    CHECK(GaussianSplatting::tile_prefix_any_pass_requires_cpu_fallback(total_workgroups, dispatch_counts.pass2_dispatch_x - 1u));
}
