#include "test_macros.h"

#define private public
#include "../renderer/tile_renderer.h"
#undef private

#include "../renderer/tile_prefix_scan_utils.h"
#include "servers/rendering_server.h"

#include <cstring>

namespace {

static bool _read_tile_range_pair(RenderingDevice *p_device, RID p_ranges_buffer, uint32_t p_tile_index,
        uint32_t &r_prefix, uint32_t &r_count) {
    if (!p_device || !p_ranges_buffer.is_valid()) {
        return false;
    }

    const uint64_t bytes_per_tile = sizeof(uint32_t) * 2u;
    const uint64_t offset = uint64_t(p_tile_index) * bytes_per_tile;
    Vector<uint8_t> data = p_device->buffer_get_data(p_ranges_buffer, offset, bytes_per_tile);
    if (data.size() != int(bytes_per_tile)) {
        return false;
    }

    uint32_t words[2] = { 0u, 0u };
    std::memcpy(words, data.ptr(), sizeof(words));
    r_prefix = words[0];
    r_count = words[1];
    return true;
}

} // namespace

TEST_CASE("[TileRenderer] Prefix pass2 dispatch limit boundary is explicit") {
    const uint32_t total_workgroups = 1025u;
    const uint32_t pass2_dispatch_x = GaussianSplatting::tile_prefix_compute_dispatch_groups(total_workgroups);

    CHECK(pass2_dispatch_x == 5u);

    // Exactly-at-limit dispatch is valid.
    CHECK_FALSE(GaussianSplatting::tile_prefix_pass2_requires_cpu_fallback(total_workgroups, pass2_dispatch_x));
    // One less than required must trigger deterministic CPU fallback.
    CHECK(GaussianSplatting::tile_prefix_pass2_requires_cpu_fallback(total_workgroups, pass2_dispatch_x - 1u));
}

TEST_CASE("[TileRenderer] Prefix CPU fallback writes deterministic renderer range buffers") {
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

    TileRenderer::AdaptiveSettings adaptive_settings;
    adaptive_settings.enable_adaptive_tile_size = false;
    adaptive_settings.clamp_to_power_of_two = false;
    adaptive_settings.min_tile_size = 1;
    adaptive_settings.max_tile_size = 1;
    renderer->set_adaptive_settings(adaptive_settings);

    Error err = renderer->initialize(local_device, Vector2i(64, 16), 1);
    CHECK_MESSAGE(err == OK, "TileRenderer initialization should succeed for CPU prefix fallback test");
    if (err != OK) {
        memdelete(local_device);
        return;
    }

    const uint32_t total_tiles = renderer->grid_state.total_tiles;
    REQUIRE(total_tiles >= TileRenderer::BINNING_GROUP_SIZE + 1u);

    renderer->_ensure_global_sort_resources(64u);
    REQUIRE(renderer->global_sort_resources.tile_buffer_tiles == total_tiles);
    REQUIRE(renderer->global_sort_resources.get_tile_counts_buffer().is_valid());
    REQUIRE(renderer->global_sort_resources.tile_ranges_buffer.is_valid());
    REQUIRE(renderer->global_sort_resources.prefix_total_buffer.is_valid());

    Vector<uint32_t> tile_counts;
    tile_counts.resize(total_tiles);
    for (uint32_t i = 0; i < total_tiles; i++) {
        tile_counts.write[i] = 0u;
    }

    const uint32_t idx0 = 0u;
    const uint32_t idx255 = TileRenderer::BINNING_GROUP_SIZE - 1u;
    const uint32_t idx256 = TileRenderer::BINNING_GROUP_SIZE;
    const uint32_t idx_last = total_tiles - 1u;
    tile_counts.write[idx0] = 2u;
    tile_counts.write[idx255] = 1u;
    tile_counts.write[idx256] = 4u;
    tile_counts.write[idx_last] = 3u;

    const uint32_t expected_raw_total = 10u;
    const uint64_t counts_bytes = uint64_t(total_tiles) * sizeof(uint32_t);
    local_device->buffer_update(renderer->global_sort_resources.get_tile_counts_buffer(), 0, counts_bytes, tile_counts.ptr());

    renderer->timing_state.prefix_timestamp.reset();
    renderer->async_readback.overflow_state.pending_readback = true;
    renderer->async_readback.overflow_state.requested_frame_serial = 99u;
    renderer->async_readback.overflow_state.first_frame_complete = false;
    renderer->async_readback.overflow_state.overflow_detected = false;
    renderer->async_readback.overflow_state.last_unclamped_total = 0u;

    uint32_t record_count = 0u;
    uint32_t raw_record_count = 0u;
    const TileRenderer::TilePrefixScanStage::PrefixParams prefix_params = renderer->prefix_scan_stage.build_prefix_params();
    const bool ok = renderer->prefix_scan_stage.run_cpu_prefix_fallback(local_device, prefix_params, record_count, raw_record_count);
    REQUIRE(ok);

    const uint32_t effective_capacity = renderer->_get_effective_overlap_capacity();
    const uint32_t expected_record_count = effective_capacity > 0u ? MIN(expected_raw_total, effective_capacity) : expected_raw_total;
    const bool expected_overflow = (effective_capacity > 0u) && (expected_raw_total > effective_capacity);

    CHECK(raw_record_count == expected_raw_total);
    CHECK(record_count == expected_record_count);
    CHECK_FALSE(renderer->timing_state.prefix_timestamp.is_valid());

    CHECK_FALSE(renderer->async_readback.overflow_state.pending_readback);
    CHECK(renderer->async_readback.overflow_state.requested_frame_serial == 0u);
    CHECK(renderer->async_readback.overflow_state.first_frame_complete);
    CHECK(renderer->async_readback.overflow_state.overflow_detected == expected_overflow);
    CHECK(renderer->async_readback.overflow_state.last_unclamped_total == expected_raw_total);

    Vector<uint8_t> total_bytes = local_device->buffer_get_data(renderer->global_sort_resources.prefix_total_buffer, 0, sizeof(uint32_t));
    REQUIRE(total_bytes.size() == int(sizeof(uint32_t)));
    uint32_t prefix_total = 0u;
    std::memcpy(&prefix_total, total_bytes.ptr(), sizeof(prefix_total));
    CHECK(prefix_total == expected_raw_total);

    uint32_t prefix = 0u;
    uint32_t count = 0u;
    REQUIRE(_read_tile_range_pair(local_device, renderer->global_sort_resources.tile_ranges_buffer, idx0, prefix, count));
    CHECK(prefix == 0u);
    CHECK(count == 2u);
    REQUIRE(_read_tile_range_pair(local_device, renderer->global_sort_resources.tile_ranges_buffer, idx255, prefix, count));
    CHECK(prefix == 2u);
    CHECK(count == 1u);
    REQUIRE(_read_tile_range_pair(local_device, renderer->global_sort_resources.tile_ranges_buffer, idx256, prefix, count));
    CHECK(prefix == 3u);
    CHECK(count == 4u);
    REQUIRE(_read_tile_range_pair(local_device, renderer->global_sort_resources.tile_ranges_buffer, idx_last, prefix, count));
    CHECK(prefix == 7u);
    CHECK(count == 3u);

    renderer->cleanup();
    memdelete(local_device);
}
