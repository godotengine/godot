#include "test_macros.h"

#include "../renderer/tile_prefix_scan_utils.h"
#include "core/templates/vector.h"

#include <cstddef>
#include <limits>

TEST_CASE("[TileRenderer] Prefix scan ABI contracts stay aligned") {
    CHECK(sizeof(GaussianSplatting::TilePrefixParamsLayout) == 16u);
    CHECK(offsetof(GaussianSplatting::TilePrefixParamsLayout, total_tiles) == 0u);
    CHECK(offsetof(GaussianSplatting::TilePrefixParamsLayout, workgroup_stride) == 4u);
    CHECK(offsetof(GaussianSplatting::TilePrefixParamsLayout, total_workgroups) == 8u);
    CHECK(offsetof(GaussianSplatting::TilePrefixParamsLayout, global_sort_capacity) == 12u);

    CHECK(sizeof(GaussianSplatting::TilePrefixPass2ControlLayout) == 16u);
    CHECK(offsetof(GaussianSplatting::TilePrefixPass2ControlLayout, operation) == 0u);
    CHECK(offsetof(GaussianSplatting::TilePrefixPass2ControlLayout, source_buffer) == 4u);
    CHECK(offsetof(GaussianSplatting::TilePrefixPass2ControlLayout, stride) == 8u);
    CHECK(offsetof(GaussianSplatting::TilePrefixPass2ControlLayout, reserved) == 12u);
}

TEST_CASE("[TileRenderer] Prefix dispatch guards cover all GPU passes") {
    const uint32_t total_workgroups = 4097u;
    const GaussianSplatting::TilePrefixDispatchCounts dispatch_counts =
            GaussianSplatting::tile_prefix_compute_dispatch_counts(total_workgroups);

    CHECK(dispatch_counts.pass1_dispatch_x == total_workgroups);
    CHECK(dispatch_counts.pass2_dispatch_x > 1u);
    CHECK(dispatch_counts.pass3_dispatch_x == total_workgroups);
    CHECK(GaussianSplatting::tile_prefix_compute_pass2_levels(total_workgroups) >= 12u);
    CHECK(!GaussianSplatting::tile_prefix_pass1_requires_cpu_fallback(total_workgroups, total_workgroups));
    CHECK(!GaussianSplatting::tile_prefix_pass2_requires_cpu_fallback(total_workgroups, dispatch_counts.pass2_dispatch_x));
    CHECK(!GaussianSplatting::tile_prefix_pass3_requires_cpu_fallback(total_workgroups, total_workgroups));
    CHECK(!GaussianSplatting::tile_prefix_any_pass_requires_cpu_fallback(total_workgroups, total_workgroups));
    CHECK(GaussianSplatting::tile_prefix_pass1_requires_cpu_fallback(total_workgroups, total_workgroups - 1u));
    CHECK(GaussianSplatting::tile_prefix_pass3_requires_cpu_fallback(total_workgroups, total_workgroups - 1u));
    CHECK(!GaussianSplatting::tile_prefix_pass2_requires_cpu_fallback(total_workgroups, total_workgroups - 1u));
    CHECK(GaussianSplatting::tile_prefix_any_pass_requires_cpu_fallback(total_workgroups, total_workgroups - 1u));
    CHECK(GaussianSplatting::tile_prefix_pass2_requires_cpu_fallback(total_workgroups, dispatch_counts.pass2_dispatch_x - 1u));
    CHECK(GaussianSplatting::tile_prefix_any_pass_requires_cpu_fallback(total_workgroups, dispatch_counts.pass2_dispatch_x - 1u));
}

TEST_CASE("[TileRenderer] CPU prefix fallback keeps large-scene correctness") {
    const uint32_t total_workgroups = 4097u;
    const uint32_t total_tiles = total_workgroups * GaussianSplatting::kTilePrefixPassLocalSize;

    Vector<uint32_t> tile_counts;
    tile_counts.resize(total_tiles);
    for (uint32_t i = 0; i < total_tiles; i++) {
        tile_counts.write[i] = (i % 11u == 0u) ? 3u : 1u;
    }

    Vector<uint32_t> tile_ranges_words;
    tile_ranges_words.resize(total_tiles * 2u);

    GaussianSplatting::TilePrefixCpuScanResult result;
    const bool ok = GaussianSplatting::compute_tile_prefix_cpu(tile_counts.ptr(), total_tiles,
            tile_ranges_words.ptrw(), 0u, result);
    REQUIRE(ok);

    uint64_t running_total = 0;
    const uint64_t max_u32 = uint64_t(std::numeric_limits<uint32_t>::max());
    for (uint32_t i = 0; i < total_tiles; i++) {
        const uint32_t expected_prefix = running_total > max_u32 ? std::numeric_limits<uint32_t>::max() : uint32_t(running_total);
        CHECK(tile_ranges_words[i * 2u + 0u] == expected_prefix);
        CHECK(tile_ranges_words[i * 2u + 1u] == tile_counts[i]);
        running_total += tile_counts[i];
    }

    const uint32_t expected_total = running_total > max_u32 ? std::numeric_limits<uint32_t>::max() : uint32_t(running_total);
    uint32_t expected_dispatch_x = (expected_total + GaussianSplatting::kTilePrefixPassLocalSize - 1u) /
            GaussianSplatting::kTilePrefixPassLocalSize;
    if (expected_dispatch_x == 0u) {
        expected_dispatch_x = 1u;
    }

    CHECK(result.raw_total == expected_total);
    CHECK(result.raw_total_saturated == false);
    CHECK(result.indirect_dispatch.unclamped_total == expected_total);
    CHECK(result.indirect_dispatch.element_count == expected_total);
    CHECK(result.indirect_dispatch.dispatch_x == expected_dispatch_x);
    CHECK(result.indirect_dispatch.dispatch_y == 1u);
    CHECK(result.indirect_dispatch.dispatch_z == 1u);
    CHECK(result.indirect_dispatch.overflow_flag == 0u);
}

TEST_CASE("[TileRenderer] CPU prefix fallback clamps element count to overlap capacity") {
    const uint32_t tile_counts[] = { 4u, 2u, 3u };
    Vector<uint32_t> tile_ranges_words;
    tile_ranges_words.resize(6);

    GaussianSplatting::TilePrefixCpuScanResult result;
    const bool ok = GaussianSplatting::compute_tile_prefix_cpu(tile_counts, 3u, tile_ranges_words.ptrw(), 5u, result);
    REQUIRE(ok);

    CHECK(tile_ranges_words[0] == 0u);
    CHECK(tile_ranges_words[1] == 4u);
    CHECK(tile_ranges_words[2] == 4u);
    CHECK(tile_ranges_words[3] == 2u);
    CHECK(tile_ranges_words[4] == 6u);
    CHECK(tile_ranges_words[5] == 3u);

    CHECK(result.raw_total == 9u);
    CHECK(result.indirect_dispatch.unclamped_total == 9u);
    CHECK(result.indirect_dispatch.element_count == 5u);
    CHECK(result.indirect_dispatch.dispatch_x == 1u);
    CHECK(result.indirect_dispatch.overflow_flag == 1u);
}

TEST_CASE("[TileRenderer] CPU prefix fallback computes saturated dispatch groups without overflow") {
    const uint32_t tile_counts[] = { std::numeric_limits<uint32_t>::max(), 1u };
    Vector<uint32_t> tile_ranges_words;
    tile_ranges_words.resize(4);

    GaussianSplatting::TilePrefixCpuScanResult result;
    const bool ok = GaussianSplatting::compute_tile_prefix_cpu(tile_counts, 2u, tile_ranges_words.ptrw(), 0u, result);
    REQUIRE(ok);

    CHECK(result.raw_total_saturated == true);
    CHECK(result.raw_total == std::numeric_limits<uint32_t>::max());
    CHECK(result.indirect_dispatch.unclamped_total == std::numeric_limits<uint32_t>::max());
    CHECK(result.indirect_dispatch.element_count == std::numeric_limits<uint32_t>::max());
    CHECK(result.indirect_dispatch.dispatch_x == 16777216u); // ceil(UINT32_MAX / 256)
    CHECK(result.indirect_dispatch.dispatch_y == 1u);
    CHECK(result.indirect_dispatch.dispatch_z == 1u);
    CHECK(result.indirect_dispatch.overflow_flag == 0u);
}
