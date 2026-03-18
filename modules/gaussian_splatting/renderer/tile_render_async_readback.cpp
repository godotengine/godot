#include "tile_renderer.h"

#include "pipeline_io_contracts.h"

#include <cstring>

void TileRenderer::TileAsyncReadback::on_overflow_flag_readback(const Vector<uint8_t> &p_data, uint64_t p_request_frame_serial) {
    if (!overflow_state.pending_readback || overflow_state.requested_frame_serial != p_request_frame_serial) {
        return;
    }

    // Callback for async readback from indirect buffer (element_count onwards).
    // Layout matches IndirectDispatchLayout (element_count, overflow_flag, unclamped_total).
    overflow_state.pending_readback = false;
    overflow_state.requested_frame_serial = 0;
    if (p_data.size() < int(GaussianSplatting::kIndirectDispatchElementCountReadbackSize)) {
        WARN_PRINT_ONCE("[TileRenderer] Async overlap readback returned insufficient data; retaining previous overlap estimate");
        return;
    }
    const uint32_t *data = reinterpret_cast<const uint32_t *>(p_data.ptr());
    overflow_state.overflow_detected = (data[1] != 0);
    overflow_state.last_unclamped_total = data[2];
    overflow_state.first_frame_complete = true;
}

void TileRenderer::TileAsyncReadback::on_tile_counts_readback(const Vector<uint8_t> &p_data, uint64_t p_request_frame_serial) {
    if (!tile_counts_state.pending_readback || tile_counts_state.requested_frame_serial != p_request_frame_serial) {
        return;
    }

    tile_counts_state.pending_readback = false;
    tile_counts_state.requested_frame_serial = 0;
    const uint32_t expected_tiles = tile_counts_state.cached_total_tiles;
    const size_t expected_bytes = size_t(expected_tiles) * sizeof(uint32_t);
    if (p_data.size() < int(expected_bytes) || expected_tiles == 0) {
        return;
    }
    // Copy the tile counts data into our cached vector
    tile_counts_state.cached_counts.resize(expected_tiles);
    memcpy(tile_counts_state.cached_counts.ptrw(), p_data.ptr(), expected_bytes);
    tile_counts_state.first_frame_complete = true;
}
