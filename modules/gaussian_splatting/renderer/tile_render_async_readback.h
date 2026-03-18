#ifndef TILE_RENDER_ASYNC_READBACK_H
#define TILE_RENDER_ASYNC_READBACK_H

// Split TileRenderer async readback state machine. Included inside TileRenderer's private section.
    struct TileAsyncReadback {
        explicit TileAsyncReadback(TileRenderer &p_owner) : owner(p_owner) {}

        struct AsyncOverflowState {
            bool pending_readback = false;
            bool overflow_detected = false;
            uint32_t last_unclamped_total = 0;    // Raw total before clamping (offset 20)
            bool first_frame_complete = false;
            uint64_t requested_frame_serial = 0;
        };

        struct AsyncTileCountsState {
            bool pending_readback = false;
            bool first_frame_complete = false;
            uint64_t requested_frame_serial = 0;
            Vector<uint32_t> cached_counts;       // Cached tile counts from previous frame
            uint32_t cached_total_tiles = 0;      // Number of tiles when counts were captured
        };

        void on_overflow_flag_readback(const Vector<uint8_t> &p_data, uint64_t p_request_frame_serial);
        void on_tile_counts_readback(const Vector<uint8_t> &p_data, uint64_t p_request_frame_serial);

        TileRenderer &owner;
        AsyncOverflowState overflow_state;
        AsyncTileCountsState tile_counts_state;
    };

#endif
