#ifndef TILE_RENDER_ADAPTIVE_CONTROLLER_H
#define TILE_RENDER_ADAPTIVE_CONTROLLER_H

// Split TileRenderer adaptive tile size controller. Included inside TileRenderer's private section.
    struct TileAdaptiveController {
        explicit TileAdaptiveController(TileRenderer &p_owner) : owner(p_owner) {}

        struct AdaptiveState {
            bool metrics_available = false;
            float smoothed_occupancy = 0.0f;
            float smoothed_dense_ratio = 0.0f;
            float smoothed_overflow_ratio = 0.0f;
            float smoothed_average_splats = 0.0f;
            uint32_t frames_since_adjustment = 0;
            int last_computed_tile_size = DEFAULT_TILE_SIZE;
        };

        void reset_state(int p_tile_size);
        void set_settings(const AdaptiveSettings &p_settings, int &r_tile_size);
        const AdaptiveSettings &get_settings() const { return settings; }
        bool is_enabled() const { return settings.enable_adaptive_tile_size; }
        void set_metrics_available(bool p_available) { state.metrics_available = p_available; }
        int compute_tile_size(int p_requested_tile_size, const Vector2i &p_size) const;
        void on_tile_size_applied(int p_tile_size, bool p_changed);
        void on_allocation_failure(int p_tile_size);
        void update_state(const RenderStats &p_stats);

        TileRenderer &owner;
        AdaptiveSettings settings;
        AdaptiveState state;
    };

#endif
