#include "tile_renderer.h"

#include "core/math/math_funcs.h"
#include "core/string/print_string.h"
#include "../logger/gs_logger.h"

void TileRenderer::TileAdaptiveController::reset_state(int p_tile_size) {
    state = AdaptiveState();
    state.last_computed_tile_size = p_tile_size;
    state.frames_since_adjustment = settings.frames_before_adjustment;
    state.metrics_available = false;
}

void TileRenderer::TileAdaptiveController::set_settings(const AdaptiveSettings &p_settings, int &r_tile_size) {
    settings = p_settings;

    settings.min_tile_size = MAX(1, settings.min_tile_size);
    settings.max_tile_size = MAX(settings.min_tile_size, settings.max_tile_size);
    settings.tile_size_step = MAX(1, settings.tile_size_step);
    settings.frames_before_adjustment = MAX(uint32_t(1), settings.frames_before_adjustment);
    settings.target_occupancy_ratio = CLAMP(settings.target_occupancy_ratio, 0.0f, 1.0f);
    settings.occupancy_hysteresis = MAX(0.0f, settings.occupancy_hysteresis);
    settings.dense_ratio_threshold = CLAMP(settings.dense_ratio_threshold, 0.0f, 1.0f);
    settings.overflow_ratio_threshold = MAX(0.0f, settings.overflow_ratio_threshold);
    settings.max_average_splat_ratio = CLAMP(settings.max_average_splat_ratio, 0.0f, 1.0f);
    settings.smoothing_factor = CLAMP(settings.smoothing_factor, 0.0f, 1.0f);

    r_tile_size = CLAMP(r_tile_size, settings.min_tile_size, settings.max_tile_size);
    state.last_computed_tile_size = r_tile_size;
    state.frames_since_adjustment = MIN(state.frames_since_adjustment, settings.frames_before_adjustment);
}

void TileRenderer::TileAdaptiveController::on_tile_size_applied(int p_tile_size, bool p_changed) {
    state.last_computed_tile_size = p_tile_size;
    if (p_changed) {
        state.frames_since_adjustment = 0;
    }
}

void TileRenderer::TileAdaptiveController::on_allocation_failure(int p_tile_size) {
    state.last_computed_tile_size = p_tile_size;
    state.frames_since_adjustment = settings.frames_before_adjustment;
}

int TileRenderer::TileAdaptiveController::compute_tile_size(int p_requested_tile_size, const Vector2i &p_size) const {
    int sanitized_request = p_requested_tile_size > 0
            ? p_requested_tile_size
            : (owner.config_state.tile_size > 0 ? owner.config_state.tile_size : DEFAULT_TILE_SIZE);
    if (p_size.x > 0 && p_size.y > 0) {
        int max_dimension = MAX(1, MIN(p_size.x, p_size.y));
        sanitized_request = MIN(sanitized_request, max_dimension);
    }
    sanitized_request = CLAMP(sanitized_request, settings.min_tile_size, settings.max_tile_size);

    if (!settings.enable_adaptive_tile_size) {
        return sanitized_request;
    }

    if (!state.metrics_available || state.frames_since_adjustment < settings.frames_before_adjustment) {
        return sanitized_request;
    }

    int current_size = owner.config_state.tile_size > 0 ? owner.config_state.tile_size : sanitized_request;
    current_size = CLAMP(current_size, settings.min_tile_size, settings.max_tile_size);

    float occupancy = state.smoothed_occupancy;
    float dense_ratio = state.smoothed_dense_ratio;
    float overflow_ratio = state.smoothed_overflow_ratio;
    float average_splats = state.smoothed_average_splats;
    float average_ratio = MAX(0.0f, average_splats) / float(owner.config_state.effective_splat_capacity);

    float lower_target = MAX(0.0f, settings.target_occupancy_ratio - settings.occupancy_hysteresis);
    float upper_target = MIN(1.0f, settings.target_occupancy_ratio + settings.occupancy_hysteresis);

    bool reduce_tile_size = false;
    bool increase_tile_size = false;

    if (overflow_ratio > settings.overflow_ratio_threshold) {
        reduce_tile_size = true;
    }

    if (!reduce_tile_size && dense_ratio > settings.dense_ratio_threshold) {
        reduce_tile_size = true;
    }

    if (!reduce_tile_size && average_ratio > settings.max_average_splat_ratio) {
        reduce_tile_size = true;
    }

    if (!reduce_tile_size) {
        if (occupancy < lower_target && overflow_ratio < settings.overflow_ratio_threshold * 0.5f) {
            increase_tile_size = true;
        } else if (occupancy > upper_target && dense_ratio < settings.dense_ratio_threshold * 0.5f) {
            increase_tile_size = true;
        }
    }

    int new_size = current_size;
    int step = MAX(1, settings.tile_size_step);

    if (reduce_tile_size && current_size > settings.min_tile_size) {
        new_size = MAX(settings.min_tile_size, current_size - step);
    } else if (increase_tile_size && current_size < settings.max_tile_size) {
        new_size = MIN(settings.max_tile_size, current_size + step);
    }

    new_size = CLAMP(new_size, settings.min_tile_size, settings.max_tile_size);

    if (new_size != current_size) {
        GS_LOG_DEBUG(gs_logger::Category::GENERAL, vformat("[TileRenderer] Adaptive tile size change: %d -> %d (occ=%.2f, dense=%.2f, overflow=%.3f)",
                current_size, new_size, occupancy, dense_ratio, overflow_ratio));
    }

    return new_size;
}

void TileRenderer::TileAdaptiveController::update_state(const RenderStats &p_stats) {
    bool had_metrics = state.metrics_available;
    float smoothing = CLAMP(settings.smoothing_factor, 0.0f, 1.0f);

    auto smooth_value = [&](float &r_target, float p_value) {
        if (!had_metrics || smoothing <= 0.0f) {
            r_target = p_value;
        } else {
            r_target += (p_value - r_target) * smoothing;
        }
    };

    smooth_value(state.smoothed_occupancy, p_stats.density_metrics.occupancy_ratio);
    smooth_value(state.smoothed_dense_ratio, p_stats.density_metrics.dense_ratio);
    smooth_value(state.smoothed_overflow_ratio, p_stats.density_metrics.overflow_ratio);
    smooth_value(state.smoothed_average_splats, p_stats.average_splats_per_tile);

    state.metrics_available = p_stats.total_tiles > 0;

    if (state.metrics_available && state.frames_since_adjustment < 0xFFFFFFFF) {
        state.frames_since_adjustment++;
    }
}
