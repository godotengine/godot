// Overflow Auto-Tuner: Dynamically adjusts culling parameters based on tile overflow feedback
// Extracted from GaussianSplatRenderer::_apply_overflow_feedback()

#include "overflow_auto_tuner.h"
#include "../core/gs_project_settings.h"
#include "rasterizer_interfaces.h"
#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"
#include "core/variant/variant.h"
#include "../logger/gs_logger.h"

namespace {

static bool _is_autotune_log_enabled() {
#ifdef GS_SILENCE_LOGS
    return false;
#else
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        if (gs::settings::is_all_debug_enabled(ps)) {
            return true;
        }
        return gs::settings::get_bool(ps, "rendering/gaussian_splatting/debug/enable_autotune_logs", false);
    }
    return false;
#endif
}

} // namespace

void OverflowAutoTuner::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &OverflowAutoTuner::set_enabled);
    ClassDB::bind_method(D_METHOD("is_enabled"), &OverflowAutoTuner::is_enabled);
    ClassDB::bind_method(D_METHOD("get_importance_threshold"), &OverflowAutoTuner::get_importance_threshold);
    ClassDB::bind_method(D_METHOD("get_tiny_splat_radius"), &OverflowAutoTuner::get_tiny_splat_radius);
    ClassDB::bind_method(D_METHOD("reset_to_baselines"), &OverflowAutoTuner::reset_to_baselines);
}

OverflowAutoTuner::OverflowAutoTuner() {
}

OverflowAutoTuner::~OverflowAutoTuner() {
}

void OverflowAutoTuner::set_config(const OverflowAutoTuneConfig &p_config) {
    config = p_config;
}

void OverflowAutoTuner::set_baselines(float p_importance_baseline, float p_tiny_radius_baseline) {
    importance_baseline = p_importance_baseline;
    tiny_radius_baseline = p_tiny_radius_baseline;

    // Initialize current values to baselines if not already set
    if (current_importance_threshold < importance_baseline) {
        current_importance_threshold = importance_baseline;
    }
    if (current_tiny_splat_radius < tiny_radius_baseline) {
        current_tiny_splat_radius = tiny_radius_baseline;
    }
}

void OverflowAutoTuner::reset_to_baselines() {
    current_importance_threshold = importance_baseline;
    current_tiny_splat_radius = tiny_radius_baseline;
    smoothed_severity = 0.0f;
    smoothed_screen_coverage = 0.0f;
    overflow_active = false;
    cooldown_remaining = 0;
    close_up_mode = false;
    last_stats_frame_number = 0;
    last_fresh_stats_frame_counter = 0;
}

AutoTuneResult OverflowAutoTuner::apply_feedback(const RasterOverflowStats &p_stats,
        uint32_t p_splat_count, uint32_t p_tile_count,
        float p_avg_screen_coverage) {
    AutoTuneResult result;
    result.new_importance_threshold = current_importance_threshold;
    result.new_tiny_splat_radius = current_tiny_splat_radius;

    frame_counter++;

    if (!config.enabled || p_splat_count == 0) {
        return result;
    }

    // Discard stale overflow stats from async GPU readback (ISSUE-033).
    // Stats arrive 1-3 frames late due to GPU readback latency. If the stats
    // frame_number hasn't advanced in more than 2 auto-tuner ticks, the readback
    // is stuck or the same stale snapshot is being re-delivered. Reacting to
    // outdated overflow conditions was the root cause of exponential splat decay
    // (84K -> 0 over ~60 frames).
    static constexpr uint64_t kMaxStaleFrames = 2;
    if (p_stats.frame_number > 0) {
        if (p_stats.frame_number != last_stats_frame_number) {
            // Fresh stats arrived - update tracking.
            last_stats_frame_number = p_stats.frame_number;
            last_fresh_stats_frame_counter = frame_counter;
        } else {
            // Same stats we already processed - stale re-delivery.
            // Allow up to kMaxStaleFrames re-deliveries before discarding.
            uint64_t frames_since_fresh = frame_counter - last_fresh_stats_frame_counter;
            if (frames_since_fresh > kMaxStaleFrames) {
                if (_is_autotune_log_enabled()) {
                    GS_LOG_RENDERER_DEBUG(vformat("[AUTOTUNE] Discarding stale overflow stats: "
                            "stats.frame_number=%d unchanged for %d frames (max=%d)",
                            p_stats.frame_number, frames_since_fresh, kMaxStaleFrames));
                }
                return result;
            }
        }
    }

    // Skip warmup frames to avoid reacting to startup transients
    if (frame_counter <= config.warmup_frames) {
        return result;
    }

    // Calculate raw overflow severity
    // overflow_splats_clamped counts tile writes that overflowed
    // overflow_splats_aggregated is total attempted tile inserts (better denominator)
    float clamped_ratio = 0.0f;
    if (p_stats.overflow_splats_aggregated > 0) {
        clamped_ratio = float(p_stats.overflow_splats_clamped) / float(p_stats.overflow_splats_aggregated);
    } else {
        clamped_ratio = float(p_stats.overflow_splats_clamped) / float(p_splat_count);
    }

    float tile_ratio = p_tile_count > 0 ? float(p_stats.overflow_tile_count) / float(p_tile_count) : 0.0f;
    float raw_severity = MAX(clamped_ratio, tile_ratio);

    // Apply EMA smoothing to severity for stable decision-making
    // This prevents sudden spikes from causing aggressive culling
    float alpha = CLAMP(config.ema_alpha, 0.01f, 0.5f);
    smoothed_severity = smoothed_severity * (1.0f - alpha) + raw_severity * alpha;

    // Update close-up detection with EMA
    if (p_avg_screen_coverage > 0.0f) {
        smoothed_screen_coverage = smoothed_screen_coverage * (1.0f - alpha) + p_avg_screen_coverage * alpha;
    }

    // Detect close-up view (splats covering large screen area)
    // Close-ups naturally cause more tile overlap - we should be less aggressive
    close_up_mode = smoothed_screen_coverage > config.close_up_threshold;
    result.close_up_detected = close_up_mode;

    // Calculate effective trigger with hysteresis
    float trigger = MAX(config.trigger_ratio, 1e-5f);
    float release_threshold = trigger * (1.0f - config.hysteresis_band);

    // Update cooldown timer
    if (cooldown_remaining > 0) {
        cooldown_remaining--;
        result.in_cooldown = true;
    }

    // Debug tracing (first few frames after warmup and every 60th frame)
    bool should_log = (frame_counter <= config.warmup_frames + 5) || (frame_counter % 60 == 0);
    if (should_log && _is_autotune_log_enabled()) {
        GS_LOG_RENDERER_DEBUG(vformat("[AUTOTUNE] frame=%d raw=%.4f smoothed=%.4f trigger=%.4f "
                "closeup=%s cooldown=%d importance=%.4f tiny=%.2f",
                frame_counter, raw_severity, smoothed_severity, trigger,
                close_up_mode ? "Y" : "N", cooldown_remaining,
                current_importance_threshold, current_tiny_splat_radius));
    }

    // State machine: overflow_active with hysteresis
    if (!overflow_active && smoothed_severity > trigger) {
        // Transition to overflow state
        overflow_active = true;
        cooldown_remaining = config.cooldown_frames;
    } else if (overflow_active && smoothed_severity < release_threshold && cooldown_remaining == 0) {
        // Transition out of overflow state (only after cooldown)
        overflow_active = false;
    }

    if (overflow_active) {
        // Overflow detected - increase culling thresholds
        // Calculate step multiplier based on severity overshoot
        float overshoot = (smoothed_severity - trigger) / MAX(trigger, 1e-5f);
        float factor = CLAMP(overshoot, 0.0f, config.max_step_multiplier);

        // Apply close-up dampening: reduce aggressiveness when viewing close-up
        // This prevents over-culling when the user is inspecting detail
        float dampen = close_up_mode ? config.close_up_dampen : 1.0f;

        // Calculate importance step with dampening
        float importance_step = config.importance_step * (1.0f + factor) * dampen;
        float max_importance = importance_baseline + config.importance_max;
        float target_importance = MIN(max_importance, current_importance_threshold + importance_step);

        if (target_importance > current_importance_threshold) {
            current_importance_threshold = target_importance;
            result.parameters_changed = true;
        }

        // Calculate tiny splat radius step with dampening
        float tiny_step = config.tiny_step * (1.0f + factor) * dampen;
        float max_tiny = tiny_radius_baseline + config.tiny_max;
        float target_tiny = MIN(max_tiny, current_tiny_splat_radius + tiny_step);

        if (target_tiny > current_tiny_splat_radius) {
            current_tiny_splat_radius = target_tiny;
            result.parameters_changed = true;
        }

        // Reset cooldown whenever we increase thresholds
        cooldown_remaining = config.cooldown_frames;
        result.in_cooldown = true;
    } else if (!overflow_active && cooldown_remaining == 0) {
        // Low overflow and cooldown expired - gradually recover towards baselines
        // Recovery is always allowed (not dampened) to eventually return to baseline

        if (current_importance_threshold > importance_baseline) {
            float lowered = MAX(importance_baseline, current_importance_threshold - config.importance_decay);
            if (!Math::is_equal_approx(lowered, current_importance_threshold)) {
                current_importance_threshold = lowered;
                result.parameters_changed = true;
            }
        }

        if (current_tiny_splat_radius > tiny_radius_baseline) {
            float lowered = MAX(tiny_radius_baseline, current_tiny_splat_radius - config.tiny_decay);
            if (!Math::is_equal_approx(lowered, current_tiny_splat_radius)) {
                current_tiny_splat_radius = lowered;
                result.parameters_changed = true;
            }
        }
    }

    result.new_importance_threshold = current_importance_threshold;
    result.new_tiny_splat_radius = current_tiny_splat_radius;

    return result;
}
