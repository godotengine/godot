#ifndef GS_OVERFLOW_AUTO_TUNER_H
#define GS_OVERFLOW_AUTO_TUNER_H

#include "core/object/ref_counted.h"

// Overflow statistics from rasterizer
struct RasterOverflowStats;

// Configuration for overflow auto-tuning
// Close-up safety: Uses EMA smoothing and hysteresis to prevent aggressive culling
// when the camera moves close to surfaces (which naturally increases overlap).
struct OverflowAutoTuneConfig {
    bool enabled = true;
    float trigger_ratio = 0.01f;          // Overflow ratio that triggers adjustment
    float importance_step = 0.005f;        // How much to increase importance threshold
    float importance_max = 0.1f;           // Maximum increase from baseline
    float importance_decay = 0.001f;       // Recovery rate when overflow subsides
    float tiny_step = 0.25f;               // How much to increase tiny splat radius
    float tiny_max = 2.0f;                 // Maximum increase from baseline
    float tiny_decay = 0.05f;              // Recovery rate for tiny splat radius

    // Close-up safety parameters (added December 2024)
    float ema_alpha = 0.15f;              // EMA smoothing factor (0.1-0.3 typical, lower = more smoothing)
    float hysteresis_band = 0.3f;         // Must drop to (trigger * (1 - hysteresis_band)) before releasing
    float max_step_multiplier = 1.5f;     // Max amplification of step size (was 4.0, reduced for stability)
    uint32_t warmup_frames = 10;          // Frames before autotune activates (ignore startup spikes)
    uint32_t cooldown_frames = 30;        // Frames to wait before relaxing after overflow event
    float close_up_threshold = 0.1f;      // Average splat screen coverage above this indicates close-up
    float close_up_dampen = 0.5f;         // Reduce aggressiveness by this factor when close-up detected
};

// Result of auto-tuning step
struct AutoTuneResult {
    bool parameters_changed = false;
    float new_importance_threshold = 0.0f;
    float new_tiny_splat_radius = 0.0f;
    bool close_up_detected = false;       // True if close-up view dampening is active
    bool in_cooldown = false;             // True if waiting for cooldown before relaxing
};

// Pure abstract interface for overflow auto-tuning
class IOverflowAutoTuner {
public:
    virtual ~IOverflowAutoTuner() = default;

    // Configure auto-tuning parameters
    virtual void set_config(const OverflowAutoTuneConfig &p_config) = 0;
    virtual OverflowAutoTuneConfig get_config() const = 0;

    // Set baseline values (used as minimum thresholds during recovery)
    virtual void set_baselines(float p_importance_baseline, float p_tiny_radius_baseline) = 0;

    // Process overflow feedback and return adjusted parameters
    // p_avg_screen_coverage: average splat screen area (for close-up detection), 0.0 if unknown
    virtual AutoTuneResult apply_feedback(const RasterOverflowStats &p_stats,
            uint32_t p_splat_count, uint32_t p_tile_count,
            float p_avg_screen_coverage = 0.0f) = 0;

    // Get current adjusted values
    virtual float get_importance_threshold() const = 0;
    virtual float get_tiny_splat_radius() const = 0;

    // Reset to baseline values
    virtual void reset_to_baselines() = 0;

    // Enable/disable auto-tuning
    virtual void set_enabled(bool p_enabled) = 0;
    virtual bool is_enabled() const = 0;

    // Implementation info
    virtual String get_name() const = 0;
};

// Concrete implementation of overflow auto-tuner
class OverflowAutoTuner : public RefCounted, public IOverflowAutoTuner {
    GDCLASS(OverflowAutoTuner, RefCounted);

public:
    OverflowAutoTuner();
    ~OverflowAutoTuner();

    // IOverflowAutoTuner interface
    void set_config(const OverflowAutoTuneConfig &p_config) override;
    OverflowAutoTuneConfig get_config() const override { return config; }

    void set_baselines(float p_importance_baseline, float p_tiny_radius_baseline) override;

    AutoTuneResult apply_feedback(const RasterOverflowStats &p_stats,
            uint32_t p_splat_count, uint32_t p_tile_count,
            float p_avg_screen_coverage = 0.0f) override;

    float get_importance_threshold() const override { return current_importance_threshold; }
    float get_tiny_splat_radius() const override { return current_tiny_splat_radius; }

    void reset_to_baselines() override;

    void set_enabled(bool p_enabled) override { config.enabled = p_enabled; }
    bool is_enabled() const override { return config.enabled; }

    String get_name() const override { return "OverflowAutoTuner"; }

protected:
    static void _bind_methods();

private:
    OverflowAutoTuneConfig config;

    // Baseline values (minimum thresholds)
    float importance_baseline = 0.0f;
    float tiny_radius_baseline = 0.5f;

    // Current adjusted values
    float current_importance_threshold = 0.0f;
    float current_tiny_splat_radius = 0.5f;

    // EMA-smoothed severity (for stable triggering)
    float smoothed_severity = 0.0f;

    // Hysteresis state
    bool overflow_active = false;          // Currently in overflow state
    uint32_t cooldown_remaining = 0;       // Frames until we can start relaxing

    // Close-up detection state
    float smoothed_screen_coverage = 0.0f; // EMA of average screen coverage
    bool close_up_mode = false;            // True when close-up dampening is active

    // Debug tracking
    uint64_t frame_counter = 0;

    // Staleness detection for async GPU readback stats (ISSUE-033).
    // Tracks the frame_number from the last overflow stats we accepted, and
    // the auto-tuner frame_counter at that time. If the stats frame_number
    // hasn't changed for more than kMaxStaleFrames auto-tuner ticks, discard.
    uint64_t last_stats_frame_number = 0;
    uint64_t last_fresh_stats_frame_counter = 0;
};

#endif // GS_OVERFLOW_AUTO_TUNER_H
