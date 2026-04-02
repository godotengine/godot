#ifndef GAUSSIAN_SPLAT_QUALITY_CONFIG_H
#define GAUSSIAN_SPLAT_QUALITY_CONFIG_H

#include <cstdint>

namespace GaussianSplatting {

// Neutral node-facing LOD config kept independent from legacy LOD system types.
struct GaussianSplatLODConfig {
	float lod0_distance = 10.0f;
	float lod1_distance = 50.0f;
	float lod2_distance = 100.0f;
	float lod3_distance = 200.0f;
	float cull_distance = 500.0f;
	float far_lod_keep_ratio = 0.1f;

	uint32_t max_splats_per_frame = 500000;
	uint32_t min_splats_per_frame = 10000;

	float importance_threshold = 0.1f;
	float size_cull_threshold = 0.5f;

	float lod_bias = 1.0f;
	bool smooth_transitions = true;
	float transition_time = 0.25f;

	float target_framerate = 60.0f;
	float quality_adjustment_rate = 0.1f;
	bool enable_temporal_coherence = true;
	bool enable_painterly_mode = false;
};

// Neutral node-facing streaming config kept independent from legacy streaming manager types.
struct GaussianSplatStreamingConfig {
	uint64_t max_gpu_memory = 1024ull * 1024ull * 1024ull;
	uint64_t max_cpu_memory = 2048ull * 1024ull * 1024ull;
	uint64_t target_gpu_memory = 768ull * 1024ull * 1024ull;

	float load_ahead_distance = 50.0f;
	float unload_distance = 200.0f;
	uint32_t max_concurrent_loads = 2;
	bool enable_predictive_loading = true;
	float prediction_time = 0.5f;

	uint32_t num_lod_levels = 4;
	float lod_distance_multiplier = 2.0f;
	bool enable_adaptive_quality = true;
	bool enable_painterly_mode = false;
	uint32_t painterly_seed = 1337;
	float painterly_transition_rate = 4.0f;
	float painterly_hold_strength = 0.2f;

	uint32_t stream_budget_ms = 2;
	bool enable_async_loading = true;
	bool enable_compression = true;
};

} // namespace GaussianSplatting

#endif // GAUSSIAN_SPLAT_QUALITY_CONFIG_H
