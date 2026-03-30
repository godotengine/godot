#ifndef GAUSSIAN_SPLAT_QUALITY_TIER_CONFIG_H
#define GAUSSIAN_SPLAT_QUALITY_TIER_CONFIG_H

#include "core/string/ustring.h"
#include <cstdint>

struct QualityTierConfig {
	String name;
	uint32_t max_splats = 0;
	uint32_t max_gpu_memory_mb = 0;
	uint32_t target_gpu_memory_mb = 0;
	uint32_t stream_budget_ms = 0;
	float load_ahead_factor = 0.0f;
	float unload_factor = 1.0f;
	uint32_t max_concurrent_loads = 0;
	uint32_t streaming_upload_mb_per_frame = 0;
	uint32_t streaming_upload_mb_per_slice = 0;
	uint32_t streaming_upload_mb_per_second = 0;
	uint32_t streaming_vram_budget_mb = 0;
	uint32_t streaming_min_chunks_in_vram = 0;
	uint32_t streaming_max_chunks_in_vram = 0;
	bool enable_packed_stage_data = false;
	bool enable_tighter_bounds = false;
	bool enable_sh_amortization = false;
	int sh_amortization_divisor = 1;
	bool enable_fast_raster = false;

	// Render quality overrides (-1 = no opinion, let user/code-default decide).
	int sh_bands = -1;
	float lod_max_distance = -1.0f;
	float lod_base_threshold = -1.0f;
	int quantization_enabled = -1;
	int route_policy = -1;
};

bool get_quality_tier_config(const String &p_preset, QualityTierConfig &r_out);

#endif // GAUSSIAN_SPLAT_QUALITY_TIER_CONFIG_H
