#include "quality_tier_config.h"

static void _set_streaming_caps_low(QualityTierConfig &r_out) {
	r_out.streaming_upload_mb_per_frame = 32;
	r_out.streaming_upload_mb_per_slice = 8;
	r_out.streaming_upload_mb_per_second = 256;
	r_out.streaming_vram_budget_mb = 2048;
	r_out.streaming_min_chunks_in_vram = 2;
	r_out.streaming_max_chunks_in_vram = 32;
}

static void _set_streaming_caps_medium(QualityTierConfig &r_out) {
	r_out.streaming_upload_mb_per_frame = 64;
	r_out.streaming_upload_mb_per_slice = 12;
	r_out.streaming_upload_mb_per_second = 768;
	r_out.streaming_vram_budget_mb = 6144;
	r_out.streaming_min_chunks_in_vram = 4;
	r_out.streaming_max_chunks_in_vram = 96;
}

static void _set_streaming_caps_high(QualityTierConfig &r_out) {
	r_out.streaming_upload_mb_per_frame = 128;
	r_out.streaming_upload_mb_per_slice = 16;
	r_out.streaming_upload_mb_per_second = 0;
	r_out.streaming_vram_budget_mb = 12288;
	r_out.streaming_min_chunks_in_vram = 4;
	r_out.streaming_max_chunks_in_vram = 128;
}

static bool _fill_quality_tier_config(const String &p_preset, QualityTierConfig &r_out) {
	if (p_preset == "low" || p_preset == "tier_low") {
		r_out.name = "low";
		r_out.max_splats = 300000;
		r_out.max_gpu_memory_mb = 256;
		r_out.target_gpu_memory_mb = 192;
		r_out.stream_budget_ms = 1;
		r_out.load_ahead_factor = 0.15f;
		r_out.unload_factor = 0.95f;
		r_out.max_concurrent_loads = 1;
		_set_streaming_caps_low(r_out);
		r_out.enable_packed_stage_data = false;
		r_out.enable_tighter_bounds = true;
		r_out.enable_sh_amortization = true;
		r_out.sh_amortization_divisor = 8;
		r_out.enable_fast_raster = false;
		return true;
	}

	if (p_preset == "medium" || p_preset == "tier_medium" || p_preset == "balanced") {
		r_out.name = "medium";
		r_out.max_splats = 700000;
		r_out.max_gpu_memory_mb = 768;
		r_out.target_gpu_memory_mb = 640;
		r_out.stream_budget_ms = 2;
		r_out.load_ahead_factor = 0.25f;
		r_out.unload_factor = 1.05f;
		r_out.max_concurrent_loads = 2;
		_set_streaming_caps_medium(r_out);
		r_out.enable_packed_stage_data = false;
		r_out.enable_tighter_bounds = true;
		r_out.enable_sh_amortization = true;
		r_out.sh_amortization_divisor = 4;
		r_out.enable_fast_raster = false;
		return true;
	}

	if (p_preset == "high" || p_preset == "tier_high") {
		r_out.name = "high";
		r_out.max_splats = 1200000;
		r_out.max_gpu_memory_mb = 1024;
		r_out.target_gpu_memory_mb = 896;
		r_out.stream_budget_ms = 3;
		r_out.load_ahead_factor = 0.35f;
		r_out.unload_factor = 1.2f;
		r_out.max_concurrent_loads = 3;
		_set_streaming_caps_high(r_out);
		r_out.enable_packed_stage_data = false;
		r_out.enable_tighter_bounds = true;
		r_out.enable_sh_amortization = true;
		r_out.sh_amortization_divisor = 2;
		r_out.enable_fast_raster = false;
		return true;
	}

	if (p_preset == "rtx_3090_1080p") {
		r_out.name = "rtx_3090_1080p";
		r_out.max_splats = 5000000;
		r_out.max_gpu_memory_mb = 12288;
		r_out.target_gpu_memory_mb = 11000;
		r_out.stream_budget_ms = 6;
		r_out.load_ahead_factor = 0.45f;
		r_out.unload_factor = 1.2f;
		r_out.max_concurrent_loads = 6;
		_set_streaming_caps_high(r_out);
		r_out.enable_packed_stage_data = false;
		r_out.enable_tighter_bounds = true;
		r_out.enable_sh_amortization = true;
		r_out.sh_amortization_divisor = 2;
		r_out.enable_fast_raster = false;
		return true;
	}

	if (p_preset == "rtx_3090_4k") {
		r_out.name = "rtx_3090_4k";
		r_out.max_splats = 3500000;
		r_out.max_gpu_memory_mb = 12288;
		r_out.target_gpu_memory_mb = 10240;
		r_out.stream_budget_ms = 5;
		r_out.load_ahead_factor = 0.35f;
		r_out.unload_factor = 1.1f;
		r_out.max_concurrent_loads = 4;
		_set_streaming_caps_high(r_out);
		r_out.enable_packed_stage_data = false;
		r_out.enable_tighter_bounds = true;
		r_out.enable_sh_amortization = true;
		r_out.sh_amortization_divisor = 3;
		r_out.enable_fast_raster = false;
		return true;
	}

	if (p_preset == "desktop_1080p" || p_preset == "high_1080p" ||
			p_preset == "desktop") {
		r_out.name = "desktop_1080p";
		r_out.max_splats = 1200000;
		r_out.max_gpu_memory_mb = 1024;
		r_out.target_gpu_memory_mb = 896;
		r_out.stream_budget_ms = 3;
		r_out.load_ahead_factor = 0.35f;
		r_out.unload_factor = 1.2f;
		r_out.max_concurrent_loads = 3;
		_set_streaming_caps_high(r_out);
		r_out.enable_packed_stage_data = false;
		r_out.enable_tighter_bounds = true;
		r_out.enable_sh_amortization = true;
		r_out.sh_amortization_divisor = 2;
		r_out.enable_fast_raster = false;
		return true;
	}

	if (p_preset == "desktop_4k" || p_preset == "high_4k") {
		r_out.name = "desktop_4k";
		r_out.max_splats = 700000;
		r_out.max_gpu_memory_mb = 768;
		r_out.target_gpu_memory_mb = 640;
		r_out.stream_budget_ms = 2;
		r_out.load_ahead_factor = 0.25f;
		r_out.unload_factor = 1.05f;
		r_out.max_concurrent_loads = 2;
		_set_streaming_caps_medium(r_out);
		r_out.enable_packed_stage_data = false;
		r_out.enable_tighter_bounds = true;
		r_out.enable_sh_amortization = true;
		r_out.sh_amortization_divisor = 4;
		r_out.enable_fast_raster = false;
		return true;
	}

	if (p_preset == "steam_deck" || p_preset == "steamdeck" || p_preset == "handheld") {
		r_out.name = "steam_deck";
		r_out.max_splats = 300000;
		r_out.max_gpu_memory_mb = 256;
		r_out.target_gpu_memory_mb = 192;
		r_out.stream_budget_ms = 1;
		r_out.load_ahead_factor = 0.15f;
		r_out.unload_factor = 0.95f;
		r_out.max_concurrent_loads = 1;
		_set_streaming_caps_low(r_out);
		r_out.enable_packed_stage_data = false;
		r_out.enable_tighter_bounds = true;
		r_out.enable_sh_amortization = true;
		r_out.sh_amortization_divisor = 8;
		r_out.enable_fast_raster = false;
		return true;
	}

	return false;
}

bool get_quality_tier_config(const String &p_preset, QualityTierConfig &r_out) {
	String preset = p_preset.strip_edges().to_lower();
	if (preset.is_empty() || preset == "custom" || preset == "none") {
		return false;
	}

	return _fill_quality_tier_config(preset, r_out);
}
