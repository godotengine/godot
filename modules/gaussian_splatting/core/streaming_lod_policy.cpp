/**
 * @file streaming_lod_policy.cpp
 * @brief LOD policy implementations for GaussianStreamingSystem.
 *
 * Extracted from gaussian_streaming.cpp to isolate LOD-specific logic.
 * Contains:
 *   - LOD Blend Factor Functions (LODGE technique)
 *   - Distance-based LOD (Octree-GS) Implementation
 *
 * All functions are GaussianStreamingSystem methods declared in
 * gaussian_streaming.h; no new header is required.
 */

#include "gaussian_streaming.h"
#include "../lod/lod_config.h"
#include "core/math/math_funcs.h"
#include <cfloat>

// ==============================================================================
// LOD Blend Factor Functions (LODGE technique)
// ==============================================================================

void GaussianStreamingSystem::_load_lod_blend_config_from_project_settings() {
	visibility.load_lod_blend_config_from_project_settings();
}

float GaussianStreamingSystem::_calculate_lod_blend_factor(float distance, float lod_distance) const {
	return visibility.calculate_lod_blend_factor(distance, lod_distance);
}

void GaussianStreamingSystem::_update_chunk_lod_blend_factors(const Vector3 &camera_pos) {
	visibility.update_chunk_lod_blend_factors(*this, camera_pos);
}

// ==============================================================================
// Distance-based LOD (Octree-GS) Implementation
// ==============================================================================

void GaussianStreamingSystem::_load_lod_config_from_project_settings() {
	// Reload the global LOD configuration from project settings
	g_lod_config.load_from_project_settings();
}

void GaussianStreamingSystem::_update_chunk_lod_parameters(const Vector3 &camera_pos) {
	visibility.update_chunk_lod_parameters(*this, camera_pos);
}

Dictionary GaussianStreamingSystem::get_lod_debug_stats() const {
	Dictionary stats;
	const LODConfig &lod_config = _get_lod_config();

	uint32_t lod_level_counts[8] = {};
	uint32_t sh_band_counts[4] = {};
	uint32_t total_original_splats = 0;
	uint32_t total_effective_splats = 0;
	float min_distance = FLT_MAX;
	float max_distance = 0.0f;
	float total_distance = 0.0f;
	uint32_t visible_count = 0;

	// Aggregate chunk-level LOD metrics
	uint32_t total_lod_level = 0;
	int max_skip_factor = 1;
	float min_opacity = 1.0f;
	uint32_t chunks_in_transition = 0;

	const FrameData &frame = frame_data[current_frame_idx];
	_collect_lod_debug_stats(frame, lod_level_counts, sh_band_counts, total_original_splats,
			total_effective_splats, min_distance, max_distance, total_distance, visible_count,
			total_lod_level, max_skip_factor, min_opacity, chunks_in_transition);

	// Build stats dictionary
	stats["enabled"] = lod_config.enabled;
	stats["num_levels"] = lod_config.num_levels;
	stats["max_distance"] = lod_config.max_distance;
	stats["base_threshold"] = lod_config.base_threshold;

	stats["visible_chunks"] = visible_count;
	stats["total_original_splats"] = total_original_splats;
	stats["total_effective_splats"] = total_effective_splats;

	if (total_original_splats > 0) {
		stats["reduction_ratio"] = 1.0f - (float(total_effective_splats) / float(total_original_splats));
	} else {
		stats["reduction_ratio"] = 0.0f;
	}

	// Distance stats
	if (visible_count > 0) {
		stats["min_distance"] = (min_distance < FLT_MAX) ? min_distance : 0.0f;
		stats["max_distance"] = max_distance;
		stats["avg_distance"] = total_distance / visible_count;
	} else {
		stats["min_distance"] = 0.0f;
		stats["max_distance"] = 0.0f;
		stats["avg_distance"] = 0.0f;
	}

	// LOD level distribution as array
	Array lod_dist;
	for (int i = 0; i < 8; i++) {
		lod_dist.push_back(lod_level_counts[i]);
	}
	stats["lod_distribution"] = lod_dist;

	// SH band distribution as array
	Array sh_dist;
	for (int i = 0; i < 4; i++) {
		sh_dist.push_back(sh_band_counts[i]);
	}
	stats["sh_band_distribution"] = sh_dist;

	// Debug visualization info
	stats["debug_visualization"] = lod_config.debug_visualization;
	stats["splat_skip_enabled"] = lod_config.splat_skip_enabled;
	stats["sh_reduction_enabled"] = lod_config.sh_reduction_enabled;
	stats["opacity_fade_enabled"] = lod_config.opacity_fade_enabled;

	// Aggregate LOD metrics (for Performance Monitors Phase 2)
	if (visible_count > 0) {
		stats["current_lod_level"] = int(total_lod_level / visible_count); // Average LOD level
	} else {
		stats["current_lod_level"] = 0;
	}
	stats["lod_target_distance"] = lod_config.base_threshold; // Distance threshold for LOD 0 transition
	stats["max_splat_skip_factor"] = max_skip_factor; // Maximum skip factor among visible chunks
	stats["min_opacity_multiplier"] = min_opacity; // Minimum opacity multiplier among visible chunks
	stats["chunks_in_transition"] = chunks_in_transition; // Chunks currently blending between LOD levels
	stats["transitions_this_frame"] = visibility.lod_transitions_this_frame; // LOD level/target transitions updated this frame.

	// LOD blend configuration (from LODBlendConfig)
	stats["blend_distance"] = visibility.lod_blend_config.blend_distance;
	stats["hysteresis_zone"] = visibility.lod_blend_config.hysteresis_zone;

	return stats;
}

void GaussianStreamingSystem::_collect_lod_debug_stats(const FrameData &frame,
		uint32_t (&lod_level_counts)[8],
		uint32_t (&sh_band_counts)[4],
		uint32_t &total_original_splats,
		uint32_t &total_effective_splats,
		float &min_distance,
		float &max_distance,
		float &total_distance,
		uint32_t &visible_count,
		uint32_t &total_lod_level,
		int &max_skip_factor,
		float &min_opacity,
		uint32_t &chunks_in_transition) const {
	for (uint32_t chunk_idx : frame.visible_chunks) {
		if (chunk_idx >= chunks.size()) {
			continue;
		}

		const StreamingChunk &chunk = chunks[chunk_idx];
		visible_count++;

		if (chunk.current_lod_level < 8) {
			lod_level_counts[chunk.current_lod_level]++;
		}

		if (chunk.sh_band_level >= 0 && chunk.sh_band_level < 4) {
			sh_band_counts[chunk.sh_band_level]++;
		}

		total_original_splats += chunk.count;
		total_effective_splats += chunk.effective_count;

		if (chunk.distance < min_distance) {
			min_distance = chunk.distance;
		}
		if (chunk.distance > max_distance) {
			max_distance = chunk.distance;
		}
		total_distance += chunk.distance;

		total_lod_level += chunk.current_lod_level;
		if (chunk.splat_skip_factor > max_skip_factor) {
			max_skip_factor = chunk.splat_skip_factor;
		}
		if (chunk.opacity_multiplier < min_opacity) {
			min_opacity = chunk.opacity_multiplier;
		}
		if (chunk.lod_blend_factor < 1.0f) {
			chunks_in_transition++;
		}
	}
}

uint32_t GaussianStreamingSystem::get_effective_splat_count() const {
	uint32_t total = 0;
	const FrameData &frame = frame_data[current_frame_idx];

	for (uint32_t chunk_idx : frame.visible_chunks) {
		if (chunk_idx < chunks.size()) {
			total += chunks[chunk_idx].effective_count;
		}
	}

	return total;
}
