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
	visibility.load_lod_blend_config_from_project_settings(*this);
}

float GaussianStreamingSystem::_calculate_lod_blend_factor(float distance, float lod_distance) const {
	return visibility.calculate_lod_blend_factor(distance, lod_distance);
}

void GaussianStreamingSystem::_update_chunk_lod_blend_factors(const Vector3 &camera_pos) {
	visibility.update_chunk_lod_blend_factors(*this, camera_pos);
}

void GaussianStreamingSystem::VisibilityState::load_lod_blend_config_from_project_settings(GaussianStreamingSystem &system) {
	(void)system;
	lod_blend_config = LODBlendConfig::load_from_project_settings();
}

float GaussianStreamingSystem::VisibilityState::calculate_lod_blend_factor(float distance, float lod_distance) const {
	if (!lod_blend_config.blend_enabled) {
		return 1.0f; // No blending, full opacity
	}

	float blend_distance = lod_blend_config.blend_distance;
	if (blend_distance <= 0.0f) {
		return 1.0f;
	}

	// LODGE technique: smoothstep blend near LOD boundaries
	// lod_distance is the threshold where LOD would change
	float lower_bound = lod_distance - blend_distance;
	float upper_bound = lod_distance + blend_distance;

	// smoothstep for smooth transition
	if (distance <= lower_bound) {
		return 1.0f; // Fully visible (high quality LOD)
	} else if (distance >= upper_bound) {
		return 0.0f; // Fading out (transitioning to lower LOD)
	} else {
		// Smoothstep interpolation within the blend zone
		float t = (distance - lower_bound) / (upper_bound - lower_bound);
		// smoothstep: 3t^2 - 2t^3
		return 1.0f - (t * t * (3.0f - 2.0f * t));
	}
}

void GaussianStreamingSystem::VisibilityState::update_chunk_lod_blend_factors(GaussianStreamingSystem &system, const Vector3 &camera_pos) {
	(void)camera_pos;
	if (!lod_blend_config.blend_enabled) {
		// Reset all blend factors to 1.0 if blending disabled
		for (uint32_t i = 0; i < system.chunks.size(); i++) {
			system.chunks[i].lod_blend_factor = 1.0f;
		}
		current_lod_blend_factor = 1.0f;
		return;
	}

	// Apply LOD distance multiplier from regulator for quality degradation
	float lod_mult = system.budget.vram_regulator.is_valid() ? system.budget.vram_regulator->get_lod_distance_multiplier() : 1.0f;

	// Base LOD distance threshold (same as used in update_streaming)
	float base_lod_distance = STREAMING_LOAD_DISTANCE_BASE / lod_mult;

	float weighted_blend_sum = 0.0f;
	float total_weight = 0.0f;

	for (uint32_t i = 0; i < system.chunks.size(); i++) {
		StreamingChunk &chunk = system.chunks[i];

		// Apply hysteresis: use smoothed distance to prevent flickering
		float hysteresis = lod_blend_config.hysteresis_zone;
		float effective_distance = chunk.distance;

		// Hysteresis: only update if distance changed significantly
		if (Math::abs(effective_distance - chunk.previous_distance) > hysteresis) {
			chunk.previous_distance = effective_distance;
		} else {
			// Use previous distance within hysteresis zone
			effective_distance = chunk.previous_distance;
		}

		// Calculate blend factor for this chunk
		chunk.lod_blend_factor = calculate_lod_blend_factor(effective_distance, base_lod_distance);

		// Accumulate weighted average for global blend factor
		if (chunk.is_visible && chunk.is_loaded) {
			float weight = float(chunk.count);
			weighted_blend_sum += chunk.lod_blend_factor * weight;
			total_weight += weight;
		}
	}

	// Calculate global blend factor as weighted average of visible chunks
	if (total_weight > 0.0f) {
		current_lod_blend_factor = weighted_blend_sum / total_weight;
	} else {
		current_lod_blend_factor = 1.0f;
	}
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

void GaussianStreamingSystem::VisibilityState::update_chunk_lod_parameters(
		GaussianStreamingSystem &system, const Vector3 &camera_pos) {
	(void)camera_pos;
	lod_transitions_this_frame = 0;
	// Update LOD parameters for all chunks based on distance using Octree-GS formula
	// L = floor(min(max(log2(d_max/d), 0), K-1))

	const LODConfig &lod_config = system._get_lod_config();
	const auto mark_resident_chunk_meta_if_lod_changed = [&](uint32_t chunk_idx,
																  const StreamingChunk &chunk,
																  uint32_t prev_effective_count,
																  uint32_t prev_lod_level,
																  int prev_sh_band_level) {
		if (chunk.effective_count == prev_effective_count &&
				chunk.current_lod_level == prev_lod_level &&
				chunk.sh_band_level == prev_sh_band_level) {
			return;
		}

		const bool resident = chunk.is_loaded && !chunk.upload_pending && chunk.buffer_slot != UINT32_MAX;
		if (resident) {
			system._mark_chunk_meta_dirty(chunk_idx);
		}
	};

	if (!lod_config.enabled) {
		// LOD disabled - use full quality for all chunks
		for (uint32_t i = 0; i < system.chunks.size(); i++) {
			StreamingChunk &chunk = system.chunks[i];
			const uint32_t prev_effective_count = chunk.effective_count;
			const uint32_t prev_lod_level = chunk.current_lod_level;
			const uint32_t prev_target_lod_level = chunk.target_lod_level;
			const int prev_sh_band_level = chunk.sh_band_level;
			chunk.current_lod_level = 0;
			chunk.target_lod_level = 0;
			chunk.sh_band_level = 3; // Full SH quality
			chunk.splat_skip_factor = 1; // Render all splats
			chunk.opacity_multiplier = 1.0f;
			chunk.effective_count = chunk.count;
			if (prev_lod_level != chunk.current_lod_level ||
					prev_target_lod_level != chunk.target_lod_level) {
				lod_transitions_this_frame++;
			}

			mark_resident_chunk_meta_if_lod_changed(i, chunk, prev_effective_count, prev_lod_level, prev_sh_band_level);
		}
		return;
	}

	for (uint32_t i = 0; i < system.chunks.size(); i++) {
		StreamingChunk &chunk = system.chunks[i];
		const uint32_t prev_effective_count = chunk.effective_count;
		const uint32_t prev_lod_level = chunk.current_lod_level;
		const uint32_t prev_target_lod_level = chunk.target_lod_level;
		const int prev_sh_band_level = chunk.sh_band_level;

		// Distance is already calculated in _update_chunk_visibility, no need to recalculate
		float distance = chunk.distance;

		// Calculate LOD level using Octree-GS formula
		int lod_level = lod_config.calculate_lod_level(distance);
		if (prev_lod_level != uint32_t(lod_level)) {
			lod_transitions_this_frame++;
		}

		// Update target LOD level (for smooth transitions)
		chunk.target_lod_level = lod_level;

		// For now, immediately apply target LOD (smooth transitions handled by LODGE blend)
		chunk.current_lod_level = lod_level;
		if (prev_lod_level != chunk.current_lod_level ||
				prev_target_lod_level != chunk.target_lod_level) {
			lod_transitions_this_frame++;
		}

		// Calculate LOD reduction parameters
		chunk.sh_band_level = lod_config.get_sh_band_for_lod(lod_level);
		chunk.splat_skip_factor = lod_config.get_splat_skip_factor(lod_level);
		chunk.opacity_multiplier = lod_config.get_opacity_multiplier(distance);

		// Calculate effective splat count after LOD reduction
		chunk.effective_count = chunk.count / chunk.splat_skip_factor;
		if (chunk.effective_count == 0 && chunk.count > 0) {
			chunk.effective_count = 1; // Always render at least 1 splat
		}

		mark_resident_chunk_meta_if_lod_changed(i, chunk, prev_effective_count, prev_lod_level, prev_sh_band_level);
	}
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
