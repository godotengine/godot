/**************************************************************************/
/*  test_config_validation.h                                              */
/*  Configuration validation unit tests for Gaussian Splatting module     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "tests/test_macros.h"
#include "../renderer/gpu_sorting_config.h"
#include "../renderer/pipeline_feature_set.h"
#include "../renderer/sorting_config.h"
#include "../renderer/gpu_sorter.h"
#include "../core/gaussian_splat_quality_config.h"
#include "../interfaces/gpu_culler.h"
#include "../lod/lod_config.h"

#include <limits>

namespace TestConfigValidation {

// =============================================================================
// GPUSortingConfig Validation Tests
// =============================================================================

TEST_CASE("[GaussianSplatting][Config] GPUSortingConfig default values pass validation") {
	GPUSortingConfig config;
	config.reset_to_defaults();

	CHECK(config.validate());
	CHECK(config.get_validation_errors().is_empty());
}

TEST_CASE("[GaussianSplatting][Config] GPUSortingConfig rejects invalid target_sort_time_ms") {
	GPUSortingConfig config;
	config.reset_to_defaults();

	SUBCASE("Zero target time is invalid") {
		config.target_sort_time_ms = 0.0f;
		CHECK_FALSE(config.validate());
		CHECK(config.get_validation_errors().contains("Target sort time must be > 0.1ms"));
	}

	SUBCASE("Negative target time is invalid") {
		config.target_sort_time_ms = -1.0f;
		CHECK_FALSE(config.validate());
		CHECK(config.get_validation_errors().contains("Target sort time must be > 0.1ms"));
	}

	SUBCASE("Exactly 0.1ms is invalid (must be greater than)") {
		config.target_sort_time_ms = 0.1f;
		CHECK_FALSE(config.validate());
	}

	SUBCASE("Just above threshold is valid") {
		config.target_sort_time_ms = 0.11f;
		CHECK(config.validate());
	}
}

TEST_CASE("[GaussianSplatting][Config] GPUSortingConfig rejects invalid max_sort_elements") {
	GPUSortingConfig config;
	config.reset_to_defaults();

	SUBCASE("Zero elements is invalid") {
		config.max_sort_elements = 0;
		CHECK_FALSE(config.validate());
		CHECK(config.get_validation_errors().contains("Max sort elements must be > 1000"));
	}

	SUBCASE("1000 elements is invalid (must be greater than)") {
		config.max_sort_elements = 1000;
		CHECK_FALSE(config.validate());
	}

	SUBCASE("1001 elements is valid") {
		config.max_sort_elements = 1001;
		CHECK(config.validate());
	}

	SUBCASE("Large element count is valid") {
		config.max_sort_elements = 100000000;
		CHECK(config.validate());
	}
}

TEST_CASE("[GaussianSplatting][Config] GPUSortingConfig rejects invalid radix_bits") {
	GPUSortingConfig config;
	config.reset_to_defaults();

	SUBCASE("4-bit radix is valid") {
		config.radix_bits = 4;
		CHECK(config.validate());
	}

	SUBCASE("8-bit radix is valid") {
		config.radix_bits = 8;
		CHECK(config.validate());
	}

	SUBCASE("Other radix values are invalid") {
		uint32_t invalid_values[] = {0, 1, 2, 3, 5, 6, 7, 9, 16, 32};
		for (uint32_t value : invalid_values) {
			config.radix_bits = value;
			CHECK_FALSE(config.validate());
			CHECK(config.get_validation_errors().contains("Radix bits must be 4 or 8"));
		}
	}
}

TEST_CASE("[GaussianSplatting][Config] GPUSortingConfig rejects invalid workgroup_size") {
	GPUSortingConfig config;
	config.reset_to_defaults();

	SUBCASE("Valid workgroup sizes") {
		uint32_t valid_sizes[] = {64, 128, 256, 512};
		for (uint32_t size : valid_sizes) {
			config.workgroup_size = size;
			CHECK(config.validate());
		}
	}

	SUBCASE("Invalid workgroup sizes") {
		uint32_t invalid_sizes[] = {0, 1, 32, 63, 65, 100, 255, 257, 1024};
		for (uint32_t size : invalid_sizes) {
			config.workgroup_size = size;
			CHECK_FALSE(config.validate());
			CHECK(config.get_validation_errors().contains("Workgroup size must be 64, 128, 256, or 512"));
		}
	}
}

TEST_CASE("[GaussianSplatting][Config] GPUSortingConfig rejects invalid key_bits") {
	GPUSortingConfig config;
	config.reset_to_defaults();

	SUBCASE("32-bit keys are valid") {
		config.key_bits = 32;
		config.tile_bits = 16;
		config.depth_bits = 16;
		CHECK(config.validate());
	}

	SUBCASE("64-bit keys are valid") {
		config.key_bits = 64;
		CHECK(config.validate());
	}

	SUBCASE("Other key widths are invalid") {
		uint32_t invalid_widths[] = {0, 8, 16, 24, 48, 128};
		for (uint32_t width : invalid_widths) {
			config.key_bits = width;
			// Ensure tile+depth fits for valid test
			config.tile_bits = 1;
			config.depth_bits = 1;
			CHECK_FALSE(config.validate());
			CHECK(config.get_validation_errors().contains("Key bits must be 32 or 64"));
		}
	}
}

TEST_CASE("[GaussianSplatting][Config] GPUSortingConfig validates tile/depth bit allocation") {
	GPUSortingConfig config;
	config.reset_to_defaults();

	SUBCASE("Tile and depth bits must not exceed key_bits") {
		config.key_bits = 32;
		config.tile_bits = 20;
		config.depth_bits = 20; // 40 > 32
		CHECK_FALSE(config.validate());
		CHECK(config.get_validation_errors().contains("Tile/depth bit split must fit within key_bits"));
	}

	SUBCASE("Tile and depth bits must allocate at least one bit") {
		config.tile_bits = 0;
		config.depth_bits = 0;
		CHECK_FALSE(config.validate());
		CHECK(config.get_validation_errors().contains("Tile/depth bit split must allocate at least one bit"));
	}

	SUBCASE("Valid 32-bit allocation") {
		config.key_bits = 32;
		config.tile_bits = 16;
		config.depth_bits = 16;
		CHECK(config.validate());
	}

	SUBCASE("Valid 64-bit allocation") {
		config.key_bits = 64;
		config.tile_bits = 32;
		config.depth_bits = 32;
		CHECK(config.validate());
	}

	SUBCASE("Partial allocation is valid") {
		config.key_bits = 64;
		config.tile_bits = 24;
		config.depth_bits = 24; // Only 48 bits used of 64
		CHECK(config.validate());
	}
}

TEST_CASE("[GaussianSplatting][Config] GPUSortingConfig rejects invalid performance_log_interval") {
	GPUSortingConfig config;
	config.reset_to_defaults();

	SUBCASE("Zero interval is invalid") {
		config.performance_log_interval = 0;
		CHECK_FALSE(config.validate());
		CHECK(config.get_validation_errors().contains("Performance log interval must be > 0"));
	}

	SUBCASE("Positive interval is valid") {
		config.performance_log_interval = 1;
		CHECK(config.validate());

		config.performance_log_interval = 1000;
		CHECK(config.validate());
	}
}

TEST_CASE("[GaussianSplatting][Config] GPUSortingConfig accumulates multiple errors") {
	GPUSortingConfig config;
	config.reset_to_defaults();

	// Set multiple invalid values
	config.target_sort_time_ms = 0.0f;
	config.max_sort_elements = 0;
	config.radix_bits = 3;
	config.workgroup_size = 100;
	config.key_bits = 16;
	config.performance_log_interval = 0;

	CHECK_FALSE(config.validate());

	String errors = config.get_validation_errors();
	CHECK(errors.contains("Target sort time must be > 0.1ms"));
	CHECK(errors.contains("Max sort elements must be > 1000"));
	CHECK(errors.contains("Radix bits must be 4 or 8"));
	CHECK(errors.contains("Workgroup size must be 64, 128, 256, or 512"));
	CHECK(errors.contains("Key bits must be 32 or 64"));
	CHECK(errors.contains("Performance log interval must be > 0"));
}

TEST_CASE("[GaussianSplatting][Config] PipelineFeatureSet default values pass validation") {
	PipelineFeatureSet config;
	config.reset_to_defaults();

	CHECK(config.validate());
	CHECK(config.get_validation_errors().is_empty());
}

TEST_CASE("[GaussianSplatting][Config] PipelineFeatureSet validates SH amortization settings only when active") {
	PipelineFeatureSet config;
	config.reset_to_defaults();

	SUBCASE("Inactive SH amortization tolerates stale divisor values") {
		config.sh_amortization_divisor = 0;
		CHECK(config.validate());
	}

	SUBCASE("Divisor must be greater than one when the feature is active") {
		config.enable_sh_amortization = true;
		config.sh_amortization_divisor = 1;
		CHECK_FALSE(config.validate());
		CHECK(config.get_validation_errors().contains("SH amortization divisor must be > 1."));
	}

	SUBCASE("Visibility threshold must be finite") {
		config.enable_sh_amortization = true;
		config.sh_amortization_visibility_threshold = std::numeric_limits<float>::infinity();
		CHECK_FALSE(config.validate());
		CHECK(config.get_validation_errors().contains("SH amortization visibility threshold must be finite."));
	}

	SUBCASE("Visibility threshold must stay within normalized range") {
		config.enable_sh_amortization = true;
		config.sh_amortization_visibility_threshold = 1.5f;
		CHECK_FALSE(config.validate());
		CHECK(config.get_validation_errors().contains("SH amortization visibility threshold must be <= 1."));
	}

	SUBCASE("Disabled visibility invalidation ignores the threshold value") {
		config.enable_sh_amortization = true;
		config.disable_sh_amortization_on_visibility_change = false;
		config.sh_amortization_divisor = 2;
		config.sh_amortization_visibility_threshold = 1.5f;
		CHECK(config.validate());
	}

	SUBCASE("Normalized threshold is accepted") {
		config.enable_sh_amortization = true;
		config.sh_amortization_divisor = 4;
		config.sh_amortization_visibility_threshold = 0.5f;
		CHECK(config.validate());
	}

	SUBCASE("Experimental bundle inherits SH amortization validation") {
		config.enable_all_experimental = true;
		config.sh_amortization_divisor = 1;
		CHECK_FALSE(config.validate());
		CHECK(config.get_validation_errors().contains("SH amortization divisor must be > 1."));
	}
}

TEST_CASE("[GaussianSplatting][Config] PipelineFeatureSet validates packed stage limits when scene size is known") {
	PipelineFeatureSet config;
	config.reset_to_defaults();

	SUBCASE("Packed stage accepts unknown scene size") {
		config.enable_packed_stage_data = true;
		CHECK(config.validate());
	}

	SUBCASE("Packed stage accepts scenes within the 16-bit index budget") {
		config.enable_packed_stage_data = true;
		CHECK(config.validate(PipelineFeatureSet::PACKED_STAGE_MAX_TOTAL_SPLATS));
	}

	SUBCASE("Packed stage rejects oversized scenes") {
		config.enable_packed_stage_data = true;
		CHECK_FALSE(config.validate(PipelineFeatureSet::PACKED_STAGE_MAX_TOTAL_SPLATS + 1));
		CHECK(config.get_validation_errors(PipelineFeatureSet::PACKED_STAGE_MAX_TOTAL_SPLATS + 1)
				.contains("Packed stage data requires <="));
	}

	SUBCASE("Experimental bundle inherits packed stage limits") {
		config.enable_all_experimental = true;
		CHECK_FALSE(config.validate(PipelineFeatureSet::PACKED_STAGE_MAX_TOTAL_SPLATS + 1));
		CHECK(config.get_validation_errors(PipelineFeatureSet::PACKED_STAGE_MAX_TOTAL_SPLATS + 1)
				.contains("Packed stage data requires <="));
	}
}

// =============================================================================
// SortingStrategyConfig Sanitize Tests
// =============================================================================

TEST_CASE("[GaussianSplatting][Config] SortingStrategyConfig sanitize corrects invalid values") {
	SortingStrategyConfig config;

	SUBCASE("Zero bitonic_max_elements corrected to 1") {
		config.bitonic_max_elements = 0;
		config.sanitize();
		CHECK(config.bitonic_max_elements == 1);
	}

	SUBCASE("radix_max_elements enforced >= bitonic_max_elements") {
		config.bitonic_max_elements = 10000;
		config.radix_max_elements = 5000; // Less than bitonic
		config.sanitize();
		CHECK(config.radix_max_elements >= config.bitonic_max_elements);
	}

	SUBCASE("onesweep_max_elements enforced >= radix_max_elements") {
		config.radix_max_elements = 100000;
		config.onesweep_max_elements = 50000; // Less than radix
		config.sanitize();
		CHECK(config.onesweep_max_elements >= config.radix_max_elements);
	}

	SUBCASE("hybrid_trigger_elements enforced >= radix_max_elements") {
		config.radix_max_elements = 100000;
		config.hybrid_trigger_elements = 50000;
		config.sanitize();
		CHECK(config.hybrid_trigger_elements >= config.radix_max_elements);
	}

	SUBCASE("Zero hybrid_batch_size defaults to radix_max_elements") {
		config.radix_max_elements = 100000;
		config.hybrid_batch_size = 0;
		config.sanitize();
		CHECK(config.hybrid_batch_size == config.radix_max_elements);
	}

	SUBCASE("Zero history_size defaults to 120") {
		config.history_size = 0;
		config.sanitize();
		CHECK(config.history_size == 120);
	}

	SUBCASE("Zero log_interval_frames defaults to 60") {
		config.log_interval_frames = 0;
		config.sanitize();
		CHECK(config.log_interval_frames == 60);
	}

	SUBCASE("Negative target_sort_time_ms clamped to 0") {
		config.target_sort_time_ms = -5.0f;
		config.sanitize();
		CHECK(config.target_sort_time_ms == 0.0f);
	}
}

TEST_CASE("[GaussianSplatting][Config] SortingStrategyConfig describe_thresholds format") {
	SortingStrategyConfig config;
	config.bitonic_max_elements = 131072;
	config.radix_max_elements = 1500000;
	config.onesweep_max_elements = 3000000;
	config.hybrid_trigger_elements = 3000000;
	config.hybrid_batch_size = 1500000;

	String description = config.describe_thresholds();

	CHECK(description.contains("131072"));
	CHECK(description.contains("1500000"));
	CHECK(description.contains("3000000"));
}

// =============================================================================
// SortKeyConfig Tests
// =============================================================================

TEST_CASE("[GaussianSplatting][Config] SortKeyConfig default values") {
	SortKeyConfig config;

	CHECK(config.key_bits == 64);
	CHECK(config.tile_bits == 32);
	CHECK(config.depth_bits == 32);
	CHECK(config.enable_tie_breaker == false);
}

TEST_CASE("[GaussianSplatting][Config] SortKeyConfig bit allocation consistency") {
	SortKeyConfig config;

	SUBCASE("Default allocation fits in key") {
		CHECK(config.tile_bits + config.depth_bits <= config.key_bits);
	}

	SUBCASE("32-bit key allocation") {
		config.key_bits = 32;
		config.tile_bits = 16;
		config.depth_bits = 16;
		CHECK(config.tile_bits + config.depth_bits == config.key_bits);
	}
}

// =============================================================================
// Live LODConfig validation
// =============================================================================

TEST_CASE("[GaussianSplatting][Config] LODConfig calculate_lod_level handles disabled and near-zero distances") {
	LODConfig config;
	config.reset_to_defaults();
	config.enabled = false;

	CHECK(config.calculate_lod_level(0.0f) == 0);
	CHECK(config.calculate_lod_level(25.0f) == 0);
	CHECK(config.calculate_lod_level(100.0f) == 0);

	config.enabled = true;

	CHECK(config.calculate_lod_level(0.0f) == 0);
	CHECK(config.calculate_lod_level(0.0001f) == 0);
	CHECK(config.calculate_lod_level(0.001f) == 0);
	CHECK(config.calculate_lod_level(12.5f) == 0);
}

TEST_CASE("[GaussianSplatting][Config] LODConfig calculate_lod_level boundary mapping is explicit") {
	LODConfig config;
	config.reset_to_defaults();
	config.enabled = true;
	config.num_levels = 4;
	config.max_distance = 100.0f;
	config.base_threshold = 10.0f;

	CHECK(config.calculate_lod_level(24.9999f) == 0);
	CHECK_MESSAGE(config.calculate_lod_level(25.0f) == 1,
			"Exact 25.0 enters LOD 1 because calculate_lod_level() is driven by max_distance/num_levels.");
	CHECK(config.calculate_lod_level(25.0001f) == 1);
	CHECK(config.calculate_lod_level(49.9999f) == 1);
	CHECK_MESSAGE(config.calculate_lod_level(50.0f) == 2,
			"Exact 50.0 enters LOD 2 under the current live floor/clamp mapping.");
	CHECK(config.calculate_lod_level(50.0001f) == 2);
	CHECK(config.calculate_lod_level(99.9999f) == 2);
	CHECK_MESSAGE(config.calculate_lod_level(100.0f) == 3,
			"Exact max_distance lands on the farthest LOD level in the live implementation.");
	CHECK(config.calculate_lod_level(100.0001f) == 3);
	CHECK(config.calculate_lod_level(1000.0f) == 3);
}

TEST_CASE("[GaussianSplatting][Config] LODConfig distance thresholds follow base-threshold progression") {
	LODConfig config;
	config.reset_to_defaults();
	config.base_threshold = 10.0f;
	config.max_distance = 100.0f;

	CHECK(config.get_distance_threshold(-1) == doctest::Approx(10.0f));
	CHECK(config.get_distance_threshold(0) == doctest::Approx(10.0f));
	CHECK(config.get_distance_threshold(1) == doctest::Approx(20.0f));
	CHECK(config.get_distance_threshold(2) == doctest::Approx(40.0f));
	CHECK(config.get_distance_threshold(3) == doctest::Approx(80.0f));
	CHECK_MESSAGE(config.get_distance_threshold(4) == doctest::Approx(100.0f),
			"Distance thresholds double from base_threshold and clamp at max_distance.");
}

TEST_CASE("[GaussianSplatting][Config] LODConfig helper mappings match the live implementation") {
	LODConfig config;
	config.reset_to_defaults();
	config.base_threshold = 10.0f;
	config.max_distance = 100.0f;

	CHECK(config.get_splat_skip_factor(0) == 1);
	CHECK(config.get_splat_skip_factor(1) == 2);
	CHECK(config.get_splat_skip_factor(2) == 4);
	CHECK(config.get_splat_skip_factor(3) == 8);

	config.splat_skip_enabled = false;
	CHECK(config.get_splat_skip_factor(3) == 1);
	config.splat_skip_enabled = true;

	CHECK(config.get_sh_band_for_lod(0) == 3);
	CHECK(config.get_sh_band_for_lod(1) == 2);
	CHECK(config.get_sh_band_for_lod(2) == 1);
	CHECK(config.get_sh_band_for_lod(3) == 0);
	CHECK(config.get_sh_band_for_lod(4) == 0);

	config.sh_reduction_enabled = false;
	CHECK(config.get_sh_band_for_lod(3) == 3);
	config.sh_reduction_enabled = true;

	CHECK(config.get_opacity_multiplier(0.0f) == doctest::Approx(1.0f));
	CHECK(config.get_opacity_multiplier(10.0f) == doctest::Approx(1.0f));
	CHECK(config.get_opacity_multiplier(55.0f) == doctest::Approx(0.5f));
	CHECK(config.get_opacity_multiplier(100.0f) == doctest::Approx(0.0f));
	CHECK(config.get_opacity_multiplier(120.0f) == doctest::Approx(0.0f));

	config.opacity_fade_enabled = false;
	CHECK(config.get_opacity_multiplier(55.0f) == doctest::Approx(1.0f));
}

// =============================================================================
// Node-facing LOD/Streaming config validation
// =============================================================================

TEST_CASE("[GaussianSplatting][Config] GaussianSplatLODConfig defaults match live node expectations") {
	using namespace GaussianSplatting;
	GaussianSplatLODConfig config;

	CHECK(config.lod0_distance < config.lod1_distance);
	CHECK(config.lod1_distance < config.lod2_distance);
	CHECK(config.lod2_distance < config.lod3_distance);
	CHECK(config.lod3_distance < config.cull_distance);
	CHECK(config.min_splats_per_frame < config.max_splats_per_frame);
	CHECK(config.importance_threshold >= 0.0f);
	CHECK(config.importance_threshold <= 1.0f);
	CHECK(config.size_cull_threshold > 0.0f);
	CHECK(config.lod_bias > 0.0f);
}

TEST_CASE("[GaussianSplatting][Config] GaussianSplatStreamingConfig defaults match live node expectations") {
	using namespace GaussianSplatting;
	GaussianSplatStreamingConfig config;

	CHECK(config.max_gpu_memory > 0);
	CHECK(config.target_gpu_memory > 0);
	CHECK(config.target_gpu_memory <= config.max_gpu_memory);
	CHECK(config.max_cpu_memory >= config.max_gpu_memory);
	CHECK(config.load_ahead_distance > 0.0f);
	CHECK(config.unload_distance > config.load_ahead_distance);
	CHECK(config.max_concurrent_loads > 0);
	CHECK(config.num_lod_levels >= 2);
	CHECK(config.stream_budget_ms > 0);
}

// =============================================================================
// CullingConfig Validation Tests (GPUCuller::CullingConfig)
// =============================================================================

TEST_CASE("[GaussianSplatting][Config] CullingConfig default values are sensible") {
	GPUCuller::CullingConfig config;

	// Boolean defaults
	CHECK(config.lod_enabled == true);
	CHECK(config.frustum_culling == true);
	CHECK(config.gpu_culling_enabled == true);
	CHECK(config.temporal_coherence == true);

	// Numeric defaults should be positive where expected
	CHECK(config.lod_bias > 0.0f);
	CHECK(config.lod_min_screen_size > 0.0f);
	CHECK(config.lod_max_distance > 0.0f);
	CHECK(config.cull_radius_multiplier > 0.0f);
	CHECK(config.cull_frustum_plane_slack >= 0.0f);
}

TEST_CASE("[GaussianSplatting][Config] CullingConfig tolerance values") {
	GPUCuller::CullingConfig config;

	// Tolerances should be small positive values
	CHECK(config.cull_near_tolerance >= 0.0f);
	CHECK(config.cull_near_tolerance <= 1.0f);
	CHECK(config.cull_far_tolerance >= 0.0f);
	CHECK(config.cull_far_tolerance <= 1.0f);
}

TEST_CASE("[GaussianSplatting][Config] CullingConfig viewport size") {
	GPUCuller::CullingConfig config;

	// Default viewport size should be valid
	CHECK(config.last_cull_viewport_size.x > 0);
	CHECK(config.last_cull_viewport_size.y > 0);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_CASE("[GaussianSplatting][Config] GPUSortingConfig edge case: maximum valid values") {
	GPUSortingConfig config;
	config.reset_to_defaults();

	// Test maximum reasonable values
	config.target_sort_time_ms = 1000.0f; // 1 second
	config.max_sort_elements = UINT32_MAX;
	config.performance_log_interval = UINT32_MAX;

	CHECK(config.validate());
}

TEST_CASE("[GaussianSplatting][Config] GPUSortingConfig edge case: minimum valid values") {
	GPUSortingConfig config;
	config.reset_to_defaults();

	// Test minimum valid values
	config.target_sort_time_ms = 0.11f; // Just above 0.1
	config.max_sort_elements = 1001; // Just above 1000
	config.radix_bits = 4;
	config.workgroup_size = 64; // Smallest valid
	config.key_bits = 32; // Smallest valid
	config.tile_bits = 1;
	config.depth_bits = 1;
	config.performance_log_interval = 1;

	CHECK(config.validate());
}

TEST_CASE("[GaussianSplatting][Config] SortingStrategyConfig cascading sanitization") {
	SortingStrategyConfig config;

	// Set unreasonable ordering that should be corrected
	config.bitonic_max_elements = 1000000; // Very large bitonic
	config.radix_max_elements = 100;       // Small radix
	config.onesweep_max_elements = 50;     // Tiny onesweep

	config.sanitize();

	// After sanitization, ordering should be enforced
	CHECK(config.radix_max_elements >= config.bitonic_max_elements);
	CHECK(config.onesweep_max_elements >= config.radix_max_elements);
}

} // namespace TestConfigValidation
