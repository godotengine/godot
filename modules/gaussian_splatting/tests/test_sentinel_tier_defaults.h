/**
 * @file test_sentinel_tier_defaults.h
 * @brief Regression tests for sentinel-based quality tier defaults.
 *
 * Four project settings (sh_bands, lod_max_distance, lod_base_threshold,
 * per_chunk_quantization) now default to -1 (sentinel).  When the value is
 * -1, the system checks the active quality tier for a recommendation; if no
 * tier is active, it falls back to the hard-coded default.
 *
 * Test matrix per setting:
 *   1. sentinel (-1) + active tier   -> tier value
 *   2. explicit real-default + tier   -> explicit value wins
 *   3. sentinel (-1) + no tier        -> code default
 *
 * Plus backward-compat tests for quantization legacy bools.
 */

#pragma once

#include "tests/test_macros.h"

#include "gs_test_setting_guard.h"
#include "../core/quality_tier_config.h"
#include "../lod/lod_config.h"
#include "../renderer/quantization_config.h"
#include "../renderer/sh_config.h"

// ---------------------------------------------------------------------------
// SH bands
// ---------------------------------------------------------------------------

TEST_CASE("[GaussianSplatting][Config] SH bands: sentinel + steam_deck tier -> tier value") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_bands(ps, SHConfig::BANDS_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "steam_deck");
	ps->set_setting(SHConfig::BANDS_PATH, -1);

	SHConfig config;
	config.load_from_project_settings();

	CHECK(config.sh_bands == SH_BAND_1);
}

TEST_CASE("[GaussianSplatting][Config] SH bands: explicit code-default + tier -> explicit wins") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_bands(ps, SHConfig::BANDS_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "steam_deck");
	ps->set_setting(SHConfig::BANDS_PATH, static_cast<int>(SH_BAND_3));

	SHConfig config;
	config.load_from_project_settings();

	CHECK(config.sh_bands == SH_BAND_3);
}

TEST_CASE("[GaussianSplatting][Config] SH bands: sentinel + no tier -> code default") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_bands(ps, SHConfig::BANDS_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "custom");
	ps->set_setting(SHConfig::BANDS_PATH, -1);

	SHConfig config;
	config.load_from_project_settings();

	CHECK(config.sh_bands == SH_BAND_3);
}

// ---------------------------------------------------------------------------
// LOD max_distance
// ---------------------------------------------------------------------------

TEST_CASE("[GaussianSplatting][Config] LOD max_distance: sentinel + steam_deck tier -> tier value") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_max(ps, LODConfig::MAX_DISTANCE_PATH);
	ProjectSettingGuard guard_base(ps, LODConfig::BASE_THRESHOLD_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "steam_deck");
	ps->set_setting(LODConfig::MAX_DISTANCE_PATH, -1.0f);
	// Also set base_threshold sentinel so clamping doesn't interfere.
	ps->set_setting(LODConfig::BASE_THRESHOLD_PATH, -1.0f);

	LODConfig config;
	config.load_from_project_settings();

	CHECK(Math::is_equal_approx(config.max_distance, 60.0f));
}

TEST_CASE("[GaussianSplatting][Config] LOD max_distance: explicit code-default + tier -> explicit wins") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_max(ps, LODConfig::MAX_DISTANCE_PATH);
	ProjectSettingGuard guard_base(ps, LODConfig::BASE_THRESHOLD_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "steam_deck");
	ps->set_setting(LODConfig::MAX_DISTANCE_PATH, 100.0f);
	ps->set_setting(LODConfig::BASE_THRESHOLD_PATH, 10.0f);

	LODConfig config;
	config.load_from_project_settings();

	CHECK(Math::is_equal_approx(config.max_distance, 100.0f));
}

TEST_CASE("[GaussianSplatting][Config] LOD max_distance: sentinel + no tier -> code default") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_max(ps, LODConfig::MAX_DISTANCE_PATH);
	ProjectSettingGuard guard_base(ps, LODConfig::BASE_THRESHOLD_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "custom");
	ps->set_setting(LODConfig::MAX_DISTANCE_PATH, -1.0f);
	ps->set_setting(LODConfig::BASE_THRESHOLD_PATH, -1.0f);

	LODConfig config;
	config.load_from_project_settings();

	CHECK(Math::is_equal_approx(config.max_distance, 100.0f));
}

// ---------------------------------------------------------------------------
// LOD base_threshold
// ---------------------------------------------------------------------------

TEST_CASE("[GaussianSplatting][Config] LOD base_threshold: sentinel + steam_deck tier -> tier value") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_max(ps, LODConfig::MAX_DISTANCE_PATH);
	ProjectSettingGuard guard_base(ps, LODConfig::BASE_THRESHOLD_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "steam_deck");
	ps->set_setting(LODConfig::MAX_DISTANCE_PATH, -1.0f);
	ps->set_setting(LODConfig::BASE_THRESHOLD_PATH, -1.0f);

	LODConfig config;
	config.load_from_project_settings();

	CHECK(Math::is_equal_approx(config.base_threshold, 5.0f));
}

TEST_CASE("[GaussianSplatting][Config] LOD base_threshold: explicit code-default + tier -> explicit wins") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_max(ps, LODConfig::MAX_DISTANCE_PATH);
	ProjectSettingGuard guard_base(ps, LODConfig::BASE_THRESHOLD_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "steam_deck");
	ps->set_setting(LODConfig::MAX_DISTANCE_PATH, 100.0f);
	ps->set_setting(LODConfig::BASE_THRESHOLD_PATH, 10.0f);

	LODConfig config;
	config.load_from_project_settings();

	CHECK(Math::is_equal_approx(config.base_threshold, 10.0f));
}

TEST_CASE("[GaussianSplatting][Config] LOD base_threshold: sentinel + no tier -> code default") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_max(ps, LODConfig::MAX_DISTANCE_PATH);
	ProjectSettingGuard guard_base(ps, LODConfig::BASE_THRESHOLD_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "custom");
	ps->set_setting(LODConfig::MAX_DISTANCE_PATH, -1.0f);
	ps->set_setting(LODConfig::BASE_THRESHOLD_PATH, -1.0f);

	LODConfig config;
	config.load_from_project_settings();

	CHECK(Math::is_equal_approx(config.base_threshold, 10.0f));
}

// ---------------------------------------------------------------------------
// Quantization (per_chunk_quantization)
// ---------------------------------------------------------------------------

TEST_CASE("[GaussianSplatting][Config] quantization: sentinel + steam_deck tier -> tier value (enabled)") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_quant(ps, QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "steam_deck");
	ps->set_setting(QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH, -1);

	QuantizationConfig config;
	config.load_from_project_settings();

	CHECK(config.per_chunk_quantization == true);
}

TEST_CASE("[GaussianSplatting][Config] quantization: explicit disabled + tier -> explicit wins") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_quant(ps, QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "steam_deck");
	// Explicit 0 = disabled, overrides tier.
	ps->set_setting(QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH, 0);

	QuantizationConfig config;
	config.load_from_project_settings();

	CHECK(config.per_chunk_quantization == false);
}

TEST_CASE("[GaussianSplatting][Config] quantization: sentinel + no tier -> code default (disabled)") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_quant(ps, QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "custom");
	ps->set_setting(QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH, -1);

	QuantizationConfig config;
	config.load_from_project_settings();

	CHECK(config.per_chunk_quantization == false);
}

// ---------------------------------------------------------------------------
// Backward compatibility: legacy bool values for quantization
// ---------------------------------------------------------------------------

TEST_CASE("[GaussianSplatting][Config] quantization: legacy bool true coerced to explicit enabled") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_quant(ps, QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	// No tier active -- legacy bool should be treated as an explicit value.
	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "custom");
	// Variant(true) stores as BOOL type, simulating a legacy project file.
	ps->set_setting(QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH, Variant(true));

	QuantizationConfig config;
	config.load_from_project_settings();

	CHECK(config.per_chunk_quantization == true);
}

TEST_CASE("[GaussianSplatting][Config] quantization: legacy bool false coerced to explicit disabled") {
	ProjectSettings *ps = ProjectSettings::get_singleton();
	REQUIRE(ps != nullptr);

	ProjectSettingGuard guard_quant(ps, QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH);
	ProjectSettingGuard guard_tier(ps, "rendering/gaussian_splatting/quality/tier_preset");

	// Tier wants quantization enabled, but explicit false must win.
	ps->set_setting("rendering/gaussian_splatting/quality/tier_preset", "steam_deck");
	ps->set_setting(QuantizationConfig::PER_CHUNK_QUANTIZATION_PATH, Variant(false));

	QuantizationConfig config;
	config.load_from_project_settings();

	CHECK(config.per_chunk_quantization == false);
}
