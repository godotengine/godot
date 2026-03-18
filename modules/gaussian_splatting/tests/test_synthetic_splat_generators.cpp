/**************************************************************************/
/*  test_synthetic_splat_generators.cpp                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#include "synthetic_splat_generators.h"

#include "tests/test_macros.h"

namespace TestGaussianSplatting {

TEST_CASE("[GaussianSplatting][Synthetic] Uniform synthetic generator stays deterministic") {
	UniformSplatGenerator::Config config;
	config.splat_count = 128;
	config.seed = 42;
	config.position_min = Vector3(-3.0f, -2.0f, -1.0f);
	config.position_max = Vector3(3.0f, 2.0f, 1.0f);
	config.min_scale = 0.1f;
	config.max_scale = 0.4f;
	config.min_opacity = 0.3f;
	config.max_opacity = 0.9f;
	config.normal_tilt = 0.2f;
	config.random_rotation = true;
	config.random_colors = true;

	SyntheticSceneSummary summary_a;
	SyntheticSceneSummary summary_b;
	const LocalVector<Gaussian> splats_a = UniformSplatGenerator::generate(config, &summary_a);
	const LocalVector<Gaussian> splats_b = UniformSplatGenerator::generate(config, &summary_b);

	CHECK_EQ(splats_a.size(), static_cast<int>(config.splat_count));
	CHECK_EQ(splats_b.size(), static_cast<int>(config.splat_count));
	CHECK_EQ(summary_a.seed, config.seed);
	CHECK_EQ(summary_a.config_hash, summary_b.config_hash);
	CHECK_EQ(summary_a.scene_hash, summary_b.scene_hash);
	CHECK(summary_a.scene_hash != 0);
	CHECK(summary_a.average_scale > 0.0f);
	CHECK(summary_a.average_opacity > 0.0f);

	Dictionary summary_dict = summary_a.to_dict();
	CHECK(summary_dict.has("seed"));
	CHECK(summary_dict.has("config_hash"));
	CHECK(summary_dict.has("scene_hash"));

	UniformSplatGenerator::Config swapped_ranges = config;
	swapped_ranges.position_min = config.position_max;
	swapped_ranges.position_max = config.position_min;
	swapped_ranges.min_scale = config.max_scale;
	swapped_ranges.max_scale = config.min_scale;
	swapped_ranges.min_opacity = config.max_opacity;
	swapped_ranges.max_opacity = config.min_opacity;

	SyntheticSceneSummary swapped_summary;
	UniformSplatGenerator::generate(swapped_ranges, &swapped_summary);
	CHECK_EQ(summary_a.config_hash, swapped_summary.config_hash);
	CHECK_EQ(summary_a.scene_hash, swapped_summary.scene_hash);

	UniformSplatGenerator::Config new_seed = config;
	new_seed.seed += 17;
	SyntheticSceneSummary seeded_summary;
	UniformSplatGenerator::generate(new_seed, &seeded_summary);
	CHECK(summary_a.scene_hash != seeded_summary.scene_hash);

	UniformSplatGenerator::Config roundtrip;
	roundtrip.from_dict(config.to_dict());
	CHECK_EQ(UniformSplatGenerator::hash_config(config), UniformSplatGenerator::hash_config(roundtrip));
}

TEST_CASE("[GaussianSplatting][Synthetic] Config serialization preserves full uint64 seeds") {
	UniformSplatGenerator::Config uniform_config;
	uniform_config.seed = 0xFEDCBA9876543210ull;
	UniformSplatGenerator::Config uniform_roundtrip;
	uniform_roundtrip.seed = 1;
	uniform_roundtrip.from_dict(uniform_config.to_dict());
	CHECK_EQ(uniform_roundtrip.seed, uniform_config.seed);

	ClusteredSplatGenerator::Config clustered_config;
	clustered_config.seed = 0x876543210FEDCBA9ull;
	ClusteredSplatGenerator::Config clustered_roundtrip;
	clustered_roundtrip.seed = 1;
	clustered_roundtrip.from_dict(clustered_config.to_dict());
	CHECK_EQ(clustered_roundtrip.seed, clustered_config.seed);
}

TEST_CASE("[GaussianSplatting][Synthetic] Scene hash captures area and rotation deltas") {
	LocalVector<Gaussian> base;
	base.resize(1);
	Gaussian gaussian = {};
	gaussian.position = Vector3(1.0f, -2.0f, 3.0f);
	gaussian.scale = Vector3(0.4f, 0.4f, 0.4f);
	gaussian.area = 0.7f;
	gaussian.opacity = 0.8f;
	gaussian.rotation = Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
	gaussian.sh_dc = Color(0.5f, 0.6f, 0.7f, 0.8f);
	gaussian.normal = Vector3(0.0f, 1.0f, 0.0f);
	base[0] = gaussian;

	const uint64_t seed = 33;
	const uint64_t config_hash = 99;
	const SyntheticSceneSummary summary_base = summarize_generated_scene(base, "uniform", seed, config_hash);
	CHECK(summary_base.scene_hash != 0);

	LocalVector<Gaussian> area_variant = base;
	area_variant[0].area += 0.1f;
	const SyntheticSceneSummary summary_area = summarize_generated_scene(area_variant, "uniform", seed, config_hash);
	CHECK(summary_base.scene_hash != summary_area.scene_hash);

	LocalVector<Gaussian> rotation_variant = base;
	rotation_variant[0].rotation = Quaternion(Vector3(0.0f, 1.0f, 0.0f), 0.35f);
	const SyntheticSceneSummary summary_rotation = summarize_generated_scene(rotation_variant, "uniform", seed, config_hash);
	CHECK(summary_base.scene_hash != summary_rotation.scene_hash);
}

TEST_CASE("[GaussianSplatting][Synthetic] Clustered synthetic generator stays deterministic") {
	ClusteredSplatGenerator::Config config;
	config.splat_count = 160;
	config.seed = 84;
	config.cluster_count = 8;
	config.cluster_center_min = Vector3(-10.0f, -5.0f, -2.0f);
	config.cluster_center_max = Vector3(10.0f, 5.0f, 2.0f);
	config.center_offset = Vector3(0.5f, 0.0f, 0.0f);
	config.cluster_radius = 1.5f;
	config.min_scale = 0.05f;
	config.max_scale = 0.25f;
	config.min_opacity = 0.5f;
	config.max_opacity = 1.0f;
	config.normal_tilt = 0.0f;
	config.random_rotation = false;
	config.color_per_cluster = true;

	SyntheticSceneSummary summary_a;
	SyntheticSceneSummary summary_b;
	const LocalVector<Gaussian> splats_a = ClusteredSplatGenerator::generate(config, &summary_a);
	const LocalVector<Gaussian> splats_b = ClusteredSplatGenerator::generate(config, &summary_b);

	CHECK_EQ(splats_a.size(), static_cast<int>(config.splat_count));
	CHECK_EQ(splats_b.size(), static_cast<int>(config.splat_count));
	CHECK_EQ(summary_a.seed, config.seed);
	CHECK_EQ(summary_a.config_hash, summary_b.config_hash);
	CHECK_EQ(summary_a.scene_hash, summary_b.scene_hash);
	CHECK(summary_a.scene_hash != 0);
	CHECK(summary_a.average_scale > 0.0f);
	CHECK(summary_a.average_opacity > 0.0f);

	ClusteredSplatGenerator::Config swapped_ranges = config;
	swapped_ranges.cluster_center_min = config.cluster_center_max;
	swapped_ranges.cluster_center_max = config.cluster_center_min;
	swapped_ranges.min_scale = config.max_scale;
	swapped_ranges.max_scale = config.min_scale;
	swapped_ranges.min_opacity = config.max_opacity;
	swapped_ranges.max_opacity = config.min_opacity;

	SyntheticSceneSummary swapped_summary;
	ClusteredSplatGenerator::generate(swapped_ranges, &swapped_summary);
	CHECK_EQ(summary_a.config_hash, swapped_summary.config_hash);
	CHECK_EQ(summary_a.scene_hash, swapped_summary.scene_hash);

	ClusteredSplatGenerator::Config new_seed = config;
	new_seed.seed += 1;
	SyntheticSceneSummary seeded_summary;
	ClusteredSplatGenerator::generate(new_seed, &seeded_summary);
	CHECK(summary_a.scene_hash != seeded_summary.scene_hash);

	ClusteredSplatGenerator::Config roundtrip;
	roundtrip.from_dict(config.to_dict());
	CHECK_EQ(ClusteredSplatGenerator::hash_config(config), ClusteredSplatGenerator::hash_config(roundtrip));
}

} // namespace TestGaussianSplatting
