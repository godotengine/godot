/**************************************************************************/
/*  test_synthetic_clustered_generator.h                                 */
/**************************************************************************/

#pragma once

#include "synthetic_splat_generators.h"
#include "tests/test_macros.h"

namespace TestGaussianSplatting {

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

TEST_CASE("[GaussianSplatting][Synthetic] Clustered surface_aligned orients splats outward") {
	ClusteredSplatGenerator::Config config;
	config.splat_count = 300;
	config.seed = 200;
	config.cluster_count = 1;
	config.cluster_center_min = Vector3(0, 0, 0);
	config.cluster_center_max = Vector3(0, 0, 0);
	config.cluster_radius = 5.0f;
	config.surface_aligned = true;

	const LocalVector<Gaussian> splats = ClusteredSplatGenerator::generate(config);
	uint32_t correctly_oriented = 0;
	for (const Gaussian &g : splats) {
		const Vector3 dir = g.position.normalized();
		if (dir.dot(g.normal) > 0.5f) {
			correctly_oriented++;
		}
	}
	CHECK(correctly_oriented > config.splat_count * 0.9f);
}

TEST_CASE("[GaussianSplatting][Synthetic] Clustered cluster_flatten creates disc-shaped clusters") {
	ClusteredSplatGenerator::Config config;
	config.splat_count = 1000;
	config.seed = 300;
	config.cluster_count = 1;
	config.cluster_center_min = Vector3(0, 0, 0);
	config.cluster_center_max = Vector3(0, 0, 0);
	config.cluster_radius = 5.0f;
	config.cluster_flatten = 0.9f;

	const LocalVector<Gaussian> splats = ClusteredSplatGenerator::generate(config);
	float y_range = 0.0f;
	float xz_range = 0.0f;
	for (const Gaussian &g : splats) {
		y_range = MAX(y_range, Math::abs(g.position.y));
		xz_range = MAX(xz_range, MAX(Math::abs(g.position.x), Math::abs(g.position.z)));
	}
	CHECK(y_range < xz_range * 0.3f);
}

} // namespace TestGaussianSplatting
