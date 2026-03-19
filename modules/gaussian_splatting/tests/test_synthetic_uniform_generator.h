/**************************************************************************/
/*  test_synthetic_uniform_generator.h                                   */
/**************************************************************************/

#pragma once

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

TEST_CASE("[GaussianSplatting][Synthetic] Anisotropic scales produce non-spherical splats") {
	UniformSplatGenerator::Config config;
	config.splat_count = 500;
	config.seed = 77;
	config.anisotropy = 0.8f;

	const LocalVector<Gaussian> splats = UniformSplatGenerator::generate(config);
	uint32_t aniso_count = 0;
	for (const Gaussian &g : splats) {
		if (Math::abs(g.scale.x - g.scale.y) > CMP_EPSILON ||
				Math::abs(g.scale.y - g.scale.z) > CMP_EPSILON) {
			aniso_count++;
		}
	}
	CHECK(aniso_count > config.splat_count * 0.8f);
}

TEST_CASE("[GaussianSplatting][Synthetic] SH band-1 coefficients are generated when sh_intensity > 0") {
	UniformSplatGenerator::Config config;
	config.splat_count = 200;
	config.seed = 99;
	config.sh_intensity = 0.5f;

	const LocalVector<Gaussian> splats = UniformSplatGenerator::generate(config);
	uint32_t nonzero_sh = 0;
	for (const Gaussian &g : splats) {
		for (int band = 0; band < 3; band++) {
			if (g.sh_1[band].length_squared() > CMP_EPSILON2) {
				nonzero_sh++;
				break;
			}
		}
	}
	CHECK(nonzero_sh == config.splat_count);
}

TEST_CASE("[GaussianSplatting][Synthetic] SH band-1 stays zero when sh_intensity is zero") {
	UniformSplatGenerator::Config config;
	config.splat_count = 100;
	config.seed = 55;
	config.sh_intensity = 0.0f;

	const LocalVector<Gaussian> splats = UniformSplatGenerator::generate(config);
	for (const Gaussian &g : splats) {
		for (int band = 0; band < 3; band++) {
			CHECK(g.sh_1[band].length_squared() < CMP_EPSILON2);
		}
	}
}

TEST_CASE("[GaussianSplatting][Synthetic] Log-normal distribution skews toward small scales") {
	UniformSplatGenerator::Config config;
	config.splat_count = 2000;
	config.seed = 101;
	config.min_scale = 0.01f;
	config.max_scale = 1.0f;
	config.log_scale_distribution = true;

	const LocalVector<Gaussian> splats = UniformSplatGenerator::generate(config);
	const float midpoint = (config.min_scale + config.max_scale) * 0.5f;
	uint32_t below_mid = 0;
	for (const Gaussian &g : splats) {
		const float avg_scale = (g.scale.x + g.scale.y + g.scale.z) / 3.0f;
		if (avg_scale < midpoint) {
			below_mid++;
		}
	}
	CHECK(below_mid > config.splat_count / 2);
}

} // namespace TestGaussianSplatting
