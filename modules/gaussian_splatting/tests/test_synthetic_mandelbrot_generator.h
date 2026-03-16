/**************************************************************************/
/*  test_synthetic_mandelbrot_generator.h                                */
/**************************************************************************/

#pragma once

#include "synthetic_splat_generators.h"
#include "tests/test_macros.h"

namespace TestGaussianSplatting {

TEST_CASE("[GaussianSplatting][Synthetic] Mandelbrot generator stays deterministic") {
	MandelbrotSplatGenerator::Config config;
	config.splat_count = 500;
	config.seed = 42;
	config.max_iterations = 128;

	SyntheticSceneSummary summary_a;
	SyntheticSceneSummary summary_b;
	const LocalVector<Gaussian> splats_a = MandelbrotSplatGenerator::generate(config, &summary_a);
	const LocalVector<Gaussian> splats_b = MandelbrotSplatGenerator::generate(config, &summary_b);

	CHECK_EQ(splats_a.size(), static_cast<int>(config.splat_count));
	CHECK_EQ(summary_a.scene_hash, summary_b.scene_hash);
	CHECK(summary_a.scene_hash != 0);

	MandelbrotSplatGenerator::Config roundtrip;
	roundtrip.from_dict(config.to_dict());
	CHECK_EQ(MandelbrotSplatGenerator::hash_config(config), MandelbrotSplatGenerator::hash_config(roundtrip));
}

TEST_CASE("[GaussianSplatting][Synthetic] Mandelbrot has boundary and interior splats") {
	MandelbrotSplatGenerator::Config config;
	config.splat_count = 5000;
	config.seed = 77;
	config.center_real = -0.5;
	config.center_imag = 0.0;
	config.zoom = 1.5;
	config.max_iterations = 256;

	const LocalVector<Gaussian> splats = MandelbrotSplatGenerator::generate(config);
	uint32_t interior = 0;
	uint32_t exterior = 0;
	for (const Gaussian &g : splats) {
		if (Math::is_equal_approx(g.opacity, config.max_opacity)) {
			interior++;
		} else {
			exterior++;
		}
	}
	CHECK(interior > 0);
	CHECK(exterior > 0);
}

TEST_CASE("[GaussianSplatting][Synthetic] Mandelbrot scale varies with iteration count") {
	MandelbrotSplatGenerator::Config config;
	config.splat_count = 2000;
	config.seed = 55;
	config.min_scale = 0.01f;
	config.max_scale = 0.2f;
	config.anisotropy = 0.0f;

	const LocalVector<Gaussian> splats = MandelbrotSplatGenerator::generate(config);
	uint32_t small_splats = 0;
	uint32_t large_splats = 0;
	const float mid = (config.min_scale + config.max_scale) * 0.5f;
	for (const Gaussian &g : splats) {
		if (g.scale.x < mid) {
			small_splats++;
		} else {
			large_splats++;
		}
	}
	CHECK(small_splats > 0);
	CHECK(large_splats > 0);
}

TEST_CASE("[GaussianSplatting][Synthetic] Mandelbrot color modes produce different outputs") {
	MandelbrotSplatGenerator::Config config;
	config.splat_count = 200;
	config.seed = 42;

	config.color_mode = MandelbrotSplatGenerator::COLOR_CLASSIC_RAINBOW;
	SyntheticSceneSummary sum_rainbow;
	MandelbrotSplatGenerator::generate(config, &sum_rainbow);

	config.color_mode = MandelbrotSplatGenerator::COLOR_FIRE;
	SyntheticSceneSummary sum_fire;
	MandelbrotSplatGenerator::generate(config, &sum_fire);

	config.color_mode = MandelbrotSplatGenerator::COLOR_ICE;
	SyntheticSceneSummary sum_ice;
	MandelbrotSplatGenerator::generate(config, &sum_ice);

	CHECK(sum_rainbow.scene_hash != sum_fire.scene_hash);
	CHECK(sum_fire.scene_hash != sum_ice.scene_hash);
}

} // namespace TestGaussianSplatting
