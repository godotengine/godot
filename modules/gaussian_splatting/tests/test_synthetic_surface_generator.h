/**************************************************************************/
/*  test_synthetic_surface_generator.h                                   */
/**************************************************************************/

#pragma once

#include "synthetic_splat_generators.h"
#include "tests/test_macros.h"

namespace TestGaussianSplatting {

TEST_CASE("[GaussianSplatting][Synthetic] Surface sphere generator stays deterministic") {
	SurfaceSplatGenerator::Config config;
	config.splat_count = 500;
	config.seed = 42;
	config.shape = SurfaceSplatGenerator::SHAPE_SPHERE;
	config.shape_radius = 3.0f;

	SyntheticSceneSummary summary_a;
	SyntheticSceneSummary summary_b;
	const LocalVector<Gaussian> splats_a = SurfaceSplatGenerator::generate(config, &summary_a);
	const LocalVector<Gaussian> splats_b = SurfaceSplatGenerator::generate(config, &summary_b);

	CHECK_EQ(splats_a.size(), static_cast<int>(config.splat_count));
	CHECK_EQ(summary_a.scene_hash, summary_b.scene_hash);
	CHECK(summary_a.scene_hash != 0);

	SurfaceSplatGenerator::Config roundtrip;
	roundtrip.from_dict(config.to_dict());
	CHECK_EQ(SurfaceSplatGenerator::hash_config(config), SurfaceSplatGenerator::hash_config(roundtrip));
}

TEST_CASE("[GaussianSplatting][Synthetic] Surface sphere splats lie on sphere surface") {
	SurfaceSplatGenerator::Config config;
	config.splat_count = 1000;
	config.seed = 55;
	config.shape = SurfaceSplatGenerator::SHAPE_SPHERE;
	config.shape_radius = 4.0f;
	config.surface_noise = 0.0f;

	const LocalVector<Gaussian> splats = SurfaceSplatGenerator::generate(config);
	for (const Gaussian &g : splats) {
		CHECK(Math::abs(g.position.length() - config.shape_radius) < 0.01f);
	}
}

TEST_CASE("[GaussianSplatting][Synthetic] Surface torus splats lie on torus surface") {
	SurfaceSplatGenerator::Config config;
	config.splat_count = 500;
	config.seed = 66;
	config.shape = SurfaceSplatGenerator::SHAPE_TORUS;
	config.shape_radius = 5.0f;
	config.torus_tube_radius = 1.5f;
	config.surface_noise = 0.0f;

	const LocalVector<Gaussian> splats = SurfaceSplatGenerator::generate(config);
	for (const Gaussian &g : splats) {
		const float xz_dist = Math::sqrt(g.position.x * g.position.x + g.position.z * g.position.z);
		const float ring_dist = Math::sqrt(
				(xz_dist - config.shape_radius) * (xz_dist - config.shape_radius) +
				g.position.y * g.position.y);
		CHECK(Math::abs(ring_dist - config.torus_tube_radius) < 0.01f);
	}
}

TEST_CASE("[GaussianSplatting][Synthetic] Surface plane splats are flat") {
	SurfaceSplatGenerator::Config config;
	config.splat_count = 200;
	config.seed = 77;
	config.shape = SurfaceSplatGenerator::SHAPE_PLANE;
	config.plane_half_extent = 6.0f;
	config.surface_noise = 0.0f;

	const LocalVector<Gaussian> splats = SurfaceSplatGenerator::generate(config);
	for (const Gaussian &g : splats) {
		CHECK(Math::abs(g.position.y) < CMP_EPSILON);
		CHECK(Math::abs(g.position.x) <= config.plane_half_extent + CMP_EPSILON);
		CHECK(Math::abs(g.position.z) <= config.plane_half_extent + CMP_EPSILON);
	}
}

TEST_CASE("[GaussianSplatting][Synthetic] Surface cube splats cover all 6 faces") {
	SurfaceSplatGenerator::Config config;
	config.splat_count = 6000;
	config.seed = 88;
	config.shape = SurfaceSplatGenerator::SHAPE_CUBE;
	config.shape_radius = 2.0f;
	config.surface_noise = 0.0f;

	const LocalVector<Gaussian> splats = SurfaceSplatGenerator::generate(config);
	bool face_hit[6] = {};
	for (const Gaussian &g : splats) {
		if (Math::abs(g.position.x - 2.0f) < CMP_EPSILON) { face_hit[0] = true; }
		if (Math::abs(g.position.x + 2.0f) < CMP_EPSILON) { face_hit[1] = true; }
		if (Math::abs(g.position.y - 2.0f) < CMP_EPSILON) { face_hit[2] = true; }
		if (Math::abs(g.position.y + 2.0f) < CMP_EPSILON) { face_hit[3] = true; }
		if (Math::abs(g.position.z - 2.0f) < CMP_EPSILON) { face_hit[4] = true; }
		if (Math::abs(g.position.z + 2.0f) < CMP_EPSILON) { face_hit[5] = true; }
	}
	for (int f = 0; f < 6; f++) {
		CHECK(face_hit[f]);
	}
}

TEST_CASE("[GaussianSplatting][Synthetic] Surface splats have anisotropic scales by default") {
	SurfaceSplatGenerator::Config config;
	config.splat_count = 500;
	config.seed = 99;

	const LocalVector<Gaussian> splats = SurfaceSplatGenerator::generate(config);
	uint32_t aniso_count = 0;
	for (const Gaussian &g : splats) {
		if (Math::abs(g.scale.x - g.scale.y) > CMP_EPSILON ||
				Math::abs(g.scale.y - g.scale.z) > CMP_EPSILON) {
			aniso_count++;
		}
	}
	CHECK(aniso_count > config.splat_count * 0.7f);
}

TEST_CASE("[GaussianSplatting][Synthetic] Surface splats have non-zero SH by default") {
	SurfaceSplatGenerator::Config config;
	config.splat_count = 200;
	config.seed = 111;

	const LocalVector<Gaussian> splats = SurfaceSplatGenerator::generate(config);
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

TEST_CASE("[GaussianSplatting][Synthetic] Surface color_variation produces per-splat color jitter") {
	SurfaceSplatGenerator::Config config;
	config.splat_count = 100;
	config.seed = 222;
	config.random_colors = false;
	config.base_color = Color(0.5f, 0.5f, 0.5f, 1.0f);
	config.color_variation = 0.2f;

	const LocalVector<Gaussian> splats = SurfaceSplatGenerator::generate(config);
	uint32_t varied = 0;
	for (const Gaussian &g : splats) {
		if (Math::abs(g.sh_dc.r - 0.5f) > CMP_EPSILON ||
				Math::abs(g.sh_dc.g - 0.5f) > CMP_EPSILON ||
				Math::abs(g.sh_dc.b - 0.5f) > CMP_EPSILON) {
			varied++;
		}
	}
	CHECK(varied > config.splat_count * 0.8f);
}

} // namespace TestGaussianSplatting
