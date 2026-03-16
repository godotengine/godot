/**************************************************************************/
/*  test_synthetic_cloud_generator.h                                     */
/**************************************************************************/

#pragma once

#include "synthetic_splat_generators.h"
#include "tests/test_macros.h"

namespace TestGaussianSplatting {

TEST_CASE("[GaussianSplatting][Synthetic] Cloud generator stays deterministic") {
	CloudSplatGenerator::Config config;
	config.splat_count = 500;
	config.seed = 42;
	config.density_threshold = 0.3f;

	SyntheticSceneSummary summary_a;
	SyntheticSceneSummary summary_b;
	const LocalVector<Gaussian> splats_a = CloudSplatGenerator::generate(config, &summary_a);
	const LocalVector<Gaussian> splats_b = CloudSplatGenerator::generate(config, &summary_b);

	CHECK(splats_a.size() > 0);
	CHECK_EQ(splats_a.size(), splats_b.size());
	CHECK_EQ(summary_a.scene_hash, summary_b.scene_hash);
	CHECK(summary_a.scene_hash != 0);

	CloudSplatGenerator::Config roundtrip;
	roundtrip.from_dict(config.to_dict());
	CHECK_EQ(CloudSplatGenerator::hash_config(config), CloudSplatGenerator::hash_config(roundtrip));
}

TEST_CASE("[GaussianSplatting][Synthetic] Cloud splats are within cloud extent") {
	CloudSplatGenerator::Config config;
	config.splat_count = 1000;
	config.seed = 77;
	config.cloud_center = Vector3(0, 5, 0);
	config.cloud_extent = Vector3(10, 3, 8);
	config.density_threshold = 0.2f;

	const LocalVector<Gaussian> splats = CloudSplatGenerator::generate(config);
	for (const Gaussian &g : splats) {
		const Vector3 local = g.position - config.cloud_center;
		const float ex = local.x / config.cloud_extent.x;
		const float ey = local.y / config.cloud_extent.y;
		const float ez = local.z / config.cloud_extent.z;
		CHECK(ex * ex + ey * ey + ez * ez <= 1.01f);
	}
}

TEST_CASE("[GaussianSplatting][Synthetic] Cloud splats have vertical color gradient") {
	CloudSplatGenerator::Config config;
	config.splat_count = 2000;
	config.seed = 88;
	config.cloud_color = Color(1.0f, 1.0f, 1.0f, 1.0f);
	config.shadow_tint = Color(0.3f, 0.3f, 0.4f, 1.0f);
	config.density_threshold = 0.2f;

	const LocalVector<Gaussian> splats = CloudSplatGenerator::generate(config);
	float top_brightness = 0.0f;
	uint32_t top_count = 0;
	float bottom_brightness = 0.0f;
	uint32_t bottom_count = 0;
	const float mid_y = config.cloud_center.y;

	for (const Gaussian &g : splats) {
		const float lum = (g.sh_dc.r + g.sh_dc.g + g.sh_dc.b) / 3.0f;
		if (g.position.y > mid_y) {
			top_brightness += lum;
			top_count++;
		} else {
			bottom_brightness += lum;
			bottom_count++;
		}
	}
	if (top_count > 0 && bottom_count > 0) {
		top_brightness /= static_cast<float>(top_count);
		bottom_brightness /= static_cast<float>(bottom_count);
		CHECK(top_brightness > bottom_brightness);
	}
}

TEST_CASE("[GaussianSplatting][Synthetic] Cloud splats have SH coefficients encoding light direction") {
	CloudSplatGenerator::Config config;
	config.splat_count = 500;
	config.seed = 111;
	config.sh_intensity = 0.4f;
	config.density_threshold = 0.2f;

	const LocalVector<Gaussian> splats = CloudSplatGenerator::generate(config);
	uint32_t nonzero_sh = 0;
	for (const Gaussian &g : splats) {
		for (int band = 0; band < 3; band++) {
			if (g.sh_1[band].length_squared() > CMP_EPSILON2) {
				nonzero_sh++;
				break;
			}
		}
	}
	CHECK(nonzero_sh == splats.size());
}

TEST_CASE("[GaussianSplatting][Synthetic] Cloud density_threshold controls sparsity") {
	CloudSplatGenerator::Config config_dense;
	config_dense.splat_count = 2000;
	config_dense.seed = 42;
	config_dense.density_threshold = 0.2f;

	CloudSplatGenerator::Config config_sparse;
	config_sparse.splat_count = 2000;
	config_sparse.seed = 42;
	config_sparse.density_threshold = 0.6f;

	const LocalVector<Gaussian> dense = CloudSplatGenerator::generate(config_dense);
	const LocalVector<Gaussian> sparse = CloudSplatGenerator::generate(config_sparse);
	CHECK(dense.size() > sparse.size());
}

} // namespace TestGaussianSplatting
