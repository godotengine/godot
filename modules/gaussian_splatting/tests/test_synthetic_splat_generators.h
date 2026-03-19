/**************************************************************************/
/*  test_synthetic_splat_generators.h                                     */
/*  Shared tests: scene hash, config serialization, determinism basics.  */
/**************************************************************************/

#pragma once

#include "synthetic_splat_generators.h"

#include "tests/test_macros.h"

namespace TestGaussianSplatting {

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

} // namespace TestGaussianSplatting
