/**************************************************************************/
/*  test_synthetic_bml_traffic_generator.h                               */
/**************************************************************************/

#pragma once

#include "synthetic_splat_generators.h"
#include "tests/test_macros.h"

namespace TestGaussianSplatting {

TEST_CASE("[GaussianSplatting][Synthetic] BML traffic generator stays deterministic") {
	BMLTrafficSplatGenerator::Config config;
	config.grid_width = 32;
	config.grid_height = 32;
	config.density = 0.35f;
	config.steps = 100;
	config.seed = 42;

	SyntheticSceneSummary summary_a;
	SyntheticSceneSummary summary_b;
	const LocalVector<Gaussian> splats_a = BMLTrafficSplatGenerator::generate(config, &summary_a);
	const LocalVector<Gaussian> splats_b = BMLTrafficSplatGenerator::generate(config, &summary_b);

	CHECK(splats_a.size() > 0);
	CHECK_EQ(splats_a.size(), splats_b.size());
	CHECK_EQ(summary_a.scene_hash, summary_b.scene_hash);
	CHECK(summary_a.scene_hash != 0);

	BMLTrafficSplatGenerator::Config roundtrip;
	roundtrip.from_dict(config.to_dict());
	CHECK_EQ(BMLTrafficSplatGenerator::hash_config(config), BMLTrafficSplatGenerator::hash_config(roundtrip));
}

TEST_CASE("[GaussianSplatting][Synthetic] BML traffic has both particle types") {
	BMLTrafficSplatGenerator::Config config;
	config.grid_width = 64;
	config.grid_height = 64;
	config.density = 0.3f;
	config.steps = 200;
	config.seed = 99;

	const LocalVector<Gaussian> splats = BMLTrafficSplatGenerator::generate(config);
	uint32_t red_count = 0;
	uint32_t blue_count = 0;
	for (const Gaussian &g : splats) {
		if (g.sh_dc.r > g.sh_dc.b) {
			red_count++;
		} else {
			blue_count++;
		}
	}
	CHECK(red_count > 0);
	CHECK(blue_count > 0);
}

TEST_CASE("[GaussianSplatting][Synthetic] BML traffic particle count matches density") {
	BMLTrafficSplatGenerator::Config config;
	config.grid_width = 64;
	config.grid_height = 64;
	config.density = 0.4f;
	config.steps = 0;
	config.seed = 42;

	const LocalVector<Gaussian> splats = BMLTrafficSplatGenerator::generate(config);
	const float expected = config.density * static_cast<float>(config.grid_width * config.grid_height);
	CHECK(static_cast<float>(splats.size()) > expected * 0.9f);
	CHECK(static_cast<float>(splats.size()) < expected * 1.1f);
}

TEST_CASE("[GaussianSplatting][Synthetic] BML traffic splats are on flat grid") {
	BMLTrafficSplatGenerator::Config config;
	config.grid_width = 16;
	config.grid_height = 16;
	config.density = 0.5f;
	config.steps = 10;
	config.seed = 55;

	const LocalVector<Gaussian> splats = BMLTrafficSplatGenerator::generate(config);
	for (const Gaussian &g : splats) {
		CHECK(Math::abs(g.position.y) < CMP_EPSILON);
	}
}

} // namespace TestGaussianSplatting
