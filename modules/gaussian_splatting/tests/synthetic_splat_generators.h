/**************************************************************************/
/*  synthetic_splat_generators.h                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "../core/gaussian_data.h"

#include "core/string/ustring.h"
#include "core/variant/dictionary.h"

#include <cstdint>

namespace TestGaussianSplatting {

struct SyntheticSceneSummary {
	String generator_name;
	uint32_t splat_count = 0;
	uint64_t seed = 0;
	uint64_t config_hash = 0;
	uint64_t scene_hash = 0;
	Vector3 bounds_min = Vector3();
	Vector3 bounds_max = Vector3();
	float average_scale = 0.0f;
	float average_opacity = 0.0f;

	Dictionary to_dict() const;
};

class UniformSplatGenerator {
public:
	struct Config {
		uint32_t splat_count = 100000;
		uint64_t seed = 42;
		Vector3 position_min = Vector3(-10.0f, -10.0f, -10.0f);
		Vector3 position_max = Vector3(10.0f, 10.0f, 10.0f);
		float min_scale = 0.1f;
		float max_scale = 1.0f;
		float min_opacity = 0.3f;
		float max_opacity = 1.0f;
		float normal_tilt = 0.2f;
		bool random_rotation = true;
		bool random_colors = true;
		Color base_color = Color(0.7f, 0.7f, 0.7f, 1.0f);

		Dictionary to_dict() const;
		void from_dict(const Dictionary &p_dict);
	};

	static LocalVector<Gaussian> generate(const Config &p_config, SyntheticSceneSummary *r_summary = nullptr);
	static uint64_t hash_config(const Config &p_config);
};

class ClusteredSplatGenerator {
public:
	struct Config {
		uint32_t splat_count = 100000;
		uint64_t seed = 42;
		uint32_t cluster_count = 10;
		Vector3 cluster_center_min = Vector3(-20.0f, -20.0f, -20.0f);
		Vector3 cluster_center_max = Vector3(20.0f, 20.0f, 20.0f);
		Vector3 center_offset = Vector3();
		float cluster_radius = 2.0f;
		float min_scale = 0.05f;
		float max_scale = 0.5f;
		float min_opacity = 0.5f;
		float max_opacity = 1.0f;
		float normal_tilt = 0.0f;
		bool random_rotation = false;
		bool color_per_cluster = true;

		Dictionary to_dict() const;
		void from_dict(const Dictionary &p_dict);
	};

	static LocalVector<Gaussian> generate(const Config &p_config, SyntheticSceneSummary *r_summary = nullptr);
	static uint64_t hash_config(const Config &p_config);
};

SyntheticSceneSummary summarize_generated_scene(const LocalVector<Gaussian> &p_splats, const String &p_generator_name, uint64_t p_seed, uint64_t p_config_hash);

} // namespace TestGaussianSplatting

