/**************************************************************************/
/*  synthetic_uniform_generator.cpp                                      */
/**************************************************************************/

#include "synthetic_splat_generators.h"

namespace TestGaussianSplatting {

using namespace detail;

Dictionary UniformSplatGenerator::Config::to_dict() const {
	Dictionary dict;
	dict["splat_count"] = static_cast<int64_t>(splat_count);
	dict["seed"] = hash_hex(seed);
	dict["position_min"] = position_min;
	dict["position_max"] = position_max;
	dict["min_scale"] = min_scale;
	dict["max_scale"] = max_scale;
	dict["min_opacity"] = min_opacity;
	dict["max_opacity"] = max_opacity;
	dict["normal_tilt"] = normal_tilt;
	dict["random_rotation"] = random_rotation;
	dict["random_colors"] = random_colors;
	dict["base_color"] = base_color;
	dict["anisotropy"] = anisotropy;
	dict["sh_intensity"] = sh_intensity;
	dict["log_scale_distribution"] = log_scale_distribution;
	return dict;
}

void UniformSplatGenerator::Config::from_dict(const Dictionary &p_dict) {
	if (p_dict.has("splat_count")) {
		splat_count = read_non_negative_u32(p_dict["splat_count"], splat_count);
	}
	if (p_dict.has("seed")) {
		seed = read_non_negative_u64(p_dict["seed"], seed);
	}
	if (p_dict.has("position_min")) {
		position_min = p_dict["position_min"];
	}
	if (p_dict.has("position_max")) {
		position_max = p_dict["position_max"];
	}
	if (p_dict.has("min_scale")) {
		min_scale = p_dict["min_scale"];
	}
	if (p_dict.has("max_scale")) {
		max_scale = p_dict["max_scale"];
	}
	if (p_dict.has("min_opacity")) {
		min_opacity = p_dict["min_opacity"];
	}
	if (p_dict.has("max_opacity")) {
		max_opacity = p_dict["max_opacity"];
	}
	if (p_dict.has("normal_tilt")) {
		normal_tilt = p_dict["normal_tilt"];
	}
	if (p_dict.has("random_rotation")) {
		random_rotation = p_dict["random_rotation"];
	}
	if (p_dict.has("random_colors")) {
		random_colors = p_dict["random_colors"];
	}
	if (p_dict.has("base_color")) {
		base_color = p_dict["base_color"];
	}
	if (p_dict.has("anisotropy")) {
		anisotropy = p_dict["anisotropy"];
	}
	if (p_dict.has("sh_intensity")) {
		sh_intensity = p_dict["sh_intensity"];
	}
	if (p_dict.has("log_scale_distribution")) {
		log_scale_distribution = p_dict["log_scale_distribution"];
	}
}

uint64_t UniformSplatGenerator::hash_config(const Config &p_config) {
	uint64_t hash = HASH_BASIS;
	hash = hash_u64(hash, p_config.splat_count);
	hash = hash_u64(hash, p_config.seed);
	hash = hash_vector3(hash, p_config.position_min);
	hash = hash_vector3(hash, p_config.position_max);
	hash = hash_float(hash, p_config.min_scale);
	hash = hash_float(hash, p_config.max_scale);
	hash = hash_float(hash, p_config.min_opacity);
	hash = hash_float(hash, p_config.max_opacity);
	hash = hash_float(hash, p_config.normal_tilt);
	hash = hash_bool(hash, p_config.random_rotation);
	hash = hash_bool(hash, p_config.random_colors);
	hash = hash_color(hash, p_config.base_color);
	hash = hash_float(hash, p_config.anisotropy);
	hash = hash_float(hash, p_config.sh_intensity);
	hash = hash_bool(hash, p_config.log_scale_distribution);
	return hash;
}

LocalVector<Gaussian> UniformSplatGenerator::generate(const Config &p_config, SyntheticSceneSummary *r_summary) {
	Config config = p_config;
	normalize_range_vec3(config.position_min, config.position_max);
	normalize_range(config.min_scale, config.max_scale);
	normalize_range(config.min_opacity, config.max_opacity);
	config.anisotropy = CLAMP(config.anisotropy, 0.0f, 1.0f);
	config.sh_intensity = MAX(config.sh_intensity, 0.0f);

	LocalVector<Gaussian> splats;
	splats.resize(config.splat_count);

	DeterministicRng rng(config.seed);
	for (uint32_t i = 0; i < config.splat_count; i++) {
		Gaussian gaussian = make_base_gaussian();

		gaussian.position = Vector3(
				rng.range(config.position_min.x, config.position_max.x),
				rng.range(config.position_min.y, config.position_max.y),
				rng.range(config.position_min.z, config.position_max.z));

		float base_scale;
		if (config.log_scale_distribution) {
			base_scale = log_normal_scale(rng, config.min_scale, config.max_scale);
		} else {
			base_scale = rng.range(config.min_scale, config.max_scale);
		}

		gaussian.scale = anisotropic_scale(rng, base_scale, config.anisotropy);
		gaussian.area = gaussian.scale.x * gaussian.scale.z * static_cast<float>(Math::PI);
		gaussian.opacity = rng.range(config.min_opacity, config.max_opacity);
		gaussian.normal = make_normal(rng, config.normal_tilt);

		if (config.random_rotation) {
			gaussian.rotation = random_unit_quaternion(rng);
		}

		if (config.random_colors) {
			gaussian.sh_dc = Color(rng.next_unit_float(), rng.next_unit_float(), rng.next_unit_float(), gaussian.opacity);
		} else {
			gaussian.sh_dc = Color(config.base_color.r, config.base_color.g, config.base_color.b, gaussian.opacity);
		}

		generate_sh1(rng, gaussian, config.sh_intensity);

		splats[i] = gaussian;
	}

	if (r_summary != nullptr) {
		*r_summary = summarize_generated_scene(splats, "uniform", config.seed, hash_config(config));
	}
	return splats;
}

} // namespace TestGaussianSplatting
