/**************************************************************************/
/*  synthetic_clustered_generator.cpp                                    */
/**************************************************************************/

#include "synthetic_splat_generators.h"

namespace TestGaussianSplatting {

using namespace detail;

namespace {

static Color _cluster_color(uint64_t p_seed, uint32_t p_cluster_index, float p_opacity) {
	DeterministicRng rng(p_seed ^ (static_cast<uint64_t>(p_cluster_index + 1) * 0xA24BAED4963EE407ull));
	const float r = 0.2f + rng.next_unit_float() * 0.8f;
	const float g = 0.2f + rng.next_unit_float() * 0.8f;
	const float b = 0.2f + rng.next_unit_float() * 0.8f;
	return Color(r, g, b, p_opacity);
}

} // namespace

Dictionary ClusteredSplatGenerator::Config::to_dict() const {
	Dictionary dict;
	dict["splat_count"] = static_cast<int64_t>(splat_count);
	dict["seed"] = hash_hex(seed);
	dict["cluster_count"] = static_cast<int64_t>(cluster_count);
	dict["cluster_center_min"] = cluster_center_min;
	dict["cluster_center_max"] = cluster_center_max;
	dict["center_offset"] = center_offset;
	dict["cluster_radius"] = cluster_radius;
	dict["min_scale"] = min_scale;
	dict["max_scale"] = max_scale;
	dict["min_opacity"] = min_opacity;
	dict["max_opacity"] = max_opacity;
	dict["normal_tilt"] = normal_tilt;
	dict["random_rotation"] = random_rotation;
	dict["color_per_cluster"] = color_per_cluster;
	dict["anisotropy"] = anisotropy;
	dict["sh_intensity"] = sh_intensity;
	dict["log_scale_distribution"] = log_scale_distribution;
	dict["surface_aligned"] = surface_aligned;
	dict["cluster_flatten"] = cluster_flatten;
	return dict;
}

void ClusteredSplatGenerator::Config::from_dict(const Dictionary &p_dict) {
	if (p_dict.has("splat_count")) {
		splat_count = read_non_negative_u32(p_dict["splat_count"], splat_count);
	}
	if (p_dict.has("seed")) {
		seed = read_non_negative_u64(p_dict["seed"], seed);
	}
	if (p_dict.has("cluster_count")) {
		cluster_count = read_non_negative_u32(p_dict["cluster_count"], cluster_count);
	}
	if (p_dict.has("cluster_center_min")) {
		cluster_center_min = p_dict["cluster_center_min"];
	}
	if (p_dict.has("cluster_center_max")) {
		cluster_center_max = p_dict["cluster_center_max"];
	}
	if (p_dict.has("center_offset")) {
		center_offset = p_dict["center_offset"];
	}
	if (p_dict.has("cluster_radius")) {
		cluster_radius = p_dict["cluster_radius"];
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
	if (p_dict.has("color_per_cluster")) {
		color_per_cluster = p_dict["color_per_cluster"];
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
	if (p_dict.has("surface_aligned")) {
		surface_aligned = p_dict["surface_aligned"];
	}
	if (p_dict.has("cluster_flatten")) {
		cluster_flatten = p_dict["cluster_flatten"];
	}
}

uint64_t ClusteredSplatGenerator::hash_config(const Config &p_config) {
	uint64_t hash = HASH_BASIS;
	hash = hash_u64(hash, p_config.splat_count);
	hash = hash_u64(hash, p_config.seed);
	hash = hash_u64(hash, p_config.cluster_count);
	hash = hash_vector3(hash, p_config.cluster_center_min);
	hash = hash_vector3(hash, p_config.cluster_center_max);
	hash = hash_vector3(hash, p_config.center_offset);
	hash = hash_float(hash, p_config.cluster_radius);
	hash = hash_float(hash, p_config.min_scale);
	hash = hash_float(hash, p_config.max_scale);
	hash = hash_float(hash, p_config.min_opacity);
	hash = hash_float(hash, p_config.max_opacity);
	hash = hash_float(hash, p_config.normal_tilt);
	hash = hash_bool(hash, p_config.random_rotation);
	hash = hash_bool(hash, p_config.color_per_cluster);
	hash = hash_float(hash, p_config.anisotropy);
	hash = hash_float(hash, p_config.sh_intensity);
	hash = hash_bool(hash, p_config.log_scale_distribution);
	hash = hash_bool(hash, p_config.surface_aligned);
	hash = hash_float(hash, p_config.cluster_flatten);
	return hash;
}

LocalVector<Gaussian> ClusteredSplatGenerator::generate(const Config &p_config, SyntheticSceneSummary *r_summary) {
	Config config = p_config;
	config.cluster_count = MAX<uint32_t>(1u, config.cluster_count);
	normalize_range_vec3(config.cluster_center_min, config.cluster_center_max);
	normalize_range(config.min_scale, config.max_scale);
	normalize_range(config.min_opacity, config.max_opacity);
	if (config.cluster_radius < 0.0f) {
		config.cluster_radius = 0.0f;
	}
	config.anisotropy = CLAMP(config.anisotropy, 0.0f, 1.0f);
	config.sh_intensity = MAX(config.sh_intensity, 0.0f);
	config.cluster_flatten = CLAMP(config.cluster_flatten, 0.0f, 1.0f);

	LocalVector<Gaussian> splats;
	splats.resize(config.splat_count);

	LocalVector<Vector3> centers;
	centers.resize(config.cluster_count);

	DeterministicRng rng(config.seed);
	for (uint32_t cluster_idx = 0; cluster_idx < config.cluster_count; cluster_idx++) {
		centers[cluster_idx] = Vector3(
									  rng.range(config.cluster_center_min.x, config.cluster_center_max.x),
									  rng.range(config.cluster_center_min.y, config.cluster_center_max.y),
									  rng.range(config.cluster_center_min.z, config.cluster_center_max.z)) +
				config.center_offset;
	}

	for (uint32_t i = 0; i < config.splat_count; i++) {
		Gaussian gaussian = make_base_gaussian();
		const uint32_t cluster_idx = i % config.cluster_count;
		const Vector3 &center = centers[cluster_idx];

		Vector3 offset(
				rng.range(-config.cluster_radius, config.cluster_radius),
				rng.range(-config.cluster_radius, config.cluster_radius),
				rng.range(-config.cluster_radius, config.cluster_radius));

		if (config.cluster_flatten > 0.0f) {
			offset.y *= (1.0f - config.cluster_flatten);
		}

		gaussian.position = center + offset;

		float base_scale;
		if (config.log_scale_distribution) {
			base_scale = log_normal_scale(rng, config.min_scale, config.max_scale);
		} else {
			base_scale = rng.range(config.min_scale, config.max_scale);
		}

		gaussian.scale = anisotropic_scale(rng, base_scale, config.anisotropy);
		gaussian.area = gaussian.scale.x * gaussian.scale.z * static_cast<float>(Math::PI);
		gaussian.opacity = rng.range(config.min_opacity, config.max_opacity);

		if (config.surface_aligned) {
			const Vector3 dir = (gaussian.position - center);
			if (dir.length_squared() > CMP_EPSILON2) {
				gaussian.normal = dir.normalized();
				gaussian.rotation = orient_to_normal(gaussian.normal);
			}
		} else {
			gaussian.normal = make_normal(rng, config.normal_tilt);
			if (config.random_rotation) {
				gaussian.rotation = random_unit_quaternion(rng);
			}
		}

		if (config.color_per_cluster) {
			gaussian.sh_dc = _cluster_color(config.seed, cluster_idx, gaussian.opacity);
		} else {
			gaussian.sh_dc = Color(rng.next_unit_float(), rng.next_unit_float(), rng.next_unit_float(), gaussian.opacity);
		}

		generate_sh1(rng, gaussian, config.sh_intensity);

		splats[i] = gaussian;
	}

	if (r_summary != nullptr) {
		*r_summary = summarize_generated_scene(splats, "clustered", config.seed, hash_config(config));
	}
	return splats;
}

} // namespace TestGaussianSplatting
