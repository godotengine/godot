/**************************************************************************/
/*  synthetic_splat_generators.cpp                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#include "synthetic_splat_generators.h"

#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"

namespace TestGaussianSplatting {

namespace {

constexpr uint64_t HASH_BASIS = 1469598103934665603ull;
constexpr uint64_t HASH_PRIME = 1099511628211ull;
constexpr float HASH_QUANTIZATION_SCALE = 100000.0f;

class DeterministicRng {
	uint64_t state = 0;

public:
	explicit DeterministicRng(uint64_t p_seed) :
			state(p_seed) {}

	uint64_t next_u64() {
		state += 0x9E3779B97F4A7C15ull;
		uint64_t z = state;
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
		return z ^ (z >> 31);
	}

	float next_unit_float() {
		constexpr double inv = 1.0 / static_cast<double>(1ull << 24);
		const uint64_t bits = next_u64() >> 40;
		return static_cast<float>(bits * inv);
	}

	float range(float p_min, float p_max) {
		return p_min + (p_max - p_min) * next_unit_float();
	}
};

static void _normalize_range(float &r_min, float &r_max) {
	if (r_max < r_min) {
		SWAP(r_min, r_max);
	}
}

static void _normalize_range_vec3(Vector3 &r_min, Vector3 &r_max) {
	if (r_max.x < r_min.x) {
		SWAP(r_min.x, r_max.x);
	}
	if (r_max.y < r_min.y) {
		SWAP(r_min.y, r_max.y);
	}
	if (r_max.z < r_min.z) {
		SWAP(r_min.z, r_max.z);
	}
}

static Quaternion _random_unit_quaternion(DeterministicRng &p_rng) {
	const float x = p_rng.range(-1.0f, 1.0f);
	const float y = p_rng.range(-1.0f, 1.0f);
	const float z = p_rng.range(-1.0f, 1.0f);
	const float w = p_rng.range(-1.0f, 1.0f);
	const float length_sq = x * x + y * y + z * z + w * w;
	if (length_sq < CMP_EPSILON2) {
		return Quaternion();
	}
	const float inv_len = 1.0f / Math::sqrt(length_sq);
	return Quaternion(x * inv_len, y * inv_len, z * inv_len, w * inv_len);
}

static Vector3 _make_normal(DeterministicRng &p_rng, float p_normal_tilt) {
	if (p_normal_tilt <= 0.0f) {
		return Vector3(0.0f, 1.0f, 0.0f);
	}
	Vector3 normal(
			p_rng.range(-p_normal_tilt, p_normal_tilt),
			1.0f,
			p_rng.range(-p_normal_tilt, p_normal_tilt));
	const float length_sq = normal.length_squared();
	if (length_sq < CMP_EPSILON2) {
		return Vector3(0.0f, 1.0f, 0.0f);
	}
	return normal / Math::sqrt(length_sq);
}

static uint64_t _fnv1a_u64(uint64_t p_hash, uint64_t p_value) {
	return (p_hash ^ p_value) * HASH_PRIME;
}

static uint64_t _hash_bool(uint64_t p_hash, bool p_value) {
	return _fnv1a_u64(p_hash, p_value ? 1ull : 0ull);
}

static uint64_t _hash_i64(uint64_t p_hash, int64_t p_value) {
	return _fnv1a_u64(p_hash, static_cast<uint64_t>(p_value));
}

static uint64_t _hash_u64(uint64_t p_hash, uint64_t p_value) {
	return _fnv1a_u64(p_hash, p_value);
}

static uint64_t _hash_float(uint64_t p_hash, float p_value) {
	const int64_t quantized = static_cast<int64_t>(Math::round(p_value * HASH_QUANTIZATION_SCALE));
	return _hash_i64(p_hash, quantized);
}

static uint64_t _hash_vector3(uint64_t p_hash, const Vector3 &p_vec) {
	p_hash = _hash_float(p_hash, p_vec.x);
	p_hash = _hash_float(p_hash, p_vec.y);
	p_hash = _hash_float(p_hash, p_vec.z);
	return p_hash;
}

static uint64_t _hash_vector2(uint64_t p_hash, const Vector2 &p_vec) {
	p_hash = _hash_float(p_hash, p_vec.x);
	p_hash = _hash_float(p_hash, p_vec.y);
	return p_hash;
}

static uint64_t _hash_quaternion(uint64_t p_hash, const Quaternion &p_quat) {
	p_hash = _hash_float(p_hash, p_quat.x);
	p_hash = _hash_float(p_hash, p_quat.y);
	p_hash = _hash_float(p_hash, p_quat.z);
	p_hash = _hash_float(p_hash, p_quat.w);
	return p_hash;
}

static uint64_t _hash_color(uint64_t p_hash, const Color &p_color) {
	p_hash = _hash_float(p_hash, p_color.r);
	p_hash = _hash_float(p_hash, p_color.g);
	p_hash = _hash_float(p_hash, p_color.b);
	p_hash = _hash_float(p_hash, p_color.a);
	return p_hash;
}

static Color _cluster_color(uint64_t p_seed, uint32_t p_cluster_index, float p_opacity) {
	DeterministicRng rng(p_seed ^ (static_cast<uint64_t>(p_cluster_index + 1) * 0xA24BAED4963EE407ull));
	const float r = 0.2f + rng.next_unit_float() * 0.8f;
	const float g = 0.2f + rng.next_unit_float() * 0.8f;
	const float b = 0.2f + rng.next_unit_float() * 0.8f;
	return Color(r, g, b, p_opacity);
}

static String _hash_hex(uint64_t p_value) {
	return "0x" + String::num_uint64(p_value, 16).pad_zeros(16);
}

static uint64_t _hash_gaussian(uint64_t p_hash, const Gaussian &p_gaussian) {
	p_hash = _hash_vector3(p_hash, p_gaussian.position);
	p_hash = _hash_vector3(p_hash, p_gaussian.scale);
	p_hash = _hash_float(p_hash, p_gaussian.area);
	p_hash = _hash_float(p_hash, p_gaussian.opacity);
	p_hash = _hash_quaternion(p_hash, p_gaussian.rotation);
	p_hash = _hash_color(p_hash, p_gaussian.sh_dc);
	for (uint32_t i = 0; i < 3; i++) {
		p_hash = _hash_vector3(p_hash, p_gaussian.sh_1[i]);
	}
	p_hash = _hash_vector3(p_hash, p_gaussian.normal);
	p_hash = _hash_float(p_hash, p_gaussian.stroke_age);
	p_hash = _hash_vector2(p_hash, p_gaussian.brush_axes);
	p_hash = _hash_u64(p_hash, p_gaussian.painterly_meta);
	return p_hash;
}

static Gaussian _make_base_gaussian() {
	Gaussian gaussian = {};
	gaussian.rotation = Quaternion();
	gaussian.sh_dc = Color(1.0f, 1.0f, 1.0f, 1.0f);
	gaussian.normal = Vector3(0.0f, 1.0f, 0.0f);
	gaussian.stroke_age = 0.0f;
	gaussian._padding = 0.0f;
	gaussian.brush_axes = Vector2(1.0f, 0.0f);
	gaussian.painterly_meta = gaussian_pack_painterly_meta(0);
	gaussian._padding2[0] = 0.0f;
	gaussian._padding2[1] = 0.0f;
	gaussian._padding2[2] = 0.0f;
	return gaussian;
}

static uint32_t _read_non_negative_u32(const Variant &p_value, uint32_t p_default) {
	if (p_value.get_type() == Variant::NIL) {
		return p_default;
	}
	const int64_t parsed = static_cast<int64_t>(p_value);
	if (parsed < 0) {
		return 0;
	}
	if (static_cast<uint64_t>(parsed) > static_cast<uint64_t>(UINT32_MAX)) {
		return UINT32_MAX;
	}
	return static_cast<uint32_t>(parsed);
}

static uint64_t _read_non_negative_u64(const Variant &p_value, uint64_t p_default) {
	if (p_value.get_type() == Variant::NIL) {
		return p_default;
	}
	const uint64_t max_u64 = uint64_t(-1);

	switch (p_value.get_type()) {
		case Variant::INT: {
			const int64_t parsed = static_cast<int64_t>(p_value);
			if (parsed < 0) {
				return 0;
			}
			return static_cast<uint64_t>(parsed);
		}
		case Variant::FLOAT: {
			const double parsed = static_cast<double>(p_value);
			if (parsed < 0.0) {
				return 0;
			}
			if (parsed >= static_cast<double>(max_u64)) {
				return max_u64;
			}
			return static_cast<uint64_t>(Math::round(parsed));
		}
		case Variant::STRING: {
			const String raw_text = static_cast<String>(p_value).strip_edges();
			if (raw_text.is_empty()) {
				return p_default;
			}
			int index = 0;
			uint64_t base = 10;
			if (raw_text.length() > 2 && raw_text[0] == '0' &&
					(raw_text[1] == 'x' || raw_text[1] == 'X')) {
				base = 16;
				index = 2;
			}
			if (index >= raw_text.length()) {
				return p_default;
			}

			uint64_t value = 0;
			for (int i = index; i < raw_text.length(); i++) {
				const char32_t c = raw_text[i];
				uint64_t digit = 0;
				if (c >= '0' && c <= '9') {
					digit = static_cast<uint64_t>(c - '0');
				} else if (base == 16 && c >= 'a' && c <= 'f') {
					digit = static_cast<uint64_t>((c - 'a') + 10);
				} else if (base == 16 && c >= 'A' && c <= 'F') {
					digit = static_cast<uint64_t>((c - 'A') + 10);
				} else {
					return p_default;
				}

				if (digit >= base) {
					return p_default;
				}
				if (value > (max_u64 - digit) / base) {
					return p_default;
				}
				value = (value * base) + digit;
			}
			return value;
		}
		default:
			return p_default;
	}
}

} // namespace

Dictionary SyntheticSceneSummary::to_dict() const {
	Dictionary dict;
	dict["generator"] = generator_name;
	dict["splat_count"] = static_cast<int64_t>(splat_count);
	dict["seed"] = _hash_hex(seed);
	dict["config_hash"] = _hash_hex(config_hash);
	dict["scene_hash"] = _hash_hex(scene_hash);
	dict["bounds_min"] = bounds_min;
	dict["bounds_max"] = bounds_max;
	dict["average_scale"] = average_scale;
	dict["average_opacity"] = average_opacity;
	return dict;
}

Dictionary UniformSplatGenerator::Config::to_dict() const {
	Dictionary dict;
	dict["splat_count"] = static_cast<int64_t>(splat_count);
	dict["seed"] = _hash_hex(seed);
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
	return dict;
}

void UniformSplatGenerator::Config::from_dict(const Dictionary &p_dict) {
	if (p_dict.has("splat_count")) {
		splat_count = _read_non_negative_u32(p_dict["splat_count"], splat_count);
	}
	if (p_dict.has("seed")) {
		seed = _read_non_negative_u64(p_dict["seed"], seed);
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
}

uint64_t UniformSplatGenerator::hash_config(const Config &p_config) {
	uint64_t hash = HASH_BASIS;
	hash = _hash_u64(hash, p_config.splat_count);
	hash = _hash_u64(hash, p_config.seed);
	hash = _hash_vector3(hash, p_config.position_min);
	hash = _hash_vector3(hash, p_config.position_max);
	hash = _hash_float(hash, p_config.min_scale);
	hash = _hash_float(hash, p_config.max_scale);
	hash = _hash_float(hash, p_config.min_opacity);
	hash = _hash_float(hash, p_config.max_opacity);
	hash = _hash_float(hash, p_config.normal_tilt);
	hash = _hash_bool(hash, p_config.random_rotation);
	hash = _hash_bool(hash, p_config.random_colors);
	hash = _hash_color(hash, p_config.base_color);
	return hash;
}

LocalVector<Gaussian> UniformSplatGenerator::generate(const Config &p_config, SyntheticSceneSummary *r_summary) {
	Config config = p_config;
	_normalize_range_vec3(config.position_min, config.position_max);
	_normalize_range(config.min_scale, config.max_scale);
	_normalize_range(config.min_opacity, config.max_opacity);

	LocalVector<Gaussian> splats;
	splats.resize(config.splat_count);

	DeterministicRng rng(config.seed);
	for (uint32_t i = 0; i < config.splat_count; i++) {
		Gaussian gaussian = _make_base_gaussian();

		gaussian.position = Vector3(
				rng.range(config.position_min.x, config.position_max.x),
				rng.range(config.position_min.y, config.position_max.y),
				rng.range(config.position_min.z, config.position_max.z));

		const float scale = rng.range(config.min_scale, config.max_scale);
		gaussian.scale = Vector3(scale, scale, scale);
		gaussian.area = scale * scale * static_cast<float>(Math::PI);
		gaussian.opacity = rng.range(config.min_opacity, config.max_opacity);
		gaussian.normal = _make_normal(rng, config.normal_tilt);

		if (config.random_rotation) {
			gaussian.rotation = _random_unit_quaternion(rng);
		}

		if (config.random_colors) {
			gaussian.sh_dc = Color(rng.next_unit_float(), rng.next_unit_float(), rng.next_unit_float(), gaussian.opacity);
		} else {
			gaussian.sh_dc = Color(config.base_color.r, config.base_color.g, config.base_color.b, gaussian.opacity);
		}

		splats[i] = gaussian;
	}

	if (r_summary != nullptr) {
		*r_summary = summarize_generated_scene(splats, "uniform", config.seed, hash_config(config));
	}

	return splats;
}

Dictionary ClusteredSplatGenerator::Config::to_dict() const {
	Dictionary dict;
	dict["splat_count"] = static_cast<int64_t>(splat_count);
	dict["seed"] = _hash_hex(seed);
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
	return dict;
}

void ClusteredSplatGenerator::Config::from_dict(const Dictionary &p_dict) {
	if (p_dict.has("splat_count")) {
		splat_count = _read_non_negative_u32(p_dict["splat_count"], splat_count);
	}
	if (p_dict.has("seed")) {
		seed = _read_non_negative_u64(p_dict["seed"], seed);
	}
	if (p_dict.has("cluster_count")) {
		cluster_count = _read_non_negative_u32(p_dict["cluster_count"], cluster_count);
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
}

uint64_t ClusteredSplatGenerator::hash_config(const Config &p_config) {
	uint64_t hash = HASH_BASIS;
	hash = _hash_u64(hash, p_config.splat_count);
	hash = _hash_u64(hash, p_config.seed);
	hash = _hash_u64(hash, p_config.cluster_count);
	hash = _hash_vector3(hash, p_config.cluster_center_min);
	hash = _hash_vector3(hash, p_config.cluster_center_max);
	hash = _hash_vector3(hash, p_config.center_offset);
	hash = _hash_float(hash, p_config.cluster_radius);
	hash = _hash_float(hash, p_config.min_scale);
	hash = _hash_float(hash, p_config.max_scale);
	hash = _hash_float(hash, p_config.min_opacity);
	hash = _hash_float(hash, p_config.max_opacity);
	hash = _hash_float(hash, p_config.normal_tilt);
	hash = _hash_bool(hash, p_config.random_rotation);
	hash = _hash_bool(hash, p_config.color_per_cluster);
	return hash;
}

LocalVector<Gaussian> ClusteredSplatGenerator::generate(const Config &p_config, SyntheticSceneSummary *r_summary) {
	Config config = p_config;
	config.cluster_count = MAX<uint32_t>(1u, config.cluster_count);
	_normalize_range_vec3(config.cluster_center_min, config.cluster_center_max);
	_normalize_range(config.min_scale, config.max_scale);
	_normalize_range(config.min_opacity, config.max_opacity);
	if (config.cluster_radius < 0.0f) {
		config.cluster_radius = 0.0f;
	}

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
		Gaussian gaussian = _make_base_gaussian();
		const uint32_t cluster_idx = i % config.cluster_count;
		const Vector3 &center = centers[cluster_idx];

		gaussian.position = center + Vector3(
											rng.range(-config.cluster_radius, config.cluster_radius),
											rng.range(-config.cluster_radius, config.cluster_radius),
											rng.range(-config.cluster_radius, config.cluster_radius));

		const float scale = rng.range(config.min_scale, config.max_scale);
		gaussian.scale = Vector3(scale, scale, scale);
		gaussian.area = scale * scale * static_cast<float>(Math::PI);
		gaussian.opacity = rng.range(config.min_opacity, config.max_opacity);
		gaussian.normal = _make_normal(rng, config.normal_tilt);

		if (config.random_rotation) {
			gaussian.rotation = _random_unit_quaternion(rng);
		}

		if (config.color_per_cluster) {
			gaussian.sh_dc = _cluster_color(config.seed, cluster_idx, gaussian.opacity);
		} else {
			gaussian.sh_dc = Color(rng.next_unit_float(), rng.next_unit_float(), rng.next_unit_float(), gaussian.opacity);
		}

		splats[i] = gaussian;
	}

	if (r_summary != nullptr) {
		*r_summary = summarize_generated_scene(splats, "clustered", config.seed, hash_config(config));
	}

	return splats;
}

SyntheticSceneSummary summarize_generated_scene(const LocalVector<Gaussian> &p_splats, const String &p_generator_name, uint64_t p_seed, uint64_t p_config_hash) {
	SyntheticSceneSummary summary;
	summary.generator_name = p_generator_name;
	summary.splat_count = p_splats.size();
	summary.seed = p_seed;
	summary.config_hash = p_config_hash;
	summary.scene_hash = HASH_BASIS;
	summary.scene_hash = _hash_u64(summary.scene_hash, p_seed);
	summary.scene_hash = _hash_u64(summary.scene_hash, p_config_hash);
	summary.scene_hash = _hash_u64(summary.scene_hash, p_splats.size());

	if (p_splats.is_empty()) {
		return summary;
	}

	Vector3 bounds_min = p_splats[0].position;
	Vector3 bounds_max = p_splats[0].position;
	double scale_accum = 0.0;
	double opacity_accum = 0.0;

	for (const Gaussian &gaussian : p_splats) {
		bounds_min = bounds_min.min(gaussian.position);
		bounds_max = bounds_max.max(gaussian.position);
		scale_accum += (gaussian.scale.x + gaussian.scale.y + gaussian.scale.z) / 3.0;
		opacity_accum += gaussian.opacity;
		summary.scene_hash = _hash_gaussian(summary.scene_hash, gaussian);
	}

	summary.bounds_min = bounds_min;
	summary.bounds_max = bounds_max;
	summary.average_scale = static_cast<float>(scale_accum / static_cast<double>(p_splats.size()));
	summary.average_opacity = static_cast<float>(opacity_accum / static_cast<double>(p_splats.size()));
	return summary;
}

} // namespace TestGaussianSplatting
