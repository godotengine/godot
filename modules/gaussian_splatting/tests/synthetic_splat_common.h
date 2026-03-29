/**************************************************************************/
/*  synthetic_splat_common.h                                             */
/*  Shared infrastructure for deterministic synthetic splat generators.   */
/**************************************************************************/

#pragma once

#include "../core/gaussian_data.h"

#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/string/ustring.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

#include <cstdint>

namespace TestGaussianSplatting {

// ── Scene summary ────────────────────────────────────────────────────────

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

SyntheticSceneSummary summarize_generated_scene(
		const LocalVector<Gaussian> &p_splats,
		const String &p_generator_name,
		uint64_t p_seed,
		uint64_t p_config_hash);

// ── Deterministic RNG (SplitMix64) ──────────────────────────────────────

namespace detail {

constexpr uint64_t HASH_BASIS = 1469598103934665603ull;
constexpr uint64_t HASH_PRIME = 1099511628211ull;
constexpr float HASH_QUANTIZATION_SCALE = 100000.0f;

class DeterministicRng {
	uint64_t state = 0;

public:
	explicit DeterministicRng(uint64_t p_seed) :
			state(p_seed) {}

	inline uint64_t next_u64() {
		state += 0x9E3779B97F4A7C15ull;
		uint64_t z = state;
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
		return z ^ (z >> 31);
	}

	inline float next_unit_float() {
		constexpr double inv = 1.0 / static_cast<double>(1ull << 24);
		const uint64_t bits = next_u64() >> 40;
		return static_cast<float>(bits * inv);
	}

	inline float range(float p_min, float p_max) {
		return p_min + (p_max - p_min) * next_unit_float();
	}

	inline float next_gaussian() {
		const float u1 = MAX(next_unit_float(), 1e-7f);
		const float u2 = next_unit_float();
		return Math::sqrt(-2.0f * Math::log(u1)) * Math::cos(2.0f * static_cast<float>(Math::PI) * u2);
	}
};

// ── Range normalization ─────────────────────────────────────────────────

inline void normalize_range(float &r_min, float &r_max) {
	if (r_max < r_min) {
		SWAP(r_min, r_max);
	}
}

inline void normalize_range_vec3(Vector3 &r_min, Vector3 &r_max) {
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

// ── Hashing helpers (FNV-1a) ────────────────────────────────────────────

inline uint64_t fnv1a_u64(uint64_t p_hash, uint64_t p_value) {
	return (p_hash ^ p_value) * HASH_PRIME;
}

inline uint64_t hash_bool(uint64_t p_hash, bool p_value) {
	return fnv1a_u64(p_hash, p_value ? 1ull : 0ull);
}

inline uint64_t hash_i64(uint64_t p_hash, int64_t p_value) {
	return fnv1a_u64(p_hash, static_cast<uint64_t>(p_value));
}

inline uint64_t hash_u64(uint64_t p_hash, uint64_t p_value) {
	return fnv1a_u64(p_hash, p_value);
}

inline uint64_t hash_float(uint64_t p_hash, float p_value) {
	const int64_t quantized = static_cast<int64_t>(Math::round(p_value * HASH_QUANTIZATION_SCALE));
	return hash_i64(p_hash, quantized);
}

inline uint64_t hash_vector3(uint64_t p_hash, const Vector3 &p_vec) {
	p_hash = hash_float(p_hash, p_vec.x);
	p_hash = hash_float(p_hash, p_vec.y);
	p_hash = hash_float(p_hash, p_vec.z);
	return p_hash;
}

inline uint64_t hash_vector2(uint64_t p_hash, const Vector2 &p_vec) {
	p_hash = hash_float(p_hash, p_vec.x);
	p_hash = hash_float(p_hash, p_vec.y);
	return p_hash;
}

inline uint64_t hash_quaternion(uint64_t p_hash, const Quaternion &p_quat) {
	p_hash = hash_float(p_hash, p_quat.x);
	p_hash = hash_float(p_hash, p_quat.y);
	p_hash = hash_float(p_hash, p_quat.z);
	p_hash = hash_float(p_hash, p_quat.w);
	return p_hash;
}

inline uint64_t hash_color(uint64_t p_hash, const Color &p_color) {
	p_hash = hash_float(p_hash, p_color.r);
	p_hash = hash_float(p_hash, p_color.g);
	p_hash = hash_float(p_hash, p_color.b);
	p_hash = hash_float(p_hash, p_color.a);
	return p_hash;
}

inline uint64_t hash_gaussian(uint64_t p_hash, const Gaussian &p_gaussian) {
	p_hash = hash_vector3(p_hash, p_gaussian.position);
	p_hash = hash_vector3(p_hash, p_gaussian.scale);
	p_hash = hash_float(p_hash, p_gaussian.area);
	p_hash = hash_float(p_hash, p_gaussian.opacity);
	p_hash = hash_quaternion(p_hash, p_gaussian.rotation);
	p_hash = hash_color(p_hash, p_gaussian.sh_dc);
	for (uint32_t i = 0; i < 3; i++) {
		p_hash = hash_vector3(p_hash, p_gaussian.sh_1[i]);
	}
	p_hash = hash_vector3(p_hash, p_gaussian.normal);
	p_hash = hash_float(p_hash, p_gaussian.stroke_age);
	p_hash = hash_vector2(p_hash, p_gaussian.brush_axes);
	p_hash = hash_u64(p_hash, p_gaussian.painterly_meta);
	return p_hash;
}

String hash_hex(uint64_t p_value);

// ── Geometry helpers ────────────────────────────────────────────────────

inline Quaternion random_unit_quaternion(DeterministicRng &p_rng) {
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

inline Vector3 make_normal(DeterministicRng &p_rng, float p_normal_tilt) {
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

inline Quaternion orient_to_normal(const Vector3 &p_normal) {
	const Vector3 up(0.0f, 1.0f, 0.0f);
	const Vector3 n = p_normal.normalized();
	const float dot = up.dot(n);
	if (dot > 0.9999f) {
		return Quaternion();
	}
	if (dot < -0.9999f) {
		return Quaternion(1.0f, 0.0f, 0.0f, 0.0f);
	}
	const Vector3 axis = up.cross(n).normalized();
	const float angle = Math::acos(CLAMP(dot, -1.0f, 1.0f));
	return Quaternion(axis, angle);
}

// ── Splat property helpers ──────────────────────────────────────────────

inline Gaussian make_base_gaussian() {
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
	return gaussian;
}

inline Vector3 anisotropic_scale(DeterministicRng &p_rng, float p_base_scale, float p_anisotropy) {
	if (p_anisotropy <= 0.0f) {
		return Vector3(p_base_scale, p_base_scale, p_base_scale);
	}
	const float r1 = Math::lerp(1.0f, p_rng.range(0.1f, 1.0f), p_anisotropy);
	const float r2 = Math::lerp(1.0f, p_rng.range(0.1f, 1.0f), p_anisotropy);
	return Vector3(p_base_scale * r1, p_base_scale, p_base_scale * r2);
}

inline void generate_sh1(DeterministicRng &p_rng, Gaussian &r_gaussian, float p_intensity) {
	if (p_intensity <= 0.0f) {
		return;
	}
	for (int band = 0; band < 3; band++) {
		r_gaussian.sh_1[band] = Vector3(
				p_rng.range(-p_intensity, p_intensity),
				p_rng.range(-p_intensity, p_intensity),
				p_rng.range(-p_intensity, p_intensity));
	}
}

inline float log_normal_scale(DeterministicRng &p_rng, float p_min, float p_max) {
	const float log_min = Math::log(MAX(p_min, 1e-6f));
	const float log_max = Math::log(MAX(p_max, 1e-6f));
	const float log_mid = (log_min + log_max) * 0.5f;
	const float log_range = (log_max - log_min) * 0.25f;
	const float log_scale = log_mid + p_rng.next_gaussian() * log_range;
	return CLAMP(Math::exp(log_scale), p_min, p_max);
}

// ── Dictionary read helpers ─────────────────────────────────────────────

uint32_t read_non_negative_u32(const Variant &p_value, uint32_t p_default);
uint64_t read_non_negative_u64(const Variant &p_value, uint64_t p_default);

} // namespace detail

} // namespace TestGaussianSplatting
