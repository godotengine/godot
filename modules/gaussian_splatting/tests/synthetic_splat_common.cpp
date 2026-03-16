/**************************************************************************/
/*  synthetic_splat_common.cpp                                           */
/*  Shared implementations for synthetic splat generators.               */
/**************************************************************************/

#include "synthetic_splat_common.h"

namespace TestGaussianSplatting {

Dictionary SyntheticSceneSummary::to_dict() const {
	Dictionary dict;
	dict["generator"] = generator_name;
	dict["splat_count"] = static_cast<int64_t>(splat_count);
	dict["seed"] = detail::hash_hex(seed);
	dict["config_hash"] = detail::hash_hex(config_hash);
	dict["scene_hash"] = detail::hash_hex(scene_hash);
	dict["bounds_min"] = bounds_min;
	dict["bounds_max"] = bounds_max;
	dict["average_scale"] = average_scale;
	dict["average_opacity"] = average_opacity;
	return dict;
}

SyntheticSceneSummary summarize_generated_scene(
		const LocalVector<Gaussian> &p_splats,
		const String &p_generator_name,
		uint64_t p_seed,
		uint64_t p_config_hash) {
	SyntheticSceneSummary summary;
	summary.generator_name = p_generator_name;
	summary.splat_count = p_splats.size();
	summary.seed = p_seed;
	summary.config_hash = p_config_hash;
	summary.scene_hash = detail::HASH_BASIS;
	summary.scene_hash = detail::hash_u64(summary.scene_hash, p_seed);
	summary.scene_hash = detail::hash_u64(summary.scene_hash, p_config_hash);
	summary.scene_hash = detail::hash_u64(summary.scene_hash, p_splats.size());

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
		summary.scene_hash = detail::hash_gaussian(summary.scene_hash, gaussian);
	}

	summary.bounds_min = bounds_min;
	summary.bounds_max = bounds_max;
	summary.average_scale = static_cast<float>(scale_accum / static_cast<double>(p_splats.size()));
	summary.average_opacity = static_cast<float>(opacity_accum / static_cast<double>(p_splats.size()));
	return summary;
}

namespace detail {

String hash_hex(uint64_t p_value) {
	return "0x" + String::num_uint64(p_value, 16).pad_zeros(16);
}

uint32_t read_non_negative_u32(const Variant &p_value, uint32_t p_default) {
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

uint64_t read_non_negative_u64(const Variant &p_value, uint64_t p_default) {
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

} // namespace detail

} // namespace TestGaussianSplatting
