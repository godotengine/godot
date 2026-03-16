/**************************************************************************/
/*  synthetic_surface_generator.cpp                                      */
/**************************************************************************/

#include "synthetic_splat_generators.h"

namespace TestGaussianSplatting {

using namespace detail;

Dictionary SurfaceSplatGenerator::Config::to_dict() const {
	Dictionary dict;
	dict["splat_count"] = static_cast<int64_t>(splat_count);
	dict["seed"] = hash_hex(seed);
	dict["shape"] = static_cast<int64_t>(shape);
	dict["center"] = center;
	dict["shape_radius"] = shape_radius;
	dict["torus_tube_radius"] = torus_tube_radius;
	dict["plane_half_extent"] = plane_half_extent;
	dict["min_scale"] = min_scale;
	dict["max_scale"] = max_scale;
	dict["anisotropy"] = anisotropy;
	dict["surface_noise"] = surface_noise;
	dict["min_opacity"] = min_opacity;
	dict["max_opacity"] = max_opacity;
	dict["sh_intensity"] = sh_intensity;
	dict["random_colors"] = random_colors;
	dict["base_color"] = base_color;
	dict["color_variation"] = color_variation;
	return dict;
}

void SurfaceSplatGenerator::Config::from_dict(const Dictionary &p_dict) {
	if (p_dict.has("splat_count")) { splat_count = read_non_negative_u32(p_dict["splat_count"], splat_count); }
	if (p_dict.has("seed")) { seed = read_non_negative_u64(p_dict["seed"], seed); }
	if (p_dict.has("shape")) { const int s = static_cast<int>(p_dict["shape"]); if (s >= 0 && s <= SHAPE_CUBE) { shape = static_cast<Shape>(s); } }
	if (p_dict.has("center")) { center = p_dict["center"]; }
	if (p_dict.has("shape_radius")) { shape_radius = p_dict["shape_radius"]; }
	if (p_dict.has("torus_tube_radius")) { torus_tube_radius = p_dict["torus_tube_radius"]; }
	if (p_dict.has("plane_half_extent")) { plane_half_extent = p_dict["plane_half_extent"]; }
	if (p_dict.has("min_scale")) { min_scale = p_dict["min_scale"]; }
	if (p_dict.has("max_scale")) { max_scale = p_dict["max_scale"]; }
	if (p_dict.has("anisotropy")) { anisotropy = p_dict["anisotropy"]; }
	if (p_dict.has("surface_noise")) { surface_noise = p_dict["surface_noise"]; }
	if (p_dict.has("min_opacity")) { min_opacity = p_dict["min_opacity"]; }
	if (p_dict.has("max_opacity")) { max_opacity = p_dict["max_opacity"]; }
	if (p_dict.has("sh_intensity")) { sh_intensity = p_dict["sh_intensity"]; }
	if (p_dict.has("random_colors")) { random_colors = p_dict["random_colors"]; }
	if (p_dict.has("base_color")) { base_color = p_dict["base_color"]; }
	if (p_dict.has("color_variation")) { color_variation = p_dict["color_variation"]; }
}

uint64_t SurfaceSplatGenerator::hash_config(const Config &p_config) {
	uint64_t hash = HASH_BASIS;
	hash = hash_u64(hash, p_config.splat_count);
	hash = hash_u64(hash, p_config.seed);
	hash = hash_u64(hash, static_cast<uint64_t>(p_config.shape));
	hash = hash_vector3(hash, p_config.center);
	hash = hash_float(hash, p_config.shape_radius);
	hash = hash_float(hash, p_config.torus_tube_radius);
	hash = hash_float(hash, p_config.plane_half_extent);
	hash = hash_float(hash, p_config.min_scale);
	hash = hash_float(hash, p_config.max_scale);
	hash = hash_float(hash, p_config.anisotropy);
	hash = hash_float(hash, p_config.surface_noise);
	hash = hash_float(hash, p_config.min_opacity);
	hash = hash_float(hash, p_config.max_opacity);
	hash = hash_float(hash, p_config.sh_intensity);
	hash = hash_bool(hash, p_config.random_colors);
	hash = hash_color(hash, p_config.base_color);
	hash = hash_float(hash, p_config.color_variation);
	return hash;
}

LocalVector<Gaussian> SurfaceSplatGenerator::generate(const Config &p_config, SyntheticSceneSummary *r_summary) {
	Config config = p_config;
	normalize_range(config.min_scale, config.max_scale);
	normalize_range(config.min_opacity, config.max_opacity);
	config.anisotropy = CLAMP(config.anisotropy, 0.0f, 1.0f);
	config.sh_intensity = MAX(config.sh_intensity, 0.0f);
	config.surface_noise = MAX(config.surface_noise, 0.0f);
	config.color_variation = MAX(config.color_variation, 0.0f);
	config.shape_radius = MAX(config.shape_radius, 0.01f);
	config.torus_tube_radius = CLAMP(config.torus_tube_radius, 0.01f, config.shape_radius * 0.95f);
	config.plane_half_extent = MAX(config.plane_half_extent, 0.01f);

	LocalVector<Gaussian> splats;
	splats.resize(config.splat_count);

	DeterministicRng rng(config.seed);
	const float pi = static_cast<float>(Math::PI);
	const float two_pi = 2.0f * pi;

	for (uint32_t i = 0; i < config.splat_count; i++) {
		Gaussian gaussian = make_base_gaussian();
		Vector3 surface_pos;
		Vector3 surface_normal;

		switch (config.shape) {
			case SHAPE_SPHERE: {
				const float theta = rng.range(0.0f, two_pi);
				const float phi = Math::acos(rng.range(-1.0f, 1.0f));
				const float sp = Math::sin(phi);
				surface_normal = Vector3(sp * Math::cos(theta), Math::cos(phi), sp * Math::sin(theta));
				surface_pos = surface_normal * config.shape_radius;
			} break;
			case SHAPE_TORUS: {
				const float u = rng.range(0.0f, two_pi);
				const float v = rng.range(0.0f, two_pi);
				const float R = config.shape_radius;
				const float r = config.torus_tube_radius;
				const float cv = Math::cos(v), sv = Math::sin(v);
				const float cu = Math::cos(u), su = Math::sin(u);
				surface_pos = Vector3((R + r * cv) * cu, r * sv, (R + r * cv) * su);
				surface_normal = Vector3(cv * cu, sv, cv * su);
			} break;
			case SHAPE_PLANE: {
				const float hx = config.plane_half_extent;
				surface_pos = Vector3(rng.range(-hx, hx), 0.0f, rng.range(-hx, hx));
				surface_normal = Vector3(0.0f, 1.0f, 0.0f);
			} break;
			case SHAPE_CUBE: {
				const int face = static_cast<int>(rng.next_u64() % 6);
				const float rv = config.shape_radius;
				const float a = rng.range(-rv, rv);
				const float b = rng.range(-rv, rv);
				switch (face) {
					case 0: surface_pos = Vector3(rv, a, b); surface_normal = Vector3(1, 0, 0); break;
					case 1: surface_pos = Vector3(-rv, a, b); surface_normal = Vector3(-1, 0, 0); break;
					case 2: surface_pos = Vector3(a, rv, b); surface_normal = Vector3(0, 1, 0); break;
					case 3: surface_pos = Vector3(a, -rv, b); surface_normal = Vector3(0, -1, 0); break;
					case 4: surface_pos = Vector3(a, b, rv); surface_normal = Vector3(0, 0, 1); break;
					default: surface_pos = Vector3(a, b, -rv); surface_normal = Vector3(0, 0, -1); break;
				}
			} break;
		}

		if (config.surface_noise > 0.0f) {
			surface_pos += surface_normal * rng.range(-config.surface_noise, config.surface_noise);
		}

		gaussian.position = config.center + surface_pos;
		gaussian.normal = surface_normal;
		gaussian.rotation = orient_to_normal(surface_normal);

		const float base_scale = log_normal_scale(rng, config.min_scale, config.max_scale);
		gaussian.scale = anisotropic_scale(rng, base_scale, config.anisotropy);
		gaussian.area = gaussian.scale.x * gaussian.scale.z * pi;
		gaussian.opacity = rng.range(config.min_opacity, config.max_opacity);

		if (config.random_colors) {
			gaussian.sh_dc = Color(rng.next_unit_float(), rng.next_unit_float(), rng.next_unit_float(), gaussian.opacity);
		} else {
			const float v_r = config.color_variation > 0.0f ? rng.range(-config.color_variation, config.color_variation) : 0.0f;
			const float v_g = config.color_variation > 0.0f ? rng.range(-config.color_variation, config.color_variation) : 0.0f;
			const float v_b = config.color_variation > 0.0f ? rng.range(-config.color_variation, config.color_variation) : 0.0f;
			gaussian.sh_dc = Color(
					CLAMP(config.base_color.r + v_r, 0.0f, 1.0f),
					CLAMP(config.base_color.g + v_g, 0.0f, 1.0f),
					CLAMP(config.base_color.b + v_b, 0.0f, 1.0f),
					gaussian.opacity);
		}

		generate_sh1(rng, gaussian, config.sh_intensity);
		splats[i] = gaussian;
	}

	if (r_summary != nullptr) {
		*r_summary = summarize_generated_scene(splats, "surface", config.seed, hash_config(config));
	}
	return splats;
}

} // namespace TestGaussianSplatting
