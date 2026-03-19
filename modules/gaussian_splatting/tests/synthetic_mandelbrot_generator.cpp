/**************************************************************************/
/*  synthetic_mandelbrot_generator.cpp                                   */
/**************************************************************************/

#include "synthetic_splat_generators.h"

namespace TestGaussianSplatting {

using namespace detail;

namespace {

static float _mandelbrot_iterate(double p_cr, double p_ci, uint32_t p_max_iter) {
	double zr = 0.0, zi = 0.0;
	double zr2 = 0.0, zi2 = 0.0;
	uint32_t iter = 0;
	while (zr2 + zi2 <= 4.0 && iter < p_max_iter) {
		zi = 2.0 * zr * zi + p_ci;
		zr = zr2 - zi2 + p_cr;
		zr2 = zr * zr;
		zi2 = zi * zi;
		iter++;
	}
	if (iter == p_max_iter) {
		return -1.0f;
	}
	const double log_zn = Math::log(zr2 + zi2) * 0.5;
	const double nu = Math::log(log_zn / Math::log(2.0)) / Math::log(2.0);
	return static_cast<float>((static_cast<double>(iter) + 1.0 - nu) / static_cast<double>(p_max_iter));
}

static Color _mandelbrot_color(float p_t, MandelbrotSplatGenerator::ColorMode p_mode) {
	if (p_t < 0.0f) {
		return Color(0.02f, 0.02f, 0.05f, 1.0f);
	}
	const float t = CLAMP(p_t, 0.0f, 1.0f);
	switch (p_mode) {
		case MandelbrotSplatGenerator::COLOR_FIRE:
			return Color(CLAMP(t * 3.0f, 0.0f, 1.0f), CLAMP(t * 3.0f - 1.0f, 0.0f, 1.0f), CLAMP(t * 3.0f - 2.0f, 0.0f, 1.0f), 1.0f);
		case MandelbrotSplatGenerator::COLOR_ICE:
			return Color(CLAMP(0.5f + 0.5f * Math::sin(t * 6.0f), 0.0f, 1.0f), CLAMP(0.7f + 0.3f * Math::cos(t * 4.0f), 0.0f, 1.0f), CLAMP(0.8f + 0.2f * Math::sin(t * 8.0f + 1.0f), 0.0f, 1.0f), 1.0f);
		case MandelbrotSplatGenerator::COLOR_MONOCHROME: {
			const float v = CLAMP(Math::sqrt(t), 0.0f, 1.0f);
			return Color(v, v, v, 1.0f);
		}
		default: {
			const float phase = t * 10.0f;
			return Color(CLAMP(0.5f + 0.5f * Math::sin(phase), 0.0f, 1.0f), CLAMP(0.5f + 0.5f * Math::sin(phase + 2.094f), 0.0f, 1.0f), CLAMP(0.5f + 0.5f * Math::sin(phase + 4.189f), 0.0f, 1.0f), 1.0f);
		}
	}
}

} // namespace

Dictionary MandelbrotSplatGenerator::Config::to_dict() const {
	Dictionary dict;
	dict["splat_count"] = static_cast<int64_t>(splat_count);
	dict["seed"] = hash_hex(seed);
	dict["center_real"] = center_real;
	dict["center_imag"] = center_imag;
	dict["zoom"] = zoom;
	dict["max_iterations"] = static_cast<int64_t>(max_iterations);
	dict["min_scale"] = min_scale;
	dict["max_scale"] = max_scale;
	dict["depth_extrude"] = depth_extrude;
	dict["anisotropy"] = anisotropy;
	dict["sh_intensity"] = sh_intensity;
	dict["min_opacity"] = min_opacity;
	dict["max_opacity"] = max_opacity;
	dict["color_mode"] = static_cast<int64_t>(color_mode);
	return dict;
}

void MandelbrotSplatGenerator::Config::from_dict(const Dictionary &p_dict) {
	if (p_dict.has("splat_count")) { splat_count = read_non_negative_u32(p_dict["splat_count"], splat_count); }
	if (p_dict.has("seed")) { seed = read_non_negative_u64(p_dict["seed"], seed); }
	if (p_dict.has("center_real")) { center_real = p_dict["center_real"]; }
	if (p_dict.has("center_imag")) { center_imag = p_dict["center_imag"]; }
	if (p_dict.has("zoom")) { zoom = p_dict["zoom"]; }
	if (p_dict.has("max_iterations")) { max_iterations = read_non_negative_u32(p_dict["max_iterations"], max_iterations); }
	if (p_dict.has("min_scale")) { min_scale = p_dict["min_scale"]; }
	if (p_dict.has("max_scale")) { max_scale = p_dict["max_scale"]; }
	if (p_dict.has("depth_extrude")) { depth_extrude = p_dict["depth_extrude"]; }
	if (p_dict.has("anisotropy")) { anisotropy = p_dict["anisotropy"]; }
	if (p_dict.has("sh_intensity")) { sh_intensity = p_dict["sh_intensity"]; }
	if (p_dict.has("min_opacity")) { min_opacity = p_dict["min_opacity"]; }
	if (p_dict.has("max_opacity")) { max_opacity = p_dict["max_opacity"]; }
	if (p_dict.has("color_mode")) { const int m = static_cast<int>(p_dict["color_mode"]); if (m >= 0 && m <= COLOR_MONOCHROME) { color_mode = static_cast<ColorMode>(m); } }
}

uint64_t MandelbrotSplatGenerator::hash_config(const Config &p_config) {
	uint64_t hash = HASH_BASIS;
	hash = hash_u64(hash, p_config.splat_count);
	hash = hash_u64(hash, p_config.seed);
	hash = hash_float(hash, static_cast<float>(p_config.center_real));
	hash = hash_float(hash, static_cast<float>(p_config.center_imag));
	hash = hash_float(hash, static_cast<float>(p_config.zoom));
	hash = hash_u64(hash, p_config.max_iterations);
	hash = hash_float(hash, p_config.min_scale);
	hash = hash_float(hash, p_config.max_scale);
	hash = hash_float(hash, p_config.depth_extrude);
	hash = hash_float(hash, p_config.anisotropy);
	hash = hash_float(hash, p_config.sh_intensity);
	hash = hash_float(hash, p_config.min_opacity);
	hash = hash_float(hash, p_config.max_opacity);
	hash = hash_u64(hash, static_cast<uint64_t>(p_config.color_mode));
	return hash;
}

LocalVector<Gaussian> MandelbrotSplatGenerator::generate(const Config &p_config, SyntheticSceneSummary *r_summary) {
	Config config = p_config;
	normalize_range(config.min_scale, config.max_scale);
	normalize_range(config.min_opacity, config.max_opacity);
	config.anisotropy = CLAMP(config.anisotropy, 0.0f, 1.0f);
	config.sh_intensity = MAX(config.sh_intensity, 0.0f);
	config.max_iterations = MAX<uint32_t>(1u, config.max_iterations);
	config.zoom = MAX(config.zoom, 1e-12);

	LocalVector<Gaussian> splats;
	splats.resize(config.splat_count);

	DeterministicRng rng(config.seed);
	const float world_scale = 10.0f;

	for (uint32_t i = 0; i < config.splat_count; i++) {
		Gaussian gaussian = make_base_gaussian();

		const double cr = config.center_real + (rng.range(-1.0f, 1.0f)) * config.zoom;
		const double ci = config.center_imag + (rng.range(-1.0f, 1.0f)) * config.zoom * 0.75;
		const float t = _mandelbrot_iterate(cr, ci, config.max_iterations);

		const float norm_x = static_cast<float>((cr - config.center_real) / config.zoom);
		const float norm_y = static_cast<float>((ci - config.center_imag) / (config.zoom * 0.75));
		const float depth = (t >= 0.0f) ? (t * config.depth_extrude) : 0.0f;
		gaussian.position = Vector3(norm_x * world_scale, norm_y * world_scale, depth);

		float base_scale;
		if (t < 0.0f) {
			base_scale = config.max_scale;
		} else {
			base_scale = Math::lerp(config.min_scale, config.max_scale, 1.0f - t);
		}
		gaussian.scale = anisotropic_scale(rng, base_scale, config.anisotropy);
		gaussian.area = gaussian.scale.x * gaussian.scale.z * static_cast<float>(Math::PI);

		if (t < 0.0f) {
			gaussian.opacity = config.max_opacity;
		} else {
			gaussian.opacity = Math::lerp(config.min_opacity, config.max_opacity, CLAMP(1.0f - t * 0.7f, 0.0f, 1.0f));
		}

		const Color base_col = _mandelbrot_color(t, config.color_mode);
		gaussian.sh_dc = Color(base_col.r, base_col.g, base_col.b, gaussian.opacity);
		gaussian.normal = Vector3(0.0f, 0.0f, 1.0f);

		generate_sh1(rng, gaussian, config.sh_intensity);
		splats[i] = gaussian;
	}

	if (r_summary != nullptr) {
		*r_summary = summarize_generated_scene(splats, "mandelbrot", config.seed, hash_config(config));
	}
	return splats;
}

} // namespace TestGaussianSplatting
