/**************************************************************************/
/*  synthetic_cloud_generator.cpp                                        */
/**************************************************************************/

#include "synthetic_splat_generators.h"

namespace TestGaussianSplatting {

using namespace detail;

namespace {

static float _noise_hash(int p_x, int p_y, int p_z, uint64_t p_seed) {
	uint64_t h = HASH_BASIS;
	h = fnv1a_u64(h, static_cast<uint64_t>(static_cast<uint32_t>(p_x)));
	h = fnv1a_u64(h, static_cast<uint64_t>(static_cast<uint32_t>(p_y)));
	h = fnv1a_u64(h, static_cast<uint64_t>(static_cast<uint32_t>(p_z)));
	h = fnv1a_u64(h, p_seed);
	return static_cast<float>(static_cast<int64_t>(h >> 40)) / static_cast<float>(1 << 23);
}

static float _smoothstep(float t) {
	return t * t * (3.0f - 2.0f * t);
}

static float _value_noise_3d(float p_x, float p_y, float p_z, uint64_t p_seed) {
	const int ix = static_cast<int>(Math::floor(p_x));
	const int iy = static_cast<int>(Math::floor(p_y));
	const int iz = static_cast<int>(Math::floor(p_z));
	const float fx = p_x - static_cast<float>(ix);
	const float fy = p_y - static_cast<float>(iy);
	const float fz = p_z - static_cast<float>(iz);

	const float u = _smoothstep(fx);
	const float v = _smoothstep(fy);
	const float w = _smoothstep(fz);

	const float c000 = _noise_hash(ix, iy, iz, p_seed);
	const float c100 = _noise_hash(ix + 1, iy, iz, p_seed);
	const float c010 = _noise_hash(ix, iy + 1, iz, p_seed);
	const float c110 = _noise_hash(ix + 1, iy + 1, iz, p_seed);
	const float c001 = _noise_hash(ix, iy, iz + 1, p_seed);
	const float c101 = _noise_hash(ix + 1, iy, iz + 1, p_seed);
	const float c011 = _noise_hash(ix, iy + 1, iz + 1, p_seed);
	const float c111 = _noise_hash(ix + 1, iy + 1, iz + 1, p_seed);

	const float x00 = Math::lerp(c000, c100, u);
	const float x10 = Math::lerp(c010, c110, u);
	const float x01 = Math::lerp(c001, c101, u);
	const float x11 = Math::lerp(c011, c111, u);
	const float y0 = Math::lerp(x00, x10, v);
	const float y1 = Math::lerp(x01, x11, v);
	return Math::lerp(y0, y1, w);
}

static float _fbm_3d(float p_x, float p_y, float p_z, uint64_t p_seed,
		uint32_t p_octaves, float p_frequency, float p_lacunarity, float p_persistence) {
	float amplitude = 1.0f;
	float frequency = p_frequency;
	float value = 0.0f;
	float max_amp = 0.0f;

	for (uint32_t oct = 0; oct < p_octaves; oct++) {
		value += amplitude * _value_noise_3d(
									 p_x * frequency, p_y * frequency, p_z * frequency,
									 p_seed ^ (static_cast<uint64_t>(oct) * 0x517CC1B727220A95ull));
		max_amp += amplitude;
		amplitude *= p_persistence;
		frequency *= p_lacunarity;
	}
	return value / max_amp;
}

} // namespace

Dictionary CloudSplatGenerator::Config::to_dict() const {
	Dictionary dict;
	dict["splat_count"] = static_cast<int64_t>(splat_count);
	dict["seed"] = hash_hex(seed);
	dict["cloud_center"] = cloud_center;
	dict["cloud_extent"] = cloud_extent;
	dict["density_threshold"] = density_threshold;
	dict["noise_octaves"] = static_cast<int64_t>(noise_octaves);
	dict["noise_frequency"] = noise_frequency;
	dict["noise_lacunarity"] = noise_lacunarity;
	dict["noise_persistence"] = noise_persistence;
	dict["min_scale"] = min_scale;
	dict["max_scale"] = max_scale;
	dict["min_opacity"] = min_opacity;
	dict["max_opacity"] = max_opacity;
	dict["sh_intensity"] = sh_intensity;
	dict["cloud_color"] = cloud_color;
	dict["shadow_tint"] = shadow_tint;
	dict["light_direction"] = light_direction;
	return dict;
}

void CloudSplatGenerator::Config::from_dict(const Dictionary &p_dict) {
	if (p_dict.has("splat_count")) { splat_count = read_non_negative_u32(p_dict["splat_count"], splat_count); }
	if (p_dict.has("seed")) { seed = read_non_negative_u64(p_dict["seed"], seed); }
	if (p_dict.has("cloud_center")) { cloud_center = p_dict["cloud_center"]; }
	if (p_dict.has("cloud_extent")) { cloud_extent = p_dict["cloud_extent"]; }
	if (p_dict.has("density_threshold")) { density_threshold = p_dict["density_threshold"]; }
	if (p_dict.has("noise_octaves")) { noise_octaves = read_non_negative_u32(p_dict["noise_octaves"], noise_octaves); }
	if (p_dict.has("noise_frequency")) { noise_frequency = p_dict["noise_frequency"]; }
	if (p_dict.has("noise_lacunarity")) { noise_lacunarity = p_dict["noise_lacunarity"]; }
	if (p_dict.has("noise_persistence")) { noise_persistence = p_dict["noise_persistence"]; }
	if (p_dict.has("min_scale")) { min_scale = p_dict["min_scale"]; }
	if (p_dict.has("max_scale")) { max_scale = p_dict["max_scale"]; }
	if (p_dict.has("min_opacity")) { min_opacity = p_dict["min_opacity"]; }
	if (p_dict.has("max_opacity")) { max_opacity = p_dict["max_opacity"]; }
	if (p_dict.has("sh_intensity")) { sh_intensity = p_dict["sh_intensity"]; }
	if (p_dict.has("cloud_color")) { cloud_color = p_dict["cloud_color"]; }
	if (p_dict.has("shadow_tint")) { shadow_tint = p_dict["shadow_tint"]; }
	if (p_dict.has("light_direction")) { light_direction = p_dict["light_direction"]; }
}

uint64_t CloudSplatGenerator::hash_config(const Config &p_config) {
	uint64_t hash = HASH_BASIS;
	hash = hash_u64(hash, p_config.splat_count);
	hash = hash_u64(hash, p_config.seed);
	hash = hash_vector3(hash, p_config.cloud_center);
	hash = hash_vector3(hash, p_config.cloud_extent);
	hash = hash_float(hash, p_config.density_threshold);
	hash = hash_u64(hash, p_config.noise_octaves);
	hash = hash_float(hash, p_config.noise_frequency);
	hash = hash_float(hash, p_config.noise_lacunarity);
	hash = hash_float(hash, p_config.noise_persistence);
	hash = hash_float(hash, p_config.min_scale);
	hash = hash_float(hash, p_config.max_scale);
	hash = hash_float(hash, p_config.min_opacity);
	hash = hash_float(hash, p_config.max_opacity);
	hash = hash_float(hash, p_config.sh_intensity);
	hash = hash_color(hash, p_config.cloud_color);
	hash = hash_color(hash, p_config.shadow_tint);
	hash = hash_vector3(hash, p_config.light_direction);
	return hash;
}

LocalVector<Gaussian> CloudSplatGenerator::generate(const Config &p_config, SyntheticSceneSummary *r_summary) {
	Config config = p_config;
	normalize_range(config.min_scale, config.max_scale);
	normalize_range(config.min_opacity, config.max_opacity);
	config.sh_intensity = MAX(config.sh_intensity, 0.0f);
	config.noise_octaves = CLAMP(config.noise_octaves, 1u, 12u);
	config.density_threshold = CLAMP(config.density_threshold, -1.0f, 1.0f);
	config.cloud_extent.x = MAX(config.cloud_extent.x, 0.1f);
	config.cloud_extent.y = MAX(config.cloud_extent.y, 0.1f);
	config.cloud_extent.z = MAX(config.cloud_extent.z, 0.1f);

	const Vector3 light_dir = config.light_direction.normalized();

	LocalVector<Gaussian> splats;
	splats.reserve(config.splat_count);

	DeterministicRng rng(config.seed);
	uint32_t attempts = 0;
	const uint32_t max_attempts = config.splat_count * 20;

	while (splats.size() < config.splat_count && attempts < max_attempts) {
		attempts++;

		float sx, sy, sz;
		do {
			sx = rng.range(-1.0f, 1.0f);
			sy = rng.range(-1.0f, 1.0f);
			sz = rng.range(-1.0f, 1.0f);
		} while (sx * sx + sy * sy + sz * sz > 1.0f);

		const Vector3 local_pos(sx * config.cloud_extent.x, sy * config.cloud_extent.y, sz * config.cloud_extent.z);
		const Vector3 world_pos = config.cloud_center + local_pos;

		const float noise_val = _fbm_3d(world_pos.x, world_pos.y, world_pos.z, config.seed,
				config.noise_octaves, config.noise_frequency, config.noise_lacunarity, config.noise_persistence);

		const float dist_sq = sx * sx + sy * sy + sz * sz;
		const float falloff = 1.0f - dist_sq;
		const float density = (noise_val * 0.5f + 0.5f) * falloff;

		if (density < config.density_threshold) {
			continue;
		}

		Gaussian gaussian = make_base_gaussian();
		gaussian.position = world_pos;

		const float density_norm = (density - config.density_threshold) / (1.0f - config.density_threshold);
		const float base_scale = Math::lerp(config.min_scale, config.max_scale, CLAMP(density_norm, 0.0f, 1.0f));
		gaussian.scale = Vector3(base_scale, base_scale, base_scale);
		gaussian.area = base_scale * base_scale * static_cast<float>(Math::PI);
		gaussian.opacity = Math::lerp(config.min_opacity, config.max_opacity, CLAMP(density_norm, 0.0f, 1.0f));

		const float height_factor = CLAMP(sy * 0.5f + 0.5f, 0.0f, 1.0f);
		const float sun_dot = CLAMP(Vector3(sx, sy, sz).normalized().dot(light_dir) * 0.5f + 0.5f, 0.0f, 1.0f);
		const float brightness = height_factor * 0.6f + sun_dot * 0.4f;
		gaussian.sh_dc = Color(
				Math::lerp(config.shadow_tint.r, config.cloud_color.r, brightness),
				Math::lerp(config.shadow_tint.g, config.cloud_color.g, brightness),
				Math::lerp(config.shadow_tint.b, config.cloud_color.b, brightness),
				gaussian.opacity);

		if (dist_sq > CMP_EPSILON2) {
			gaussian.normal = Vector3(sx, sy, sz).normalized();
		}

		if (config.sh_intensity > 0.0f) {
			const float sh_sun = config.sh_intensity * sun_dot;
			gaussian.sh_1[0] = Vector3(light_dir.x * sh_sun, light_dir.x * sh_sun * 0.9f, light_dir.x * sh_sun * 0.85f);
			gaussian.sh_1[1] = Vector3(light_dir.y * sh_sun, light_dir.y * sh_sun * 0.9f, light_dir.y * sh_sun * 0.85f);
			gaussian.sh_1[2] = Vector3(light_dir.z * sh_sun, light_dir.z * sh_sun * 0.9f, light_dir.z * sh_sun * 0.85f);
			for (int band = 0; band < 3; band++) {
				gaussian.sh_1[band].x += rng.range(-config.sh_intensity * 0.3f, config.sh_intensity * 0.3f);
				gaussian.sh_1[band].y += rng.range(-config.sh_intensity * 0.3f, config.sh_intensity * 0.3f);
				gaussian.sh_1[band].z += rng.range(-config.sh_intensity * 0.3f, config.sh_intensity * 0.3f);
			}
		}

		splats.push_back(gaussian);
	}

	if (r_summary != nullptr) {
		*r_summary = summarize_generated_scene(splats, "cloud", config.seed, hash_config(config));
	}
	return splats;
}

} // namespace TestGaussianSplatting
