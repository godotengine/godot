/**************************************************************************/
/*  synthetic_bml_traffic_generator.cpp                                  */
/**************************************************************************/

#include "synthetic_splat_generators.h"

namespace TestGaussianSplatting {

using namespace detail;

namespace {

enum BMLCell : uint8_t {
	BML_EMPTY = 0,
	BML_RIGHT = 1,
	BML_UP = 2,
};

} // namespace

Dictionary BMLTrafficSplatGenerator::Config::to_dict() const {
	Dictionary dict;
	dict["grid_width"] = static_cast<int64_t>(grid_width);
	dict["grid_height"] = static_cast<int64_t>(grid_height);
	dict["density"] = density;
	dict["steps"] = static_cast<int64_t>(steps);
	dict["seed"] = hash_hex(seed);
	dict["cell_size"] = cell_size;
	dict["min_scale"] = min_scale;
	dict["max_scale"] = max_scale;
	dict["anisotropy"] = anisotropy;
	dict["sh_intensity"] = sh_intensity;
	dict["min_opacity"] = min_opacity;
	dict["max_opacity"] = max_opacity;
	dict["right_mover_color"] = right_mover_color;
	dict["up_mover_color"] = up_mover_color;
	return dict;
}

void BMLTrafficSplatGenerator::Config::from_dict(const Dictionary &p_dict) {
	if (p_dict.has("grid_width")) { grid_width = read_non_negative_u32(p_dict["grid_width"], grid_width); }
	if (p_dict.has("grid_height")) { grid_height = read_non_negative_u32(p_dict["grid_height"], grid_height); }
	if (p_dict.has("density")) { density = p_dict["density"]; }
	if (p_dict.has("steps")) { steps = read_non_negative_u32(p_dict["steps"], steps); }
	if (p_dict.has("seed")) { seed = read_non_negative_u64(p_dict["seed"], seed); }
	if (p_dict.has("cell_size")) { cell_size = p_dict["cell_size"]; }
	if (p_dict.has("min_scale")) { min_scale = p_dict["min_scale"]; }
	if (p_dict.has("max_scale")) { max_scale = p_dict["max_scale"]; }
	if (p_dict.has("anisotropy")) { anisotropy = p_dict["anisotropy"]; }
	if (p_dict.has("sh_intensity")) { sh_intensity = p_dict["sh_intensity"]; }
	if (p_dict.has("min_opacity")) { min_opacity = p_dict["min_opacity"]; }
	if (p_dict.has("max_opacity")) { max_opacity = p_dict["max_opacity"]; }
	if (p_dict.has("right_mover_color")) { right_mover_color = p_dict["right_mover_color"]; }
	if (p_dict.has("up_mover_color")) { up_mover_color = p_dict["up_mover_color"]; }
}

uint64_t BMLTrafficSplatGenerator::hash_config(const Config &p_config) {
	uint64_t hash = HASH_BASIS;
	hash = hash_u64(hash, p_config.grid_width);
	hash = hash_u64(hash, p_config.grid_height);
	hash = hash_float(hash, p_config.density);
	hash = hash_u64(hash, p_config.steps);
	hash = hash_u64(hash, p_config.seed);
	hash = hash_float(hash, p_config.cell_size);
	hash = hash_float(hash, p_config.min_scale);
	hash = hash_float(hash, p_config.max_scale);
	hash = hash_float(hash, p_config.anisotropy);
	hash = hash_float(hash, p_config.sh_intensity);
	hash = hash_float(hash, p_config.min_opacity);
	hash = hash_float(hash, p_config.max_opacity);
	hash = hash_color(hash, p_config.right_mover_color);
	hash = hash_color(hash, p_config.up_mover_color);
	return hash;
}

LocalVector<Gaussian> BMLTrafficSplatGenerator::generate(const Config &p_config, SyntheticSceneSummary *r_summary) {
	Config config = p_config;
	config.grid_width = CLAMP(config.grid_width, 4u, 2048u);
	config.grid_height = CLAMP(config.grid_height, 4u, 2048u);
	config.density = CLAMP(config.density, 0.01f, 0.99f);
	config.cell_size = MAX(config.cell_size, 0.001f);
	normalize_range(config.min_scale, config.max_scale);
	normalize_range(config.min_opacity, config.max_opacity);
	config.anisotropy = CLAMP(config.anisotropy, 0.0f, 1.0f);
	config.sh_intensity = MAX(config.sh_intensity, 0.0f);

	const uint32_t w = config.grid_width;
	const uint32_t h = config.grid_height;
	const uint32_t area = w * h;

	// Initialize grid with random particles.
	LocalVector<BMLCell> grid;
	grid.resize(area);
	DeterministicRng rng(config.seed);

	for (uint32_t idx = 0; idx < area; idx++) {
		const float r = rng.next_unit_float();
		if (r < config.density * 0.5f) {
			grid[idx] = BML_RIGHT;
		} else if (r < config.density) {
			grid[idx] = BML_UP;
		} else {
			grid[idx] = BML_EMPTY;
		}
	}

	// Run BML simulation.
	LocalVector<BMLCell> next_grid;
	next_grid.resize(area);

	for (uint32_t step = 0; step < config.steps; step++) {
		for (uint32_t idx = 0; idx < area; idx++) {
			next_grid[idx] = grid[idx];
		}

		if (step % 2 == 0) {
			for (uint32_t y = 0; y < h; y++) {
				for (uint32_t x = 0; x < w; x++) {
					const uint32_t idx = y * w + x;
					if (grid[idx] == BML_RIGHT) {
						const uint32_t nx = (x + 1) % w;
						const uint32_t nidx = y * w + nx;
						if (grid[nidx] == BML_EMPTY) {
							next_grid[idx] = BML_EMPTY;
							next_grid[nidx] = BML_RIGHT;
						}
					}
				}
			}
		} else {
			for (uint32_t y = 0; y < h; y++) {
				for (uint32_t x = 0; x < w; x++) {
					const uint32_t idx = y * w + x;
					if (grid[idx] == BML_UP) {
						const uint32_t ny = (y + 1) % h;
						const uint32_t nidx = ny * w + x;
						if (grid[nidx] == BML_EMPTY) {
							next_grid[idx] = BML_EMPTY;
							next_grid[nidx] = BML_UP;
						}
					}
				}
			}
		}

		for (uint32_t idx = 0; idx < area; idx++) {
			grid[idx] = next_grid[idx];
		}
	}

	// Convert occupied cells to splats.
	uint32_t occupied = 0;
	for (uint32_t idx = 0; idx < area; idx++) {
		if (grid[idx] != BML_EMPTY) {
			occupied++;
		}
	}

	LocalVector<Gaussian> splats;
	splats.resize(occupied);

	const float half_w = static_cast<float>(w) * config.cell_size * 0.5f;
	const float half_h = static_cast<float>(h) * config.cell_size * 0.5f;

	uint32_t splat_idx = 0;
	for (uint32_t y = 0; y < h; y++) {
		for (uint32_t x = 0; x < w; x++) {
			const uint32_t idx = y * w + x;
			if (grid[idx] == BML_EMPTY) {
				continue;
			}

			Gaussian gaussian = make_base_gaussian();
			gaussian.position = Vector3(
					static_cast<float>(x) * config.cell_size - half_w,
					0.0f,
					static_cast<float>(y) * config.cell_size - half_h);
			gaussian.normal = Vector3(0.0f, 1.0f, 0.0f);

			const float base_scale = rng.range(config.min_scale, config.max_scale);
			gaussian.scale = anisotropic_scale(rng, base_scale, config.anisotropy);
			gaussian.area = gaussian.scale.x * gaussian.scale.z * static_cast<float>(Math::PI);
			gaussian.opacity = rng.range(config.min_opacity, config.max_opacity);

			if (grid[idx] == BML_RIGHT) {
				gaussian.sh_dc = Color(config.right_mover_color.r, config.right_mover_color.g,
						config.right_mover_color.b, gaussian.opacity);
			} else {
				gaussian.sh_dc = Color(config.up_mover_color.r, config.up_mover_color.g,
						config.up_mover_color.b, gaussian.opacity);
			}

			generate_sh1(rng, gaussian, config.sh_intensity);
			splats[splat_idx++] = gaussian;
		}
	}

	if (r_summary != nullptr) {
		*r_summary = summarize_generated_scene(splats, "bml_traffic", config.seed, hash_config(config));
	}
	return splats;
}

} // namespace TestGaussianSplatting
