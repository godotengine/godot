/**************************************************************************/
/*  synthetic_splat_generators.h                                         */
/*  Facade header — includes all synthetic splat generator declarations.  */
/**************************************************************************/

#pragma once

#include "synthetic_splat_common.h"

#include <cstdint>

namespace TestGaussianSplatting {

// ── Uniform distribution ────────────────────────────────────────────────

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

		float anisotropy = 0.0f;
		float sh_intensity = 0.0f;
		bool log_scale_distribution = false;

		Dictionary to_dict() const;
		void from_dict(const Dictionary &p_dict);
	};

	static LocalVector<Gaussian> generate(const Config &p_config, SyntheticSceneSummary *r_summary = nullptr);
	static uint64_t hash_config(const Config &p_config);
};

// ── Clustered distribution ──────────────────────────────────────────────

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

		float anisotropy = 0.0f;
		float sh_intensity = 0.0f;
		bool log_scale_distribution = false;
		bool surface_aligned = false;
		float cluster_flatten = 0.0f;

		Dictionary to_dict() const;
		void from_dict(const Dictionary &p_dict);
	};

	static LocalVector<Gaussian> generate(const Config &p_config, SyntheticSceneSummary *r_summary = nullptr);
	static uint64_t hash_config(const Config &p_config);
};

// ── Geometric surfaces ──────────────────────────────────────────────────

class SurfaceSplatGenerator {
public:
	enum Shape {
		SHAPE_SPHERE,
		SHAPE_TORUS,
		SHAPE_PLANE,
		SHAPE_CUBE,
	};

	struct Config {
		uint32_t splat_count = 50000;
		uint64_t seed = 42;
		Shape shape = SHAPE_SPHERE;
		Vector3 center = Vector3();
		float shape_radius = 5.0f;
		float torus_tube_radius = 1.5f;
		float plane_half_extent = 8.0f;

		float min_scale = 0.02f;
		float max_scale = 0.15f;
		float anisotropy = 0.6f;
		float surface_noise = 0.05f;
		float min_opacity = 0.7f;
		float max_opacity = 1.0f;
		float sh_intensity = 0.3f;

		bool random_colors = false;
		Color base_color = Color(0.6f, 0.5f, 0.4f, 1.0f);
		float color_variation = 0.15f;

		Dictionary to_dict() const;
		void from_dict(const Dictionary &p_dict);
	};

	static LocalVector<Gaussian> generate(const Config &p_config, SyntheticSceneSummary *r_summary = nullptr);
	static uint64_t hash_config(const Config &p_config);
};

// ── Mandelbrot fractal ──────────────────────────────────────────────────

class MandelbrotSplatGenerator {
public:
	enum ColorMode {
		COLOR_CLASSIC_RAINBOW,
		COLOR_FIRE,
		COLOR_ICE,
		COLOR_MONOCHROME,
	};

	struct Config {
		uint32_t splat_count = 100000;
		uint64_t seed = 42;
		double center_real = -0.5;
		double center_imag = 0.0;
		double zoom = 1.0;
		uint32_t max_iterations = 256;
		float min_scale = 0.005f;
		float max_scale = 0.1f;
		float depth_extrude = 0.5f;
		float anisotropy = 0.3f;
		float sh_intensity = 0.2f;
		float min_opacity = 0.4f;
		float max_opacity = 1.0f;
		ColorMode color_mode = COLOR_CLASSIC_RAINBOW;

		Dictionary to_dict() const;
		void from_dict(const Dictionary &p_dict);
	};

	static LocalVector<Gaussian> generate(const Config &p_config, SyntheticSceneSummary *r_summary = nullptr);
	static uint64_t hash_config(const Config &p_config);
};

// ── Biham-Middleton-Levine traffic model ────────────────────────────────

class BMLTrafficSplatGenerator {
public:
	struct Config {
		uint32_t grid_width = 128;
		uint32_t grid_height = 128;
		float density = 0.35f;
		uint32_t steps = 500;
		uint64_t seed = 42;
		float cell_size = 0.1f;
		float min_scale = 0.03f;
		float max_scale = 0.06f;
		float anisotropy = 0.2f;
		float sh_intensity = 0.1f;
		float min_opacity = 0.7f;
		float max_opacity = 1.0f;
		Color right_mover_color = Color(0.9f, 0.2f, 0.15f, 1.0f);
		Color up_mover_color = Color(0.15f, 0.3f, 0.9f, 1.0f);

		Dictionary to_dict() const;
		void from_dict(const Dictionary &p_dict);
	};

	static LocalVector<Gaussian> generate(const Config &p_config, SyntheticSceneSummary *r_summary = nullptr);
	static uint64_t hash_config(const Config &p_config);
};

// ── Volumetric cloud (fBm noise) ────────────────────────────────────────

class CloudSplatGenerator {
public:
	struct Config {
		uint32_t splat_count = 80000;
		uint64_t seed = 42;
		Vector3 cloud_center = Vector3(0.0f, 5.0f, 0.0f);
		Vector3 cloud_extent = Vector3(15.0f, 3.0f, 10.0f);
		float density_threshold = 0.4f;
		uint32_t noise_octaves = 6;
		float noise_frequency = 0.3f;
		float noise_lacunarity = 2.0f;
		float noise_persistence = 0.5f;
		float min_scale = 0.05f;
		float max_scale = 0.4f;
		float min_opacity = 0.1f;
		float max_opacity = 0.8f;
		float sh_intensity = 0.15f;
		Color cloud_color = Color(0.95f, 0.95f, 0.97f, 1.0f);
		Color shadow_tint = Color(0.6f, 0.65f, 0.75f, 1.0f);
		Vector3 light_direction = Vector3(0.5f, 1.0f, 0.3f);

		Dictionary to_dict() const;
		void from_dict(const Dictionary &p_dict);
	};

	static LocalVector<Gaussian> generate(const Config &p_config, SyntheticSceneSummary *r_summary = nullptr);
	static uint64_t hash_config(const Config &p_config);
};

} // namespace TestGaussianSplatting
