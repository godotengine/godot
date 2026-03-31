#ifndef GENERATE_SYNTHETIC_PLY_FIXTURES_H
#define GENERATE_SYNTHETIC_PLY_FIXTURES_H

#include "tests/test_macros.h"

#include "synthetic_ply_writer.h"
#include "synthetic_splat_generators.h"
#include "core/os/os.h"
#include "core/string/print_string.h"

namespace TestGaussianSplatting {

static String _ply_output_dir() {
	String dir = OS::get_singleton()->get_environment("SYNTHETIC_PLY_OUTPUT_DIR");
	if (dir.is_empty()) {
		dir = OS::get_singleton()->get_executable_path().get_base_dir().path_join("..").path_join("tests").path_join("fixtures");
	}
	return dir;
}

TEST_CASE("[GaussianSplatting][GeneratePLY] Generate high-quality synthetic PLY fixtures") {
	const String output_dir = _ply_output_dir();
	print_line(vformat("[GeneratePLY] output_dir=%s", output_dir));

	// ── Mandelbrot (100K splats, fire palette, zoomed into interesting region) ──
	{
		MandelbrotSplatGenerator::Config cfg;
		cfg.splat_count = 100000;
		cfg.seed = 3601;
		cfg.center_real = -0.7436;
		cfg.center_imag = 0.1318;
		cfg.zoom = 0.02;
		cfg.max_iterations = 512;
		cfg.color_mode = MandelbrotSplatGenerator::COLOR_FIRE;
		cfg.depth_extrude = 2.0f;
		cfg.anisotropy = 0.4f;
		cfg.sh_intensity = 0.3f;
		cfg.min_scale = 0.003f;
		cfg.max_scale = 0.08f;
		cfg.min_opacity = 0.5f;
		cfg.max_opacity = 1.0f;

		LocalVector<Gaussian> splats = MandelbrotSplatGenerator::generate(cfg);
		CHECK(splats.size() == cfg.splat_count);

		const String path = output_dir.path_join("synthetic_mandelbulb.ply");
		CHECK_MESSAGE(write_gaussian_ply(path, splats, true, false),
				vformat("Failed to write %s", path));
		print_line(vformat("[GeneratePLY] wrote synthetic_mandelbulb.ply (%d splats)", splats.size()));
	}

	// ── Cloud (80K splats, volumetric fBm) ──
	{
		CloudSplatGenerator::Config cfg;
		cfg.splat_count = 80000;
		cfg.seed = 3701;
		cfg.cloud_center = Vector3(0.0f, 5.0f, -4.0f);
		cfg.cloud_extent = Vector3(18.0f, 4.0f, 12.0f);
		cfg.density_threshold = 0.35f;
		cfg.noise_octaves = 6;
		cfg.noise_frequency = 0.25f;
		cfg.min_scale = 0.04f;
		cfg.max_scale = 0.35f;
		cfg.min_opacity = 0.12f;
		cfg.max_opacity = 0.75f;
		cfg.sh_intensity = 0.2f;

		LocalVector<Gaussian> splats = CloudSplatGenerator::generate(cfg);
		CHECK(splats.size() == cfg.splat_count);

		const String path = output_dir.path_join("synthetic_cloud.ply");
		CHECK_MESSAGE(write_gaussian_ply(path, splats, true, false),
				vformat("Failed to write %s", path));
		print_line(vformat("[GeneratePLY] wrote synthetic_cloud.ply (%d splats)", splats.size()));
	}

	// ── Sphere (50K splats) ──
	{
		SurfaceSplatGenerator::Config cfg;
		cfg.splat_count = 50000;
		cfg.seed = 3101;
		cfg.shape = SurfaceSplatGenerator::SHAPE_SPHERE;
		cfg.shape_radius = 5.0f;
		cfg.min_scale = 0.015f;
		cfg.max_scale = 0.1f;
		cfg.anisotropy = 0.5f;
		cfg.sh_intensity = 0.3f;
		cfg.random_colors = true;

		LocalVector<Gaussian> splats = SurfaceSplatGenerator::generate(cfg);
		CHECK(splats.size() == cfg.splat_count);

		const String path = output_dir.path_join("synthetic_sphere.ply");
		CHECK_MESSAGE(write_gaussian_ply(path, splats, true, true),
				vformat("Failed to write %s", path));
		print_line(vformat("[GeneratePLY] wrote synthetic_sphere.ply (%d splats)", splats.size()));
	}

	// ── Torus (50K splats) ──
	{
		SurfaceSplatGenerator::Config cfg;
		cfg.splat_count = 50000;
		cfg.seed = 3401;
		cfg.shape = SurfaceSplatGenerator::SHAPE_TORUS;
		cfg.shape_radius = 5.0f;
		cfg.torus_tube_radius = 1.8f;
		cfg.min_scale = 0.01f;
		cfg.max_scale = 0.08f;
		cfg.anisotropy = 0.6f;
		cfg.sh_intensity = 0.3f;
		cfg.base_color = Color(0.7f, 0.4f, 0.3f, 1.0f);
		cfg.color_variation = 0.2f;

		LocalVector<Gaussian> splats = SurfaceSplatGenerator::generate(cfg);
		CHECK(splats.size() == cfg.splat_count);

		const String path = output_dir.path_join("synthetic_torus.ply");
		CHECK_MESSAGE(write_gaussian_ply(path, splats, true, true),
				vformat("Failed to write %s", path));
		print_line(vformat("[GeneratePLY] wrote synthetic_torus.ply (%d splats)", splats.size()));
	}

	// ── Cube (50K splats) ──
	{
		SurfaceSplatGenerator::Config cfg;
		cfg.splat_count = 50000;
		cfg.seed = 3201;
		cfg.shape = SurfaceSplatGenerator::SHAPE_CUBE;
		cfg.shape_radius = 4.0f;
		cfg.min_scale = 0.01f;
		cfg.max_scale = 0.08f;
		cfg.anisotropy = 0.5f;
		cfg.sh_intensity = 0.25f;
		cfg.base_color = Color(0.5f, 0.6f, 0.8f, 1.0f);
		cfg.color_variation = 0.2f;

		LocalVector<Gaussian> splats = SurfaceSplatGenerator::generate(cfg);
		CHECK(splats.size() == cfg.splat_count);

		const String path = output_dir.path_join("synthetic_cube.ply");
		CHECK_MESSAGE(write_gaussian_ply(path, splats, true, true),
				vformat("Failed to write %s", path));
		print_line(vformat("[GeneratePLY] wrote synthetic_cube.ply (%d splats)", splats.size()));
	}

	// ── Plane (50K splats) ──
	{
		SurfaceSplatGenerator::Config cfg;
		cfg.splat_count = 50000;
		cfg.seed = 3301;
		cfg.shape = SurfaceSplatGenerator::SHAPE_PLANE;
		cfg.plane_half_extent = 10.0f;
		cfg.min_scale = 0.01f;
		cfg.max_scale = 0.08f;
		cfg.anisotropy = 0.7f;
		cfg.sh_intensity = 0.2f;
		cfg.base_color = Color(0.4f, 0.65f, 0.35f, 1.0f);
		cfg.color_variation = 0.15f;

		LocalVector<Gaussian> splats = SurfaceSplatGenerator::generate(cfg);
		CHECK(splats.size() == cfg.splat_count);

		const String path = output_dir.path_join("synthetic_plane.ply");
		CHECK_MESSAGE(write_gaussian_ply(path, splats, true, true),
				vformat("Failed to write %s", path));
		print_line(vformat("[GeneratePLY] wrote synthetic_plane.ply (%d splats)", splats.size()));
	}

	// ── Uniform baseline (10K splats for test_splats.ply) ──
	{
		UniformSplatGenerator::Config cfg;
		cfg.splat_count = 10000;
		cfg.seed = 1101;
		cfg.position_min = Vector3(-3.0f, -3.0f, -3.0f);
		cfg.position_max = Vector3(3.0f, 3.0f, 3.0f);
		cfg.random_colors = true;
		cfg.sh_intensity = 0.15f;

		LocalVector<Gaussian> splats = UniformSplatGenerator::generate(cfg);
		CHECK(splats.size() == cfg.splat_count);

		const String path = output_dir.path_join("test_splats.ply");
		CHECK_MESSAGE(write_gaussian_ply(path, splats, true, false),
				vformat("Failed to write %s", path));
		print_line(vformat("[GeneratePLY] wrote test_splats.ply (%d splats)", splats.size()));
	}

	print_line("[GeneratePLY] All fixtures generated successfully.");
}

} // namespace TestGaussianSplatting

#endif // GENERATE_SYNTHETIC_PLY_FIXTURES_H
