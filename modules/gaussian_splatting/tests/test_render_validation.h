#pragma once

#include "test_macros.h"

#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_manager.h"
#include "../renderer/gaussian_splat_renderer.h"

#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/string/ustring.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering_server.h"

namespace TestGaussianSplatting {

class ScopedGaussianManagerValidation {
	GaussianSplatManager *manager = nullptr;
	bool owns_instance = false;

public:
	ScopedGaussianManagerValidation() {
		manager = GaussianSplatManager::get_singleton();
		if (!manager) {
			manager = memnew(GaussianSplatManager);
			owns_instance = true;
		}
	}

	~ScopedGaussianManagerValidation() {
		if (owns_instance && manager) {
			memdelete(manager);
		}
	}

	GaussianSplatManager *get() const { return manager; }
};

static Ref<GaussianData> build_debug_gaussians() {
	LocalVector<Gaussian> gaussians;
	gaussians.resize(4);

	for (uint32_t i = 0; i < gaussians.size(); i++) {
		Gaussian &g = gaussians[i];
		g = Gaussian{};
		const float offset = static_cast<float>(i) * 0.35f;
		g.position = Vector3(offset - 0.5f, 0.0f, -3.0f - offset);
		g.scale = Vector3(0.15f, 0.15f, 0.15f);
		g.opacity = 0.85f;
		g.rotation = Quaternion();
		g.normal = Vector3(0.0f, 0.0f, 1.0f);
		g.area = g.scale.x * g.scale.y;
		g.sh_dc = Color(0.4f + 0.1f * i, 0.5f, 0.6f, g.opacity);
		g.brush_axes = Vector2(1.0f, 0.0f);
		g.stroke_age = 0.0f;
		g.painterly_meta = gaussian_pack_painterly_meta(0);
	}

	Ref<GaussianData> data;
	data.instantiate();
	data->set_gaussians(gaussians);
	return data;
}

static RID create_validation_texture(RenderingDevice *p_rd, const Vector2i &p_size) {
	RD::TextureFormat format;
	format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	format.width = p_size.x;
	format.height = p_size.y;
	format.depth = 1;
	format.array_layers = 1;
	format.mipmaps = 1;
	format.samples = RD::TEXTURE_SAMPLES_1;
	format.usage_bits = RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT |
			RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
	return p_rd->texture_create(format, RD::TextureView());
}

static uint8_t float_to_u8(float p_value) {
	float clamped = CLAMP(p_value, 0.0f, 1.0f);
	return static_cast<uint8_t>(CLAMP(Math::round(clamped * 255.0f), 0.0f, 255.0f));
}

static bool compare_debug_projection(const Vector<uint8_t> &p_pixels, int p_width, int p_height,
		int p_tolerance, int &r_max_diff, int &r_bad_pixels) {
	struct CompareStats {
		int max_diff = 0;
		int bad_pixels = 0;
	};

	const int pixel_stride = 4;
	const int expected_size = p_width * p_height * pixel_stride;
	ERR_FAIL_COND_V(p_pixels.size() < expected_size, false);

	auto evaluate = [&](bool p_flip_y) {
		CompareStats stats;
		for (int y = 0; y < p_height; y++) {
			const float norm_y = (static_cast<float>(y) + 0.5f) / static_cast<float>(p_height);
			const float ny = p_flip_y ? (1.0f - norm_y) : norm_y;
			for (int x = 0; x < p_width; x++) {
				const float nx = (static_cast<float>(x) + 0.5f) / static_cast<float>(p_width);
				const uint8_t expected_r = float_to_u8(nx);
				const uint8_t expected_g = float_to_u8(ny);
				const uint8_t expected_b = float_to_u8(0.25f + 0.5f * ny);

				const int idx = (y * p_width + x) * pixel_stride;
				const uint8_t actual_r = p_pixels[idx + 0];
				const uint8_t actual_g = p_pixels[idx + 1];
				const uint8_t actual_b = p_pixels[idx + 2];
				const uint8_t actual_a = p_pixels[idx + 3];

				const int diff_r = Math::abs(int(actual_r) - int(expected_r));
				const int diff_g = Math::abs(int(actual_g) - int(expected_g));
				const int diff_b = Math::abs(int(actual_b) - int(expected_b));
				const int diff_a = Math::abs(int(actual_a) - 255);
				stats.max_diff = MAX(stats.max_diff, MAX(MAX(diff_r, diff_g), MAX(diff_b, diff_a)));

				if (diff_r > p_tolerance || diff_g > p_tolerance || diff_b > p_tolerance || diff_a > p_tolerance) {
					stats.bad_pixels++;
				}
			}
		}
		return stats;
	};

	CompareStats normal = evaluate(false);
	CompareStats flipped = evaluate(true);
	if (flipped.bad_pixels < normal.bad_pixels) {
		r_max_diff = flipped.max_diff;
		r_bad_pixels = flipped.bad_pixels;
	} else {
		r_max_diff = normal.max_diff;
		r_bad_pixels = normal.bad_pixels;
	}

	return r_bad_pixels == 0;
}

TEST_CASE("[GaussianSplatting] Debug projection output matches golden gradient") {
	RenderingServer *rs = RenderingServer::get_singleton();
	if (rs == nullptr) {
		MESSAGE("Skipping test - Rendering server unavailable");
		return;
	}

	ScopedGaussianManagerValidation manager_scope;
	GaussianSplatManager *manager = manager_scope.get();
	if (manager == nullptr) {
		MESSAGE("Skipping test - GaussianSplatManager unavailable");
		return;
	}

	bool owns_device = false;
	RenderingDevice *primary_rd = manager->get_primary_rendering_device();
	if (!primary_rd) {
		primary_rd = rs->create_local_rendering_device();
		owns_device = true;
	}
	if (primary_rd == nullptr) {
		MESSAGE("Skipping test - Rendering device unavailable");
		return;
	}

	Ref<GaussianSplatRenderer> renderer;
	renderer.instantiate(primary_rd);
	CHECK(renderer.is_valid());
	if (!renderer.is_valid()) {
		if (owns_device && primary_rd) {
			memdelete(primary_rd);
		}
		return;
	}
	renderer->initialize();

	renderer->set_debug_show_projection_issues(true);
	renderer->set_debug_show_tile_grid(false);
	renderer->set_debug_show_overflow_tiles(false);
	renderer->set_debug_show_density_heatmap(false);
	renderer->set_painterly_enabled(false);

	Ref<GaussianData> data = build_debug_gaussians();
	renderer->set_max_splats(data->get_count());
	bool data_set_ok = (renderer->set_gaussian_data(data) == OK);
	CHECK(data_set_ok);
	if (!data_set_ok) {
		renderer.unref();
		if (owns_device && primary_rd) {
			memdelete(primary_rd);
		}
		return;
	}

	const Vector2i resolution(64, 64);
	const Transform3D camera_transform;
	Projection projection;
	projection.set_perspective(60.0f, 1.0f, 0.1f, 50.0f);

	bool rendered = renderer->render_for_view(camera_transform, projection, RID(), resolution);
	CHECK(rendered);
	if (!rendered) {
		renderer.unref();
		if (owns_device && primary_rd) {
			memdelete(primary_rd);
		}
		return;
	}

	RID target = create_validation_texture(primary_rd, resolution);
	CHECK(target.is_valid());
	if (!target.is_valid()) {
		renderer.unref();
		if (owns_device && primary_rd) {
			memdelete(primary_rd);
		}
		return;
	}

	bool copy_ok = renderer->copy_final_texture_to_target(target, resolution);
	CHECK(copy_ok);
	if (!copy_ok) {
		primary_rd->free(target);
		renderer.unref();
		if (owns_device && primary_rd) {
			memdelete(primary_rd);
		}
		return;
	}

	Vector<uint8_t> pixel_data = primary_rd->texture_get_data(target, 0);
	bool pixel_size_ok = (pixel_data.size() == resolution.x * resolution.y * 4);
	CHECK(pixel_size_ok);
	if (!pixel_size_ok) {
		primary_rd->free(target);
		renderer.unref();
		if (owns_device && primary_rd) {
			memdelete(primary_rd);
		}
		return;
	}

	int max_diff = 0;
	int bad_pixels = 0;
	const int tolerance = 2;
	bool matched = compare_debug_projection(pixel_data, resolution.x, resolution.y, tolerance, max_diff, bad_pixels);

	if (!matched) {
		String message = "Debug projection mismatch: bad_pixels=" + itos(bad_pixels) +
				" max_diff=" + itos(max_diff) + " tolerance=" + itos(tolerance);
		CHECK_MESSAGE(false, message);
	}

	if (target.is_valid()) {
		primary_rd->free(target);
	}
	renderer.unref();
	if (owns_device && primary_rd) {
		memdelete(primary_rd);
	}
}

} // namespace TestGaussianSplatting
