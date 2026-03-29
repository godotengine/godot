/**************************************************************************/
/*  synthetic_ply_writer.cpp                                              */
/**************************************************************************/

#include "synthetic_ply_writer.h"
#include "core/io/file_access.h"
#include <cmath>

namespace TestGaussianSplatting {

static constexpr float SH_C0 = 0.28209479177387814f;

static float _logit(float p_x) {
	const float clamped = CLAMP(p_x, 1e-6f, 1.0f - 1e-6f);
	return std::log(clamped / (1.0f - clamped));
}

bool write_gaussian_ply(const String &p_path, const LocalVector<Gaussian> &p_splats, bool p_write_sh1, bool p_write_normals) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	if (!f.is_valid()) {
		return false;
	}

	const uint32_t count = p_splats.size();

	// Build header.
	String header = "ply\n";
	header += "format binary_little_endian 1.0\n";
	header += vformat("element vertex %d\n", count);
	header += "property float x\n";
	header += "property float y\n";
	header += "property float z\n";
	if (p_write_normals) {
		header += "property float nx\n";
		header += "property float ny\n";
		header += "property float nz\n";
	}
	header += "property float f_dc_0\n";
	header += "property float f_dc_1\n";
	header += "property float f_dc_2\n";
	if (p_write_sh1) {
		// First-order SH: 9 f_rest coefficients (3 per channel, channel-major).
		// f_rest_0..2 = R channel, f_rest_3..5 = G channel, f_rest_6..8 = B channel.
		// Note: standard 3DGS PLY uses f_rest_0..44 for full SH with channel-major
		// layout across all 45 coefficients. We write only the first 9 (band 1).
		for (int i = 0; i < 9; i++) {
			header += vformat("property float f_rest_%d\n", i);
		}
	}
	header += "property float opacity\n";
	header += "property float scale_0\n";
	header += "property float scale_1\n";
	header += "property float scale_2\n";
	header += "property float rot_0\n";
	header += "property float rot_1\n";
	header += "property float rot_2\n";
	header += "property float rot_3\n";
	header += "end_header\n";

	f->store_buffer((const uint8_t *)header.utf8().get_data(), header.utf8().length());

	// Write binary vertex data.
	for (uint32_t i = 0; i < count; i++) {
		const Gaussian &g = p_splats[i];

		f->store_float(g.position.x);
		f->store_float(g.position.y);
		f->store_float(g.position.z);

		if (p_write_normals) {
			f->store_float(g.normal.x);
			f->store_float(g.normal.y);
			f->store_float(g.normal.z);
		}

		// DC SH coefficients: inverse of load transform (color / SH_C0).
		f->store_float(g.sh_dc.r / SH_C0);
		f->store_float(g.sh_dc.g / SH_C0);
		f->store_float(g.sh_dc.b / SH_C0);

		if (p_write_sh1) {
			// Channel-major order: R coefficients, then G, then B.
			// sh_1[0..2] stores coefficient-major RGB triplets, so:
			//   f_rest_0/1/2 = sh_1[0/1/2].x (R channel)
			//   f_rest_3/4/5 = sh_1[0/1/2].y (G channel)
			//   f_rest_6/7/8 = sh_1[0/1/2].z (B channel)
			for (int ch = 0; ch < 3; ch++) {
				for (int coeff = 0; coeff < 3; coeff++) {
					f->store_float(g.sh_1[coeff][ch]);
				}
			}
		}

		// Opacity: logit transform (inverse of sigmoid).
		f->store_float(_logit(g.opacity));

		// Scale: log transform (inverse of exp).
		f->store_float(std::log(MAX(g.scale.x, 1e-7f)));
		f->store_float(std::log(MAX(g.scale.y, 1e-7f)));
		f->store_float(std::log(MAX(g.scale.z, 1e-7f)));

		// Rotation: WXYZ quaternion written directly.
		f->store_float(g.rotation.w);
		f->store_float(g.rotation.x);
		f->store_float(g.rotation.y);
		f->store_float(g.rotation.z);
	}

	return true;
}

} // namespace TestGaussianSplatting
