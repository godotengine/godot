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
	f->set_big_endian(false);

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
		// Standard 3DGS PLY uses f_rest_0..44 with channel-major layout:
		// R: indices 0..14, G: indices 15..29, B: indices 30..44.
		// We only populate band-1 (3 coefficients per channel) but must
		// declare all 45 slots so the loader finds G/B at the right indices.
		for (int i = 0; i < 45; i++) {
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

	const CharString header_utf8 = header.utf8();
	f->store_buffer((const uint8_t *)header_utf8.get_data(), header_utf8.length());

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
			// Write all 45 f_rest slots in channel-major order:
			// R band-1 at indices 0-2, G band-1 at indices 15-17,
			// B band-1 at indices 30-32.  All other slots are zero.
			float sh_rest[45] = {};
			for (int coeff = 0; coeff < 3; coeff++) {
				sh_rest[coeff] = g.sh_1[coeff][0]; // R
				sh_rest[coeff + 15] = g.sh_1[coeff][1]; // G
				sh_rest[coeff + 30] = g.sh_1[coeff][2]; // B
			}
			for (int j = 0; j < 45; j++) {
				f->store_float(sh_rest[j]);
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
