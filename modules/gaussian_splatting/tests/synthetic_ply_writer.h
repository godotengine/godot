#ifndef SYNTHETIC_PLY_WRITER_H
#define SYNTHETIC_PLY_WRITER_H

#include "../core/gaussian_data.h"
#include "core/templates/local_vector.h"
#include "core/string/ustring.h"

namespace TestGaussianSplatting {

// Write a LocalVector<Gaussian> to a binary PLY file.
// Encoding is the exact inverse of PLYLoader::parse_binary_data:
//   f_dc     = sh_dc.rgb / SH_C0
//   opacity  = logit(opacity) = log(o / (1 - o))
//   scale    = log(scale)
//   rotation = quaternion WXYZ
// Returns true on success.
bool write_gaussian_ply(const String &p_path, const LocalVector<Gaussian> &p_splats, bool p_write_sh1 = true, bool p_write_normals = false);

} // namespace TestGaussianSplatting

#endif // SYNTHETIC_PLY_WRITER_H
