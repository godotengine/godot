/// @ref core
/// @file glm/ext/matrix_float3x4.hpp

#pragma once
#include "../detail/type_mat3x4.hpp"

namespace glm
{
	/// @addtogroup core_matrix
	/// @{

	/// 3 columns of 4 components matrix of single-precision floating-point numbers.
	///
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.1.6 Matrices</a>
	typedef mat<3, 4, float, defaultp>			mat3x4;

	/// @}
}//namespace glm
