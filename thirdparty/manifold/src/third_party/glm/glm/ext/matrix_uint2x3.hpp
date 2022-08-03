/// @ref ext_matrix_uint2x3
/// @file glm/ext/matrix_uint2x3.hpp
///
/// @see core (dependence)
///
/// @defgroup ext_matrix_int2x3 GLM_EXT_matrix_uint2x3
/// @ingroup ext
///
/// Include <glm/ext/matrix_uint2x3.hpp> to use the features of this extension.
///
/// Defines a number of matrices with integer types.

#pragma once

// Dependency:
#include "../mat2x3.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_uint2x3 extension included")
#endif

namespace glm
{
	/// @addtogroup ext_matrix_uint2x3
	/// @{

	/// Unsigned integer 2x3 matrix.
	///
	/// @see ext_matrix_uint2x3
	typedef mat<2, 3, uint, defaultp>	umat2x3;

	/// @}
}//namespace glm
