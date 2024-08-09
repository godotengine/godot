/// @ref ext_matrix_uint4x3
/// @file glm/ext/matrix_uint4x3.hpp
///
/// @see core (dependence)
///
/// @defgroup ext_matrix_uint4x3 GLM_EXT_matrix_uint4x3
/// @ingroup ext
///
/// Include <glm/ext/matrix_uint4x3.hpp> to use the features of this extension.
///
/// Defines a number of matrices with integer types.

#pragma once

// Dependency:
#include "../mat4x3.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_uint4x3 extension included")
#endif

namespace glm
{
	/// @addtogroup ext_matrix_uint4x3
	/// @{

	/// Unsigned integer 4x3 matrix.
	///
	/// @see ext_matrix_uint4x3
	typedef mat<4, 3, uint, defaultp>	umat4x3;

	/// @}
}//namespace glm
