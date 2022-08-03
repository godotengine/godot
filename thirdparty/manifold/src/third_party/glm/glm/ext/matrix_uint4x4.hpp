/// @ref ext_matrix_uint4x4
/// @file glm/ext/matrix_uint4x4.hpp
///
/// @see core (dependence)
///
/// @defgroup ext_matrix_uint4x4 GLM_EXT_matrix_uint4x4
/// @ingroup ext
///
/// Include <glm/ext/matrix_uint4x4.hpp> to use the features of this extension.
///
/// Defines a number of matrices with integer types.

#pragma once

// Dependency:
#include "../mat4x4.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_uint4x4 extension included")
#endif

namespace glm
{
	/// @addtogroup ext_matrix_uint4x4
	/// @{

	/// Unsigned integer 4x4 matrix.
	///
	/// @see ext_matrix_uint4x4
	typedef mat<4, 4, uint, defaultp>	umat4x4;

	/// Unsigned integer 4x4 matrix.
	///
	/// @see ext_matrix_uint4x4
	typedef mat<4, 4, uint, defaultp>	umat4;

	/// @}
}//namespace glm
