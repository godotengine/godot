/// @ref ext_matrix_int2x2
/// @file glm/ext/matrix_int2x2.hpp
///
/// @see core (dependence)
///
/// @defgroup ext_matrix_int2x2 GLM_EXT_matrix_int2x2
/// @ingroup ext
///
/// Include <glm/ext/matrix_int2x2.hpp> to use the features of this extension.
///
/// Defines a number of matrices with integer types.

#pragma once

// Dependency:
#include "../mat2x2.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_int2x2 extension included")
#endif

namespace glm
{
	/// @addtogroup ext_matrix_int2x2
	/// @{

	/// Signed integer 2x2 matrix.
	///
	/// @see ext_matrix_int2x2
	typedef mat<2, 2, int, defaultp>	imat2x2;

	/// Signed integer 2x2 matrix.
	///
	/// @see ext_matrix_int2x2
	typedef mat<2, 2, int, defaultp>	imat2;

	/// @}
}//namespace glm
