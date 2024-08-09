/// @ref ext_matrix_int2x4_sized
/// @file glm/ext/matrix_int2x4_sized.hpp
///
/// @see core (dependence)
///
/// @defgroup ext_matrix_int2x4_sized GLM_EXT_matrix_int2x4_sized
/// @ingroup ext
///
/// Include <glm/ext/matrix_int2x4_sized.hpp> to use the features of this extension.
///
/// Defines a number of matrices with integer types.

#pragma once

// Dependency:
#include "../mat2x4.hpp"
#include "../ext/scalar_int_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_int2x4_sized extension included")
#endif

namespace glm
{
	/// @addtogroup ext_matrix_int2x4_sized
	/// @{

	/// 8 bit signed integer 2x4 matrix.
	///
	/// @see ext_matrix_int2x4_sized
	typedef mat<2, 4, int8, defaultp>				i8mat2x4;

	/// 16 bit signed integer 2x4 matrix.
	///
	/// @see ext_matrix_int2x4_sized
	typedef mat<2, 4, int16, defaultp>				i16mat2x4;

	/// 32 bit signed integer 2x4 matrix.
	///
	/// @see ext_matrix_int2x4_sized
	typedef mat<2, 4, int32, defaultp>				i32mat2x4;

	/// 64 bit signed integer 2x4 matrix.
	///
	/// @see ext_matrix_int2x4_sized
	typedef mat<2, 4, int64, defaultp>				i64mat2x4;

	/// @}
}//namespace glm
