/// @ref ext_matrix_int4x3_sized
/// @file glm/ext/matrix_int4x3_sized.hpp
///
/// @see core (dependence)
///
/// @defgroup ext_matrix_int4x3_sized GLM_EXT_matrix_int4x3_sized
/// @ingroup ext
///
/// Include <glm/ext/matrix_int4x3_sized.hpp> to use the features of this extension.
///
/// Defines a number of matrices with integer types.

#pragma once

// Dependency:
#include "../mat4x3.hpp"
#include "../ext/scalar_int_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_int4x3_sized extension included")
#endif

namespace glm
{
	/// @addtogroup ext_matrix_int4x3_sized
	/// @{

	/// 8 bit signed integer 4x3 matrix.
	///
	/// @see ext_matrix_int4x3_sized
	typedef mat<4, 3, int8, defaultp>				i8mat4x3;

	/// 16 bit signed integer 4x3 matrix.
	///
	/// @see ext_matrix_int4x3_sized
	typedef mat<4, 3, int16, defaultp>				i16mat4x3;

	/// 32 bit signed integer 4x3 matrix.
	///
	/// @see ext_matrix_int4x3_sized
	typedef mat<4, 3, int32, defaultp>				i32mat4x3;

	/// 64 bit signed integer 4x3 matrix.
	///
	/// @see ext_matrix_int4x3_sized
	typedef mat<4, 3, int64, defaultp>				i64mat4x3;

	/// @}
}//namespace glm
