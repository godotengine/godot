/// @ref ext_matrix_int4x2_sized
/// @file glm/ext/matrix_int4x2_sized.hpp
///
/// @see core (dependence)
///
/// @defgroup ext_matrix_int4x2_sized GLM_EXT_matrix_int4x2_sized
/// @ingroup ext
///
/// Include <glm/ext/matrix_int4x2_sized.hpp> to use the features of this extension.
///
/// Defines a number of matrices with integer types.

#pragma once

// Dependency:
#include "../mat4x2.hpp"
#include "../ext/scalar_int_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_int4x2_sized extension included")
#endif

namespace glm
{
	/// @addtogroup ext_matrix_int4x2_sized
	/// @{

	/// 8 bit signed integer 4x2 matrix.
	///
	/// @see ext_matrix_int4x2_sized
	typedef mat<4, 2, int8, defaultp>				i8mat4x2;

	/// 16 bit signed integer 4x2 matrix.
	///
	/// @see ext_matrix_int4x2_sized
	typedef mat<4, 2, int16, defaultp>				i16mat4x2;

	/// 32 bit signed integer 4x2 matrix.
	///
	/// @see ext_matrix_int4x2_sized
	typedef mat<4, 2, int32, defaultp>				i32mat4x2;

	/// 64 bit signed integer 4x2 matrix.
	///
	/// @see ext_matrix_int4x2_sized
	typedef mat<4, 2, int64, defaultp>				i64mat4x2;

	/// @}
}//namespace glm
