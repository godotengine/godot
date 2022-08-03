/// @ref ext_matrix_int3x4_sized
/// @file glm/ext/matrix_int3x2_sized.hpp
///
/// @see core (dependence)
///
/// @defgroup ext_matrix_int3x4_sized GLM_EXT_matrix_int3x4_sized
/// @ingroup ext
///
/// Include <glm/ext/matrix_int3x4_sized.hpp> to use the features of this extension.
///
/// Defines a number of matrices with integer types.

#pragma once

// Dependency:
#include "../mat3x4.hpp"
#include "../ext/scalar_int_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_int3x4_sized extension included")
#endif

namespace glm
{
	/// @addtogroup ext_matrix_int3x4_sized
	/// @{

	/// 8 bit signed integer 3x4 matrix.
	///
	/// @see ext_matrix_int3x4_sized
	typedef mat<3, 4, int8, defaultp>				i8mat3x4;

	/// 16 bit signed integer 3x4 matrix.
	///
	/// @see ext_matrix_int3x4_sized
	typedef mat<3, 4, int16, defaultp>				i16mat3x4;

	/// 32 bit signed integer 3x4 matrix.
	///
	/// @see ext_matrix_int3x4_sized
	typedef mat<3, 4, int32, defaultp>				i32mat3x4;

	/// 64 bit signed integer 3x4 matrix.
	///
	/// @see ext_matrix_int3x4_sized
	typedef mat<3, 4, int64, defaultp>				i64mat3x4;

	/// @}
}//namespace glm
