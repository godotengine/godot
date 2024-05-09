/// @ref ext_matrix_int2x2_sized
/// @file glm/ext/matrix_int2x2_sized.hpp
///
/// @see core (dependence)
///
/// @defgroup ext_matrix_int2x2_sized GLM_EXT_matrix_int2x2_sized
/// @ingroup ext
///
/// Include <glm/ext/matrix_int2x2_sized.hpp> to use the features of this extension.
///
/// Defines a number of matrices with integer types.

#pragma once

// Dependency:
#include "../mat2x2.hpp"
#include "../ext/scalar_int_sized.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_matrix_int2x2_sized extension included")
#endif

namespace glm
{
	/// @addtogroup ext_matrix_int2x2_sized
	/// @{

	/// 8 bit signed integer 2x2 matrix.
	///
	/// @see ext_matrix_int2x2_sized
	typedef mat<2, 2, int8, defaultp>				i8mat2x2;

	/// 16 bit signed integer 2x2 matrix.
	///
	/// @see ext_matrix_int2x2_sized
	typedef mat<2, 2, int16, defaultp>				i16mat2x2;

	/// 32 bit signed integer 2x2 matrix.
	///
	/// @see ext_matrix_int2x2_sized
	typedef mat<2, 2, int32, defaultp>				i32mat2x2;

	/// 64 bit signed integer 2x2 matrix.
	///
	/// @see ext_matrix_int2x2_sized
	typedef mat<2, 2, int64, defaultp>				i64mat2x2;


	/// 8 bit signed integer 2x2 matrix.
	///
	/// @see ext_matrix_int2x2_sized
	typedef mat<2, 2, int8, defaultp>				i8mat2;

	/// 16 bit signed integer 2x2 matrix.
	///
	/// @see ext_matrix_int2x2_sized
	typedef mat<2, 2, int16, defaultp>				i16mat2;

	/// 32 bit signed integer 2x2 matrix.
	///
	/// @see ext_matrix_int2x2_sized
	typedef mat<2, 2, int32, defaultp>				i32mat2;

	/// 64 bit signed integer 2x2 matrix.
	///
	/// @see ext_matrix_int2x2_sized
	typedef mat<2, 2, int64, defaultp>				i64mat2;

	/// @}
}//namespace glm
