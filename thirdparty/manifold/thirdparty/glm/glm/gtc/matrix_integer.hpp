/// @ref gtc_matrix_integer
/// @file glm/gtc/matrix_integer.hpp
///
/// @see core (dependence)
///
/// @defgroup gtc_matrix_integer GLM_GTC_matrix_integer
/// @ingroup gtc
///
/// Include <glm/gtc/matrix_integer.hpp> to use the features of this extension.
///
/// Defines a number of matrices with integer types.

#pragma once

// Dependency:
#include "../mat2x2.hpp"
#include "../mat2x3.hpp"
#include "../mat2x4.hpp"
#include "../mat3x2.hpp"
#include "../mat3x3.hpp"
#include "../mat3x4.hpp"
#include "../mat4x2.hpp"
#include "../mat4x3.hpp"
#include "../mat4x4.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTC_matrix_integer extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_matrix_integer
	/// @{

	/// High-qualifier signed integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, int, highp>				highp_imat2;

	/// High-qualifier signed integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, int, highp>				highp_imat3;

	/// High-qualifier signed integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, int, highp>				highp_imat4;

	/// High-qualifier signed integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, int, highp>				highp_imat2x2;

	/// High-qualifier signed integer 2x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 3, int, highp>				highp_imat2x3;

	/// High-qualifier signed integer 2x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 4, int, highp>				highp_imat2x4;

	/// High-qualifier signed integer 3x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 2, int, highp>				highp_imat3x2;

	/// High-qualifier signed integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, int, highp>				highp_imat3x3;

	/// High-qualifier signed integer 3x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 4, int, highp>				highp_imat3x4;

	/// High-qualifier signed integer 4x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 2, int, highp>				highp_imat4x2;

	/// High-qualifier signed integer 4x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 3, int, highp>				highp_imat4x3;

	/// High-qualifier signed integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, int, highp>				highp_imat4x4;


	/// Medium-qualifier signed integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, int, mediump>			mediump_imat2;

	/// Medium-qualifier signed integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, int, mediump>			mediump_imat3;

	/// Medium-qualifier signed integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, int, mediump>			mediump_imat4;


	/// Medium-qualifier signed integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, int, mediump>			mediump_imat2x2;

	/// Medium-qualifier signed integer 2x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 3, int, mediump>			mediump_imat2x3;

	/// Medium-qualifier signed integer 2x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 4, int, mediump>			mediump_imat2x4;

	/// Medium-qualifier signed integer 3x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 2, int, mediump>			mediump_imat3x2;

	/// Medium-qualifier signed integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, int, mediump>			mediump_imat3x3;

	/// Medium-qualifier signed integer 3x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 4, int, mediump>			mediump_imat3x4;

	/// Medium-qualifier signed integer 4x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 2, int, mediump>			mediump_imat4x2;

	/// Medium-qualifier signed integer 4x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 3, int, mediump>			mediump_imat4x3;

	/// Medium-qualifier signed integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, int, mediump>			mediump_imat4x4;


	/// Low-qualifier signed integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, int, lowp>				lowp_imat2;

	/// Low-qualifier signed integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, int, lowp>				lowp_imat3;

	/// Low-qualifier signed integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, int, lowp>				lowp_imat4;


	/// Low-qualifier signed integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, int, lowp>				lowp_imat2x2;

	/// Low-qualifier signed integer 2x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 3, int, lowp>				lowp_imat2x3;

	/// Low-qualifier signed integer 2x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 4, int, lowp>				lowp_imat2x4;

	/// Low-qualifier signed integer 3x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 2, int, lowp>				lowp_imat3x2;

	/// Low-qualifier signed integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, int, lowp>				lowp_imat3x3;

	/// Low-qualifier signed integer 3x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 4, int, lowp>				lowp_imat3x4;

	/// Low-qualifier signed integer 4x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 2, int, lowp>				lowp_imat4x2;

	/// Low-qualifier signed integer 4x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 3, int, lowp>				lowp_imat4x3;

	/// Low-qualifier signed integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, int, lowp>				lowp_imat4x4;


	/// High-qualifier unsigned integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, uint, highp>				highp_umat2;

	/// High-qualifier unsigned integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, uint, highp>				highp_umat3;

	/// High-qualifier unsigned integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, uint, highp>				highp_umat4;

	/// High-qualifier unsigned integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, uint, highp>				highp_umat2x2;

	/// High-qualifier unsigned integer 2x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 3, uint, highp>				highp_umat2x3;

	/// High-qualifier unsigned integer 2x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 4, uint, highp>				highp_umat2x4;

	/// High-qualifier unsigned integer 3x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 2, uint, highp>				highp_umat3x2;

	/// High-qualifier unsigned integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, uint, highp>				highp_umat3x3;

	/// High-qualifier unsigned integer 3x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 4, uint, highp>				highp_umat3x4;

	/// High-qualifier unsigned integer 4x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 2, uint, highp>				highp_umat4x2;

	/// High-qualifier unsigned integer 4x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 3, uint, highp>				highp_umat4x3;

	/// High-qualifier unsigned integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, uint, highp>				highp_umat4x4;


	/// Medium-qualifier unsigned integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, uint, mediump>			mediump_umat2;

	/// Medium-qualifier unsigned integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, uint, mediump>			mediump_umat3;

	/// Medium-qualifier unsigned integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, uint, mediump>			mediump_umat4;


	/// Medium-qualifier unsigned integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, uint, mediump>			mediump_umat2x2;

	/// Medium-qualifier unsigned integer 2x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 3, uint, mediump>			mediump_umat2x3;

	/// Medium-qualifier unsigned integer 2x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 4, uint, mediump>			mediump_umat2x4;

	/// Medium-qualifier unsigned integer 3x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 2, uint, mediump>			mediump_umat3x2;

	/// Medium-qualifier unsigned integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, uint, mediump>			mediump_umat3x3;

	/// Medium-qualifier unsigned integer 3x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 4, uint, mediump>			mediump_umat3x4;

	/// Medium-qualifier unsigned integer 4x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 2, uint, mediump>			mediump_umat4x2;

	/// Medium-qualifier unsigned integer 4x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 3, uint, mediump>			mediump_umat4x3;

	/// Medium-qualifier unsigned integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, uint, mediump>			mediump_umat4x4;


	/// Low-qualifier unsigned integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, uint, lowp>				lowp_umat2;

	/// Low-qualifier unsigned integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, uint, lowp>				lowp_umat3;

	/// Low-qualifier unsigned integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, uint, lowp>				lowp_umat4;


	/// Low-qualifier unsigned integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, uint, lowp>				lowp_umat2x2;

	/// Low-qualifier unsigned integer 2x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 3, uint, lowp>				lowp_umat2x3;

	/// Low-qualifier unsigned integer 2x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 4, uint, lowp>				lowp_umat2x4;

	/// Low-qualifier unsigned integer 3x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 2, uint, lowp>				lowp_umat3x2;

	/// Low-qualifier unsigned integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, uint, lowp>				lowp_umat3x3;

	/// Low-qualifier unsigned integer 3x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 4, uint, lowp>				lowp_umat3x4;

	/// Low-qualifier unsigned integer 4x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 2, uint, lowp>				lowp_umat4x2;

	/// Low-qualifier unsigned integer 4x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 3, uint, lowp>				lowp_umat4x3;

	/// Low-qualifier unsigned integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, uint, lowp>				lowp_umat4x4;



	/// Signed integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, int, defaultp>				imat2;

	/// Signed integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, int, defaultp>				imat3;

	/// Signed integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, int, defaultp>				imat4;

	/// Signed integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, int, defaultp>				imat2x2;

	/// Signed integer 2x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 3, int, defaultp>				imat2x3;

	/// Signed integer 2x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 4, int, defaultp>				imat2x4;

	/// Signed integer 3x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 2, int, defaultp>				imat3x2;

	/// Signed integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, int, defaultp>				imat3x3;

	/// Signed integer 3x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 4, int, defaultp>				imat3x4;

	/// Signed integer 4x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 2, int, defaultp>				imat4x2;

	/// Signed integer 4x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 3, int, defaultp>				imat4x3;

	/// Signed integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, int, defaultp>				imat4x4;



	/// Unsigned integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, uint, defaultp>				umat2;

	/// Unsigned integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, uint, defaultp>				umat3;

	/// Unsigned integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, uint, defaultp>				umat4;

	/// Unsigned integer 2x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 2, uint, defaultp>				umat2x2;

	/// Unsigned integer 2x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 3, uint, defaultp>				umat2x3;

	/// Unsigned integer 2x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<2, 4, uint, defaultp>				umat2x4;

	/// Unsigned integer 3x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 2, uint, defaultp>				umat3x2;

	/// Unsigned integer 3x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 3, uint, defaultp>				umat3x3;

	/// Unsigned integer 3x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<3, 4, uint, defaultp>				umat3x4;

	/// Unsigned integer 4x2 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 2, uint, defaultp>				umat4x2;

	/// Unsigned integer 4x3 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 3, uint, defaultp>				umat4x3;

	/// Unsigned integer 4x4 matrix.
	/// @see gtc_matrix_integer
	typedef mat<4, 4, uint, defaultp>				umat4x4;

	/// @}
}//namespace glm
