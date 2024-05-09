/// @ref gtc_matrix_inverse
/// @file glm/gtc/matrix_inverse.hpp
///
/// @see core (dependence)
///
/// @defgroup gtc_matrix_inverse GLM_GTC_matrix_inverse
/// @ingroup gtc
///
/// Include <glm/gtc/matrix_inverse.hpp> to use the features of this extension.
///
/// Defines additional matrix inverting functions.

#pragma once

// Dependencies
#include "../detail/setup.hpp"
#include "../matrix.hpp"
#include "../mat2x2.hpp"
#include "../mat3x3.hpp"
#include "../mat4x4.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTC_matrix_inverse extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_matrix_inverse
	/// @{

	/// Fast matrix inverse for affine matrix.
	///
	/// @param m Input matrix to invert.
	/// @tparam genType Squared floating-point matrix: half, float or double. Inverse of matrix based of half-qualifier floating point value is highly inaccurate.
	/// @see gtc_matrix_inverse
	template<typename genType>
	GLM_FUNC_DECL genType affineInverse(genType const& m);

	/// Compute the inverse transpose of a matrix.
	///
	/// @param m Input matrix to invert transpose.
	/// @tparam genType Squared floating-point matrix: half, float or double. Inverse of matrix based of half-qualifier floating point value is highly inaccurate.
	/// @see gtc_matrix_inverse
	template<typename genType>
	GLM_FUNC_DECL genType inverseTranspose(genType const& m);

	/// @}
}//namespace glm

#include "matrix_inverse.inl"
