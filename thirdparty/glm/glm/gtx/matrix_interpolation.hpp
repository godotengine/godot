/// @ref gtx_matrix_interpolation
/// @file glm/gtx/matrix_interpolation.hpp
/// @author Ghenadii Ursachi (the.asteroth@gmail.com)
///
/// @see core (dependence)
///
/// @defgroup gtx_matrix_interpolation GLM_GTX_matrix_interpolation
/// @ingroup gtx
///
/// Include <glm/gtx/matrix_interpolation.hpp> to use the features of this extension.
///
/// Allows to directly interpolate two matrices.

#pragma once

// Dependency:
#include "../glm.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_matrix_interpolation is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_matrix_interpolation extension included")
#	endif
#endif

namespace glm
{
	/// @addtogroup gtx_matrix_interpolation
	/// @{

	/// Get the axis and angle of the rotation from a matrix.
	/// From GLM_GTX_matrix_interpolation extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL void axisAngle(
		mat<4, 4, T, Q> const& Mat, vec<3, T, Q> & Axis, T & Angle);

	/// Build a matrix from axis and angle.
	/// From GLM_GTX_matrix_interpolation extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL mat<4, 4, T, Q> axisAngleMatrix(
		vec<3, T, Q> const& Axis, T const Angle);

	/// Extracts the rotation part of a matrix.
	/// From GLM_GTX_matrix_interpolation extension.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL mat<4, 4, T, Q> extractMatrixRotation(
		mat<4, 4, T, Q> const& Mat);

	/// Build a interpolation of 4 * 4 matrixes.
	/// From GLM_GTX_matrix_interpolation extension.
	/// Warning! works only with rotation and/or translation matrixes, scale will generate unexpected results.
	template<typename T, qualifier Q>
	GLM_FUNC_DECL mat<4, 4, T, Q> interpolate(
		mat<4, 4, T, Q> const& m1, mat<4, 4, T, Q> const& m2, T const Delta);

	/// @}
}//namespace glm

#include "matrix_interpolation.inl"
