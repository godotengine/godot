/// @ref ext_quaternion_trigonometric
/// @file glm/ext/quaternion_trigonometric.hpp
///
/// @defgroup ext_quaternion_trigonometric GLM_EXT_quaternion_trigonometric
/// @ingroup ext
///
/// Provides trigonometric functions for quaternion types
///
/// Include <glm/ext/quaternion_trigonometric.hpp> to use the features of this extension.
///
/// @see ext_quaternion_float
/// @see ext_quaternion_double
/// @see ext_quaternion_exponential
/// @see ext_quaternion_geometric
/// @see ext_quaternion_relational
/// @see ext_quaternion_transform

#pragma once

// Dependency:
#include "../trigonometric.hpp"
#include "../exponential.hpp"
#include "scalar_constants.hpp"
#include "vector_relational.hpp"
#include <limits>

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_quaternion_trigonometric extension included")
#endif

namespace glm
{
	/// @addtogroup ext_quaternion_trigonometric
	/// @{

	/// Returns the quaternion rotation angle.
	///
	/// @tparam T A floating-point scalar type
	/// @tparam Q A value from qualifier enum
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T angle(qua<T, Q> const& x);

	/// Returns the q rotation axis.
	///
	/// @tparam T A floating-point scalar type
	/// @tparam Q A value from qualifier enum
	template<typename T, qualifier Q>
	GLM_FUNC_DECL vec<3, T, Q> axis(qua<T, Q> const& x);

	/// Build a quaternion from an angle and a normalized axis.
	///
	/// @param angle Angle expressed in radians.
	/// @param axis Axis of the quaternion, must be normalized.
	///
	/// @tparam T A floating-point scalar type
	/// @tparam Q A value from qualifier enum
	template<typename T, qualifier Q>
	GLM_FUNC_DECL qua<T, Q> angleAxis(T const& angle, vec<3, T, Q> const& axis);

	/// @}
} //namespace glm

#include "quaternion_trigonometric.inl"
