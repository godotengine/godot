/// @ref gtc_quaternion
/// @file glm/gtc/quaternion.hpp
///
/// @see core (dependence)
/// @see gtc_constants (dependence)
///
/// @defgroup gtc_quaternion GLM_GTC_quaternion
/// @ingroup gtc
///
/// Include <glm/gtc/quaternion.hpp> to use the features of this extension.
///
/// Defines a templated quaternion type and several quaternion operations.

#pragma once

// Dependency:
#include "../gtc/constants.hpp"
#include "../gtc/matrix_transform.hpp"
#include "../ext/vector_relational.hpp"
#include "../ext/quaternion_common.hpp"
#include "../ext/quaternion_float.hpp"
#include "../ext/quaternion_float_precision.hpp"
#include "../ext/quaternion_double.hpp"
#include "../ext/quaternion_double_precision.hpp"
#include "../ext/quaternion_relational.hpp"
#include "../ext/quaternion_geometric.hpp"
#include "../ext/quaternion_trigonometric.hpp"
#include "../ext/quaternion_transform.hpp"
#include "../detail/type_mat3x3.hpp"
#include "../detail/type_mat4x4.hpp"
#include "../detail/type_vec3.hpp"
#include "../detail/type_vec4.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTC_quaternion extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_quaternion
	/// @{

	/// Returns euler angles, pitch as x, yaw as y, roll as z.
	/// The result is expressed in radians.
	///
	/// @tparam T Floating-point scalar types.
	///
	/// @see gtc_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL vec<3, T, Q> eulerAngles(qua<T, Q> const& x);

	/// Returns roll value of euler angles expressed in radians.
	///
	/// @tparam T Floating-point scalar types.
	///
	/// @see gtc_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T roll(qua<T, Q> const& x);

	/// Returns pitch value of euler angles expressed in radians.
	///
	/// @tparam T Floating-point scalar types.
	///
	/// @see gtc_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T pitch(qua<T, Q> const& x);

	/// Returns yaw value of euler angles expressed in radians.
	///
	/// @tparam T Floating-point scalar types.
	///
	/// @see gtc_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T yaw(qua<T, Q> const& x);

	/// Converts a quaternion to a 3 * 3 matrix.
	///
	/// @tparam T Floating-point scalar types.
	///
	/// @see gtc_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL mat<3, 3, T, Q> mat3_cast(qua<T, Q> const& x);

	/// Converts a quaternion to a 4 * 4 matrix.
	///
	/// @tparam T Floating-point scalar types.
	///
	/// @see gtc_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL mat<4, 4, T, Q> mat4_cast(qua<T, Q> const& x);

	/// Converts a pure rotation 3 * 3 matrix to a quaternion.
	///
	/// @tparam T Floating-point scalar types.
	///
	/// @see gtc_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL qua<T, Q> quat_cast(mat<3, 3, T, Q> const& x);

	/// Converts a pure rotation 4 * 4 matrix to a quaternion.
	///
	/// @tparam T Floating-point scalar types.
	///
	/// @see gtc_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL qua<T, Q> quat_cast(mat<4, 4, T, Q> const& x);

	/// Returns the component-wise comparison result of x < y.
	///
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see ext_quaternion_relational
	template<typename T, qualifier Q>
	GLM_FUNC_DECL vec<4, bool, Q> lessThan(qua<T, Q> const& x, qua<T, Q> const& y);

	/// Returns the component-wise comparison of result x <= y.
	///
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see ext_quaternion_relational
	template<typename T, qualifier Q>
	GLM_FUNC_DECL vec<4, bool, Q> lessThanEqual(qua<T, Q> const& x, qua<T, Q> const& y);

	/// Returns the component-wise comparison of result x > y.
	///
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see ext_quaternion_relational
	template<typename T, qualifier Q>
	GLM_FUNC_DECL vec<4, bool, Q> greaterThan(qua<T, Q> const& x, qua<T, Q> const& y);

	/// Returns the component-wise comparison of result x >= y.
	///
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see ext_quaternion_relational
	template<typename T, qualifier Q>
	GLM_FUNC_DECL vec<4, bool, Q> greaterThanEqual(qua<T, Q> const& x, qua<T, Q> const& y);

	/// Build a look at quaternion based on the default handedness.
	///
	/// @param direction Desired forward direction. Needs to be normalized.
	/// @param up Up vector, how the camera is oriented. Typically (0, 1, 0).
	template<typename T, qualifier Q>
	GLM_FUNC_DECL qua<T, Q> quatLookAt(
		vec<3, T, Q> const& direction,
		vec<3, T, Q> const& up);

	/// Build a right-handed look at quaternion.
	///
	/// @param direction Desired forward direction onto which the -z-axis gets mapped. Needs to be normalized.
	/// @param up Up vector, how the camera is oriented. Typically (0, 1, 0).
	template<typename T, qualifier Q>
	GLM_FUNC_DECL qua<T, Q> quatLookAtRH(
		vec<3, T, Q> const& direction,
		vec<3, T, Q> const& up);

	/// Build a left-handed look at quaternion.
	///
	/// @param direction Desired forward direction onto which the +z-axis gets mapped. Needs to be normalized.
	/// @param up Up vector, how the camera is oriented. Typically (0, 1, 0).
	template<typename T, qualifier Q>
	GLM_FUNC_DECL qua<T, Q> quatLookAtLH(
		vec<3, T, Q> const& direction,
		vec<3, T, Q> const& up);
	/// @}
} //namespace glm

#include "quaternion.inl"
