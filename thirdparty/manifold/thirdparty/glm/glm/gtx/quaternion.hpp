/// @ref gtx_quaternion
/// @file glm/gtx/quaternion.hpp
///
/// @see core (dependence)
/// @see gtx_extented_min_max (dependence)
///
/// @defgroup gtx_quaternion GLM_GTX_quaternion
/// @ingroup gtx
///
/// Include <glm/gtx/quaternion.hpp> to use the features of this extension.
///
/// Extended quaternion types and functions

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtc/constants.hpp"
#include "../gtc/quaternion.hpp"
#include "../ext/quaternion_exponential.hpp"
#include "../gtx/norm.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_quaternion is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_quaternion extension included")
#	endif
#endif

namespace glm
{
	/// @addtogroup gtx_quaternion
	/// @{

	/// Create an identity quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR qua<T, Q> quat_identity();

	/// Compute a cross product between a quaternion and a vector.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR vec<3, T, Q> cross(
		qua<T, Q> const& q,
		vec<3, T, Q> const& v);

	//! Compute a cross product between a vector and a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR vec<3, T, Q> cross(
		vec<3, T, Q> const& v,
		qua<T, Q> const& q);

	//! Compute a point on a path according squad equation.
	//! q1 and q2 are control points; s1 and s2 are intermediate control points.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL qua<T, Q> squad(
		qua<T, Q> const& q1,
		qua<T, Q> const& q2,
		qua<T, Q> const& s1,
		qua<T, Q> const& s2,
		T const& h);

	//! Returns an intermediate control point for squad interpolation.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL qua<T, Q> intermediate(
		qua<T, Q> const& prev,
		qua<T, Q> const& curr,
		qua<T, Q> const& next);

	//! Returns quarternion square root.
	///
	/// @see gtx_quaternion
	//template<typename T, qualifier Q>
	//qua<T, Q> sqrt(
	//	qua<T, Q> const& q);

	//! Rotates a 3 components vector by a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL vec<3, T, Q> rotate(
		qua<T, Q> const& q,
		vec<3, T, Q> const& v);

	/// Rotates a 4 components vector by a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL vec<4, T, Q> rotate(
		qua<T, Q> const& q,
		vec<4, T, Q> const& v);

	/// Extract the real component of a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL T extractRealComponent(
		qua<T, Q> const& q);

	/// Converts a quaternion to a 3 * 3 matrix.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> toMat3(
		qua<T, Q> const& x){return mat3_cast(x);}

	/// Converts a quaternion to a 4 * 4 matrix.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> toMat4(
		qua<T, Q> const& x){return mat4_cast(x);}

	/// Converts a 3 * 3 matrix to a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> toQuat(
		mat<3, 3, T, Q> const& x){return quat_cast(x);}

	/// Converts a 4 * 4 matrix to a quaternion.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> toQuat(
		mat<4, 4, T, Q> const& x){return quat_cast(x);}

	/// Quaternion interpolation using the rotation short path.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL qua<T, Q> shortMix(
		qua<T, Q> const& x,
		qua<T, Q> const& y,
		T const& a);

	/// Quaternion normalized linear interpolation.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL qua<T, Q> fastMix(
		qua<T, Q> const& x,
		qua<T, Q> const& y,
		T const& a);

	/// Compute the rotation between two vectors.
	/// @param orig vector, needs to be normalized
	/// @param dest vector, needs to be normalized
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL qua<T, Q> rotation(
		vec<3, T, Q> const& orig,
		vec<3, T, Q> const& dest);

	/// Returns the squared length of x.
	///
	/// @see gtx_quaternion
	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR T length2(qua<T, Q> const& q);

	/// @}
}//namespace glm

#include "quaternion.inl"
