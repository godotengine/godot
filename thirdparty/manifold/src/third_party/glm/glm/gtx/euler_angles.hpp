/// @ref gtx_euler_angles
/// @file glm/gtx/euler_angles.hpp
///
/// @see core (dependence)
///
/// @defgroup gtx_euler_angles GLM_GTX_euler_angles
/// @ingroup gtx
///
/// Include <glm/gtx/euler_angles.hpp> to use the features of this extension.
///
/// Build matrices from Euler angles.
///
/// Extraction of Euler angles from rotation matrix.
/// Based on the original paper 2014 Mike Day - Extracting Euler Angles from a Rotation Matrix.

#pragma once

// Dependency:
#include "../glm.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_euler_angles is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_euler_angles extension included")
#	endif
#endif

namespace glm
{
	/// @addtogroup gtx_euler_angles
	/// @{

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from an euler angle X.
	/// @see gtx_euler_angles
	template<typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleX(
		T const& angleX);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from an euler angle Y.
	/// @see gtx_euler_angles
	template<typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleY(
		T const& angleY);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from an euler angle Z.
	/// @see gtx_euler_angles
	template<typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleZ(
		T const& angleZ);

	/// Creates a 3D 4 * 4 homogeneous derived matrix from the rotation matrix about X-axis.
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> derivedEulerAngleX(
		T const & angleX, T const & angularVelocityX);

	/// Creates a 3D 4 * 4 homogeneous derived matrix from the rotation matrix about Y-axis.
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> derivedEulerAngleY(
		T const & angleY, T const & angularVelocityY);

	/// Creates a 3D 4 * 4 homogeneous derived matrix from the rotation matrix about Z-axis.
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> derivedEulerAngleZ(
		T const & angleZ, T const & angularVelocityZ);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (X * Y).
	/// @see gtx_euler_angles
	template<typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleXY(
		T const& angleX,
		T const& angleY);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Y * X).
	/// @see gtx_euler_angles
	template<typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleYX(
		T const& angleY,
		T const& angleX);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (X * Z).
	/// @see gtx_euler_angles
	template<typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleXZ(
		T const& angleX,
		T const& angleZ);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Z * X).
	/// @see gtx_euler_angles
	template<typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleZX(
		T const& angle,
		T const& angleX);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Y * Z).
	/// @see gtx_euler_angles
	template<typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleYZ(
		T const& angleY,
		T const& angleZ);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Z * Y).
	/// @see gtx_euler_angles
	template<typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleZY(
		T const& angleZ,
		T const& angleY);

    /// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (X * Y * Z).
    /// @see gtx_euler_angles
    template<typename T>
    GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleXYZ(
        T const& t1,
        T const& t2,
        T const& t3);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Y * X * Z).
	/// @see gtx_euler_angles
	template<typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleYXZ(
		T const& yaw,
		T const& pitch,
		T const& roll);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (X * Z * X).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleXZX(
		T const & t1,
		T const & t2,
		T const & t3);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (X * Y * X).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleXYX(
		T const & t1,
		T const & t2,
		T const & t3);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Y * X * Y).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleYXY(
		T const & t1,
		T const & t2,
		T const & t3);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Y * Z * Y).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleYZY(
		T const & t1,
		T const & t2,
		T const & t3);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Z * Y * Z).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleZYZ(
		T const & t1,
		T const & t2,
		T const & t3);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Z * X * Z).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleZXZ(
		T const & t1,
		T const & t2,
		T const & t3);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (X * Z * Y).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleXZY(
		T const & t1,
		T const & t2,
		T const & t3);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Y * Z * X).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleYZX(
		T const & t1,
		T const & t2,
		T const & t3);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Z * Y * X).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleZYX(
		T const & t1,
		T const & t2,
		T const & t3);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Z * X * Y).
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> eulerAngleZXY(
		T const & t1,
		T const & t2,
		T const & t3);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Y * X * Z).
	/// @see gtx_euler_angles
	template<typename T>
	GLM_FUNC_DECL mat<4, 4, T, defaultp> yawPitchRoll(
		T const& yaw,
		T const& pitch,
		T const& roll);

	/// Creates a 2D 2 * 2 rotation matrix from an euler angle.
	/// @see gtx_euler_angles
	template<typename T>
	GLM_FUNC_DECL mat<2, 2, T, defaultp> orientate2(T const& angle);

	/// Creates a 2D 4 * 4 homogeneous rotation matrix from an euler angle.
	/// @see gtx_euler_angles
	template<typename T>
	GLM_FUNC_DECL mat<3, 3, T, defaultp> orientate3(T const& angle);

	/// Creates a 3D 3 * 3 rotation matrix from euler angles (Y * X * Z).
	/// @see gtx_euler_angles
	template<typename T, qualifier Q>
	GLM_FUNC_DECL mat<3, 3, T, Q> orientate3(vec<3, T, Q> const& angles);

	/// Creates a 3D 4 * 4 homogeneous rotation matrix from euler angles (Y * X * Z).
	/// @see gtx_euler_angles
	template<typename T, qualifier Q>
	GLM_FUNC_DECL mat<4, 4, T, Q> orientate4(vec<3, T, Q> const& angles);

    /// Extracts the (X * Y * Z) Euler angles from the rotation matrix M
    /// @see gtx_euler_angles
    template<typename T>
    GLM_FUNC_DECL void extractEulerAngleXYZ(mat<4, 4, T, defaultp> const& M,
                                            T & t1,
                                            T & t2,
                                            T & t3);

	/// Extracts the (Y * X * Z) Euler angles from the rotation matrix M
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL void extractEulerAngleYXZ(mat<4, 4, T, defaultp> const & M,
											T & t1,
											T & t2,
											T & t3);

	/// Extracts the (X * Z * X) Euler angles from the rotation matrix M
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL void extractEulerAngleXZX(mat<4, 4, T, defaultp> const & M,
											T & t1,
											T & t2,
											T & t3);

	/// Extracts the (X * Y * X) Euler angles from the rotation matrix M
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL void extractEulerAngleXYX(mat<4, 4, T, defaultp> const & M,
											T & t1,
											T & t2,
											T & t3);

	/// Extracts the (Y * X * Y) Euler angles from the rotation matrix M
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL void extractEulerAngleYXY(mat<4, 4, T, defaultp> const & M,
											T & t1,
											T & t2,
											T & t3);

	/// Extracts the (Y * Z * Y) Euler angles from the rotation matrix M
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL void extractEulerAngleYZY(mat<4, 4, T, defaultp> const & M,
											T & t1,
											T & t2,
											T & t3);

	/// Extracts the (Z * Y * Z) Euler angles from the rotation matrix M
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL void extractEulerAngleZYZ(mat<4, 4, T, defaultp> const & M,
											T & t1,
											T & t2,
											T & t3);

	/// Extracts the (Z * X * Z) Euler angles from the rotation matrix M
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL void extractEulerAngleZXZ(mat<4, 4, T, defaultp> const & M,
											T & t1,
											T & t2,
											T & t3);

	/// Extracts the (X * Z * Y) Euler angles from the rotation matrix M
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL void extractEulerAngleXZY(mat<4, 4, T, defaultp> const & M,
											T & t1,
											T & t2,
											T & t3);

	/// Extracts the (Y * Z * X) Euler angles from the rotation matrix M
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL void extractEulerAngleYZX(mat<4, 4, T, defaultp> const & M,
											T & t1,
											T & t2,
											T & t3);

	/// Extracts the (Z * Y * X) Euler angles from the rotation matrix M
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL void extractEulerAngleZYX(mat<4, 4, T, defaultp> const & M,
											T & t1,
											T & t2,
											T & t3);

	/// Extracts the (Z * X * Y) Euler angles from the rotation matrix M
	/// @see gtx_euler_angles
	template <typename T>
	GLM_FUNC_DECL void extractEulerAngleZXY(mat<4, 4, T, defaultp> const & M,
											T & t1,
											T & t2,
											T & t3);

	/// @}
}//namespace glm

#include "euler_angles.inl"
