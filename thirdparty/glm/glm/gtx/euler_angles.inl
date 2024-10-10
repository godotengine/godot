/// @ref gtx_euler_angles

#include "compatibility.hpp" // glm::atan2

namespace glm
{
	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleX
	(
		T const& angleX
	)
	{
		T cosX = glm::cos(angleX);
		T sinX = glm::sin(angleX);

		return mat<4, 4, T, defaultp>(
			T(1), T(0), T(0), T(0),
			T(0), cosX, sinX, T(0),
			T(0),-sinX, cosX, T(0),
			T(0), T(0), T(0), T(1));
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleY
	(
		T const& angleY
	)
	{
		T cosY = glm::cos(angleY);
		T sinY = glm::sin(angleY);

		return mat<4, 4, T, defaultp>(
			cosY,	T(0),	-sinY,	T(0),
			T(0),	T(1),	T(0),	T(0),
			sinY,	T(0),	cosY,	T(0),
			T(0),	T(0),	T(0),	T(1));
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleZ
	(
		T const& angleZ
	)
	{
		T cosZ = glm::cos(angleZ);
		T sinZ = glm::sin(angleZ);

		return mat<4, 4, T, defaultp>(
			cosZ,	sinZ,	T(0), T(0),
			-sinZ,	cosZ,	T(0), T(0),
			T(0),	T(0),	T(1), T(0),
			T(0),	T(0),	T(0), T(1));
	}

	template <typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> derivedEulerAngleX
	(
		T const & angleX,
		T const & angularVelocityX
	)
	{
		T cosX = glm::cos(angleX) * angularVelocityX;
		T sinX = glm::sin(angleX) * angularVelocityX;

		return mat<4, 4, T, defaultp>(
			T(0), T(0), T(0), T(0),
			T(0),-sinX, cosX, T(0),
			T(0),-cosX,-sinX, T(0),
			T(0), T(0), T(0), T(0));
	}

	template <typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> derivedEulerAngleY
	(
		T const & angleY,
		T const & angularVelocityY
	)
	{
		T cosY = glm::cos(angleY) * angularVelocityY;
		T sinY = glm::sin(angleY) * angularVelocityY;

		return mat<4, 4, T, defaultp>(
			-sinY, T(0), -cosY, T(0),
			 T(0), T(0),  T(0), T(0),
			 cosY, T(0), -sinY, T(0),
			 T(0), T(0),  T(0), T(0));
	}

	template <typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> derivedEulerAngleZ
	(
		T const & angleZ,
		T const & angularVelocityZ
	)
	{
		T cosZ = glm::cos(angleZ) * angularVelocityZ;
		T sinZ = glm::sin(angleZ) * angularVelocityZ;

		return mat<4, 4, T, defaultp>(
			-sinZ,  cosZ, T(0), T(0),
			-cosZ, -sinZ, T(0), T(0),
			 T(0),  T(0), T(0), T(0),
			 T(0),  T(0), T(0), T(0));
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleXY
	(
		T const& angleX,
		T const& angleY
	)
	{
		T cosX = glm::cos(angleX);
		T sinX = glm::sin(angleX);
		T cosY = glm::cos(angleY);
		T sinY = glm::sin(angleY);

		return mat<4, 4, T, defaultp>(
			cosY,   -sinX * -sinY,  cosX * -sinY,   T(0),
			T(0),   cosX,           sinX,           T(0),
			sinY,   -sinX * cosY,   cosX * cosY,    T(0),
			T(0),   T(0),           T(0),           T(1));
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleYX
	(
		T const& angleY,
		T const& angleX
	)
	{
		T cosX = glm::cos(angleX);
		T sinX = glm::sin(angleX);
		T cosY = glm::cos(angleY);
		T sinY = glm::sin(angleY);

		return mat<4, 4, T, defaultp>(
			cosY,          0,      -sinY,    T(0),
			sinY * sinX,  cosX, cosY * sinX, T(0),
			sinY * cosX, -sinX, cosY * cosX, T(0),
			T(0),         T(0),     T(0),    T(1));
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleXZ
	(
		T const& angleX,
		T const& angleZ
	)
	{
		return eulerAngleX(angleX) * eulerAngleZ(angleZ);
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleZX
	(
		T const& angleZ,
		T const& angleX
	)
	{
		return eulerAngleZ(angleZ) * eulerAngleX(angleX);
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleYZ
	(
		T const& angleY,
		T const& angleZ
	)
	{
		return eulerAngleY(angleY) * eulerAngleZ(angleZ);
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleZY
	(
		T const& angleZ,
		T const& angleY
	)
	{
		return eulerAngleZ(angleZ) * eulerAngleY(angleY);
	}

    template<typename T>
    GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleXYZ
    (
     T const& t1,
     T const& t2,
     T const& t3
     )
    {
        T c1 = glm::cos(-t1);
        T c2 = glm::cos(-t2);
        T c3 = glm::cos(-t3);
        T s1 = glm::sin(-t1);
        T s2 = glm::sin(-t2);
        T s3 = glm::sin(-t3);

        mat<4, 4, T, defaultp> Result;
        Result[0][0] = c2 * c3;
        Result[0][1] =-c1 * s3 + s1 * s2 * c3;
        Result[0][2] = s1 * s3 + c1 * s2 * c3;
        Result[0][3] = static_cast<T>(0);
        Result[1][0] = c2 * s3;
        Result[1][1] = c1 * c3 + s1 * s2 * s3;
        Result[1][2] =-s1 * c3 + c1 * s2 * s3;
        Result[1][3] = static_cast<T>(0);
        Result[2][0] =-s2;
        Result[2][1] = s1 * c2;
        Result[2][2] = c1 * c2;
        Result[2][3] = static_cast<T>(0);
        Result[3][0] = static_cast<T>(0);
        Result[3][1] = static_cast<T>(0);
        Result[3][2] = static_cast<T>(0);
        Result[3][3] = static_cast<T>(1);
        return Result;
    }

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleYXZ
	(
		T const& yaw,
		T const& pitch,
		T const& roll
	)
	{
		T tmp_ch = glm::cos(yaw);
		T tmp_sh = glm::sin(yaw);
		T tmp_cp = glm::cos(pitch);
		T tmp_sp = glm::sin(pitch);
		T tmp_cb = glm::cos(roll);
		T tmp_sb = glm::sin(roll);

		mat<4, 4, T, defaultp> Result;
		Result[0][0] = tmp_ch * tmp_cb + tmp_sh * tmp_sp * tmp_sb;
		Result[0][1] = tmp_sb * tmp_cp;
		Result[0][2] = -tmp_sh * tmp_cb + tmp_ch * tmp_sp * tmp_sb;
		Result[0][3] = static_cast<T>(0);
		Result[1][0] = -tmp_ch * tmp_sb + tmp_sh * tmp_sp * tmp_cb;
		Result[1][1] = tmp_cb * tmp_cp;
		Result[1][2] = tmp_sb * tmp_sh + tmp_ch * tmp_sp * tmp_cb;
		Result[1][3] = static_cast<T>(0);
		Result[2][0] = tmp_sh * tmp_cp;
		Result[2][1] = -tmp_sp;
		Result[2][2] = tmp_ch * tmp_cp;
		Result[2][3] = static_cast<T>(0);
		Result[3][0] = static_cast<T>(0);
		Result[3][1] = static_cast<T>(0);
		Result[3][2] = static_cast<T>(0);
		Result[3][3] = static_cast<T>(1);
		return Result;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleXZX
	(
		T const & t1,
		T const & t2,
		T const & t3
	)
	{
		T c1 = glm::cos(t1);
		T s1 = glm::sin(t1);
		T c2 = glm::cos(t2);
		T s2 = glm::sin(t2);
		T c3 = glm::cos(t3);
		T s3 = glm::sin(t3);

		mat<4, 4, T, defaultp> Result;
		Result[0][0] = c2;
		Result[0][1] = c1 * s2;
		Result[0][2] = s1 * s2;
		Result[0][3] = static_cast<T>(0);
		Result[1][0] =-c3 * s2;
		Result[1][1] = c1 * c2 * c3 - s1 * s3;
		Result[1][2] = c1 * s3 + c2 * c3 * s1;
		Result[1][3] = static_cast<T>(0);
		Result[2][0] = s2 * s3;
		Result[2][1] =-c3 * s1 - c1 * c2 * s3;
		Result[2][2] = c1 * c3 - c2 * s1 * s3;
		Result[2][3] = static_cast<T>(0);
		Result[3][0] = static_cast<T>(0);
		Result[3][1] = static_cast<T>(0);
		Result[3][2] = static_cast<T>(0);
		Result[3][3] = static_cast<T>(1);
		return Result;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleXYX
	(
		T const & t1,
		T const & t2,
		T const & t3
	)
	{
		T c1 = glm::cos(t1);
		T s1 = glm::sin(t1);
		T c2 = glm::cos(t2);
		T s2 = glm::sin(t2);
		T c3 = glm::cos(t3);
		T s3 = glm::sin(t3);

		mat<4, 4, T, defaultp> Result;
		Result[0][0] = c2;
		Result[0][1] = s1 * s2;
		Result[0][2] =-c1 * s2;
		Result[0][3] = static_cast<T>(0);
		Result[1][0] = s2 * s3;
		Result[1][1] = c1 * c3 - c2 * s1 * s3;
		Result[1][2] = c3 * s1 + c1 * c2 * s3;
		Result[1][3] = static_cast<T>(0);
		Result[2][0] = c3 * s2;
		Result[2][1] =-c1 * s3 - c2 * c3 * s1;
		Result[2][2] = c1 * c2 * c3 - s1 * s3;
		Result[2][3] = static_cast<T>(0);
		Result[3][0] = static_cast<T>(0);
		Result[3][1] = static_cast<T>(0);
		Result[3][2] = static_cast<T>(0);
		Result[3][3] = static_cast<T>(1);
		return Result;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleYXY
	(
		T const & t1,
		T const & t2,
		T const & t3
	)
	{
		T c1 = glm::cos(t1);
		T s1 = glm::sin(t1);
		T c2 = glm::cos(t2);
		T s2 = glm::sin(t2);
		T c3 = glm::cos(t3);
		T s3 = glm::sin(t3);

		mat<4, 4, T, defaultp> Result;
		Result[0][0] = c1 * c3 - c2 * s1 * s3;
		Result[0][1] = s2* s3;
		Result[0][2] =-c3 * s1 - c1 * c2 * s3;
		Result[0][3] = static_cast<T>(0);
		Result[1][0] = s1 * s2;
		Result[1][1] = c2;
		Result[1][2] = c1 * s2;
		Result[1][3] = static_cast<T>(0);
		Result[2][0] = c1 * s3 + c2 * c3 * s1;
		Result[2][1] =-c3 * s2;
		Result[2][2] = c1 * c2 * c3 - s1 * s3;
		Result[2][3] = static_cast<T>(0);
		Result[3][0] = static_cast<T>(0);
		Result[3][1] = static_cast<T>(0);
		Result[3][2] = static_cast<T>(0);
		Result[3][3] = static_cast<T>(1);
		return Result;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleYZY
	(
		T const & t1,
		T const & t2,
		T const & t3
	)
	{
		T c1 = glm::cos(t1);
		T s1 = glm::sin(t1);
		T c2 = glm::cos(t2);
		T s2 = glm::sin(t2);
		T c3 = glm::cos(t3);
		T s3 = glm::sin(t3);

		mat<4, 4, T, defaultp> Result;
		Result[0][0] = c1 * c2 * c3 - s1 * s3;
		Result[0][1] = c3 * s2;
		Result[0][2] =-c1 * s3 - c2 * c3 * s1;
		Result[0][3] = static_cast<T>(0);
		Result[1][0] =-c1 * s2;
		Result[1][1] = c2;
		Result[1][2] = s1 * s2;
		Result[1][3] = static_cast<T>(0);
		Result[2][0] = c3 * s1 + c1 * c2 * s3;
		Result[2][1] = s2 * s3;
		Result[2][2] = c1 * c3 - c2 * s1 * s3;
		Result[2][3] = static_cast<T>(0);
		Result[3][0] = static_cast<T>(0);
		Result[3][1] = static_cast<T>(0);
		Result[3][2] = static_cast<T>(0);
		Result[3][3] = static_cast<T>(1);
		return Result;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleZYZ
	(
		T const & t1,
		T const & t2,
		T const & t3
	)
	{
		T c1 = glm::cos(t1);
		T s1 = glm::sin(t1);
		T c2 = glm::cos(t2);
		T s2 = glm::sin(t2);
		T c3 = glm::cos(t3);
		T s3 = glm::sin(t3);

		mat<4, 4, T, defaultp> Result;
		Result[0][0] = c1 * c2 * c3 - s1 * s3;
		Result[0][1] = c1 * s3 + c2 * c3 * s1;
		Result[0][2] =-c3 * s2;
		Result[0][3] = static_cast<T>(0);
		Result[1][0] =-c3 * s1 - c1 * c2 * s3;
		Result[1][1] = c1 * c3 - c2 * s1 * s3;
		Result[1][2] = s2 * s3;
		Result[1][3] = static_cast<T>(0);
		Result[2][0] = c1 * s2;
		Result[2][1] = s1 * s2;
		Result[2][2] = c2;
		Result[2][3] = static_cast<T>(0);
		Result[3][0] = static_cast<T>(0);
		Result[3][1] = static_cast<T>(0);
		Result[3][2] = static_cast<T>(0);
		Result[3][3] = static_cast<T>(1);
		return Result;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleZXZ
	(
		T const & t1,
		T const & t2,
		T const & t3
	)
	{
		T c1 = glm::cos(t1);
		T s1 = glm::sin(t1);
		T c2 = glm::cos(t2);
		T s2 = glm::sin(t2);
		T c3 = glm::cos(t3);
		T s3 = glm::sin(t3);

		mat<4, 4, T, defaultp> Result;
		Result[0][0] = c1 * c3 - c2 * s1 * s3;
		Result[0][1] = c3 * s1 + c1 * c2 * s3;
		Result[0][2] = s2 *s3;
		Result[0][3] = static_cast<T>(0);
		Result[1][0] =-c1 * s3 - c2 * c3 * s1;
		Result[1][1] = c1 * c2 * c3 - s1 * s3;
		Result[1][2] = c3 * s2;
		Result[1][3] = static_cast<T>(0);
		Result[2][0] = s1 * s2;
		Result[2][1] =-c1 * s2;
		Result[2][2] = c2;
		Result[2][3] = static_cast<T>(0);
		Result[3][0] = static_cast<T>(0);
		Result[3][1] = static_cast<T>(0);
		Result[3][2] = static_cast<T>(0);
		Result[3][3] = static_cast<T>(1);
		return Result;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleXZY
	(
		T const & t1,
		T const & t2,
		T const & t3
	)
	{
		T c1 = glm::cos(t1);
		T s1 = glm::sin(t1);
		T c2 = glm::cos(t2);
		T s2 = glm::sin(t2);
		T c3 = glm::cos(t3);
		T s3 = glm::sin(t3);

		mat<4, 4, T, defaultp> Result;
		Result[0][0] = c2 * c3;
		Result[0][1] = s1 * s3 + c1 * c3 * s2;
		Result[0][2] = c3 * s1 * s2 - c1 * s3;
		Result[0][3] = static_cast<T>(0);
		Result[1][0] =-s2;
		Result[1][1] = c1 * c2;
		Result[1][2] = c2 * s1;
		Result[1][3] = static_cast<T>(0);
		Result[2][0] = c2 * s3;
		Result[2][1] = c1 * s2 * s3 - c3 * s1;
		Result[2][2] = c1 * c3 + s1 * s2 *s3;
		Result[2][3] = static_cast<T>(0);
		Result[3][0] = static_cast<T>(0);
		Result[3][1] = static_cast<T>(0);
		Result[3][2] = static_cast<T>(0);
		Result[3][3] = static_cast<T>(1);
		return Result;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleYZX
	(
		T const & t1,
		T const & t2,
		T const & t3
	)
	{
		T c1 = glm::cos(t1);
		T s1 = glm::sin(t1);
		T c2 = glm::cos(t2);
		T s2 = glm::sin(t2);
		T c3 = glm::cos(t3);
		T s3 = glm::sin(t3);

		mat<4, 4, T, defaultp> Result;
		Result[0][0] = c1 * c2;
		Result[0][1] = s2;
		Result[0][2] =-c2 * s1;
		Result[0][3] = static_cast<T>(0);
		Result[1][0] = s1 * s3 - c1 * c3 * s2;
		Result[1][1] = c2 * c3;
		Result[1][2] = c1 * s3 + c3 * s1 * s2;
		Result[1][3] = static_cast<T>(0);
		Result[2][0] = c3 * s1 + c1 * s2 * s3;
		Result[2][1] =-c2 * s3;
		Result[2][2] = c1 * c3 - s1 * s2 * s3;
		Result[2][3] = static_cast<T>(0);
		Result[3][0] = static_cast<T>(0);
		Result[3][1] = static_cast<T>(0);
		Result[3][2] = static_cast<T>(0);
		Result[3][3] = static_cast<T>(1);
		return Result;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleZYX
	(
		T const & t1,
		T const & t2,
		T const & t3
	)
	{
		T c1 = glm::cos(t1);
		T s1 = glm::sin(t1);
		T c2 = glm::cos(t2);
		T s2 = glm::sin(t2);
		T c3 = glm::cos(t3);
		T s3 = glm::sin(t3);

		mat<4, 4, T, defaultp> Result;
		Result[0][0] = c1 * c2;
		Result[0][1] = c2 * s1;
		Result[0][2] =-s2;
		Result[0][3] = static_cast<T>(0);
		Result[1][0] = c1 * s2 * s3 - c3 * s1;
		Result[1][1] = c1 * c3 + s1 * s2 * s3;
		Result[1][2] = c2 * s3;
		Result[1][3] = static_cast<T>(0);
		Result[2][0] = s1 * s3 + c1 * c3 * s2;
		Result[2][1] = c3 * s1 * s2 - c1 * s3;
		Result[2][2] = c2 * c3;
		Result[2][3] = static_cast<T>(0);
		Result[3][0] = static_cast<T>(0);
		Result[3][1] = static_cast<T>(0);
		Result[3][2] = static_cast<T>(0);
		Result[3][3] = static_cast<T>(1);
		return Result;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> eulerAngleZXY
	(
		T const & t1,
		T const & t2,
		T const & t3
	)
	{
		T c1 = glm::cos(t1);
		T s1 = glm::sin(t1);
		T c2 = glm::cos(t2);
		T s2 = glm::sin(t2);
		T c3 = glm::cos(t3);
		T s3 = glm::sin(t3);

		mat<4, 4, T, defaultp> Result;
		Result[0][0] = c1 * c3 - s1 * s2 * s3;
		Result[0][1] = c3 * s1 + c1 * s2 * s3;
		Result[0][2] =-c2 * s3;
		Result[0][3] = static_cast<T>(0);
		Result[1][0] =-c2 * s1;
		Result[1][1] = c1 * c2;
		Result[1][2] = s2;
		Result[1][3] = static_cast<T>(0);
		Result[2][0] = c1 * s3 + c3 * s1 * s2;
		Result[2][1] = s1 * s3 - c1 * c3 * s2;
		Result[2][2] = c2 * c3;
		Result[2][3] = static_cast<T>(0);
		Result[3][0] = static_cast<T>(0);
		Result[3][1] = static_cast<T>(0);
		Result[3][2] = static_cast<T>(0);
		Result[3][3] = static_cast<T>(1);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> yawPitchRoll
	(
		T const& yaw,
		T const& pitch,
		T const& roll
	)
	{
		T tmp_ch = glm::cos(yaw);
		T tmp_sh = glm::sin(yaw);
		T tmp_cp = glm::cos(pitch);
		T tmp_sp = glm::sin(pitch);
		T tmp_cb = glm::cos(roll);
		T tmp_sb = glm::sin(roll);

		mat<4, 4, T, defaultp> Result;
		Result[0][0] = tmp_ch * tmp_cb + tmp_sh * tmp_sp * tmp_sb;
		Result[0][1] = tmp_sb * tmp_cp;
		Result[0][2] = -tmp_sh * tmp_cb + tmp_ch * tmp_sp * tmp_sb;
		Result[0][3] = static_cast<T>(0);
		Result[1][0] = -tmp_ch * tmp_sb + tmp_sh * tmp_sp * tmp_cb;
		Result[1][1] = tmp_cb * tmp_cp;
		Result[1][2] = tmp_sb * tmp_sh + tmp_ch * tmp_sp * tmp_cb;
		Result[1][3] = static_cast<T>(0);
		Result[2][0] = tmp_sh * tmp_cp;
		Result[2][1] = -tmp_sp;
		Result[2][2] = tmp_ch * tmp_cp;
		Result[2][3] = static_cast<T>(0);
		Result[3][0] = static_cast<T>(0);
		Result[3][1] = static_cast<T>(0);
		Result[3][2] = static_cast<T>(0);
		Result[3][3] = static_cast<T>(1);
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<2, 2, T, defaultp> orientate2
	(
		T const& angle
	)
	{
		T c = glm::cos(angle);
		T s = glm::sin(angle);

		mat<2, 2, T, defaultp> Result;
		Result[0][0] = c;
		Result[0][1] = s;
		Result[1][0] = -s;
		Result[1][1] = c;
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<3, 3, T, defaultp> orientate3
	(
		T const& angle
	)
	{
		T c = glm::cos(angle);
		T s = glm::sin(angle);

		mat<3, 3, T, defaultp> Result;
		Result[0][0] = c;
		Result[0][1] = s;
		Result[0][2] = T(0.0);
		Result[1][0] = -s;
		Result[1][1] = c;
		Result[1][2] = T(0.0);
		Result[2][0] = T(0.0);
		Result[2][1] = T(0.0);
		Result[2][2] = T(1.0);
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> orientate3
	(
		vec<3, T, Q> const& angles
	)
	{
		return mat<3, 3, T, Q>(yawPitchRoll(angles.z, angles.x, angles.y));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> orientate4
	(
		vec<3, T, Q> const& angles
	)
	{
		return yawPitchRoll(angles.z, angles.x, angles.y);
	}

    template<typename T>
    GLM_FUNC_DECL void extractEulerAngleXYZ(mat<4, 4, T, defaultp> const& M,
                                            T & t1,
                                            T & t2,
                                            T & t3)
    {
        T T1 = glm::atan2(M[2][1], M[2][2]);
        T C2 = glm::sqrt(M[0][0]*M[0][0] + M[1][0]*M[1][0]);
        T T2 = glm::atan2(-M[2][0], C2);
        T S1 = glm::sin(T1);
        T C1 = glm::cos(T1);
        T T3 = glm::atan2(S1*M[0][2] - C1*M[0][1], C1*M[1][1] - S1*M[1][2  ]);
        t1 = -T1;
        t2 = -T2;
        t3 = -T3;
    }

	template <typename T>
	GLM_FUNC_QUALIFIER void extractEulerAngleYXZ(mat<4, 4, T, defaultp> const & M,
												 T & t1,
												 T & t2,
												 T & t3)
	{
		T T1 = glm::atan2(M[2][0], M[2][2]);
		T C2 = glm::sqrt(M[0][1]*M[0][1] + M[1][1]*M[1][1]);
		T T2 = glm::atan2(-M[2][1], C2);
		T S1 = glm::sin(T1);
		T C1 = glm::cos(T1);
		T T3 = glm::atan2(S1*M[1][2] - C1*M[1][0], C1*M[0][0] - S1*M[0][2]);
		t1 = T1;
		t2 = T2;
		t3 = T3;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER void extractEulerAngleXZX(mat<4, 4, T, defaultp> const & M,
												 T & t1,
												 T & t2,
												 T & t3)
	{
		T T1 = glm::atan2(M[0][2], M[0][1]);
		T S2 = glm::sqrt(M[1][0]*M[1][0] + M[2][0]*M[2][0]);
		T T2 = glm::atan2(S2, M[0][0]);
		T S1 = glm::sin(T1);
		T C1 = glm::cos(T1);
		T T3 = glm::atan2(C1*M[1][2] - S1*M[1][1], C1*M[2][2] - S1*M[2][1]);
		t1 = T1;
		t2 = T2;
		t3 = T3;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER void extractEulerAngleXYX(mat<4, 4, T, defaultp> const & M,
												 T & t1,
												 T & t2,
												 T & t3)
	{
		T T1 = glm::atan2(M[0][1], -M[0][2]);
		T S2 = glm::sqrt(M[1][0]*M[1][0] + M[2][0]*M[2][0]);
		T T2 = glm::atan2(S2, M[0][0]);
		T S1 = glm::sin(T1);
		T C1 = glm::cos(T1);
		T T3 = glm::atan2(-C1*M[2][1] - S1*M[2][2], C1*M[1][1] + S1*M[1][2]);
		t1 = T1;
		t2 = T2;
		t3 = T3;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER void extractEulerAngleYXY(mat<4, 4, T, defaultp> const & M,
												 T & t1,
												 T & t2,
												 T & t3)
	{
		T T1 = glm::atan2(M[1][0], M[1][2]);
		T S2 = glm::sqrt(M[0][1]*M[0][1] + M[2][1]*M[2][1]);
		T T2 = glm::atan2(S2, M[1][1]);
		T S1 = glm::sin(T1);
		T C1 = glm::cos(T1);
		T T3 = glm::atan2(C1*M[2][0] - S1*M[2][2], C1*M[0][0] - S1*M[0][2]);
		t1 = T1;
		t2 = T2;
		t3 = T3;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER void extractEulerAngleYZY(mat<4, 4, T, defaultp> const & M,
												 T & t1,
												 T & t2,
												 T & t3)
	{
		T T1 = glm::atan2(M[1][2], -M[1][0]);
		T S2 = glm::sqrt(M[0][1]*M[0][1] + M[2][1]*M[2][1]);
		T T2 = glm::atan2(S2, M[1][1]);
		T S1 = glm::sin(T1);
		T C1 = glm::cos(T1);
		T T3 = glm::atan2(-S1*M[0][0] - C1*M[0][2], S1*M[2][0] + C1*M[2][2]);
		t1 = T1;
		t2 = T2;
		t3 = T3;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER void extractEulerAngleZYZ(mat<4, 4, T, defaultp> const & M,
												 T & t1,
												 T & t2,
												 T & t3)
	{
		T T1 = glm::atan2(M[2][1], M[2][0]);
		T S2 = glm::sqrt(M[0][2]*M[0][2] + M[1][2]*M[1][2]);
		T T2 = glm::atan2(S2, M[2][2]);
		T S1 = glm::sin(T1);
		T C1 = glm::cos(T1);
		T T3 = glm::atan2(C1*M[0][1] - S1*M[0][0], C1*M[1][1] - S1*M[1][0]);
		t1 = T1;
		t2 = T2;
		t3 = T3;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER void extractEulerAngleZXZ(mat<4, 4, T, defaultp> const & M,
												 T & t1,
												 T & t2,
												 T & t3)
	{
		T T1 = glm::atan2(M[2][0], -M[2][1]);
		T S2 = glm::sqrt(M[0][2]*M[0][2] + M[1][2]*M[1][2]);
		T T2 = glm::atan2(S2, M[2][2]);
		T S1 = glm::sin(T1);
		T C1 = glm::cos(T1);
		T T3 = glm::atan2(-C1*M[1][0] - S1*M[1][1], C1*M[0][0] + S1*M[0][1]);
		t1 = T1;
		t2 = T2;
		t3 = T3;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER void extractEulerAngleXZY(mat<4, 4, T, defaultp> const & M,
												 T & t1,
												 T & t2,
												 T & t3)
	{
		T T1 = glm::atan2(M[1][2], M[1][1]);
		T C2 = glm::sqrt(M[0][0]*M[0][0] + M[2][0]*M[2][0]);
		T T2 = glm::atan2(-M[1][0], C2);
		T S1 = glm::sin(T1);
		T C1 = glm::cos(T1);
		T T3 = glm::atan2(S1*M[0][1] - C1*M[0][2], C1*M[2][2] - S1*M[2][1]);
		t1 = T1;
		t2 = T2;
		t3 = T3;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER void extractEulerAngleYZX(mat<4, 4, T, defaultp> const & M,
												 T & t1,
												 T & t2,
												 T & t3)
	{
		T T1 = glm::atan2(-M[0][2], M[0][0]);
		T C2 = glm::sqrt(M[1][1]*M[1][1] + M[2][1]*M[2][1]);
		T T2 = glm::atan2(M[0][1], C2);
		T S1 = glm::sin(T1);
		T C1 = glm::cos(T1);
		T T3 = glm::atan2(S1*M[1][0] + C1*M[1][2], S1*M[2][0] + C1*M[2][2]);
		t1 = T1;
		t2 = T2;
		t3 = T3;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER void extractEulerAngleZYX(mat<4, 4, T, defaultp> const & M,
												 T & t1,
												 T & t2,
												 T & t3)
	{
		T T1 = glm::atan2(M[0][1], M[0][0]);
		T C2 = glm::sqrt(M[1][2]*M[1][2] + M[2][2]*M[2][2]);
		T T2 = glm::atan2(-M[0][2], C2);
		T S1 = glm::sin(T1);
		T C1 = glm::cos(T1);
		T T3 = glm::atan2(S1*M[2][0] - C1*M[2][1], C1*M[1][1] - S1*M[1][0]);
		t1 = T1;
		t2 = T2;
		t3 = T3;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER void extractEulerAngleZXY(mat<4, 4, T, defaultp> const & M,
												 T & t1,
												 T & t2,
												 T & t3)
	{
		T T1 = glm::atan2(-M[1][0], M[1][1]);
		T C2 = glm::sqrt(M[0][2]*M[0][2] + M[2][2]*M[2][2]);
		T T2 = glm::atan2(M[1][2], C2);
		T S1 = glm::sin(T1);
		T C1 = glm::cos(T1);
		T T3 = glm::atan2(C1*M[2][0] + S1*M[2][1], C1*M[0][0] + S1*M[0][1]);
		t1 = T1;
		t2 = T2;
		t3 = T3;
	}
}//namespace glm
