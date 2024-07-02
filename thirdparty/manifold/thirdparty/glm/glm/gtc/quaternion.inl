#include "../trigonometric.hpp"
#include "../geometric.hpp"
#include "../exponential.hpp"
#include "epsilon.hpp"
#include <limits>

namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> eulerAngles(qua<T, Q> const& x)
	{
		return vec<3, T, Q>(pitch(x), yaw(x), roll(x));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T roll(qua<T, Q> const& q)
	{
		T const y = static_cast<T>(2) * (q.x * q.y + q.w * q.z);
		T const x = q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z;

		if(all(equal(vec<2, T, Q>(x, y), vec<2, T, Q>(0), epsilon<T>()))) //avoid atan2(0,0) - handle singularity - Matiis
			return static_cast<T>(0);

		return static_cast<T>(atan(y, x));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T pitch(qua<T, Q> const& q)
	{
		//return T(atan(T(2) * (q.y * q.z + q.w * q.x), q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z));
		T const y = static_cast<T>(2) * (q.y * q.z + q.w * q.x);
		T const x = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z;

		if(all(equal(vec<2, T, Q>(x, y), vec<2, T, Q>(0), epsilon<T>()))) //avoid atan2(0,0) - handle singularity - Matiis
			return static_cast<T>(static_cast<T>(2) * atan(q.x, q.w));

		return static_cast<T>(atan(y, x));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T yaw(qua<T, Q> const& q)
	{
		return asin(clamp(static_cast<T>(-2) * (q.x * q.z - q.w * q.y), static_cast<T>(-1), static_cast<T>(1)));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> mat3_cast(qua<T, Q> const& q)
	{
		mat<3, 3, T, Q> Result(T(1));
		T qxx(q.x * q.x);
		T qyy(q.y * q.y);
		T qzz(q.z * q.z);
		T qxz(q.x * q.z);
		T qxy(q.x * q.y);
		T qyz(q.y * q.z);
		T qwx(q.w * q.x);
		T qwy(q.w * q.y);
		T qwz(q.w * q.z);

		Result[0][0] = T(1) - T(2) * (qyy +  qzz);
		Result[0][1] = T(2) * (qxy + qwz);
		Result[0][2] = T(2) * (qxz - qwy);

		Result[1][0] = T(2) * (qxy - qwz);
		Result[1][1] = T(1) - T(2) * (qxx +  qzz);
		Result[1][2] = T(2) * (qyz + qwx);

		Result[2][0] = T(2) * (qxz + qwy);
		Result[2][1] = T(2) * (qyz - qwx);
		Result[2][2] = T(1) - T(2) * (qxx +  qyy);
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> mat4_cast(qua<T, Q> const& q)
	{
		return mat<4, 4, T, Q>(mat3_cast(q));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> quat_cast(mat<3, 3, T, Q> const& m)
	{
		T fourXSquaredMinus1 = m[0][0] - m[1][1] - m[2][2];
		T fourYSquaredMinus1 = m[1][1] - m[0][0] - m[2][2];
		T fourZSquaredMinus1 = m[2][2] - m[0][0] - m[1][1];
		T fourWSquaredMinus1 = m[0][0] + m[1][1] + m[2][2];

		int biggestIndex = 0;
		T fourBiggestSquaredMinus1 = fourWSquaredMinus1;
		if(fourXSquaredMinus1 > fourBiggestSquaredMinus1)
		{
			fourBiggestSquaredMinus1 = fourXSquaredMinus1;
			biggestIndex = 1;
		}
		if(fourYSquaredMinus1 > fourBiggestSquaredMinus1)
		{
			fourBiggestSquaredMinus1 = fourYSquaredMinus1;
			biggestIndex = 2;
		}
		if(fourZSquaredMinus1 > fourBiggestSquaredMinus1)
		{
			fourBiggestSquaredMinus1 = fourZSquaredMinus1;
			biggestIndex = 3;
		}

		T biggestVal = sqrt(fourBiggestSquaredMinus1 + static_cast<T>(1)) * static_cast<T>(0.5);
		T mult = static_cast<T>(0.25) / biggestVal;

		switch(biggestIndex)
		{
		case 0:
			return qua<T, Q>::wxyz(biggestVal, (m[1][2] - m[2][1]) * mult, (m[2][0] - m[0][2]) * mult, (m[0][1] - m[1][0]) * mult);
		case 1:
			return qua<T, Q>::wxyz((m[1][2] - m[2][1]) * mult, biggestVal, (m[0][1] + m[1][0]) * mult, (m[2][0] + m[0][2]) * mult);
		case 2:
			return qua<T, Q>::wxyz((m[2][0] - m[0][2]) * mult, (m[0][1] + m[1][0]) * mult, biggestVal, (m[1][2] + m[2][1]) * mult);
		case 3:
			return qua<T, Q>::wxyz((m[0][1] - m[1][0]) * mult, (m[2][0] + m[0][2]) * mult, (m[1][2] + m[2][1]) * mult, biggestVal);
		default: // Silence a -Wswitch-default warning in GCC. Should never actually get here. Assert is just for sanity.
			assert(false);
			return qua<T, Q>::wxyz(1, 0, 0, 0);
		}
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> quat_cast(mat<4, 4, T, Q> const& m4)
	{
		return quat_cast(mat<3, 3, T, Q>(m4));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, bool, Q> lessThan(qua<T, Q> const& x, qua<T, Q> const& y)
	{
		vec<4, bool, Q> Result(false, false, false, false);
		for(length_t i = 0; i < x.length(); ++i)
			Result[i] = x[i] < y[i];
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, bool, Q> lessThanEqual(qua<T, Q> const& x, qua<T, Q> const& y)
	{
		vec<4, bool, Q> Result(false, false, false, false);
		for(length_t i = 0; i < x.length(); ++i)
			Result[i] = x[i] <= y[i];
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, bool, Q> greaterThan(qua<T, Q> const& x, qua<T, Q> const& y)
	{
		vec<4, bool, Q> Result(false, false, false, false);
		for(length_t i = 0; i < x.length(); ++i)
			Result[i] = x[i] > y[i];
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, bool, Q> greaterThanEqual(qua<T, Q> const& x, qua<T, Q> const& y)
	{
		vec<4, bool, Q> Result(false, false, false, false);
		for(length_t i = 0; i < x.length(); ++i)
			Result[i] = x[i] >= y[i];
		return Result;
	}


	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> quatLookAt(vec<3, T, Q> const& direction, vec<3, T, Q> const& up)
	{
#		if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
			return quatLookAtLH(direction, up);
#		else
			return quatLookAtRH(direction, up);
# 		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> quatLookAtRH(vec<3, T, Q> const& direction, vec<3, T, Q> const& up)
	{
		mat<3, 3, T, Q> Result;

		Result[2] = -direction;
		vec<3, T, Q> const& Right = cross(up, Result[2]);
		Result[0] = Right * inversesqrt(max(static_cast<T>(0.00001), dot(Right, Right)));
		Result[1] = cross(Result[2], Result[0]);

		return quat_cast(Result);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> quatLookAtLH(vec<3, T, Q> const& direction, vec<3, T, Q> const& up)
	{
		mat<3, 3, T, Q> Result;

		Result[2] = direction;
		vec<3, T, Q> const& Right = cross(up, Result[2]);
		Result[0] = Right * inversesqrt(max(static_cast<T>(0.00001), dot(Right, Right)));
		Result[1] = cross(Result[2], Result[0]);

		return quat_cast(Result);
	}
}//namespace glm

#if GLM_CONFIG_SIMD == GLM_ENABLE
#	include "quaternion_simd.inl"
#endif

