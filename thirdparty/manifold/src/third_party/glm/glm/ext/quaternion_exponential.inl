#include "scalar_constants.hpp"

namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> exp(qua<T, Q> const& q)
	{
		vec<3, T, Q> u(q.x, q.y, q.z);
		T const Angle = glm::length(u);
		if (Angle < epsilon<T>())
			return qua<T, Q>();

		vec<3, T, Q> const v(u / Angle);
		return qua<T, Q>(cos(Angle), sin(Angle) * v);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> log(qua<T, Q> const& q)
	{
		vec<3, T, Q> u(q.x, q.y, q.z);
		T Vec3Len = length(u);

		if (Vec3Len < epsilon<T>())
		{
			if(q.w > static_cast<T>(0))
				return qua<T, Q>(log(q.w), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
			else if(q.w < static_cast<T>(0))
				return qua<T, Q>(log(-q.w), pi<T>(), static_cast<T>(0), static_cast<T>(0));
			else
				return qua<T, Q>(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity());
		}
		else
		{
			T t = atan(Vec3Len, T(q.w)) / Vec3Len;
			T QuatLen2 = Vec3Len * Vec3Len + q.w * q.w;
			return qua<T, Q>(static_cast<T>(0.5) * log(QuatLen2), t * q.x, t * q.y, t * q.z);
		}
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> pow(qua<T, Q> const& x, T y)
	{
		//Raising to the power of 0 should yield 1
		//Needed to prevent a division by 0 error later on
		if(y > -epsilon<T>() && y < epsilon<T>())
			return qua<T, Q>(1,0,0,0);

		//To deal with non-unit quaternions
		T magnitude = sqrt(x.x * x.x + x.y * x.y + x.z * x.z + x.w *x.w);

		T Angle;
		if(abs(x.w / magnitude) > cos_one_over_two<T>())
		{
			//Scalar component is close to 1; using it to recover angle would lose precision
			//Instead, we use the non-scalar components since sin() is accurate around 0

			//Prevent a division by 0 error later on
			T VectorMagnitude = x.x * x.x + x.y * x.y + x.z * x.z;
			//Despite the compiler might say, we actually want to compare
			//VectorMagnitude to 0. here; we could use denorm_int() compiling a
			//project with unsafe maths optimizations might make the comparison
			//always false, even when VectorMagnitude is 0.
			if (VectorMagnitude < std::numeric_limits<T>::min()) {
				//Equivalent to raising a real number to a power
				return qua<T, Q>(pow(x.w, y), 0, 0, 0);
			}

			Angle = asin(sqrt(VectorMagnitude) / magnitude);
		}
		else
		{
			//Scalar component is small, shouldn't cause loss of precision
			Angle = acos(x.w / magnitude);
		}

		T NewAngle = Angle * y;
		T Div = sin(NewAngle) / sin(Angle);
		T Mag = pow(magnitude, y - static_cast<T>(1));
		return qua<T, Q>(cos(NewAngle) * magnitude * Mag, x.x * Div * Mag, x.y * Div * Mag, x.z * Div * Mag);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> sqrt(qua<T, Q> const& x)
	{
		return pow(x, static_cast<T>(0.5));
	}
}//namespace glm


