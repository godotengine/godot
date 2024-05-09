/// @ref gtx_quaternion

#include <limits>
#include "../gtc/constants.hpp"

namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> quat_identity()
	{
		return qua<T, Q>::wxyz(static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> cross(vec<3, T, Q> const& v, qua<T, Q> const& q)
	{
		return inverse(q) * v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> cross(qua<T, Q> const& q, vec<3, T, Q> const& v)
	{
		return q * v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> squad
	(
		qua<T, Q> const& q1,
		qua<T, Q> const& q2,
		qua<T, Q> const& s1,
		qua<T, Q> const& s2,
		T const& h)
	{
		return mix(mix(q1, q2, h), mix(s1, s2, h), static_cast<T>(2) * (static_cast<T>(1) - h) * h);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> intermediate
	(
		qua<T, Q> const& prev,
		qua<T, Q> const& curr,
		qua<T, Q> const& next
	)
	{
		qua<T, Q> invQuat = inverse(curr);
		return exp((log(next * invQuat) + log(prev * invQuat)) / static_cast<T>(-4)) * curr;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> rotate(qua<T, Q> const& q, vec<3, T, Q> const& v)
	{
		return q * v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<4, T, Q> rotate(qua<T, Q> const& q, vec<4, T, Q> const& v)
	{
		return q * v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T extractRealComponent(qua<T, Q> const& q)
	{
		T w = static_cast<T>(1) - q.x * q.x - q.y * q.y - q.z * q.z;
		if(w < T(0))
			return T(0);
		else
			return -sqrt(w);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR T length2(qua<T, Q> const& q)
	{
		return q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> shortMix(qua<T, Q> const& x, qua<T, Q> const& y, T const& a)
	{
		if(a <= static_cast<T>(0)) return x;
		if(a >= static_cast<T>(1)) return y;

		T fCos = dot(x, y);
		qua<T, Q> y2(y); //BUG!!! qua<T> y2;
		if(fCos < static_cast<T>(0))
		{
			y2 = -y;
			fCos = -fCos;
		}

		//if(fCos > 1.0f) // problem
		T k0, k1;
		if(fCos > (static_cast<T>(1) - epsilon<T>()))
		{
			k0 = static_cast<T>(1) - a;
			k1 = static_cast<T>(0) + a; //BUG!!! 1.0f + a;
		}
		else
		{
			T fSin = sqrt(T(1) - fCos * fCos);
			T fAngle = atan(fSin, fCos);
			T fOneOverSin = static_cast<T>(1) / fSin;
			k0 = sin((static_cast<T>(1) - a) * fAngle) * fOneOverSin;
			k1 = sin((static_cast<T>(0) + a) * fAngle) * fOneOverSin;
		}

		return qua<T, Q>::wxyz(
			k0 * x.w + k1 * y2.w,
			k0 * x.x + k1 * y2.x,
			k0 * x.y + k1 * y2.y,
			k0 * x.z + k1 * y2.z);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> fastMix(qua<T, Q> const& x, qua<T, Q> const& y, T const& a)
	{
		return glm::normalize(x * (static_cast<T>(1) - a) + (y * a));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> rotation(vec<3, T, Q> const& orig, vec<3, T, Q> const& dest)
	{
		T cosTheta = dot(orig, dest);
		vec<3, T, Q> rotationAxis;

		if(cosTheta >= static_cast<T>(1) - epsilon<T>()) {
			// orig and dest point in the same direction
			return quat_identity<T,Q>();
		}

		if(cosTheta < static_cast<T>(-1) + epsilon<T>())
		{
			// special case when vectors in opposite directions :
			// there is no "ideal" rotation axis
			// So guess one; any will do as long as it's perpendicular to start
			// This implementation favors a rotation around the Up axis (Y),
			// since it's often what you want to do.
			rotationAxis = cross(vec<3, T, Q>(0, 0, 1), orig);
			if(length2(rotationAxis) < epsilon<T>()) // bad luck, they were parallel, try again!
				rotationAxis = cross(vec<3, T, Q>(1, 0, 0), orig);

			rotationAxis = normalize(rotationAxis);
			return angleAxis(pi<T>(), rotationAxis);
		}

		// Implementation from Stan Melax's Game Programming Gems 1 article
		rotationAxis = cross(orig, dest);

		T s = sqrt((T(1) + cosTheta) * static_cast<T>(2));
		T invs = static_cast<T>(1) / s;

		return qua<T, Q>::wxyz(
			s * static_cast<T>(0.5f),
			rotationAxis.x * invs,
			rotationAxis.y * invs,
			rotationAxis.z * invs);
	}
}//namespace glm
