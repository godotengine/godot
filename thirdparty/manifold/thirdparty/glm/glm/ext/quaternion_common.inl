namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> mix(qua<T, Q> const& x, qua<T, Q> const& y, T a)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'mix' only accept floating-point inputs");

		T const cosTheta = dot(x, y);

		// Perform a linear interpolation when cosTheta is close to 1 to avoid side effect of sin(angle) becoming a zero denominator
		if(cosTheta > static_cast<T>(1) - epsilon<T>())
		{
			// Linear interpolation
			return qua<T, Q>::wxyz(
				mix(x.w, y.w, a),
				mix(x.x, y.x, a),
				mix(x.y, y.y, a),
				mix(x.z, y.z, a));
		}
		else
		{
			// Essential Mathematics, page 467
			T angle = acos(cosTheta);
			return (sin((static_cast<T>(1) - a) * angle) * x + sin(a * angle) * y) / sin(angle);
		}
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> lerp(qua<T, Q> const& x, qua<T, Q> const& y, T a)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'lerp' only accept floating-point inputs");

		// Lerp is only defined in [0, 1]
		assert(a >= static_cast<T>(0));
		assert(a <= static_cast<T>(1));

		return x * (static_cast<T>(1) - a) + (y * a);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> slerp(qua<T, Q> const& x, qua<T, Q> const& y, T a)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'slerp' only accept floating-point inputs");

		qua<T, Q> z = y;

		T cosTheta = dot(x, y);

		// If cosTheta < 0, the interpolation will take the long way around the sphere.
		// To fix this, one quat must be negated.
		if(cosTheta < static_cast<T>(0))
		{
			z = -y;
			cosTheta = -cosTheta;
		}

		// Perform a linear interpolation when cosTheta is close to 1 to avoid side effect of sin(angle) becoming a zero denominator
		if(cosTheta > static_cast<T>(1) - epsilon<T>())
		{
			// Linear interpolation
			return qua<T, Q>::wxyz(
				mix(x.w, z.w, a),
				mix(x.x, z.x, a),
				mix(x.y, z.y, a),
				mix(x.z, z.z, a));
		}
		else
		{
			// Essential Mathematics, page 467
			T angle = acos(cosTheta);
			return (sin((static_cast<T>(1) - a) * angle) * x + sin(a * angle) * z) / sin(angle);
		}
	}

    template<typename T, typename S, qualifier Q>
    GLM_FUNC_QUALIFIER qua<T, Q> slerp(qua<T, Q> const& x, qua<T, Q> const& y, T a, S k)
    {
        GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'slerp' only accept floating-point inputs");
        GLM_STATIC_ASSERT(std::numeric_limits<S>::is_integer, "'slerp' only accept integer for spin count");

        qua<T, Q> z = y;

        T cosTheta = dot(x, y);

        // If cosTheta < 0, the interpolation will take the long way around the sphere.
        // To fix this, one quat must be negated.
        if (cosTheta < static_cast<T>(0))
        {
            z = -y;
            cosTheta = -cosTheta;
        }

        // Perform a linear interpolation when cosTheta is close to 1 to avoid side effect of sin(angle) becoming a zero denominator
        if (cosTheta > static_cast<T>(1) - epsilon<T>())
        {
            // Linear interpolation
            return qua<T, Q>::wxyz(
                mix(x.w, z.w, a),
                mix(x.x, z.x, a),
                mix(x.y, z.y, a),
                mix(x.z, z.z, a));
        }
        else
        {
            // Graphics Gems III, page 96
            T angle = acos(cosTheta);
            T phi = angle + static_cast<T>(k) * glm::pi<T>();
            return (sin(angle - a * phi)* x + sin(a * phi) * z) / sin(angle);
        }
    }

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> conjugate(qua<T, Q> const& q)
	{
		return qua<T, Q>::wxyz(q.w, -q.x, -q.y, -q.z);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> inverse(qua<T, Q> const& q)
	{
		return conjugate(q) / dot(q, q);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<4, bool, Q> isnan(qua<T, Q> const& q)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'isnan' only accept floating-point inputs");

		return vec<4, bool, Q>(isnan(q.x), isnan(q.y), isnan(q.z), isnan(q.w));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<4, bool, Q> isinf(qua<T, Q> const& q)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'isinf' only accept floating-point inputs");

		return vec<4, bool, Q>(isinf(q.x), isinf(q.y), isinf(q.z), isinf(q.w));
	}
}//namespace glm

#if GLM_CONFIG_SIMD == GLM_ENABLE
#	include "quaternion_common_simd.inl"
#endif

