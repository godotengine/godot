namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR T dot(qua<T, Q> const& x, qua<T, Q> const& y)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'dot' accepts only floating-point inputs");
		return detail::compute_dot<qua<T, Q>, T, detail::is_aligned<Q>::value>::call(x, y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T length(qua<T, Q> const& q)
	{
		return glm::sqrt(dot(q, q));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q> normalize(qua<T, Q> const& q)
	{
		T len = length(q);
		if(len <= static_cast<T>(0)) // Problem
			return qua<T, Q>::wxyz(static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
		T oneOverLen = static_cast<T>(1) / len;
		return qua<T, Q>::wxyz(q.w * oneOverLen, q.x * oneOverLen, q.y * oneOverLen, q.z * oneOverLen);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> cross(qua<T, Q> const& q1, qua<T, Q> const& q2)
	{
		return qua<T, Q>::wxyz(
			q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
			q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
			q1.w * q2.y + q1.y * q2.w + q1.z * q2.x - q1.x * q2.z,
			q1.w * q2.z + q1.z * q2.w + q1.x * q2.y - q1.y * q2.x);
	}
}//namespace glm

