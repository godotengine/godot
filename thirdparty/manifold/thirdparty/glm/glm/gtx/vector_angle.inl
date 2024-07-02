/// @ref gtx_vector_angle

namespace glm
{
	template<typename genType>
	GLM_FUNC_QUALIFIER genType angle
	(
		genType const& x,
		genType const& y
	)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'angle' only accept floating-point inputs");
		return acos(clamp(dot(x, y), genType(-1), genType(1)));
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T angle(vec<L, T, Q> const& x, vec<L, T, Q> const& y)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'angle' only accept floating-point inputs");
		return acos(clamp(dot(x, y), T(-1), T(1)));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T orientedAngle(vec<2, T, Q> const& x, vec<2, T, Q> const& y)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'orientedAngle' only accept floating-point inputs");
		T const Angle(acos(clamp(dot(x, y), T(-1), T(1))));

		T const partialCross = x.x * y.y - y.x * x.y;

		if (partialCross > T(0))
			return Angle;
		else
			return -Angle;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T orientedAngle(vec<3, T, Q> const& x, vec<3, T, Q> const& y, vec<3, T, Q> const& ref)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'orientedAngle' only accept floating-point inputs");

		T const Angle(acos(clamp(dot(x, y), T(-1), T(1))));
		return mix(Angle, -Angle, dot(ref, cross(x, y)) < T(0));
	}
}//namespace glm
