/// @ref gtx_closest_point

namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> closestPointOnLine
	(
		vec<3, T, Q> const& point,
		vec<3, T, Q> const& a,
		vec<3, T, Q> const& b
	)
	{
		T LineLength = distance(a, b);
		vec<3, T, Q> Vector = point - a;
		vec<3, T, Q> LineDirection = (b - a) / LineLength;

		// Project Vector to LineDirection to get the distance of point from a
		T Distance = dot(Vector, LineDirection);

		if(Distance <= T(0)) return a;
		if(Distance >= LineLength) return b;
		return a + LineDirection * Distance;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<2, T, Q> closestPointOnLine
	(
		vec<2, T, Q> const& point,
		vec<2, T, Q> const& a,
		vec<2, T, Q> const& b
	)
	{
		T LineLength = distance(a, b);
		vec<2, T, Q> Vector = point - a;
		vec<2, T, Q> LineDirection = (b - a) / LineLength;

		// Project Vector to LineDirection to get the distance of point from a
		T Distance = dot(Vector, LineDirection);

		if(Distance <= T(0)) return a;
		if(Distance >= LineLength) return b;
		return a + LineDirection * Distance;
	}

}//namespace glm
