/// @ref gtx_gradient_paint

namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T radialGradient
	(
		vec<2, T, Q> const& Center,
		T const& Radius,
		vec<2, T, Q> const& Focal,
		vec<2, T, Q> const& Position
	)
	{
		vec<2, T, Q> F = Focal - Center;
		vec<2, T, Q> D = Position - Focal;
		T Radius2 = pow2(Radius);
		T Fx2 = pow2(F.x);
		T Fy2 = pow2(F.y);

		T Numerator = (D.x * F.x + D.y * F.y) + sqrt(Radius2 * (pow2(D.x) + pow2(D.y)) - pow2(D.x * F.y - D.y * F.x));
		T Denominator = Radius2 - (Fx2 + Fy2);
		return Numerator / Denominator;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T linearGradient
	(
		vec<2, T, Q> const& Point0,
		vec<2, T, Q> const& Point1,
		vec<2, T, Q> const& Position
	)
	{
		vec<2, T, Q> Dist = Point1 - Point0;
		return (Dist.x * (Position.x - Point0.x) + Dist.y * (Position.y - Point0.y)) / glm::dot(Dist, Dist);
	}
}//namespace glm
