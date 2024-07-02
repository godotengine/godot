/// @ref gtx_handed_coordinate_space

namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool rightHanded
	(
		vec<3, T, Q> const& tangent,
		vec<3, T, Q> const& binormal,
		vec<3, T, Q> const& normal
	)
	{
		return dot(cross(normal, tangent), binormal) > T(0);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool leftHanded
	(
		vec<3, T, Q> const& tangent,
		vec<3, T, Q> const& binormal,
		vec<3, T, Q> const& normal
	)
	{
		return dot(cross(normal, tangent), binormal) < T(0);
	}
}//namespace glm
