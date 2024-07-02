/// @ref gtx_transform2

namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> shearX2D(mat<3, 3, T, Q> const& m, T s)
	{
		mat<3, 3, T, Q> r(1);
		r[1][0] = s;
		return m * r;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> shearY2D(mat<3, 3, T, Q> const& m, T s)
	{
		mat<3, 3, T, Q> r(1);
		r[0][1] = s;
		return m * r;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> shearX3D(mat<4, 4, T, Q> const& m, T s, T t)
	{
		mat<4, 4, T, Q> r(1);
		r[0][1] = s;
		r[0][2] = t;
		return m * r;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> shearY3D(mat<4, 4, T, Q> const& m, T s, T t)
	{
		mat<4, 4, T, Q> r(1);
		r[1][0] = s;
		r[1][2] = t;
		return m * r;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> shearZ3D(mat<4, 4, T, Q> const& m, T s, T t)
	{
		mat<4, 4, T, Q> r(1);
		r[2][0] = s;
		r[2][1] = t;
		return m * r;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> reflect2D(mat<3, 3, T, Q> const& m, vec<3, T, Q> const& normal)
	{
		mat<3, 3, T, Q> r(static_cast<T>(1));
		r[0][0] = static_cast<T>(1) - static_cast<T>(2) * normal.x * normal.x;
		r[0][1] = -static_cast<T>(2) * normal.x * normal.y;
		r[1][0] = -static_cast<T>(2) * normal.x * normal.y;
		r[1][1] = static_cast<T>(1) - static_cast<T>(2) * normal.y * normal.y;
		return m * r;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> reflect3D(mat<4, 4, T, Q> const& m, vec<3, T, Q> const& normal)
	{
		mat<4, 4, T, Q> r(static_cast<T>(1));
		r[0][0] = static_cast<T>(1) - static_cast<T>(2) * normal.x * normal.x;
		r[0][1] = -static_cast<T>(2) * normal.x * normal.y;
		r[0][2] = -static_cast<T>(2) * normal.x * normal.z;

		r[1][0] = -static_cast<T>(2) * normal.x * normal.y;
		r[1][1] = static_cast<T>(1) - static_cast<T>(2) * normal.y * normal.y;
		r[1][2] = -static_cast<T>(2) * normal.y * normal.z;

		r[2][0] = -static_cast<T>(2) * normal.x * normal.z;
		r[2][1] = -static_cast<T>(2) * normal.y * normal.z;
		r[2][2] = static_cast<T>(1) - static_cast<T>(2) * normal.z * normal.z;
		return m * r;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> proj2D(
		const mat<3, 3, T, Q>& m,
		const vec<3, T, Q>& normal)
	{
		mat<3, 3, T, Q> r(static_cast<T>(1));
		r[0][0] = static_cast<T>(1) - normal.x * normal.x;
		r[0][1] = - normal.x * normal.y;
		r[1][0] = - normal.x * normal.y;
		r[1][1] = static_cast<T>(1) - normal.y * normal.y;
		return m * r;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> proj3D(
		const mat<4, 4, T, Q>& m,
		const vec<3, T, Q>& normal)
	{
		mat<4, 4, T, Q> r(static_cast<T>(1));
		r[0][0] = static_cast<T>(1) - normal.x * normal.x;
		r[0][1] = - normal.x * normal.y;
		r[0][2] = - normal.x * normal.z;
		r[1][0] = - normal.x * normal.y;
		r[1][1] = static_cast<T>(1) - normal.y * normal.y;
		r[1][2] = - normal.y * normal.z;
		r[2][0] = - normal.x * normal.z;
		r[2][1] = - normal.y * normal.z;
		r[2][2] = static_cast<T>(1) - normal.z * normal.z;
		return m * r;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> scaleBias(T scale, T bias)
	{
		mat<4, 4, T, Q> result;
		result[3] = vec<4, T, Q>(vec<3, T, Q>(bias), static_cast<T>(1));
		result[0][0] = scale;
		result[1][1] = scale;
		result[2][2] = scale;
		return result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> scaleBias(mat<4, 4, T, Q> const& m, T scale, T bias)
	{
		return m * scaleBias<T, Q>(scale, bias);
	}
}//namespace glm

