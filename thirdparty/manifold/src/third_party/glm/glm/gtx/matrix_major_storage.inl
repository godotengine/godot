/// @ref gtx_matrix_major_storage

namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> rowMajor2
	(
		vec<2, T, Q> const& v1,
		vec<2, T, Q> const& v2
	)
	{
		mat<2, 2, T, Q> Result;
		Result[0][0] = v1.x;
		Result[1][0] = v1.y;
		Result[0][1] = v2.x;
		Result[1][1] = v2.y;
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> rowMajor2(
		const mat<2, 2, T, Q>& m)
	{
		mat<2, 2, T, Q> Result;
		Result[0][0] = m[0][0];
		Result[0][1] = m[1][0];
		Result[1][0] = m[0][1];
		Result[1][1] = m[1][1];
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> rowMajor3(
		const vec<3, T, Q>& v1,
		const vec<3, T, Q>& v2,
		const vec<3, T, Q>& v3)
	{
		mat<3, 3, T, Q> Result;
		Result[0][0] = v1.x;
		Result[1][0] = v1.y;
		Result[2][0] = v1.z;
		Result[0][1] = v2.x;
		Result[1][1] = v2.y;
		Result[2][1] = v2.z;
		Result[0][2] = v3.x;
		Result[1][2] = v3.y;
		Result[2][2] = v3.z;
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> rowMajor3(
		const mat<3, 3, T, Q>& m)
	{
		mat<3, 3, T, Q> Result;
		Result[0][0] = m[0][0];
		Result[0][1] = m[1][0];
		Result[0][2] = m[2][0];
		Result[1][0] = m[0][1];
		Result[1][1] = m[1][1];
		Result[1][2] = m[2][1];
		Result[2][0] = m[0][2];
		Result[2][1] = m[1][2];
		Result[2][2] = m[2][2];
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> rowMajor4(
		const vec<4, T, Q>& v1,
		const vec<4, T, Q>& v2,
		const vec<4, T, Q>& v3,
		const vec<4, T, Q>& v4)
	{
		mat<4, 4, T, Q> Result;
		Result[0][0] = v1.x;
		Result[1][0] = v1.y;
		Result[2][0] = v1.z;
		Result[3][0] = v1.w;
		Result[0][1] = v2.x;
		Result[1][1] = v2.y;
		Result[2][1] = v2.z;
		Result[3][1] = v2.w;
		Result[0][2] = v3.x;
		Result[1][2] = v3.y;
		Result[2][2] = v3.z;
		Result[3][2] = v3.w;
		Result[0][3] = v4.x;
		Result[1][3] = v4.y;
		Result[2][3] = v4.z;
		Result[3][3] = v4.w;
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> rowMajor4(
		const mat<4, 4, T, Q>& m)
	{
		mat<4, 4, T, Q> Result;
		Result[0][0] = m[0][0];
		Result[0][1] = m[1][0];
		Result[0][2] = m[2][0];
		Result[0][3] = m[3][0];
		Result[1][0] = m[0][1];
		Result[1][1] = m[1][1];
		Result[1][2] = m[2][1];
		Result[1][3] = m[3][1];
		Result[2][0] = m[0][2];
		Result[2][1] = m[1][2];
		Result[2][2] = m[2][2];
		Result[2][3] = m[3][2];
		Result[3][0] = m[0][3];
		Result[3][1] = m[1][3];
		Result[3][2] = m[2][3];
		Result[3][3] = m[3][3];
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> colMajor2(
		const vec<2, T, Q>& v1,
		const vec<2, T, Q>& v2)
	{
		return mat<2, 2, T, Q>(v1, v2);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> colMajor2(
		const mat<2, 2, T, Q>& m)
	{
		return mat<2, 2, T, Q>(m);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> colMajor3(
		const vec<3, T, Q>& v1,
		const vec<3, T, Q>& v2,
		const vec<3, T, Q>& v3)
	{
		return mat<3, 3, T, Q>(v1, v2, v3);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> colMajor3(
		const mat<3, 3, T, Q>& m)
	{
		return mat<3, 3, T, Q>(m);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> colMajor4(
		const vec<4, T, Q>& v1,
		const vec<4, T, Q>& v2,
		const vec<4, T, Q>& v3,
		const vec<4, T, Q>& v4)
	{
		return mat<4, 4, T, Q>(v1, v2, v3, v4);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> colMajor4(
		const mat<4, 4, T, Q>& m)
	{
		return mat<4, 4, T, Q>(m);
	}
}//namespace glm
