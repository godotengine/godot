/// @ref gtx_matrix_cross_product

namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> matrixCross3
	(
		vec<3, T, Q> const& x
	)
	{
		mat<3, 3, T, Q> Result(T(0));
		Result[0][1] = x.z;
		Result[1][0] = -x.z;
		Result[0][2] = -x.y;
		Result[2][0] = x.y;
		Result[1][2] = x.x;
		Result[2][1] = -x.x;
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> matrixCross4
	(
		vec<3, T, Q> const& x
	)
	{
		mat<4, 4, T, Q> Result(T(0));
		Result[0][1] = x.z;
		Result[1][0] = -x.z;
		Result[0][2] = -x.y;
		Result[2][0] = x.y;
		Result[1][2] = x.x;
		Result[2][1] = -x.x;
		return Result;
	}

}//namespace glm
