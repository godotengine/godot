/// @ref gtx_matrix_transform_2d
/// @author Miguel Ángel Pérez Martínez

#include "../trigonometric.hpp"

namespace glm
{

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> translate(
		mat<3, 3, T, Q> const& m,
		vec<2, T, Q> const& v)
	{
		mat<3, 3, T, Q> Result(m);
		Result[2] = m[0] * v[0] + m[1] * v[1] + m[2];
		return Result;
	}


	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> rotate(
		mat<3, 3, T, Q> const& m,
		T angle)
	{
		T const a = angle;
		T const c = cos(a);
		T const s = sin(a);

		mat<3, 3, T, Q> Result;
		Result[0] = m[0] * c + m[1] * s;
		Result[1] = m[0] * -s + m[1] * c;
		Result[2] = m[2];
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> scale(
		mat<3, 3, T, Q> const& m,
		vec<2, T, Q> const& v)
	{
		mat<3, 3, T, Q> Result;
		Result[0] = m[0] * v[0];
		Result[1] = m[1] * v[1];
		Result[2] = m[2];
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> shearX(
		mat<3, 3, T, Q> const& m,
		T y)
	{
		mat<3, 3, T, Q> Result(1);
		Result[0][1] = y;
		return m * Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 3, T, Q> shearY(
		mat<3, 3, T, Q> const& m,
		T x)
	{
		mat<3, 3, T, Q> Result(1);
		Result[1][0] = x;
		return m * Result;
	}

}//namespace glm
