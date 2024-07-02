/// @ref gtx_spline

namespace glm
{
	template<typename genType>
	GLM_FUNC_QUALIFIER genType catmullRom
	(
		genType const& v1,
		genType const& v2,
		genType const& v3,
		genType const& v4,
		typename genType::value_type const& s
	)
	{
		typename genType::value_type s2 = pow2(s);
		typename genType::value_type s3 = pow3(s);

		typename genType::value_type f1 = -s3 + typename genType::value_type(2) * s2 - s;
		typename genType::value_type f2 = typename genType::value_type(3) * s3 - typename genType::value_type(5) * s2 + typename genType::value_type(2);
		typename genType::value_type f3 = typename genType::value_type(-3) * s3 + typename genType::value_type(4) * s2 + s;
		typename genType::value_type f4 = s3 - s2;

		return (f1 * v1 + f2 * v2 + f3 * v3 + f4 * v4) / typename genType::value_type(2);

	}

	template<typename genType>
	GLM_FUNC_QUALIFIER genType hermite
	(
		genType const& v1,
		genType const& t1,
		genType const& v2,
		genType const& t2,
		typename genType::value_type const& s
	)
	{
		typename genType::value_type s2 = pow2(s);
		typename genType::value_type s3 = pow3(s);

		typename genType::value_type f1 = typename genType::value_type(2) * s3 - typename genType::value_type(3) * s2 + typename genType::value_type(1);
		typename genType::value_type f2 = typename genType::value_type(-2) * s3 + typename genType::value_type(3) * s2;
		typename genType::value_type f3 = s3 - typename genType::value_type(2) * s2 + s;
		typename genType::value_type f4 = s3 - s2;

		return f1 * v1 + f2 * v2 + f3 * t1 + f4 * t2;
	}

	template<typename genType>
	GLM_FUNC_QUALIFIER genType cubic
	(
		genType const& v1,
		genType const& v2,
		genType const& v3,
		genType const& v4,
		typename genType::value_type const& s
	)
	{
		return ((v1 * s + v2) * s + v3) * s + v4;
	}
}//namespace glm
