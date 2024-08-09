/// @ref gtc_matrix_access

namespace glm
{
	template<typename genType>
	GLM_FUNC_QUALIFIER genType row
	(
		genType const& m,
		length_t index,
		typename genType::row_type const& x
	)
	{
		assert(index >= 0 && index < m[0].length());

		genType Result = m;
		for(length_t i = 0; i < m.length(); ++i)
			Result[i][index] = x[i];
		return Result;
	}

	template<typename genType>
	GLM_FUNC_QUALIFIER typename genType::row_type row
	(
		genType const& m,
		length_t index
	)
	{
		assert(index >= 0 && index < m[0].length());

		typename genType::row_type Result(0);
		for(length_t i = 0; i < m.length(); ++i)
			Result[i] = m[i][index];
		return Result;
	}

	template<typename genType>
	GLM_FUNC_QUALIFIER genType column
	(
		genType const& m,
		length_t index,
		typename genType::col_type const& x
	)
	{
		assert(index >= 0 && index < m.length());

		genType Result = m;
		Result[index] = x;
		return Result;
	}

	template<typename genType>
	GLM_FUNC_QUALIFIER typename genType::col_type column
	(
		genType const& m,
		length_t index
	)
	{
		assert(index >= 0 && index < m.length());

		return m[index];
	}
}//namespace glm
