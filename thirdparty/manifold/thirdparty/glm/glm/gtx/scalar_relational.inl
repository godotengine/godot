/// @ref gtx_scalar_relational

namespace glm
{
	template<typename T>
	GLM_FUNC_QUALIFIER bool lessThan
	(
		T const& x,
		T const& y
	)
	{
		return x < y;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER bool lessThanEqual
	(
		T const& x,
		T const& y
	)
	{
		return x <= y;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER bool greaterThan
	(
		T const& x,
		T const& y
	)
	{
		return x > y;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER bool greaterThanEqual
	(
		T const& x,
		T const& y
	)
	{
		return x >= y;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER bool equal
	(
		T const& x,
		T const& y
	)
	{
		return detail::compute_equal<T, std::numeric_limits<T>::is_iec559>::call(x, y);
	}

	template<typename T>
	GLM_FUNC_QUALIFIER bool notEqual
	(
		T const& x,
		T const& y
	)
	{
		return !detail::compute_equal<T, std::numeric_limits<T>::is_iec559>::call(x, y);
	}

	GLM_FUNC_QUALIFIER bool any
	(
		bool const& x
	)
	{
		return x;
	}

	GLM_FUNC_QUALIFIER bool all
	(
		bool const& x
	)
	{
		return x;
	}

	GLM_FUNC_QUALIFIER bool not_
	(
		bool const& x
	)
	{
		return !x;
	}
}//namespace glm
