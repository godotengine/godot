/// @ref gtx_fast_square_root

namespace glm
{
	// fastSqrt
	template<typename genType>
	GLM_FUNC_QUALIFIER genType fastSqrt(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'fastSqrt' only accept floating-point input");

		return genType(1) / fastInverseSqrt(x);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> fastSqrt(vec<L, T, Q> const& x)
	{
		return detail::functor1<vec, L, T, T, Q>::call(fastSqrt, x);
	}

	// fastInversesqrt
	template<typename genType>
	GLM_FUNC_QUALIFIER genType fastInverseSqrt(genType x)
	{
		return detail::compute_inversesqrt<1, genType, lowp, detail::is_aligned<lowp>::value>::call(vec<1, genType, lowp>(x)).x;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> fastInverseSqrt(vec<L, T, Q> const& x)
	{
		return detail::compute_inversesqrt<L, T, Q, detail::is_aligned<Q>::value>::call(x);
	}

	// fastLength
	template<typename genType>
	GLM_FUNC_QUALIFIER genType fastLength(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559, "'fastLength' only accept floating-point inputs");

		return abs(x);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T fastLength(vec<L, T, Q> const& x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'fastLength' only accept floating-point inputs");

		return fastSqrt(dot(x, x));
	}

	// fastDistance
	template<typename genType>
	GLM_FUNC_QUALIFIER genType fastDistance(genType x, genType y)
	{
		return fastLength(y - x);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T fastDistance(vec<L, T, Q> const& x, vec<L, T, Q> const& y)
	{
		return fastLength(y - x);
	}

	// fastNormalize
	template<typename genType>
	GLM_FUNC_QUALIFIER genType fastNormalize(genType x)
	{
		return x > genType(0) ? genType(1) : -genType(1);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> fastNormalize(vec<L, T, Q> const& x)
	{
		return x * fastInverseSqrt(dot(x, x));
	}
}//namespace glm
