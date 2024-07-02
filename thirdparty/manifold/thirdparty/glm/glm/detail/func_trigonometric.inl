#include "_vectorize.hpp"
#include <cmath>
#include <limits>

namespace glm
{
	// radians
	template<typename genType>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR genType radians(genType degrees)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'radians' only accept floating-point input");

		return degrees * static_cast<genType>(0.01745329251994329576923690768489);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, T, Q> radians(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(radians, v);
	}

	// degrees
	template<typename genType>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR genType degrees(genType radians)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'degrees' only accept floating-point input");

		return radians * static_cast<genType>(57.295779513082320876798154814105);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<L, T, Q> degrees(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(degrees, v);
	}

	// sin
	using ::std::sin;

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> sin(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(sin, v);
	}

	// cos
	using std::cos;

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> cos(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(cos, v);
	}

	// tan
	using std::tan;

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> tan(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(tan, v);
	}

	// asin
	using std::asin;

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> asin(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(asin, v);
	}

	// acos
	using std::acos;

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> acos(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(acos, v);
	}

	// atan
	template<typename genType>
	GLM_FUNC_QUALIFIER genType atan(genType y, genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'atan' only accept floating-point input");

		return ::std::atan2(y, x);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> atan(vec<L, T, Q> const& y, vec<L, T, Q> const& x)
	{
		return detail::functor2<vec, L, T, Q>::call(::std::atan2, y, x);
	}

	using std::atan;

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> atan(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(atan, v);
	}

	// sinh
	using std::sinh;

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> sinh(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(sinh, v);
	}

	// cosh
	using std::cosh;

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> cosh(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(cosh, v);
	}

	// tanh
	using std::tanh;

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> tanh(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(tanh, v);
	}

	// asinh
#	if GLM_HAS_CXX11_STL
		using std::asinh;
#	else
		template<typename genType>
		GLM_FUNC_QUALIFIER genType asinh(genType x)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'asinh' only accept floating-point input");

			return (x < static_cast<genType>(0) ? static_cast<genType>(-1) : (x > static_cast<genType>(0) ? static_cast<genType>(1) : static_cast<genType>(0))) * log(std::abs(x) + sqrt(static_cast<genType>(1) + x * x));
		}
#	endif

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> asinh(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(asinh, v);
	}

	// acosh
#	if GLM_HAS_CXX11_STL
		using std::acosh;
#	else
		template<typename genType>
		GLM_FUNC_QUALIFIER genType acosh(genType x)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'acosh' only accept floating-point input");

			if(x < static_cast<genType>(1))
				return static_cast<genType>(0);
			return log(x + sqrt(x * x - static_cast<genType>(1)));
		}
#	endif

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> acosh(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(acosh, v);
	}

	// atanh
#	if GLM_HAS_CXX11_STL
		using std::atanh;
#	else
		template<typename genType>
		GLM_FUNC_QUALIFIER genType atanh(genType x)
		{
			GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'atanh' only accept floating-point input");

			if(std::abs(x) >= static_cast<genType>(1))
				return 0;
			return static_cast<genType>(0.5) * log((static_cast<genType>(1) + x) / (static_cast<genType>(1) - x));
		}
#	endif

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> atanh(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(atanh, v);
	}
}//namespace glm

#if GLM_CONFIG_SIMD == GLM_ENABLE
#	include "func_trigonometric_simd.inl"
#endif

