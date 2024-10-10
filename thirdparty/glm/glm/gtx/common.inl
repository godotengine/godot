/// @ref gtx_common

#include <cmath>
#include "../gtc/epsilon.hpp"
#include "../gtc/constants.hpp"

namespace glm{
namespace detail
{
	template<length_t L, typename T, qualifier Q, bool isFloat = true>
	struct compute_fmod
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
		{
			return detail::functor2<vec, L, T, Q>::call(std::fmod, a, b);
		}
	};

	template<length_t L, typename T, qualifier Q>
	struct compute_fmod<L, T, Q, false>
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& a, vec<L, T, Q> const& b)
		{
			return a % b;
		}
	};
}//namespace detail

	template<typename T>
	GLM_FUNC_QUALIFIER bool isdenormal(T const& x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'isdenormal' only accept floating-point inputs");

#		if GLM_HAS_CXX11_STL
			return std::fpclassify(x) == FP_SUBNORMAL;
#		else
			return epsilonNotEqual(x, static_cast<T>(0), epsilon<T>()) && std::fabs(x) < std::numeric_limits<T>::min();
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename vec<1, T, Q>::bool_type isdenormal
	(
		vec<1, T, Q> const& x
	)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'isdenormal' only accept floating-point inputs");

		return typename vec<1, T, Q>::bool_type(
			isdenormal(x.x));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename vec<2, T, Q>::bool_type isdenormal
	(
		vec<2, T, Q> const& x
	)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'isdenormal' only accept floating-point inputs");

		return typename vec<2, T, Q>::bool_type(
			isdenormal(x.x),
			isdenormal(x.y));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename vec<3, T, Q>::bool_type isdenormal
	(
		vec<3, T, Q> const& x
	)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'isdenormal' only accept floating-point inputs");

		return typename vec<3, T, Q>::bool_type(
			isdenormal(x.x),
			isdenormal(x.y),
			isdenormal(x.z));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename vec<4, T, Q>::bool_type isdenormal
	(
		vec<4, T, Q> const& x
	)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'isdenormal' only accept floating-point inputs");

		return typename vec<4, T, Q>::bool_type(
			isdenormal(x.x),
			isdenormal(x.y),
			isdenormal(x.z),
			isdenormal(x.w));
	}

	// fmod
	template<typename genType>
	GLM_FUNC_QUALIFIER genType fmod(genType x, genType y)
	{
		return fmod(vec<1, genType>(x), y).x;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> fmod(vec<L, T, Q> const& x, T y)
	{
		return detail::compute_fmod<L, T, Q, std::numeric_limits<T>::is_iec559>::call(x, vec<L, T, Q>(y));
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> fmod(vec<L, T, Q> const& x, vec<L, T, Q> const& y)
	{
		return detail::compute_fmod<L, T, Q, std::numeric_limits<T>::is_iec559>::call(x, y);
	}

	template <length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, bool, Q> openBounded(vec<L, T, Q> const& Value, vec<L, T, Q> const& Min, vec<L, T, Q> const& Max)
	{
		return greaterThan(Value, Min) && lessThan(Value, Max);
	}

	template <length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, bool, Q> closeBounded(vec<L, T, Q> const& Value, vec<L, T, Q> const& Min, vec<L, T, Q> const& Max)
	{
		return greaterThanEqual(Value, Min) && lessThanEqual(Value, Max);
	}
}//namespace glm
