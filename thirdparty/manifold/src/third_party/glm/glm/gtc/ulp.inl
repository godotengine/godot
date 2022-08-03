/// @ref gtc_ulp

#include "../ext/scalar_ulp.hpp"

namespace glm
{
	template<>
	GLM_FUNC_QUALIFIER float next_float(float x)
	{
#		if GLM_HAS_CXX11_STL
		return std::nextafter(x, std::numeric_limits<float>::max());
#		elif((GLM_COMPILER & GLM_COMPILER_VC) || ((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_PLATFORM & GLM_PLATFORM_WINDOWS)))
		return detail::nextafterf(x, FLT_MAX);
#		elif(GLM_PLATFORM & GLM_PLATFORM_ANDROID)
		return __builtin_nextafterf(x, FLT_MAX);
#		else
		return nextafterf(x, FLT_MAX);
#		endif
	}

	template<>
	GLM_FUNC_QUALIFIER double next_float(double x)
	{
#		if GLM_HAS_CXX11_STL
		return std::nextafter(x, std::numeric_limits<double>::max());
#		elif((GLM_COMPILER & GLM_COMPILER_VC) || ((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_PLATFORM & GLM_PLATFORM_WINDOWS)))
		return detail::nextafter(x, std::numeric_limits<double>::max());
#		elif(GLM_PLATFORM & GLM_PLATFORM_ANDROID)
		return __builtin_nextafter(x, DBL_MAX);
#		else
		return nextafter(x, DBL_MAX);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER T next_float(T x, int ULPs)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'next_float' only accept floating-point input");
		assert(ULPs >= 0);

		T temp = x;
		for (int i = 0; i < ULPs; ++i)
			temp = next_float(temp);
		return temp;
	}

	GLM_FUNC_QUALIFIER float prev_float(float x)
	{
#		if GLM_HAS_CXX11_STL
		return std::nextafter(x, std::numeric_limits<float>::min());
#		elif((GLM_COMPILER & GLM_COMPILER_VC) || ((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_PLATFORM & GLM_PLATFORM_WINDOWS)))
		return detail::nextafterf(x, FLT_MIN);
#		elif(GLM_PLATFORM & GLM_PLATFORM_ANDROID)
		return __builtin_nextafterf(x, FLT_MIN);
#		else
		return nextafterf(x, FLT_MIN);
#		endif
	}

	GLM_FUNC_QUALIFIER double prev_float(double x)
	{
#		if GLM_HAS_CXX11_STL
		return std::nextafter(x, std::numeric_limits<double>::min());
#		elif((GLM_COMPILER & GLM_COMPILER_VC) || ((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_PLATFORM & GLM_PLATFORM_WINDOWS)))
		return _nextafter(x, DBL_MIN);
#		elif(GLM_PLATFORM & GLM_PLATFORM_ANDROID)
		return __builtin_nextafter(x, DBL_MIN);
#		else
		return nextafter(x, DBL_MIN);
#		endif
	}

	template<typename T>
	GLM_FUNC_QUALIFIER T prev_float(T x, int ULPs)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_iec559, "'prev_float' only accept floating-point input");
		assert(ULPs >= 0);

		T temp = x;
		for (int i = 0; i < ULPs; ++i)
			temp = prev_float(temp);
		return temp;
	}

	GLM_FUNC_QUALIFIER int float_distance(float x, float y)
	{
		detail::float_t<float> const a(x);
		detail::float_t<float> const b(y);

		return abs(a.i - b.i);
	}

	GLM_FUNC_QUALIFIER int64 float_distance(double x, double y)
	{
		detail::float_t<double> const a(x);
		detail::float_t<double> const b(y);

		return abs(a.i - b.i);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> next_float(vec<L, T, Q> const& x)
	{
		vec<L, T, Q> Result;
		for (length_t i = 0, n = Result.length(); i < n; ++i)
			Result[i] = next_float(x[i]);
		return Result;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> next_float(vec<L, T, Q> const& x, int ULPs)
	{
		vec<L, T, Q> Result;
		for (length_t i = 0, n = Result.length(); i < n; ++i)
			Result[i] = next_float(x[i], ULPs);
		return Result;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> next_float(vec<L, T, Q> const& x, vec<L, int, Q> const& ULPs)
	{
		vec<L, T, Q> Result;
		for (length_t i = 0, n = Result.length(); i < n; ++i)
			Result[i] = next_float(x[i], ULPs[i]);
		return Result;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> prev_float(vec<L, T, Q> const& x)
	{
		vec<L, T, Q> Result;
		for (length_t i = 0, n = Result.length(); i < n; ++i)
			Result[i] = prev_float(x[i]);
		return Result;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> prev_float(vec<L, T, Q> const& x, int ULPs)
	{
		vec<L, T, Q> Result;
		for (length_t i = 0, n = Result.length(); i < n; ++i)
			Result[i] = prev_float(x[i], ULPs);
		return Result;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> prev_float(vec<L, T, Q> const& x, vec<L, int, Q> const& ULPs)
	{
		vec<L, T, Q> Result;
		for (length_t i = 0, n = Result.length(); i < n; ++i)
			Result[i] = prev_float(x[i], ULPs[i]);
		return Result;
	}

	template<length_t L, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, int, Q> float_distance(vec<L, float, Q> const& x, vec<L, float, Q> const& y)
	{
		vec<L, int, Q> Result;
		for (length_t i = 0, n = Result.length(); i < n; ++i)
			Result[i] = float_distance(x[i], y[i]);
		return Result;
	}

	template<length_t L, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, int64, Q> float_distance(vec<L, double, Q> const& x, vec<L, double, Q> const& y)
	{
		vec<L, int64, Q> Result;
		for (length_t i = 0, n = Result.length(); i < n; ++i)
			Result[i] = float_distance(x[i], y[i]);
		return Result;
	}
}//namespace glm

