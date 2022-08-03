/// @ref gtx_associated_min_max

namespace glm{

// Min comparison between 2 variables
template<typename T, typename U>
GLM_FUNC_QUALIFIER U associatedMin(T x, U a, T y, U b)
{
	return x < y ? a : b;
}

template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMin
(
	vec<L, T, Q> const& x, vec<L, U, Q> const& a,
	vec<L, T, Q> const& y, vec<L, U, Q> const& b
)
{
	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
		Result[i] = x[i] < y[i] ? a[i] : b[i];
	return Result;
}

template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMin
(
	T x, const vec<L, U, Q>& a,
	T y, const vec<L, U, Q>& b
)
{
	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
		Result[i] = x < y ? a[i] : b[i];
	return Result;
}

template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMin
(
	vec<L, T, Q> const& x, U a,
	vec<L, T, Q> const& y, U b
)
{
	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
		Result[i] = x[i] < y[i] ? a : b;
	return Result;
}

// Min comparison between 3 variables
template<typename T, typename U>
GLM_FUNC_QUALIFIER U associatedMin
(
	T x, U a,
	T y, U b,
	T z, U c
)
{
	U Result = x < y ? (x < z ? a : c) : (y < z ? b : c);
	return Result;
}

template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMin
(
	vec<L, T, Q> const& x, vec<L, U, Q> const& a,
	vec<L, T, Q> const& y, vec<L, U, Q> const& b,
	vec<L, T, Q> const& z, vec<L, U, Q> const& c
)
{
	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
		Result[i] = x[i] < y[i] ? (x[i] < z[i] ? a[i] : c[i]) : (y[i] < z[i] ? b[i] : c[i]);
	return Result;
}

// Min comparison between 4 variables
template<typename T, typename U>
GLM_FUNC_QUALIFIER U associatedMin
(
	T x, U a,
	T y, U b,
	T z, U c,
	T w, U d
)
{
	T Test1 = min(x, y);
	T Test2 = min(z, w);
	U Result1 = x < y ? a : b;
	U Result2 = z < w ? c : d;
	U Result = Test1 < Test2 ? Result1 : Result2;
	return Result;
}

// Min comparison between 4 variables
template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMin
(
	vec<L, T, Q> const& x, vec<L, U, Q> const& a,
	vec<L, T, Q> const& y, vec<L, U, Q> const& b,
	vec<L, T, Q> const& z, vec<L, U, Q> const& c,
	vec<L, T, Q> const& w, vec<L, U, Q> const& d
)
{
	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
	{
		T Test1 = min(x[i], y[i]);
		T Test2 = min(z[i], w[i]);
		U Result1 = x[i] < y[i] ? a[i] : b[i];
		U Result2 = z[i] < w[i] ? c[i] : d[i];
		Result[i] = Test1 < Test2 ? Result1 : Result2;
	}
	return Result;
}

// Min comparison between 4 variables
template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMin
(
	T x, vec<L, U, Q> const& a,
	T y, vec<L, U, Q> const& b,
	T z, vec<L, U, Q> const& c,
	T w, vec<L, U, Q> const& d
)
{
	T Test1 = min(x, y);
	T Test2 = min(z, w);

	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
	{
		U Result1 = x < y ? a[i] : b[i];
		U Result2 = z < w ? c[i] : d[i];
		Result[i] = Test1 < Test2 ? Result1 : Result2;
	}
	return Result;
}

// Min comparison between 4 variables
template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMin
(
	vec<L, T, Q> const& x, U a,
	vec<L, T, Q> const& y, U b,
	vec<L, T, Q> const& z, U c,
	vec<L, T, Q> const& w, U d
)
{
	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
	{
		T Test1 = min(x[i], y[i]);
		T Test2 = min(z[i], w[i]);
		U Result1 = x[i] < y[i] ? a : b;
		U Result2 = z[i] < w[i] ? c : d;
		Result[i] = Test1 < Test2 ? Result1 : Result2;
	}
	return Result;
}

// Max comparison between 2 variables
template<typename T, typename U>
GLM_FUNC_QUALIFIER U associatedMax(T x, U a, T y, U b)
{
	return x > y ? a : b;
}

// Max comparison between 2 variables
template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMax
(
	vec<L, T, Q> const& x, vec<L, U, Q> const& a,
	vec<L, T, Q> const& y, vec<L, U, Q> const& b
)
{
	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
		Result[i] = x[i] > y[i] ? a[i] : b[i];
	return Result;
}

// Max comparison between 2 variables
template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, T, Q> associatedMax
(
	T x, vec<L, U, Q> const& a,
	T y, vec<L, U, Q> const& b
)
{
	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
		Result[i] = x > y ? a[i] : b[i];
	return Result;
}

// Max comparison between 2 variables
template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMax
(
	vec<L, T, Q> const& x, U a,
	vec<L, T, Q> const& y, U b
)
{
	vec<L, T, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
		Result[i] = x[i] > y[i] ? a : b;
	return Result;
}

// Max comparison between 3 variables
template<typename T, typename U>
GLM_FUNC_QUALIFIER U associatedMax
(
	T x, U a,
	T y, U b,
	T z, U c
)
{
	U Result = x > y ? (x > z ? a : c) : (y > z ? b : c);
	return Result;
}

// Max comparison between 3 variables
template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMax
(
	vec<L, T, Q> const& x, vec<L, U, Q> const& a,
	vec<L, T, Q> const& y, vec<L, U, Q> const& b,
	vec<L, T, Q> const& z, vec<L, U, Q> const& c
)
{
	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
		Result[i] = x[i] > y[i] ? (x[i] > z[i] ? a[i] : c[i]) : (y[i] > z[i] ? b[i] : c[i]);
	return Result;
}

// Max comparison between 3 variables
template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, T, Q> associatedMax
(
	T x, vec<L, U, Q> const& a,
	T y, vec<L, U, Q> const& b,
	T z, vec<L, U, Q> const& c
)
{
	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
		Result[i] = x > y ? (x > z ? a[i] : c[i]) : (y > z ? b[i] : c[i]);
	return Result;
}

// Max comparison between 3 variables
template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMax
(
	vec<L, T, Q> const& x, U a,
	vec<L, T, Q> const& y, U b,
	vec<L, T, Q> const& z, U c
)
{
	vec<L, T, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
		Result[i] = x[i] > y[i] ? (x[i] > z[i] ? a : c) : (y[i] > z[i] ? b : c);
	return Result;
}

// Max comparison between 4 variables
template<typename T, typename U>
GLM_FUNC_QUALIFIER U associatedMax
(
	T x, U a,
	T y, U b,
	T z, U c,
	T w, U d
)
{
	T Test1 = max(x, y);
	T Test2 = max(z, w);
	U Result1 = x > y ? a : b;
	U Result2 = z > w ? c : d;
	U Result = Test1 > Test2 ? Result1 : Result2;
	return Result;
}

// Max comparison between 4 variables
template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMax
(
	vec<L, T, Q> const& x, vec<L, U, Q> const& a,
	vec<L, T, Q> const& y, vec<L, U, Q> const& b,
	vec<L, T, Q> const& z, vec<L, U, Q> const& c,
	vec<L, T, Q> const& w, vec<L, U, Q> const& d
)
{
	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
	{
		T Test1 = max(x[i], y[i]);
		T Test2 = max(z[i], w[i]);
		U Result1 = x[i] > y[i] ? a[i] : b[i];
		U Result2 = z[i] > w[i] ? c[i] : d[i];
		Result[i] = Test1 > Test2 ? Result1 : Result2;
	}
	return Result;
}

// Max comparison between 4 variables
template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMax
(
	T x, vec<L, U, Q> const& a,
	T y, vec<L, U, Q> const& b,
	T z, vec<L, U, Q> const& c,
	T w, vec<L, U, Q> const& d
)
{
	T Test1 = max(x, y);
	T Test2 = max(z, w);

	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
	{
		U Result1 = x > y ? a[i] : b[i];
		U Result2 = z > w ? c[i] : d[i];
		Result[i] = Test1 > Test2 ? Result1 : Result2;
	}
	return Result;
}

// Max comparison between 4 variables
template<length_t L, typename T, typename U, qualifier Q>
GLM_FUNC_QUALIFIER vec<L, U, Q> associatedMax
(
	vec<L, T, Q> const& x, U a,
	vec<L, T, Q> const& y, U b,
	vec<L, T, Q> const& z, U c,
	vec<L, T, Q> const& w, U d
)
{
	vec<L, U, Q> Result;
	for(length_t i = 0, n = Result.length(); i < n; ++i)
	{
		T Test1 = max(x[i], y[i]);
		T Test2 = max(z[i], w[i]);
		U Result1 = x[i] > y[i] ? a : b;
		U Result2 = z[i] > w[i] ? c : d;
		Result[i] = Test1 > Test2 ? Result1 : Result2;
	}
	return Result;
}
}//namespace glm
