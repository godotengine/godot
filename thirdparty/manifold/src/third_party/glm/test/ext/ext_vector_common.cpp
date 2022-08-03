#include <glm/ext/vector_common.hpp>

#include <glm/ext/vector_bool1.hpp>
#include <glm/ext/vector_bool1_precision.hpp>
#include <glm/ext/vector_bool2.hpp>
#include <glm/ext/vector_bool2_precision.hpp>
#include <glm/ext/vector_bool3.hpp>
#include <glm/ext/vector_bool3_precision.hpp>
#include <glm/ext/vector_bool4.hpp>
#include <glm/ext/vector_bool4_precision.hpp>

#include <glm/ext/vector_float1.hpp>
#include <glm/ext/vector_float1_precision.hpp>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float2_precision.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float3_precision.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_float4_precision.hpp>
#include <glm/ext/vector_double1.hpp>
#include <glm/ext/vector_double1_precision.hpp>
#include <glm/ext/vector_double2.hpp>
#include <glm/ext/vector_double2_precision.hpp>
#include <glm/ext/vector_double3.hpp>
#include <glm/ext/vector_double3_precision.hpp>
#include <glm/ext/vector_double4.hpp>
#include <glm/ext/vector_double4_precision.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/vector_relational.hpp>
#include <glm/common.hpp>

#if ((GLM_LANG & GLM_LANG_CXX11_FLAG) || (GLM_COMPILER & GLM_COMPILER_VC))
#	define GLM_NAN(T) NAN
#else
#	define GLM_NAN(T) (static_cast<T>(0.0f) / static_cast<T>(0.0f))
#endif

template <typename vecType>
static int test_min()
{
	typedef typename vecType::value_type T;

	int Error = 0;

	vecType const N(static_cast<T>(0));
	vecType const B(static_cast<T>(1));

	Error += glm::all(glm::equal(glm::min(N, B), N, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::min(B, N), N, glm::epsilon<T>())) ? 0 : 1;

	vecType const C(static_cast<T>(2));
	Error += glm::all(glm::equal(glm::min(N, B, C), N, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::min(B, N, C), N, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::min(C, N, B), N, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::min(C, B, N), N, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::min(B, C, N), N, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::min(N, C, B), N, glm::epsilon<T>())) ? 0 : 1;

	vecType const D(static_cast<T>(3));
	Error += glm::all(glm::equal(glm::min(D, N, B, C), N, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::min(B, D, N, C), N, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::min(C, N, D, B), N, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::min(C, B, D, N), N, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::min(B, C, N, D), N, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::min(N, C, B, D), N, glm::epsilon<T>())) ? 0 : 1;

	return Error;
}

template <typename vecType>
static int test_min_nan()
{
	typedef typename vecType::value_type T;

	int Error = 0;

	vecType const B(static_cast<T>(1));
	vecType const N(GLM_NAN(T));

	Error += glm::all(glm::isnan(glm::min(N, B))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::min(B, N))) ? 0 : 1;

	vecType const C(static_cast<T>(2));
	Error += glm::all(glm::isnan(glm::min(N, B, C))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::min(B, N, C))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::min(C, N, B))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::min(C, B, N))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::min(B, C, N))) ? 0 : 1;
	Error += glm::all(glm::isnan(glm::min(N, C, B))) ? 0 : 1;

	vecType const D(static_cast<T>(3));
	Error += !glm::all(glm::isnan(glm::min(D, N, B, C))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::min(B, D, N, C))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::min(C, N, D, B))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::min(C, B, D, N))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::min(B, C, N, D))) ? 0 : 1;
	Error += glm::all(glm::isnan(glm::min(N, C, B, D))) ? 0 : 1;

	return Error;
}

template <typename vecType>
static int test_max()
{
	typedef typename vecType::value_type T;

	int Error = 0;

	vecType const N(static_cast<T>(0));
	vecType const B(static_cast<T>(1));
	Error += glm::all(glm::equal(glm::max(N, B), B, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::max(B, N), B, glm::epsilon<T>())) ? 0 : 1;

	vecType const C(static_cast<T>(2));
	Error += glm::all(glm::equal(glm::max(N, B, C), C, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::max(B, N, C), C, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::max(C, N, B), C, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::max(C, B, N), C, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::max(B, C, N), C, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::max(N, C, B), C, glm::epsilon<T>())) ? 0 : 1;

	vecType const D(static_cast<T>(3));
	Error += glm::all(glm::equal(glm::max(D, N, B, C), D, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::max(B, D, N, C), D, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::max(C, N, D, B), D, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::max(C, B, D, N), D, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::max(B, C, N, D), D, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::max(N, C, B, D), D, glm::epsilon<T>())) ? 0 : 1;

	return Error;
}

template <typename vecType>
static int test_max_nan()
{
	typedef typename vecType::value_type T;

	int Error = 0;

	vecType const B(static_cast<T>(1));
	vecType const N(GLM_NAN(T));

	Error += glm::all(glm::isnan(glm::max(N, B))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::max(B, N))) ? 0 : 1;

	vecType const C(static_cast<T>(2));
	Error += glm::all(glm::isnan(glm::max(N, B, C))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::max(B, N, C))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::max(C, N, B))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::max(C, B, N))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::max(B, C, N))) ? 0 : 1;
	Error += glm::all(glm::isnan(glm::max(N, C, B))) ? 0 : 1;

	vecType const D(static_cast<T>(3));
	Error += !glm::all(glm::isnan(glm::max(D, N, B, C))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::max(B, D, N, C))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::max(C, N, D, B))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::max(C, B, D, N))) ? 0 : 1;
	Error += !glm::all(glm::isnan(glm::max(B, C, N, D))) ? 0 : 1;
	Error += glm::all(glm::isnan(glm::max(N, C, B, D))) ? 0 : 1;

	return Error;
}

template <typename vecType>
static int test_fmin()
{
	typedef typename vecType::value_type T;

	int Error = 0;

	vecType const B(static_cast<T>(1));
	vecType const N(GLM_NAN(T));

	Error += glm::all(glm::equal(glm::fmin(N, B), B, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmin(B, N), B, glm::epsilon<T>())) ? 0 : 1;

	vecType const C(static_cast<T>(2));
	Error += glm::all(glm::equal(glm::fmin(N, B, C), B, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmin(B, N, C), B, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmin(C, N, B), B, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmin(C, B, N), B, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmin(B, C, N), B, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmin(N, C, B), B, glm::epsilon<T>())) ? 0 : 1;

	vecType const D(static_cast<T>(3));
	Error += glm::all(glm::equal(glm::fmin(D, N, B, C), B, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmin(B, D, N, C), B, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmin(C, N, D, B), B, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmin(C, B, D, N), B, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmin(B, C, N, D), B, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmin(N, C, B, D), B, glm::epsilon<T>())) ? 0 : 1;

	return Error;
}

template <typename vecType>
static int test_fmax()
{
	typedef typename vecType::value_type T;

	int Error = 0;

	vecType const B(static_cast<T>(1));
	vecType const N(GLM_NAN(T));

	Error += glm::all(glm::equal(glm::fmax(N, B), B, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmax(B, N), B, glm::epsilon<T>())) ? 0 : 1;

	vecType const C(static_cast<T>(2));
	Error += glm::all(glm::equal(glm::fmax(N, B, C), C, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmax(B, N, C), C, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmax(C, N, B), C, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmax(C, B, N), C, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmax(B, C, N), C, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmax(N, C, B), C, glm::epsilon<T>())) ? 0 : 1;

	vecType const D(static_cast<T>(3));
	Error += glm::all(glm::equal(glm::fmax(D, N, B, C), D, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmax(B, D, N, C), D, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmax(C, N, D, B), D, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmax(C, B, D, N), D, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmax(B, C, N, D), D, glm::epsilon<T>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::fmax(N, C, B, D), D, glm::epsilon<T>())) ? 0 : 1;

	return Error;
}

static int test_clamp()
{
	int Error = 0;

	glm::vec2 K = glm::clamp(glm::vec2(0.5f));
	Error += glm::all(glm::equal(K, glm::vec2(0.5f), glm::vec2(0.00001f))) ? 0 : 1;

	glm::vec3 L = glm::clamp(glm::vec3(0.5f));
	Error += glm::all(glm::equal(L, glm::vec3(0.5f), glm::vec3(0.00001f))) ? 0 : 1;

	glm::vec4 M = glm::clamp(glm::vec4(0.5f));
	Error += glm::all(glm::equal(M, glm::vec4(0.5f), glm::vec4(0.00001f))) ? 0 : 1;

	glm::vec1 N = glm::clamp(glm::vec1(0.5f));
	Error += glm::all(glm::equal(N, glm::vec1(0.5f), glm::vec1(0.00001f))) ? 0 : 1;

	return Error;
}

static int test_repeat()
{
	int Error = 0;

	glm::vec2 K = glm::repeat(glm::vec2(0.5f));
	Error += glm::all(glm::equal(K, glm::vec2(0.5f), glm::vec2(0.00001f))) ? 0 : 1;

	glm::vec3 L = glm::repeat(glm::vec3(0.5f));
	Error += glm::all(glm::equal(L, glm::vec3(0.5f), glm::vec3(0.00001f))) ? 0 : 1;

	glm::vec4 M = glm::repeat(glm::vec4(0.5f));
	Error += glm::all(glm::equal(M, glm::vec4(0.5f), glm::vec4(0.00001f))) ? 0 : 1;

	glm::vec1 N = glm::repeat(glm::vec1(0.5f));
	Error += glm::all(glm::equal(N, glm::vec1(0.5f), glm::vec1(0.00001f))) ? 0 : 1;

	return Error;
}

static int test_mirrorClamp()
{
	int Error = 0;

	glm::vec2 K = glm::mirrorClamp(glm::vec2(0.5f));
	Error += glm::all(glm::equal(K, glm::vec2(0.5f), glm::vec2(0.00001f))) ? 0 : 1;

	glm::vec3 L = glm::mirrorClamp(glm::vec3(0.5f));
	Error += glm::all(glm::equal(L, glm::vec3(0.5f), glm::vec3(0.00001f))) ? 0 : 1;

	glm::vec4 M = glm::mirrorClamp(glm::vec4(0.5f));
	Error += glm::all(glm::equal(M, glm::vec4(0.5f), glm::vec4(0.00001f))) ? 0 : 1;

	glm::vec1 N = glm::mirrorClamp(glm::vec1(0.5f));
	Error += glm::all(glm::equal(N, glm::vec1(0.5f), glm::vec1(0.00001f))) ? 0 : 1;

	return Error;
}

static int test_mirrorRepeat()
{
	int Error = 0;

	glm::vec2 K = glm::mirrorRepeat(glm::vec2(0.5f));
	Error += glm::all(glm::equal(K, glm::vec2(0.5f), glm::vec2(0.00001f))) ? 0 : 1;

	glm::vec3 L = glm::mirrorRepeat(glm::vec3(0.5f));
	Error += glm::all(glm::equal(L, glm::vec3(0.5f), glm::vec3(0.00001f))) ? 0 : 1;

	glm::vec4 M = glm::mirrorRepeat(glm::vec4(0.5f));
	Error += glm::all(glm::equal(M, glm::vec4(0.5f), glm::vec4(0.00001f))) ? 0 : 1;

	glm::vec1 N = glm::mirrorRepeat(glm::vec1(0.5f));
	Error += glm::all(glm::equal(N, glm::vec1(0.5f), glm::vec1(0.00001f))) ? 0 : 1;

	return Error;
}

static int test_iround()
{
	int Error = 0;

	for(float f = 0.0f; f < 3.1f; f += 0.05f)
	{
		int RoundFast = static_cast<int>(glm::iround(f));
		int RoundSTD = static_cast<int>(glm::round(f));
		Error += RoundFast == RoundSTD ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

static int test_uround()
{
	int Error = 0;

	for(float f = 0.0f; f < 3.1f; f += 0.05f)
	{
		int RoundFast = static_cast<int>(glm::uround(f));
		int RoundSTD = static_cast<int>(glm::round(f));
		Error += RoundFast == RoundSTD ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_min<glm::vec3>();
	Error += test_min<glm::vec2>();
	Error += test_min_nan<glm::vec3>();
	Error += test_min_nan<glm::vec2>();

	Error += test_max<glm::vec3>();
	Error += test_max<glm::vec2>();
	Error += test_max_nan<glm::vec3>();
	Error += test_max_nan<glm::vec2>();

	Error += test_fmin<glm::vec3>();
	Error += test_fmin<glm::vec2>();

	Error += test_fmax<glm::vec3>();
	Error += test_fmax<glm::vec2>();

	Error += test_clamp();
	Error += test_repeat();
	Error += test_mirrorClamp();
	Error += test_mirrorRepeat();

	Error += test_iround();
	Error += test_uround();

	return Error;
}
