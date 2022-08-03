#include <glm/ext/vector_relational.hpp>
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
#include <glm/ext/vector_ulp.hpp>

template <typename vecType>
static int test_equal()
{
	typedef typename vecType::value_type valType;

	valType const A = static_cast<valType>(1.01f);
	valType const B = static_cast<valType>(1.02f);
	valType const Epsilon1 = static_cast<valType>(0.1f);
	valType const Epsilon2 = static_cast<valType>(0.001f);

	int Error = 0;

	Error += glm::all(glm::equal(vecType(A), vecType(B), Epsilon1)) ? 0 : 1;
	Error += glm::all(glm::equal(vecType(A), vecType(B), vecType(Epsilon1))) ? 0 : 1;

	Error += !glm::any(glm::equal(vecType(A), vecType(B), Epsilon2)) ? 0 : 1;
	Error += !glm::any(glm::equal(vecType(A), vecType(B), vecType(Epsilon2))) ? 0 : 1;

	return Error;
}

template <typename vecType>
static int test_notEqual()
{
	typedef typename vecType::value_type valType;

	valType const A = static_cast<valType>(1.01f);
	valType const B = static_cast<valType>(1.02f);
	valType const Epsilon1 = static_cast<valType>(0.1f);
	valType const Epsilon2 = static_cast<valType>(0.001f);

	int Error = 0;

	Error += glm::all(glm::notEqual(vecType(A), vecType(B), Epsilon2)) ? 0 : 1;
	Error += glm::all(glm::notEqual(vecType(A), vecType(B), vecType(Epsilon2))) ? 0 : 1;

	Error += !glm::any(glm::notEqual(vecType(A), vecType(B), Epsilon1)) ? 0 : 1;
	Error += !glm::any(glm::notEqual(vecType(A), vecType(B), vecType(Epsilon1))) ? 0 : 1;

	return Error;
}

template <typename genType, typename valType>
static int test_constexpr()
{
#	if GLM_CONFIG_CONSTEXP == GLM_ENABLE
		static_assert(glm::all(glm::equal(genType(static_cast<valType>(1.01f)), genType(static_cast<valType>(1.02f)), static_cast<valType>(0.1f))), "GLM: Failed constexpr");
#	endif

	return 0;
}

template <typename T>
static int test_equal_ulps()
{
	typedef glm::vec<4, T, glm::defaultp> vec4;

	T const One(1);
	vec4 const Ones(1);

	int Error = 0;

	T const ULP1Plus = glm::nextFloat(One);
	Error += glm::all(glm::equal(Ones, vec4(ULP1Plus), 1)) ? 0 : 1;

	T const ULP2Plus = glm::nextFloat(ULP1Plus);
	Error += !glm::all(glm::equal(Ones, vec4(ULP2Plus), 1)) ? 0 : 1;

	T const ULP1Minus = glm::prevFloat(One);
	Error += glm::all(glm::equal(Ones, vec4(ULP1Minus), 1)) ? 0 : 1;

	T const ULP2Minus = glm::prevFloat(ULP1Minus);
	Error += !glm::all(glm::equal(Ones, vec4(ULP2Minus), 1)) ? 0 : 1;

	return Error;
}

template <typename T>
static int test_notEqual_ulps()
{
	typedef glm::vec<4, T, glm::defaultp> vec4;

	T const One(1);
	vec4 const Ones(1);

	int Error = 0;

	T const ULP1Plus = glm::nextFloat(One);
	Error += !glm::all(glm::notEqual(Ones, vec4(ULP1Plus), 1)) ? 0 : 1;

	T const ULP2Plus = glm::nextFloat(ULP1Plus);
	Error += glm::all(glm::notEqual(Ones, vec4(ULP2Plus), 1)) ? 0 : 1;

	T const ULP1Minus = glm::prevFloat(One);
	Error += !glm::all(glm::notEqual(Ones, vec4(ULP1Minus), 1)) ? 0 : 1;

	T const ULP2Minus = glm::prevFloat(ULP1Minus);
	Error += glm::all(glm::notEqual(Ones, vec4(ULP2Minus), 1)) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_equal_ulps<float>();
	Error += test_equal_ulps<double>();
	Error += test_notEqual_ulps<float>();
	Error += test_notEqual_ulps<double>();

	Error += test_equal<glm::vec1>();
	Error += test_equal<glm::lowp_vec1>();
	Error += test_equal<glm::mediump_vec1>();
	Error += test_equal<glm::highp_vec1>();
	Error += test_equal<glm::vec2>();
	Error += test_equal<glm::lowp_vec2>();
	Error += test_equal<glm::mediump_vec2>();
	Error += test_equal<glm::highp_vec2>();
	Error += test_equal<glm::vec3>();
	Error += test_equal<glm::lowp_vec3>();
	Error += test_equal<glm::mediump_vec3>();
	Error += test_equal<glm::highp_vec3>();
	Error += test_equal<glm::vec4>();
	Error += test_equal<glm::lowp_vec4>();
	Error += test_equal<glm::mediump_vec4>();
	Error += test_equal<glm::highp_vec4>();

	Error += test_equal<glm::dvec1>();
	Error += test_equal<glm::lowp_dvec1>();
	Error += test_equal<glm::mediump_dvec1>();
	Error += test_equal<glm::highp_dvec1>();
	Error += test_equal<glm::dvec2>();
	Error += test_equal<glm::lowp_dvec2>();
	Error += test_equal<glm::mediump_dvec2>();
	Error += test_equal<glm::highp_dvec2>();
	Error += test_equal<glm::dvec3>();
	Error += test_equal<glm::lowp_dvec3>();
	Error += test_equal<glm::mediump_dvec3>();
	Error += test_equal<glm::highp_dvec3>();
	Error += test_equal<glm::dvec4>();
	Error += test_equal<glm::lowp_dvec4>();
	Error += test_equal<glm::mediump_dvec4>();
	Error += test_equal<glm::highp_dvec4>();

	Error += test_notEqual<glm::vec1>();
	Error += test_notEqual<glm::lowp_vec1>();
	Error += test_notEqual<glm::mediump_vec1>();
	Error += test_notEqual<glm::highp_vec1>();
	Error += test_notEqual<glm::vec2>();
	Error += test_notEqual<glm::lowp_vec2>();
	Error += test_notEqual<glm::mediump_vec2>();
	Error += test_notEqual<glm::highp_vec2>();
	Error += test_notEqual<glm::vec3>();
	Error += test_notEqual<glm::lowp_vec3>();
	Error += test_notEqual<glm::mediump_vec3>();
	Error += test_notEqual<glm::highp_vec3>();
	Error += test_notEqual<glm::vec4>();
	Error += test_notEqual<glm::lowp_vec4>();
	Error += test_notEqual<glm::mediump_vec4>();
	Error += test_notEqual<glm::highp_vec4>();

	Error += test_notEqual<glm::dvec1>();
	Error += test_notEqual<glm::lowp_dvec1>();
	Error += test_notEqual<glm::mediump_dvec1>();
	Error += test_notEqual<glm::highp_dvec1>();
	Error += test_notEqual<glm::dvec2>();
	Error += test_notEqual<glm::lowp_dvec2>();
	Error += test_notEqual<glm::mediump_dvec2>();
	Error += test_notEqual<glm::highp_dvec2>();
	Error += test_notEqual<glm::dvec3>();
	Error += test_notEqual<glm::lowp_dvec3>();
	Error += test_notEqual<glm::mediump_dvec3>();
	Error += test_notEqual<glm::highp_dvec3>();
	Error += test_notEqual<glm::dvec4>();
	Error += test_notEqual<glm::lowp_dvec4>();
	Error += test_notEqual<glm::mediump_dvec4>();
	Error += test_notEqual<glm::highp_dvec4>();

	Error += test_constexpr<glm::vec1, float>();
	Error += test_constexpr<glm::vec2, float>();
	Error += test_constexpr<glm::vec3, float>();
	Error += test_constexpr<glm::vec4, float>();

	return Error;
}
