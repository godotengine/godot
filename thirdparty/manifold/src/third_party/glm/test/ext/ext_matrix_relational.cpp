#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/matrix_double2x2.hpp>
#include <glm/ext/matrix_double2x3.hpp>
#include <glm/ext/matrix_double2x4.hpp>
#include <glm/ext/matrix_double3x2.hpp>
#include <glm/ext/matrix_double3x3.hpp>
#include <glm/ext/matrix_double3x4.hpp>
#include <glm/ext/matrix_double4x2.hpp>
#include <glm/ext/matrix_double4x3.hpp>
#include <glm/ext/matrix_double4x4.hpp>
#include <glm/ext/vector_double2.hpp>
#include <glm/ext/vector_double3.hpp>
#include <glm/ext/vector_double4.hpp>
#include <glm/ext/matrix_float2x2.hpp>
#include <glm/ext/matrix_float2x3.hpp>
#include <glm/ext/matrix_float2x4.hpp>
#include <glm/ext/matrix_float3x2.hpp>
#include <glm/ext/matrix_float3x3.hpp>
#include <glm/ext/matrix_float3x4.hpp>
#include <glm/ext/matrix_float4x2.hpp>
#include <glm/ext/matrix_float4x3.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/scalar_ulp.hpp>

template <typename matType, typename vecType>
static int test_equal()
{
	typedef typename matType::value_type valType;

	valType const Epsilon = static_cast<valType>(0.001f);
	valType const One = static_cast<valType>(1);
	valType const Two = static_cast<valType>(2);

	int Error = 0;

	Error += glm::all(glm::equal(matType(One), matType(One), Epsilon)) ? 0 : 1;
	Error += glm::all(glm::equal(matType(One), matType(Two), vecType(Epsilon))) ? 1 : 0;

	return Error;
}

template <typename matType, typename vecType>
static int test_notEqual()
{
	typedef typename matType::value_type valType;

	valType const Epsilon = static_cast<valType>(0.001f);
	valType const One = static_cast<valType>(1);
	valType const Two = static_cast<valType>(2);

	int Error = 0;

	Error += !glm::any(glm::notEqual(matType(One), matType(One), Epsilon)) ? 0 : 1;
	Error += !glm::any(glm::notEqual(matType(One), matType(Two), vecType(Epsilon))) ? 1 : 0;

	return Error;
}


template <typename T>
static int test_equal_ulps()
{
	typedef glm::mat<4, 4, T, glm::defaultp> mat4;

	T const One(1);
	mat4 const Ones(1);

	int Error = 0;

	T const ULP1Plus = glm::nextFloat(One);
	Error += glm::all(glm::equal(Ones, mat4(ULP1Plus), 1)) ? 0 : 1;

	T const ULP2Plus = glm::nextFloat(ULP1Plus);
	Error += !glm::all(glm::equal(Ones, mat4(ULP2Plus), 1)) ? 0 : 1;

	T const ULP1Minus = glm::prevFloat(One);
	Error += glm::all(glm::equal(Ones, mat4(ULP1Minus), 1)) ? 0 : 1;

	T const ULP2Minus = glm::prevFloat(ULP1Minus);
	Error += !glm::all(glm::equal(Ones, mat4(ULP2Minus), 1)) ? 0 : 1;

	return Error;
}

template <typename T>
static int test_notEqual_ulps()
{
	typedef glm::mat<4, 4, T, glm::defaultp> mat4;

	T const One(1);
	mat4 const Ones(1);

	int Error = 0;

	T const ULP1Plus = glm::nextFloat(One);
	Error += !glm::all(glm::notEqual(Ones, mat4(ULP1Plus), 1)) ? 0 : 1;

	T const ULP2Plus = glm::nextFloat(ULP1Plus);
	Error += glm::all(glm::notEqual(Ones, mat4(ULP2Plus), 1)) ? 0 : 1;

	T const ULP1Minus = glm::prevFloat(One);
	Error += !glm::all(glm::notEqual(Ones, mat4(ULP1Minus), 1)) ? 0 : 1;

	T const ULP2Minus = glm::prevFloat(ULP1Minus);
	Error += glm::all(glm::notEqual(Ones, mat4(ULP2Minus), 1)) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_equal_ulps<float>();
	Error += test_equal_ulps<double>();
	Error += test_notEqual_ulps<float>();
	Error += test_notEqual_ulps<double>();

	Error += test_equal<glm::mat2x2, glm::vec2>();
	Error += test_equal<glm::mat2x3, glm::vec2>();
	Error += test_equal<glm::mat2x4, glm::vec2>();
	Error += test_equal<glm::mat3x2, glm::vec3>();
	Error += test_equal<glm::mat3x3, glm::vec3>();
	Error += test_equal<glm::mat3x4, glm::vec3>();
	Error += test_equal<glm::mat4x2, glm::vec4>();
	Error += test_equal<glm::mat4x3, glm::vec4>();
	Error += test_equal<glm::mat4x4, glm::vec4>();

	Error += test_equal<glm::dmat2x2, glm::dvec2>();
	Error += test_equal<glm::dmat2x3, glm::dvec2>();
	Error += test_equal<glm::dmat2x4, glm::dvec2>();
	Error += test_equal<glm::dmat3x2, glm::dvec3>();
	Error += test_equal<glm::dmat3x3, glm::dvec3>();
	Error += test_equal<glm::dmat3x4, glm::dvec3>();
	Error += test_equal<glm::dmat4x2, glm::dvec4>();
	Error += test_equal<glm::dmat4x3, glm::dvec4>();
	Error += test_equal<glm::dmat4x4, glm::dvec4>();

	Error += test_notEqual<glm::mat2x2, glm::vec2>();
	Error += test_notEqual<glm::mat2x3, glm::vec2>();
	Error += test_notEqual<glm::mat2x4, glm::vec2>();
	Error += test_notEqual<glm::mat3x2, glm::vec3>();
	Error += test_notEqual<glm::mat3x3, glm::vec3>();
	Error += test_notEqual<glm::mat3x4, glm::vec3>();
	Error += test_notEqual<glm::mat4x2, glm::vec4>();
	Error += test_notEqual<glm::mat4x3, glm::vec4>();
	Error += test_notEqual<glm::mat4x4, glm::vec4>();

	Error += test_notEqual<glm::dmat2x2, glm::dvec2>();
	Error += test_notEqual<glm::dmat2x3, glm::dvec2>();
	Error += test_notEqual<glm::dmat2x4, glm::dvec2>();
	Error += test_notEqual<glm::dmat3x2, glm::dvec3>();
	Error += test_notEqual<glm::dmat3x3, glm::dvec3>();
	Error += test_notEqual<glm::dmat3x4, glm::dvec3>();
	Error += test_notEqual<glm::dmat4x2, glm::dvec4>();
	Error += test_notEqual<glm::dmat4x3, glm::dvec4>();
	Error += test_notEqual<glm::dmat4x4, glm::dvec4>();

	return Error;
}
