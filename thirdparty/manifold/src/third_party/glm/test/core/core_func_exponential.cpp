#include <glm/gtc/constants.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/vector_float1.hpp>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/common.hpp>
#include <glm/exponential.hpp>

static int test_pow()
{
	int Error(0);

	float A = glm::pow(2.f, 2.f);
	Error += glm::equal(A, 4.f, 0.01f) ? 0 : 1;

	glm::vec1 B = glm::pow(glm::vec1(2.f), glm::vec1(2.f));
	Error += glm::all(glm::equal(B, glm::vec1(4.f), 0.01f)) ? 0 : 1;

	glm::vec2 C = glm::pow(glm::vec2(2.f), glm::vec2(2.f));
	Error += glm::all(glm::equal(C, glm::vec2(4.f), 0.01f)) ? 0 : 1;

	glm::vec3 D = glm::pow(glm::vec3(2.f), glm::vec3(2.f));
	Error += glm::all(glm::equal(D, glm::vec3(4.f), 0.01f)) ? 0 : 1;

	glm::vec4 E = glm::pow(glm::vec4(2.f), glm::vec4(2.f));
	Error += glm::all(glm::equal(E, glm::vec4(4.f), 0.01f)) ? 0 : 1;

	return Error;
}

static int test_sqrt()
{
	int Error = 0;

	float A = glm::sqrt(4.f);
	Error += glm::equal(A, 2.f, 0.01f) ? 0 : 1;

	glm::vec1 B = glm::sqrt(glm::vec1(4.f));
	Error += glm::all(glm::equal(B, glm::vec1(2.f), 0.01f)) ? 0 : 1;

	glm::vec2 C = glm::sqrt(glm::vec2(4.f));
	Error += glm::all(glm::equal(C, glm::vec2(2.f), 0.01f)) ? 0 : 1;

	glm::vec3 D = glm::sqrt(glm::vec3(4.f));
	Error += glm::all(glm::equal(D, glm::vec3(2.f), 0.01f)) ? 0 : 1;

	glm::vec4 E = glm::sqrt(glm::vec4(4.f));
	Error += glm::all(glm::equal(E, glm::vec4(2.f), 0.01f)) ? 0 : 1;

	return Error;
}

static int test_exp()
{
	int Error = 0;

	float A = glm::exp(1.f);
	Error += glm::equal(A, glm::e<float>(), 0.01f) ? 0 : 1;

	glm::vec1 B = glm::exp(glm::vec1(1.f));
	Error += glm::all(glm::equal(B, glm::vec1(glm::e<float>()), 0.01f)) ? 0 : 1;

	glm::vec2 C = glm::exp(glm::vec2(1.f));
	Error += glm::all(glm::equal(C, glm::vec2(glm::e<float>()), 0.01f)) ? 0 : 1;

	glm::vec3 D = glm::exp(glm::vec3(1.f));
	Error += glm::all(glm::equal(D, glm::vec3(glm::e<float>()), 0.01f)) ? 0 : 1;

	glm::vec4 E = glm::exp(glm::vec4(1.f));
	Error += glm::all(glm::equal(E, glm::vec4(glm::e<float>()), 0.01f)) ? 0 : 1;

	return Error;
}

static int test_log()
{
	int Error = 0;

	float const A = glm::log(glm::e<float>());
	Error += glm::equal(A, 1.f, 0.01f) ? 0 : 1;

	glm::vec1 const B = glm::log(glm::vec1(glm::e<float>()));
	Error += glm::all(glm::equal(B, glm::vec1(1.f), 0.01f)) ? 0 : 1;

	glm::vec2 const C = glm::log(glm::vec2(glm::e<float>()));
	Error += glm::all(glm::equal(C, glm::vec2(1.f), 0.01f)) ? 0 : 1;

	glm::vec3 const D = glm::log(glm::vec3(glm::e<float>()));
	Error += glm::all(glm::equal(D, glm::vec3(1.f), 0.01f)) ? 0 : 1;

	glm::vec4 const E = glm::log(glm::vec4(glm::e<float>()));
	Error += glm::all(glm::equal(E, glm::vec4(1.f), 0.01f)) ? 0 : 1;

	return Error;
}

static int test_exp2()
{
	int Error = 0;

	float A = glm::exp2(4.f);
	Error += glm::equal(A, 16.f, 0.01f) ? 0 : 1;

	glm::vec1 B = glm::exp2(glm::vec1(4.f));
	Error += glm::all(glm::equal(B, glm::vec1(16.f), 0.01f)) ? 0 : 1;

	glm::vec2 C = glm::exp2(glm::vec2(4.f, 3.f));
	Error += glm::all(glm::equal(C, glm::vec2(16.f, 8.f), 0.01f)) ? 0 : 1;

	glm::vec3 D = glm::exp2(glm::vec3(4.f, 3.f, 2.f));
	Error += glm::all(glm::equal(D, glm::vec3(16.f, 8.f, 4.f), 0.01f)) ? 0 : 1;

	glm::vec4 E = glm::exp2(glm::vec4(4.f, 3.f, 2.f, 1.f));
	Error += glm::all(glm::equal(E, glm::vec4(16.f, 8.f, 4.f, 2.f), 0.01f)) ? 0 : 1;

#   if GLM_HAS_CXX11_STL
	//large exponent
	float F = glm::exp2(23.f);
	Error += glm::equal(F, 8388608.f, 0.01f) ? 0 : 1;
#   endif

	return Error;
}

static int test_log2()
{
	int Error = 0;

	float A = glm::log2(16.f);
	Error += glm::equal(A, 4.f, 0.01f) ? 0 : 1;

	glm::vec1 B = glm::log2(glm::vec1(16.f));
	Error += glm::all(glm::equal(B, glm::vec1(4.f), 0.01f)) ? 0 : 1;

	glm::vec2 C = glm::log2(glm::vec2(16.f, 8.f));
	Error += glm::all(glm::equal(C, glm::vec2(4.f, 3.f), 0.01f)) ? 0 : 1;

	glm::vec3 D = glm::log2(glm::vec3(16.f, 8.f, 4.f));
	Error += glm::all(glm::equal(D, glm::vec3(4.f, 3.f, 2.f), 0.01f)) ? 0 : 1;

	glm::vec4 E = glm::log2(glm::vec4(16.f, 8.f, 4.f, 2.f));
	Error += glm::all(glm::equal(E, glm::vec4(4.f, 3.f, 2.f, 1.f), 0.01f)) ? 0 : 1;

	return Error;
}

static int test_inversesqrt()
{
	int Error = 0;

	float A = glm::inversesqrt(16.f) * glm::sqrt(16.f);
	Error += glm::equal(A, 1.f, 0.01f) ? 0 : 1;

	glm::vec1 B = glm::inversesqrt(glm::vec1(16.f)) * glm::sqrt(16.f);
	Error += glm::all(glm::equal(B, glm::vec1(1.f), 0.01f)) ? 0 : 1;

	glm::vec2 C = glm::inversesqrt(glm::vec2(16.f)) * glm::sqrt(16.f);
	Error += glm::all(glm::equal(C, glm::vec2(1.f), 0.01f)) ? 0 : 1;

	glm::vec3 D = glm::inversesqrt(glm::vec3(16.f)) * glm::sqrt(16.f);
	Error += glm::all(glm::equal(D, glm::vec3(1.f), 0.01f)) ? 0 : 1;

	glm::vec4 E = glm::inversesqrt(glm::vec4(16.f)) * glm::sqrt(16.f);
	Error += glm::all(glm::equal(E, glm::vec4(1.f), 0.01f)) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_pow();
	Error += test_sqrt();
	Error += test_exp();
	Error += test_log();
	Error += test_exp2();
	Error += test_log2();
	Error += test_inversesqrt();

	return Error;
}

