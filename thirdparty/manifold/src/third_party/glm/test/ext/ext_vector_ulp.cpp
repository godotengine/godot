#include <glm/ext/vector_ulp.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_double4.hpp>
#include <glm/ext/vector_int4.hpp>

static int test_ulp_float_dist()
{
	int Error = 0;

	glm::vec4 const A(1.0f);

	glm::vec4 const B = glm::nextFloat(A);
	Error += glm::any(glm::notEqual(A, B, 0)) ? 0 : 1;
	glm::vec4 const C = glm::prevFloat(B);
	Error += glm::all(glm::equal(A, C, 0)) ? 0 : 1;

	glm::ivec4 const D = glm::floatDistance(A, B);
	Error += D == glm::ivec4(1) ? 0 : 1;
	glm::ivec4 const E = glm::floatDistance(A, C);
	Error += E == glm::ivec4(0) ? 0 : 1;

	return Error;
}

static int test_ulp_float_step()
{
	int Error = 0;

	glm::vec4 const A(1.0f);

	for(int i = 10; i < 1000; i *= 10)
	{
		glm::vec4 const B = glm::nextFloat(A, i);
		Error += glm::any(glm::notEqual(A, B, 0)) ? 0 : 1;
		glm::vec4 const C = glm::prevFloat(B, i);
		Error += glm::all(glm::equal(A, C, 0)) ? 0 : 1;

		glm::ivec4 const D = glm::floatDistance(A, B);
		Error += D == glm::ivec4(i) ? 0 : 1;
		glm::ivec4 const E = glm::floatDistance(A, C);
		Error += E == glm::ivec4(0) ? 0 : 1;
	}

	return Error;
}

static int test_ulp_double_dist()
{
	int Error = 0;

	glm::dvec4 const A(1.0);

	glm::dvec4 const B = glm::nextFloat(A);
	Error += glm::any(glm::notEqual(A, B, 0)) ? 0 : 1;
	glm::dvec4 const C = glm::prevFloat(B);
	Error += glm::all(glm::equal(A, C, 0)) ? 0 : 1;

	glm::ivec4 const D(glm::floatDistance(A, B));
	Error += D == glm::ivec4(1) ? 0 : 1;
	glm::ivec4 const E = glm::floatDistance(A, C);
	Error += E == glm::ivec4(0) ? 0 : 1;

	return Error;
}

static int test_ulp_double_step()
{
	int Error = 0;

	glm::dvec4 const A(1.0);

	for(int i = 10; i < 1000; i *= 10)
	{
		glm::dvec4 const B = glm::nextFloat(A, i);
		Error += glm::any(glm::notEqual(A, B, 0)) ? 0 : 1;
		glm::dvec4 const C = glm::prevFloat(B, i);
		Error += glm::all(glm::equal(A, C, 0)) ? 0 : 1;

		glm::ivec4 const D(glm::floatDistance(A, B));
		Error += D == glm::ivec4(i) ? 0 : 1;
		glm::ivec4 const E(glm::floatDistance(A, C));
		Error += E == glm::ivec4(0) ? 0 : 1;
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_ulp_float_dist();
	Error += test_ulp_float_step();
	Error += test_ulp_double_dist();
	Error += test_ulp_double_step();

	return Error;
}
