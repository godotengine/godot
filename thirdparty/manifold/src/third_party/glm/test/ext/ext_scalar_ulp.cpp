#include <glm/ext/scalar_ulp.hpp>
#include <glm/ext/scalar_relational.hpp>

static int test_ulp_float_dist()
{
	int Error = 0;

	float A = 1.0f;

	float B = glm::nextFloat(A);
	Error += glm::notEqual(A, B, 0) ? 0 : 1;
	float C = glm::prevFloat(B);
	Error += glm::equal(A, C, 0) ? 0 : 1;

	int D = glm::floatDistance(A, B);
	Error += D == 1 ? 0 : 1;
	int E = glm::floatDistance(A, C);
	Error += E == 0 ? 0 : 1;

	return Error;
}

static int test_ulp_float_step()
{
	int Error = 0;

	float A = 1.0f;

	for(int i = 10; i < 1000; i *= 10)
	{
		float B = glm::nextFloat(A, i);
		Error += glm::notEqual(A, B, 0) ? 0 : 1;
		float C = glm::prevFloat(B, i);
		Error += glm::equal(A, C, 0) ? 0 : 1;

		int D = glm::floatDistance(A, B);
		Error += D == i ? 0 : 1;
		int E = glm::floatDistance(A, C);
		Error += E == 0 ? 0 : 1;
	}

	return Error;
}

static int test_ulp_double_dist()
{
	int Error = 0;

	double A = 1.0;

	double B = glm::nextFloat(A);
	Error += glm::notEqual(A, B, 0) ? 0 : 1;
	double C = glm::prevFloat(B);
	Error += glm::equal(A, C, 0) ? 0 : 1;

	glm::int64 const D = glm::floatDistance(A, B);
	Error += D == 1 ? 0 : 1;
	glm::int64 const E = glm::floatDistance(A, C);
	Error += E == 0 ? 0 : 1;

	return Error;
}

static int test_ulp_double_step()
{
	int Error = 0;

	double A = 1.0;

	for(int i = 10; i < 1000; i *= 10)
	{
		double B = glm::nextFloat(A, i);
		Error += glm::notEqual(A, B, 0) ? 0 : 1;
		double C = glm::prevFloat(B, i);
		Error += glm::equal(A, C, 0) ? 0 : 1;

		glm::int64 const D = glm::floatDistance(A, B);
		Error += D == i ? 0 : 1;
		glm::int64 const E = glm::floatDistance(A, C);
		Error += E == 0 ? 0 : 1;
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
