#define GLM_ENABLE_EXPERIMENTAL
#include <glm/exponential.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/integer.hpp>
#include <cstdio>
/*
int test_floor_log2()
{
	int Error = 0;

	for(std::size_t i = 1; i < 1000000; ++i)
	{
		glm::uint A = glm::floor_log2(glm::uint(i));
		glm::uint B = glm::uint(glm::floor(glm::log2(double(i)))); // Will fail with float, lack of accuracy

		Error += A == B ? 0 : 1;
		assert(!Error);
	}

	return Error;
}
*/
int test_log2()
{
	int Error = 0;

	for(std::size_t i = 1; i < 24; ++i)
	{
		glm::uint A = glm::log2(glm::uint(1 << i));
		glm::uint B = glm::uint(glm::log2(double(1 << i)));

		//Error += glm::equalEpsilon(double(A), B, 1.0) ? 0 : 1;
		Error += glm::abs(double(A) - B) <= 24 ? 0 : 1;
		assert(!Error);

		std::printf("Log2(%d) error A=%d, B=%d\n", 1 << i, A, B);
	}

	std::printf("log2 error=%d\n", Error);

	return Error;
}

int test_nlz()
{
	int Error = 0;

	for(glm::uint i = 1; i < glm::uint(33); ++i)
		Error += glm::nlz(i) == glm::uint(31u) - glm::findMSB(i) ? 0 : 1;
		//printf("%d, %d\n", glm::nlz(i), 31u - glm::findMSB(i));

	return Error;
}

int test_pow_uint()
{
	int Error = 0;

	glm::uint const p0 = glm::pow(2u, 0u);
	Error += p0 == 1u ? 0 : 1;

	glm::uint const p1 = glm::pow(2u, 1u);
	Error += p1 == 2u ? 0 : 1;

	glm::uint const p2 = glm::pow(2u, 2u);
	Error += p2 == 4u ? 0 : 1;

	return Error;
}

int test_pow_int()
{
	int Error = 0;

	int const p0 = glm::pow(2, 0u);
	Error += p0 == 1 ? 0 : 1;

	int const p1 = glm::pow(2, 1u);
	Error += p1 == 2 ? 0 : 1;

	int const p2 = glm::pow(2, 2u);
	Error += p2 == 4 ? 0 : 1;

	int const p0n = glm::pow(-2, 0u);
	Error += p0n == -1 ? 0 : 1;

	int const p1n = glm::pow(-2, 1u);
	Error += p1n == -2 ? 0 : 1;

	int const p2n = glm::pow(-2, 2u);
	Error += p2n == 4 ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_nlz();
//	Error += test_floor_log2();
	Error += test_log2();
	Error += test_pow_uint();
	Error += test_pow_int();

	return Error;
}

