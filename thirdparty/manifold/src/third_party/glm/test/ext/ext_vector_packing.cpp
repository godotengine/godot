#include <glm/ext/vector_packing.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/vector_uint2_sized.hpp>
#include <glm/ext/vector_int2_sized.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/vec2.hpp>
#include <vector>

int test_packUnorm()
{
	int Error = 0;

	std::vector<glm::vec2> A;
	A.push_back(glm::vec2(1.0f, 0.7f));
	A.push_back(glm::vec2(0.5f, 0.1f));

	for (std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec2 B(A[i]);
		glm::u16vec2 C = glm::packUnorm<glm::uint16>(B);
		glm::vec2 D = glm::unpackUnorm<float>(C);
		Error += glm::all(glm::equal(B, D, 1.0f / 255.f)) ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int test_packSnorm()
{
	int Error = 0;

	std::vector<glm::vec2> A;
	A.push_back(glm::vec2(1.0f, 0.0f));
	A.push_back(glm::vec2(-0.5f, -0.7f));
	A.push_back(glm::vec2(-0.1f, 0.1f));

	for (std::size_t i = 0; i < A.size(); ++i)
	{
		glm::vec2 B(A[i]);
		glm::i16vec2 C = glm::packSnorm<glm::int16>(B);
		glm::vec2 D = glm::unpackSnorm<float>(C);
		Error += glm::all(glm::equal(B, D, 1.0f / 32767.0f * 2.0f)) ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_packUnorm();
	Error += test_packSnorm();

	return Error;
}
