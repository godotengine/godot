#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/gtx/vector_query.hpp>

int test_areCollinear()
{
	int Error(0);

	{
		bool TestA = glm::areCollinear(glm::vec2(-1), glm::vec2(1), 0.00001f);
		Error += TestA ? 0 : 1;
	}

	{
		bool TestA = glm::areCollinear(glm::vec3(-1), glm::vec3(1), 0.00001f);
		Error += TestA ? 0 : 1;
	}

	{
		bool TestA = glm::areCollinear(glm::vec4(-1), glm::vec4(1), 0.00001f);
		Error += TestA ? 0 : 1;
	}

	return Error;
}

int test_areOrthogonal()
{
	int Error(0);
	
	bool TestA = glm::areOrthogonal(glm::vec2(1, 0), glm::vec2(0, 1), 0.00001f);
	Error += TestA ? 0 : 1;

	return Error;
}

int test_isNormalized()
{
	int Error(0);
	
	bool TestA = glm::isNormalized(glm::vec4(1, 0, 0, 0), 0.00001f);
	Error += TestA ? 0 : 1;

	return Error;
}

int test_isNull()
{
	int Error(0);
	
	bool TestA = glm::isNull(glm::vec4(0), 0.00001f);
	Error += TestA ? 0 : 1;

	return Error;
}

int test_areOrthonormal()
{
	int Error(0);

	bool TestA = glm::areOrthonormal(glm::vec2(1, 0), glm::vec2(0, 1), 0.00001f);
	Error += TestA ? 0 : 1;

	return Error;
}

int main()
{
	int Error(0);

	Error += test_areCollinear();
	Error += test_areOrthogonal();
	Error += test_isNormalized();
	Error += test_isNull();
	Error += test_areOrthonormal();

	return Error;
}


