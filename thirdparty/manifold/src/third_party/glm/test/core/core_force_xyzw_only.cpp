#define GLM_FORCE_XYZW_ONLY

#include <glm/gtc/constants.hpp>
#include <glm/gtc/vec1.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

static int test_comp()
{
	int Error = 0;

	{
		glm::ivec1 const A(1);
		Error += A.x == 1 ? 0 : 1;
	}

	{
		glm::ivec2 const A(1, 2);
		Error += A.x == 1 ? 0 : 1;
		Error += A.y == 2 ? 0 : 1;
	}

	{
		glm::ivec3 const A(1, 2, 3);
		Error += A.x == 1 ? 0 : 1;
		Error += A.y == 2 ? 0 : 1;
		Error += A.z == 3 ? 0 : 1;
	}

	{
		glm::ivec4 const A(1, 2, 3, 4);
		Error += A.x == 1 ? 0 : 1;
		Error += A.y == 2 ? 0 : 1;
		Error += A.z == 3 ? 0 : 1;
		Error += A.w == 4 ? 0 : 1;
	}

	return Error;
}

static int test_constexpr()
{
	int Error = 0;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_comp();
	Error += test_constexpr();

	return Error;
}
