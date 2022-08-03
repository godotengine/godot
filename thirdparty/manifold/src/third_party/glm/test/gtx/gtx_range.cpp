#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/constants.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/glm.hpp>

#if GLM_HAS_RANGE_FOR

#include <glm/gtx/range.hpp>

int test_vec()
{
	int Error = 0;

	{
		glm::ivec3 const v(1, 2, 3);

		int count = 0;
		glm::ivec3 Result(0);
		for(int x : v)
		{
			Result[count] = x;
			count++;
		}
		Error += count == 3 ? 0 : 1;
		Error += v == Result ? 0 : 1;
	}

	{
		glm::ivec3 v(1, 2, 3);
		for(int& x : v)
			x = 0;
		Error += glm::all(glm::equal(v, glm::ivec3(0))) ? 0 : 1;
	}

	return Error;
}

int test_mat()
{
	int Error = 0;

	{
		glm::mat4x3 m(1.0f);

		int count = 0;
		float Sum = 0.0f;
		for(float x : m)
		{
			count++;
			Sum += x;
		}
		Error += count == 12 ? 0 : 1;
		Error += glm::equal(Sum, 3.0f, 0.001f) ? 0 : 1;
	}

	{
		glm::mat4x3 m(1.0f);

		for (float& x : m) { x = 0; }
		glm::vec4 v(1, 1, 1, 1);
		Error += glm::all(glm::equal(m*v, glm::vec3(0, 0, 0), glm::epsilon<float>())) ? 0 : 1;
	}

	return Error;
}

int main()
{
	int Error = 0;
	Error += test_vec();
	Error += test_mat();
	return Error;
}

#else

int main()
{
	return 0;
}

#endif//GLM_HAS_RANGE_FOR
