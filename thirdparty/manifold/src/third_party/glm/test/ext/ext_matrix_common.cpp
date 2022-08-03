#include <glm/ext/matrix_common.hpp>
#include <glm/ext/matrix_double4x4.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/vector_bool4.hpp>

static int test_mix()
{
	int Error = 0;

	{
		glm::mat4 A(2);
		glm::mat4 B(4);
		glm::mat4 C = glm::mix(A, B, 0.5f);
		glm::bvec4 const D = glm::equal(C, glm::mat4(3), 1);
		Error += glm::all(D) ? 0 : 1;
	}

	{
		glm::mat4 A(2);
		glm::mat4 B(4);
		glm::mat4 C = glm::mix(A, B, 0.5);
		glm::bvec4 const D = glm::equal(C, glm::mat4(3), 1);
		Error += glm::all(D) ? 0 : 1;
	}

	{
		glm::dmat4 A(2);
		glm::dmat4 B(4);
		glm::dmat4 C = glm::mix(A, B, 0.5);
		glm::bvec4 const D = glm::equal(C, glm::dmat4(3), 1);
		Error += glm::all(D) ? 0 : 1;
	}

	{
		glm::dmat4 A(2);
		glm::dmat4 B(4);
		glm::dmat4 C = glm::mix(A, B, 0.5f);
		glm::bvec4 const D = glm::equal(C, glm::dmat4(3), 1);
		Error += glm::all(D) ? 0 : 1;
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_mix();

	return Error;
}
