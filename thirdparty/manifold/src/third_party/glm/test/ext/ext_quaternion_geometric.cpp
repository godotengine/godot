#include <glm/gtc/constants.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/ext/quaternion_float_precision.hpp>
#include <glm/ext/quaternion_double.hpp>
#include <glm/ext/quaternion_double_precision.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float3_precision.hpp>
#include <glm/ext/vector_double3.hpp>
#include <glm/ext/vector_double3_precision.hpp>
#include <glm/ext/scalar_relational.hpp>

float const Epsilon = 0.001f;

static int test_length()
{
	int Error = 0;

	{
		float const A = glm::length(glm::quat(1, 0, 0, 0));
		Error += glm::equal(A, 1.0f, Epsilon) ? 0 : 1;
	}

	{
		float const A = glm::length(glm::quat(1, glm::vec3(0)));
		Error += glm::equal(A, 1.0f, Epsilon) ? 0 : 1;
	}

	{
		float const A = glm::length(glm::quat(glm::vec3(1, 0, 0), glm::vec3(0, 1, 0)));
		Error += glm::equal(A, 1.0f, Epsilon) ? 0 : 1;
	}

	return Error;
}

static int test_normalize()
{
	int Error = 0;

	{
		glm::quat const A = glm::quat(1, 0, 0, 0);
		glm::quat const N = glm::normalize(A);
		Error += glm::all(glm::equal(A, N, Epsilon)) ? 0 : 1;
	}

	{
		glm::quat const A = glm::quat(1, glm::vec3(0));
		glm::quat const N = glm::normalize(A);
		Error += glm::all(glm::equal(A, N, Epsilon)) ? 0 : 1;
	}

	return Error;
}

static int test_dot()
{
	int Error = 0;

	{
		glm::quat const A = glm::quat(1, 0, 0, 0);
		glm::quat const B = glm::quat(1, 0, 0, 0);
		float const C = glm::dot(A, B);
		Error += glm::equal(C, 1.0f, Epsilon) ? 0 : 1;
	}

	return Error;
}

static int test_cross()
{
	int Error = 0;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_length();
	Error += test_normalize();
	Error += test_dot();
	Error += test_cross();

	return Error;
}
