#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/scalar_relational.hpp>

float const Epsilon = 0.001f;

static int test_angle()
{
	int Error = 0;

	{
		glm::quat const Q = glm::quat(glm::vec3(1, 0, 0), glm::vec3(0, 1, 0));
		float const A = glm::degrees(glm::angle(Q));
		Error += glm::equal(A, 90.0f, Epsilon) ? 0 : 1;
	}

	{
		glm::quat const Q = glm::quat(glm::vec3(0, 1, 0), glm::vec3(1, 0, 0));
		float const A = glm::degrees(glm::angle(Q));
		Error += glm::equal(A, 90.0f, Epsilon) ? 0 : 1;
	}

	{
		glm::quat const Q = glm::angleAxis(glm::two_pi<float>() - 1.0f, glm::vec3(1, 0, 0));
		float const A = glm::angle(Q);
		Error += glm::equal(A, 1.0f, Epsilon) ? 1 : 0;
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_angle();

	return Error;
}
