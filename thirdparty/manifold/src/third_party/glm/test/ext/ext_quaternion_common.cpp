#include <glm/ext/vector_float3.hpp>
#include <glm/ext/quaternion_common.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/quaternion_relational.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/ext/scalar_relational.hpp>

static int test_conjugate()
{
	int Error = 0;

	glm::quat const A(glm::vec3(1, 0, 0), glm::vec3(0, 1, 0));
	glm::quat const C = glm::conjugate(A);
	Error += glm::any(glm::notEqual(A, C, glm::epsilon<float>())) ? 0 : 1;

	glm::quat const B = glm::conjugate(C);
	Error += glm::all(glm::equal(A, B, glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

static int test_mix()
{
	int Error = 0;

	glm::quat const Q1(glm::vec3(1, 0, 0), glm::vec3(1, 0, 0));
	glm::quat const Q2(glm::vec3(1, 0, 0), glm::vec3(0, 1, 0));

	{
		glm::quat const Q3 = glm::mix(Q1, Q2, 0.5f);
		float const F3 = glm::degrees(glm::angle(Q3));
		Error += glm::equal(F3, 45.0f, 0.001f) ? 0 : 1;

		glm::quat const Q4 = glm::mix(Q2, Q1, 0.5f);
		float const F4 = glm::degrees(glm::angle(Q4));
		Error += glm::equal(F4, 45.0f, 0.001f) ? 0 : 1;
	}

	{
		glm::quat const Q3 = glm::slerp(Q1, Q2, 0.5f);
		float const F3 = glm::degrees(glm::angle(Q3));
		Error += glm::equal(F3, 45.0f, 0.001f) ? 0 : 1;

		glm::quat const Q4 = glm::slerp(Q2, Q1, 0.5f);
		float const F4 = glm::degrees(glm::angle(Q4));
		Error += glm::equal(F4, 45.0f, 0.001f) ? 0 : 1;
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_conjugate();
	Error += test_mix();

	return Error;
}
