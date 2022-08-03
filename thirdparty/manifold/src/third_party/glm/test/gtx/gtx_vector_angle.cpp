#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/constants.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <limits>

int test_angle()
{
	int Error = 0;
	
	float AngleA = glm::angle(glm::vec2(1, 0), glm::normalize(glm::vec2(1, 1)));
	Error += glm::epsilonEqual(AngleA, glm::pi<float>() * 0.25f, 0.01f) ? 0 : 1;
	float AngleB = glm::angle(glm::vec3(1, 0, 0), glm::normalize(glm::vec3(1, 1, 0)));
	Error += glm::epsilonEqual(AngleB, glm::pi<float>() * 0.25f, 0.01f) ? 0 : 1;
	float AngleC = glm::angle(glm::vec4(1, 0, 0, 0), glm::normalize(glm::vec4(1, 1, 0, 0)));
	Error += glm::epsilonEqual(AngleC, glm::pi<float>() * 0.25f, 0.01f) ? 0 : 1;

	return Error;
}

int test_orientedAngle_vec2()
{
	int Error = 0;
	
	float AngleA = glm::orientedAngle(glm::vec2(1, 0), glm::normalize(glm::vec2(1, 1)));
	Error += glm::epsilonEqual(AngleA, glm::pi<float>() * 0.25f, 0.01f) ? 0 : 1;
	float AngleB = glm::orientedAngle(glm::vec2(0, 1), glm::normalize(glm::vec2(1, 1)));
	Error += glm::epsilonEqual(AngleB, -glm::pi<float>() * 0.25f, 0.01f) ? 0 : 1;
	float AngleC = glm::orientedAngle(glm::normalize(glm::vec2(1, 1)), glm::vec2(0, 1));
	Error += glm::epsilonEqual(AngleC, glm::pi<float>() * 0.25f, 0.01f) ? 0 : 1;

	return Error;
}

int test_orientedAngle_vec3()
{
	int Error = 0;
	
	float AngleA = glm::orientedAngle(glm::vec3(1, 0, 0), glm::normalize(glm::vec3(1, 1, 0)), glm::vec3(0, 0, 1));
	Error += glm::epsilonEqual(AngleA, glm::pi<float>() * 0.25f, 0.01f) ? 0 : 1;
	float AngleB = glm::orientedAngle(glm::vec3(0, 1, 0), glm::normalize(glm::vec3(1, 1, 0)), glm::vec3(0, 0, 1));
	Error += glm::epsilonEqual(AngleB, -glm::pi<float>() * 0.25f, 0.01f) ? 0 : 1;
	float AngleC = glm::orientedAngle(glm::normalize(glm::vec3(1, 1, 0)), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1));
	Error += glm::epsilonEqual(AngleC, glm::pi<float>() * 0.25f, 0.01f) ? 0 : 1;

	return Error;
}

int main()
{
	int Error(0);
	
	Error += test_angle();
	Error += test_orientedAngle_vec2();
	Error += test_orientedAngle_vec3();

	return Error;
}


