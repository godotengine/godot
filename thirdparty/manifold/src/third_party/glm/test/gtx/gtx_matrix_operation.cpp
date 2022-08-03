#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <limits>

int test_adjugate()
{
	int Error = 0;

	const float epsilon = std::numeric_limits<float>::epsilon();

	// mat2
	const glm::mat2 m2(
		2, 3,
		1, 5
	);

	const glm::mat2 eam2(
		5, -3,
		-1, 2
	);

	const glm::mat2 am2 = glm::adjugate(m2);

	Error += glm::all(glm::bvec2(
		glm::all(glm::epsilonEqual(am2[0], eam2[0], epsilon)),
		glm::all(glm::epsilonEqual(am2[1], eam2[1], epsilon))
	)) ? 0 : 1;

	// mat3
	const glm::mat3 m3(
		2, 3, 3,
		1, 5, 4,
		4, 6, 8
	);

	const glm::mat3 eam3(
		16, -6, -3,
		8, 4, -5,
		-14, 0, 7
	);

	const glm::mat3 am3 = glm::adjugate(m3);

	Error += glm::all(glm::bvec3(
		glm::all(glm::epsilonEqual(am3[0], eam3[0], epsilon)),
		glm::all(glm::epsilonEqual(am3[1], eam3[1], epsilon)),
		glm::all(glm::epsilonEqual(am3[2], eam3[2], epsilon))
	)) ? 0 : 1;

	// mat4
	const glm::mat4 m4(
		2, 3, 3, 1,
		1, 5, 4, 3,
		4, 6, 8, 5,
		-2, -3, -3, 4
	);

	const glm::mat4 eam4(
		97, -30, -15, 17,
		45, 20, -25, 5,
		-91, 0, 35, -21,
		14, 0, 0, 14
	);

	const glm::mat4 am4 = glm::adjugate(m4);

	Error += glm::all(glm::bvec4(
		glm::all(glm::epsilonEqual(am4[0], eam4[0], epsilon)),
		glm::all(glm::epsilonEqual(am4[1], eam4[1], epsilon)),
		glm::all(glm::epsilonEqual(am4[2], eam4[2], epsilon)),
		glm::all(glm::epsilonEqual(am4[3], eam4[3], epsilon))
	)) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_adjugate();

	return Error;
}
