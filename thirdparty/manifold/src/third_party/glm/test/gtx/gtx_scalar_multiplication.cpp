#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/constants.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/glm.hpp>

#if GLM_HAS_TEMPLATE_ALIASES && !(GLM_COMPILER & GLM_COMPILER_GCC)
#include <glm/gtx/scalar_multiplication.hpp>

int main()
{
	int Error(0);
	glm::vec3 v(0.5, 3.1, -9.1);

	Error += glm::all(glm::equal(v, 1.0 * v, glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(v, 1 * v, glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(v, 1u * v, glm::epsilon<float>())) ? 0 : 1;

	glm::mat3 m(1, 2, 3, 4, 5, 6, 7, 8, 9);
	glm::vec3 w = 0.5f * m * v;

	Error += glm::all(glm::equal((m*v)/2, w, glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(m*(v/2), w, glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal((m/2)*v, w, glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal((0.5*m)*v, w, glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(0.5*(m*v), w, glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

#else

int main()
{
	return 0;
}

#endif
