#include <glm/gtx/exterior_product.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/vec2.hpp>

int main()
{
	int Error = 0;

	float const f = glm::cross(glm::vec2(1.0f, 1.0f), glm::vec2(1.0f, 1.0f));
	Error += glm::epsilonEqual(f, 0.0f, 0.001f) ? 0 : 1;

	return Error;
}

