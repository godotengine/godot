#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/compatibility.hpp>

int main()
{
	int Error(0);

	Error += glm::isfinite(1.0f) ? 0 : 1;
	Error += glm::isfinite(1.0) ? 0 : 1;
	Error += glm::isfinite(-1.0f) ? 0 : 1;
	Error += glm::isfinite(-1.0) ? 0 : 1;

	Error += glm::all(glm::isfinite(glm::vec4(1.0f))) ? 0 : 1;
	Error += glm::all(glm::isfinite(glm::dvec4(1.0))) ? 0 : 1;
	Error += glm::all(glm::isfinite(glm::vec4(-1.0f))) ? 0 : 1;
	Error += glm::all(glm::isfinite(glm::dvec4(-1.0))) ? 0 : 1;

	return Error;
}
