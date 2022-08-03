#define GLM_FORCE_EXPLICIT_CTOR

#include <glm/glm.hpp>
#include <glm/ext.hpp>

int main()
{
	int Error = 0;

	glm::ivec4 B(1);
	Error += B == glm::ivec4(1) ? 0 : 1;

	//glm::vec4 A = B;

	return Error;
}

