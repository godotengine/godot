#include <glm/glm.hpp>

static int test_length_mat_non_squared()
{
	int Error = 0;

	Error += glm::mat2x3().length() == 2 ? 0 : 1;
	Error += glm::mat2x4().length() == 2 ? 0 : 1;
	Error += glm::mat3x2().length() == 3 ? 0 : 1;
	Error += glm::mat3x4().length() == 3 ? 0 : 1;
	Error += glm::mat4x2().length() == 4 ? 0 : 1;
	Error += glm::mat4x3().length() == 4 ? 0 : 1;
	
	Error += glm::dmat2x3().length() == 2 ? 0 : 1;
	Error += glm::dmat2x4().length() == 2 ? 0 : 1;
	Error += glm::dmat3x2().length() == 3 ? 0 : 1;
	Error += glm::dmat3x4().length() == 3 ? 0 : 1;
	Error += glm::dmat4x2().length() == 4 ? 0 : 1;
	Error += glm::dmat4x3().length() == 4 ? 0 : 1;
	
	return Error;
}

static int test_length_mat()
{
	int Error = 0;
	
	Error += glm::mat2().length() == 2 ? 0 : 1;
	Error += glm::mat3().length() == 3 ? 0 : 1;
	Error += glm::mat4().length() == 4 ? 0 : 1;
	Error += glm::mat2x2().length() == 2 ? 0 : 1;
	Error += glm::mat3x3().length() == 3 ? 0 : 1;
	Error += glm::mat4x4().length() == 4 ? 0 : 1;
	
	Error += glm::dmat2().length() == 2 ? 0 : 1;
	Error += glm::dmat3().length() == 3 ? 0 : 1;
	Error += glm::dmat4().length() == 4 ? 0 : 1;
	Error += glm::dmat2x2().length() == 2 ? 0 : 1;
	Error += glm::dmat3x3().length() == 3 ? 0 : 1;
	Error += glm::dmat4x4().length() == 4 ? 0 : 1;
	
	return Error;
}

static int test_length_vec()
{
	int Error = 0;
	
	Error += glm::vec2().length() == 2 ? 0 : 1;
	Error += glm::vec3().length() == 3 ? 0 : 1;
	Error += glm::vec4().length() == 4 ? 0 : 1;

	Error += glm::ivec2().length() == 2 ? 0 : 1;
	Error += glm::ivec3().length() == 3 ? 0 : 1;
	Error += glm::ivec4().length() == 4 ? 0 : 1;

	Error += glm::uvec2().length() == 2 ? 0 : 1;
	Error += glm::uvec3().length() == 3 ? 0 : 1;
	Error += glm::uvec4().length() == 4 ? 0 : 1;	
	
	Error += glm::dvec2().length() == 2 ? 0 : 1;
	Error += glm::dvec3().length() == 3 ? 0 : 1;
	Error += glm::dvec4().length() == 4 ? 0 : 1;
	
	return Error;
}

int main()
{
	int Error = 0;
	
	Error += test_length_vec();
	Error += test_length_mat();
	Error += test_length_mat_non_squared();
	
	return Error;
}

