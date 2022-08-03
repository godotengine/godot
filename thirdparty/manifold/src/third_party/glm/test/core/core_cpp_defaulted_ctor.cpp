#include <glm/glm.hpp>

#if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_ENABLE

#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/vec1.hpp>
#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/vector_relational.hpp>
#include <cstring>

static int test_vec_memcpy()
{
	int Error = 0;

	{
		glm::ivec1 const A = glm::ivec1(76);
		glm::ivec1 B;
		std::memcpy(&B, &A, sizeof(glm::ivec1));
		Error += B == A ? 0 : 1;
	}

	{
		glm::ivec2 const A = glm::ivec2(76);
		glm::ivec2 B;
		std::memcpy(&B, &A, sizeof(glm::ivec2));
		Error += B == A ? 0 : 1;
	}

	{
		glm::ivec3 const A = glm::ivec3(76);
		glm::ivec3 B;
		std::memcpy(&B, &A, sizeof(glm::ivec3));
		Error += B == A ? 0 : 1;
	}

	{
		glm::ivec4 const A = glm::ivec4(76);
		glm::ivec4 B;
		std::memcpy(&B, &A, sizeof(glm::ivec4));
		Error += B == A ? 0 : 1;
	}

	return Error;
}

static int test_mat_memcpy()
{
	int Error = 0;

	{
		glm::mat2x2 const A = glm::mat2x2(76);
		glm::mat2x2 B;
		std::memcpy(&B, &A, sizeof(glm::mat2x2));
		Error += glm::all(glm::equal(B, A, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::mat2x3 const A = glm::mat2x3(76);
		glm::mat2x3 B;
		std::memcpy(&B, &A, sizeof(glm::mat2x3));
		Error += glm::all(glm::equal(B, A, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::mat2x4 const A = glm::mat2x4(76);
		glm::mat2x4 B;
		std::memcpy(&B, &A, sizeof(glm::mat2x4));
		Error += glm::all(glm::equal(B, A, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::mat3x2 const A = glm::mat3x2(76);
		glm::mat3x2 B;
		std::memcpy(&B, &A, sizeof(glm::mat3x2));
		Error += glm::all(glm::equal(B, A, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::mat3x3 const A = glm::mat3x3(76);
		glm::mat3x3 B;
		std::memcpy(&B, &A, sizeof(glm::mat3x3));
		Error += glm::all(glm::equal(B, A, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::mat3x4 const A = glm::mat3x4(76);
		glm::mat3x4 B;
		std::memcpy(&B, &A, sizeof(glm::mat3x4));
		Error += glm::all(glm::equal(B, A, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::mat4x2 const A = glm::mat4x2(76);
		glm::mat4x2 B;
		std::memcpy(&B, &A, sizeof(glm::mat4x2));
		Error += glm::all(glm::equal(B, A, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::mat4x3 const A = glm::mat4x3(76);
		glm::mat4x3 B;
		std::memcpy(&B, &A, sizeof(glm::mat4x3));
		Error += glm::all(glm::equal(B, A, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::mat4x4 const A = glm::mat4x4(76);
		glm::mat4x4 B;
		std::memcpy(&B, &A, sizeof(glm::mat4x4));
		Error += glm::all(glm::equal(B, A, glm::epsilon<float>())) ? 0 : 1;
	}

	return Error;
}

static int test_quat_memcpy()
{
	int Error = 0;

	{
		glm::quat const A = glm::quat(1, 0, 0, 0);
		glm::quat B;
		std::memcpy(&B, &A, sizeof(glm::quat));
		Error += glm::all(glm::equal(B, A, glm::epsilon<float>())) ? 0 : 1;
	}

	return Error;
}

#endif//GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_ENABLE

int main()
{
	int Error = 0;

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_ENABLE
		Error += test_vec_memcpy();
		Error += test_mat_memcpy();
		Error += test_quat_memcpy();
#	endif//GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_ENABLE

	return Error;
}

