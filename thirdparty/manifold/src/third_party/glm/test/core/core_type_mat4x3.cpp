#include <glm/gtc/constants.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/matrix_relational.hpp>
#include <glm/mat2x2.hpp>
#include <glm/mat2x3.hpp>
#include <glm/mat2x4.hpp>
#include <glm/mat3x2.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat3x4.hpp>
#include <glm/mat4x2.hpp>
#include <glm/mat4x3.hpp>
#include <glm/mat4x4.hpp>
#include <vector>

static int test_operators()
{
	glm::mat4x3 l(1.0f);
	glm::mat4x3 m(1.0f);
	glm::vec4 u(1.0f);
	glm::vec3 v(1.0f);
	float x = 1.0f;
	glm::vec3 a = m * u;
	glm::vec4 b = v * m;
	glm::mat4x3 n = x / m;
	glm::mat4x3 o = m / x;
	glm::mat4x3 p = x * m;
	glm::mat4x3 q = m * x;
	bool R = glm::any(glm::notEqual(m, q, glm::epsilon<float>()));
	bool S = glm::all(glm::equal(m, l, glm::epsilon<float>()));

	return (S && !R) ? 0 : 1;
}

int test_ctr()
{
	int Error(0);

#if(GLM_HAS_INITIALIZER_LISTS)
	glm::mat4x3 m0(
		glm::vec3(0, 1, 2), 
		glm::vec3(3, 4, 5),
		glm::vec3(6, 7, 8),
		glm::vec3(9, 10, 11));

	glm::mat4x3 m1{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

	glm::mat4x3 m2{
		{0, 1, 2},
		{3, 4, 5},
		{6, 7, 8},
		{9, 10, 11}};

	Error += glm::all(glm::equal(m0, m2, glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(m1, m2, glm::epsilon<float>())) ? 0 : 1;

	std::vector<glm::mat4x3> v1{
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	};

	std::vector<glm::mat4x3> v2{
		{
			{ 0, 1, 2 },
			{ 4, 5, 6 },
			{ 8, 9, 10 },
			{ 12, 13, 14 }
		},
		{
			{ 0, 1, 2 },
			{ 4, 5, 6 },
			{ 8, 9, 10 },
			{ 12, 13, 14 }
		}
	};

#endif//GLM_HAS_INITIALIZER_LISTS

	return Error;
}

namespace cast
{
	template<typename genType>
	int entry()
	{
		int Error = 0;

		genType A(1.0f);
		glm::mat4x3 B(A);
		glm::mat4x3 Identity(1.0f);

		Error += glm::all(glm::equal(B, Identity, glm::epsilon<float>())) ? 0 : 1;

		return Error;
	}

	int test()
	{
		int Error = 0;
		
		Error += entry<glm::mat2x2>();
		Error += entry<glm::mat2x3>();
		Error += entry<glm::mat2x4>();
		Error += entry<glm::mat3x2>();
		Error += entry<glm::mat3x3>();
		Error += entry<glm::mat3x4>();
		Error += entry<glm::mat4x2>();
		Error += entry<glm::mat4x3>();
		Error += entry<glm::mat4x4>();

		return Error;
	}
}//namespace cast

static int test_size()
{
	int Error = 0;

	Error += 48 == sizeof(glm::mat4x3) ? 0 : 1;
	Error += 96 == sizeof(glm::dmat4x3) ? 0 : 1;
	Error += glm::mat4x3().length() == 4 ? 0 : 1;
	Error += glm::dmat4x3().length() == 4 ? 0 : 1;
	Error += glm::mat4x3::length() == 4 ? 0 : 1;
	Error += glm::dmat4x3::length() == 4 ? 0 : 1;

	return Error;
}

static int test_constexpr()
{
#if GLM_HAS_CONSTEXPR
	static_assert(glm::mat4x3::length() == 4, "GLM: Failed constexpr");
#endif

	return 0;
}

int main()
{
	int Error = 0;

	Error += cast::test();
	Error += test_ctr();
	Error += test_operators();
	Error += test_size();
	Error += test_constexpr();

	return Error;
}


