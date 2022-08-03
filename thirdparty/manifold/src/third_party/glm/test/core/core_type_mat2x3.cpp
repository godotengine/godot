#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/matrix_relational.hpp>
#include <glm/gtc/constants.hpp>
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
	glm::mat2x3 l(1.0f);
	glm::mat2x3 m(1.0f);
	glm::vec2 u(1.0f);
	glm::vec3 v(1.0f);
	float x = 1.0f;
	glm::vec3 a = m * u;
	glm::vec2 b = v * m;
	glm::mat2x3 n = x / m;
	glm::mat2x3 o = m / x;
	glm::mat2x3 p = x * m;
	glm::mat2x3 q = m * x;
	bool R = glm::any(glm::notEqual(m, q, glm::epsilon<float>()));
	bool S = glm::all(glm::equal(m, l, glm::epsilon<float>()));

	return (S && !R) ? 0 : 1;
}

int test_ctr()
{
	int Error(0);

#if GLM_HAS_INITIALIZER_LISTS
	glm::mat2x3 m0(
		glm::vec3(0, 1, 2),
		glm::vec3(3, 4, 5));
	
	glm::mat2x3 m1{0, 1, 2, 3, 4, 5};
	
	glm::mat2x3 m2{
		{0, 1, 2},
		{3, 4, 5}};
	
	Error += glm::all(glm::equal(m0, m2, glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(m1, m2, glm::epsilon<float>())) ? 0 : 1;
	
	std::vector<glm::mat2x3> v1{
		{0, 1, 2, 3, 4, 5},
		{0, 1, 2, 3, 4, 5}
	};
	
	std::vector<glm::mat2x3> v2{
		{
			{ 0, 1, 2},
			{ 4, 5, 6}
		},
		{
			{ 0, 1, 2},
			{ 4, 5, 6}
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
		glm::mat2x3 B(A);
		glm::mat2x3 Identity(1.0f);

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

int test_size()
{
	int Error = 0;

	Error += 24 == sizeof(glm::mat2x3) ? 0 : 1;
	Error += 48 == sizeof(glm::dmat2x3) ? 0 : 1;
	Error += glm::mat2x3().length() == 2 ? 0 : 1;
	Error += glm::dmat2x3().length() == 2 ? 0 : 1;
	Error += glm::mat2x3::length() == 2 ? 0 : 1;
	Error += glm::dmat2x3::length() == 2 ? 0 : 1;

	return Error;
}

int test_constexpr()
{
#if GLM_HAS_CONSTEXPR
	static_assert(glm::mat2x3::length() == 2, "GLM: Failed constexpr");
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
