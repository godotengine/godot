#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/matrix.hpp>
#include <glm/vector_relational.hpp>
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

int test_operators()
{
	glm::mat2x2 l(1.0f);
	glm::mat2x2 m(1.0f);
	glm::vec2 u(1.0f);
	glm::vec2 v(1.0f);
	float x = 1.0f;
	glm::vec2 a = m * u;
	glm::vec2 b = v * m;
	glm::mat2x2 n = x / m;
	glm::mat2x2 o = m / x;
	glm::mat2x2 p = x * m;
	glm::mat2x2 q = m * x;
	bool R = glm::any(glm::notEqual(m, q, glm::epsilon<float>()));
	bool S = glm::all(glm::equal(m, l, glm::epsilon<float>()));

	return (S && !R) ? 0 : 1;
}

int test_inverse()
{
	int Error(0);

	{
		glm::mat2 const Matrix(1, 2, 3, 4);
		glm::mat2 const Inverse = glm::inverse(Matrix);
		glm::mat2 const Identity = Matrix * Inverse;

		Error += glm::all(glm::equal(Identity[0], glm::vec2(1.0f, 0.0f), glm::vec2(0.01f))) ? 0 : 1;
		Error += glm::all(glm::equal(Identity[1], glm::vec2(0.0f, 1.0f), glm::vec2(0.01f))) ? 0 : 1;
	}

	{
		glm::mat2 const Matrix(1, 2, 3, 4);
		glm::mat2 const Identity = Matrix / Matrix;

		Error += glm::all(glm::equal(Identity[0], glm::vec2(1.0f, 0.0f), glm::vec2(0.01f))) ? 0 : 1;
		Error += glm::all(glm::equal(Identity[1], glm::vec2(0.0f, 1.0f), glm::vec2(0.01f))) ? 0 : 1;
	}

	return Error;
}

int test_ctr()
{
	int Error = 0;
	
	{
		glm::mediump_mat2x2 const A(1.0f);
		glm::highp_mat2x2 const B(A);
		glm::mediump_mat2x2 const C(B);

		Error += glm::all(glm::equal(A, C, glm::epsilon<float>())) ? 0 : 1;
	}

#if GLM_HAS_INITIALIZER_LISTS
	glm::mat2x2 m0(
		glm::vec2(0, 1),
		glm::vec2(2, 3));

	glm::mat2x2 m1{0, 1, 2, 3};

	glm::mat2x2 m2{
		{0, 1},
		{2, 3}};

	Error += glm::all(glm::equal(m0, m2, glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(m1, m2, glm::epsilon<float>())) ? 0 : 1;

	std::vector<glm::mat2x2> v1{
		{0, 1, 2, 3},
		{0, 1, 2, 3}
	};

	std::vector<glm::mat2x2> v2{
		{
			{ 0, 1},
			{ 4, 5}
		},
		{
			{ 0, 1},
			{ 4, 5}
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
		glm::mat2 B(A);
		glm::mat2 Identity(1.0f);

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

	Error += 16 == sizeof(glm::mat2x2) ? 0 : 1;
	Error += 32 == sizeof(glm::dmat2x2) ? 0 : 1;
	Error += glm::mat2x2().length() == 2 ? 0 : 1;
	Error += glm::dmat2x2().length() == 2 ? 0 : 1;
	Error += glm::mat2x2::length() == 2 ? 0 : 1;
	Error += glm::dmat2x2::length() == 2 ? 0 : 1;

	return Error;
}

int test_constexpr()
{
#if GLM_HAS_CONSTEXPR
	static_assert(glm::mat2x2::length() == 2, "GLM: Failed constexpr");
#endif

	return 0;
}

int main()
{
	int Error = 0;

	Error += cast::test();
	Error += test_ctr();
	Error += test_operators();
	Error += test_inverse();
	Error += test_size();
	Error += test_constexpr();

	return Error;
}
