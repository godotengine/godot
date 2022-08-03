#include <glm/gtc/constants.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/matrix_relational.hpp>
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

static int test_mat3x3()
{
	glm::dmat3 Mat0(
		glm::dvec3(0.6f, 0.2f, 0.3f),
		glm::dvec3(0.2f, 0.7f, 0.5f),
		glm::dvec3(0.3f, 0.5f, 0.7f));
	glm::dmat3 Inv0 = glm::inverse(Mat0);
	glm::dmat3 Res0 = Mat0 * Inv0;

	return glm::all(glm::equal(Res0, glm::dmat3(1.0), 0.01)) ? 0 : 1;
}

static int test_operators()
{
	glm::mat3x3 l(1.0f);
	glm::mat3x3 m(1.0f);
	glm::vec3 u(1.0f);
	glm::vec3 v(1.0f);
	float x = 1.0f;
	glm::vec3 a = m * u;
	glm::vec3 b = v * m;
	glm::mat3x3 n = x / m;
	glm::mat3x3 o = m / x;
	glm::mat3x3 p = x * m;
	glm::mat3x3 q = m * x;
	bool R = glm::any(glm::notEqual(m, q, glm::epsilon<float>()));
	bool S = glm::all(glm::equal(m, l, glm::epsilon<float>()));

	return (S && !R) ? 0 : 1;
}

static int test_inverse()
{
	int Error(0);

	{
		glm::mat3 const Matrix(
			glm::vec3(0.6f, 0.2f, 0.3f), 
			glm::vec3(0.2f, 0.7f, 0.5f), 
			glm::vec3(0.3f, 0.5f, 0.7f));
		glm::mat3 const Inverse = glm::inverse(Matrix);
		glm::mat3 const Identity = Matrix * Inverse;

		Error += glm::all(glm::equal(Identity[0], glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.01f))) ? 0 : 1;
		Error += glm::all(glm::equal(Identity[1], glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.01f))) ? 0 : 1;
		Error += glm::all(glm::equal(Identity[2], glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.01f))) ? 0 : 1;
	}

	{
		glm::mat3 const Matrix(
			glm::vec3(0.6f, 0.2f, 0.3f), 
			glm::vec3(0.2f, 0.7f, 0.5f), 
			glm::vec3(0.3f, 0.5f, 0.7f));
		glm::mat3 const Identity = Matrix / Matrix;

		Error += glm::all(glm::equal(Identity[0], glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.01f))) ? 0 : 1;
		Error += glm::all(glm::equal(Identity[1], glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.01f))) ? 0 : 1;
		Error += glm::all(glm::equal(Identity[2], glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.01f))) ? 0 : 1;
	}

	return Error;
}

static int test_ctr()
{
	int Error(0);

#if(GLM_HAS_INITIALIZER_LISTS)
	glm::mat3x3 m0(
		glm::vec3(0, 1, 2),
		glm::vec3(3, 4, 5),
		glm::vec3(6, 7, 8));
	
	glm::mat3x3 m1{0, 1, 2, 3, 4, 5, 6, 7, 8};
	
	glm::mat3x3 m2{
		{0, 1, 2},
		{3, 4, 5},
		{6, 7, 8}};

	Error += glm::all(glm::equal(m0, m2, glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(m1, m2, glm::epsilon<float>())) ? 0 : 1;

	std::vector<glm::mat3x3> v1{
		{0, 1, 2, 3, 4, 5, 6, 7, 8},
		{0, 1, 2, 3, 4, 5, 6, 7, 8}
	};
	
	std::vector<glm::mat3x3> v2{
		{
			{ 0, 1, 2},
			{ 3, 4, 5},
			{ 6, 7, 8}
		},
		{
			{ 0, 1, 2},
			{ 3, 4, 5},
			{ 6, 7, 8}
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
		glm::mat3x3 B(A);
		glm::mat3x3 Identity(1.0f);

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

	Error += 36 == sizeof(glm::mat3x3) ? 0 : 1;
	Error += 72 == sizeof(glm::dmat3x3) ? 0 : 1;
	Error += glm::mat3x3().length() == 3 ? 0 : 1;
	Error += glm::dmat3x3().length() == 3 ? 0 : 1;
	Error += glm::mat3x3::length() == 3 ? 0 : 1;
	Error += glm::dmat3x3::length() == 3 ? 0 : 1;

	return Error;
}

static int test_constexpr()
{
#if GLM_HAS_CONSTEXPR
	static_assert(glm::mat3x3::length() == 3, "GLM: Failed constexpr");

	constexpr glm::mat3x3 const Z(0.0f);
#endif

	return 0;
}

int main()
{
	int Error = 0;

	Error += cast::test();
	Error += test_ctr();
	Error += test_mat3x3();
	Error += test_operators();
	Error += test_inverse();
	Error += test_size();
	Error += test_constexpr();

	return Error;
}

