#define GLM_FORCE_SWIZZLE
#include <glm/vector_relational.hpp>
#include <glm/gtc/vec1.hpp>
#include <vector>

static glm::vec1 g1;
static glm::vec1 g2(1);

static int test_vec1_operators()
{
	int Error(0);

	glm::ivec1 A(1);
	glm::ivec1 B(1);
	{
		bool R = A != B;
		bool S = A == B;

		Error += (S && !R) ? 0 : 1;
	}

	{
		A *= 1;
		B *= 1;
		A += 1;
		B += 1;

		bool R = A != B;
		bool S = A == B;

		Error += (S && !R) ? 0 : 1;
	}

	return Error;
}

static int test_vec1_ctor()
{
	int Error = 0;

#	if GLM_HAS_TRIVIAL_QUERIES
	//	Error += std::is_trivially_default_constructible<glm::vec1>::value ? 0 : 1;
	//	Error += std::is_trivially_copy_assignable<glm::vec1>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::vec1>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::dvec1>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::ivec1>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::uvec1>::value ? 0 : 1;

		Error += std::is_copy_constructible<glm::vec1>::value ? 0 : 1;
#	endif


	{
		glm::ivec1 A = glm::vec1(2.0f);

		glm::ivec1 E(glm::dvec1(2.0));
		Error += A == E ? 0 : 1;

		glm::ivec1 F(glm::ivec1(2));
		Error += A == F ? 0 : 1;
	}

	return Error;
}

static int test_vec1_size()
{
	int Error = 0;

	Error += sizeof(glm::vec1) == sizeof(glm::mediump_vec1) ? 0 : 1;
	Error += 4 == sizeof(glm::mediump_vec1) ? 0 : 1;
	Error += sizeof(glm::dvec1) == sizeof(glm::highp_dvec1) ? 0 : 1;
	Error += 8 == sizeof(glm::highp_dvec1) ? 0 : 1;
	Error += glm::vec1().length() == 1 ? 0 : 1;
	Error += glm::dvec1().length() == 1 ? 0 : 1;
	Error += glm::vec1::length() == 1 ? 0 : 1;
	Error += glm::dvec1::length() == 1 ? 0 : 1;

	GLM_CONSTEXPR std::size_t Length = glm::vec1::length();
	Error += Length == 1 ? 0 : 1;

	return Error;
}

static int test_vec1_operator_increment()
{
	int Error(0);

	glm::ivec1 v0(1);
	glm::ivec1 v1(v0);
	glm::ivec1 v2(v0);
	glm::ivec1 v3 = ++v1;
	glm::ivec1 v4 = v2++;

	Error += glm::all(glm::equal(v0, v4)) ? 0 : 1;
	Error += glm::all(glm::equal(v1, v2)) ? 0 : 1;
	Error += glm::all(glm::equal(v1, v3)) ? 0 : 1;

	int i0(1);
	int i1(i0);
	int i2(i0);
	int i3 = ++i1;
	int i4 = i2++;

	Error += i0 == i4 ? 0 : 1;
	Error += i1 == i2 ? 0 : 1;
	Error += i1 == i3 ? 0 : 1;

	return Error;
}

static int test_bvec1_ctor()
{
	int Error = 0;

	glm::bvec1 const A(true);
	glm::bvec1 const B(true);
	glm::bvec1 const C(false);
	glm::bvec1 const D = A && B;
	glm::bvec1 const E = A && C;
	glm::bvec1 const F = A || C;

	Error += D == glm::bvec1(true) ? 0 : 1;
	Error += E == glm::bvec1(false) ? 0 : 1;
	Error += F == glm::bvec1(true) ? 0 : 1;

	bool const G = A == C;
	bool const H = A != C;
	Error += !G ? 0 : 1;
	Error += H ? 0 : 1;

	return Error;
}

static int test_constexpr()
{
#if GLM_HAS_CONSTEXPR
	static_assert(glm::vec1::length() == 1, "GLM: Failed constexpr");
	static_assert(glm::vec1(1.0f).x > 0.0f, "GLM: Failed constexpr");
#endif

	return 0;
}

int main()
{
	int Error = 0;

	Error += test_vec1_size();
	Error += test_vec1_ctor();
	Error += test_bvec1_ctor();
	Error += test_vec1_operators();
	Error += test_vec1_operator_increment();
	Error += test_constexpr();
	
	return Error;
}
