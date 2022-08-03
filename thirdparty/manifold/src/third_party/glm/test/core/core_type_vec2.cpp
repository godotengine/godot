#define GLM_FORCE_SWIZZLE
#include <glm/gtc/vec1.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/ext/vector_float1.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/vector_relational.hpp>
#include <glm/vec2.hpp>
#include <vector>
#if GLM_HAS_TRIVIAL_QUERIES
#	include <type_traits>
#endif

static glm::ivec2 g1;
static glm::ivec2 g2(1);
static glm::ivec2 g3(1, 1);

static int test_operators()
{
	int Error = 0;

	{
		glm::ivec2 A(1);
		glm::ivec2 B(1);
		Error += A != B ? 1 : 0;
		Error += A == B ? 0 : 1;
	}

	{
		glm::vec2 A(1.0f);
		glm::vec2 C = A + 1.0f;
		A += 1.0f;
		Error += glm::all(glm::equal(A, glm::vec2(2.0f), glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, C, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::vec2 A(1.0f);
		glm::vec2 B(2.0f,-1.0f);
		glm::vec2 C = A + B;
		A += B;
		Error += glm::all(glm::equal(A, glm::vec2(3.0f, 0.0f), glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, C, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::vec2 A(1.0f);
		glm::vec2 C = A - 1.0f;
		A -= 1.0f;
		Error += glm::all(glm::equal(A, glm::vec2(0.0f), glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, C, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::vec2 A(1.0f);
		glm::vec2 B(2.0f,-1.0f);
		glm::vec2 C = A - B;
		A -= B;
		Error += glm::all(glm::equal(A, glm::vec2(-1.0f, 2.0f), glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, C, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::vec2 A(1.0f);
		glm::vec2 C = A * 2.0f;
		A *= 2.0f;
		Error += glm::all(glm::equal(A, glm::vec2(2.0f), glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, C, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::vec2 A(2.0f);
		glm::vec2 B(2.0f);
		glm::vec2 C = A / B;
		A /= B;
		Error += glm::all(glm::equal(A, glm::vec2(1.0f), glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(A, C, glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::vec2 A(1.0f, 2.0f);
		glm::vec2 B(4.0f, 5.0f);

		glm::vec2 C = A + B;
		Error += glm::all(glm::equal(C, glm::vec2(5, 7), glm::epsilon<float>())) ? 0 : 1;

		glm::vec2 D = B - A;
		Error += glm::all(glm::equal(D, glm::vec2(3, 3), glm::epsilon<float>())) ? 0 : 1;

		glm::vec2 E = A * B;
		Error += glm::all(glm::equal(E, glm::vec2(4, 10), glm::epsilon<float>())) ? 0 : 1;

		glm::vec2 F = B / A;
		Error += glm::all(glm::equal(F, glm::vec2(4, 2.5), glm::epsilon<float>())) ? 0 : 1;

		glm::vec2 G = A + 1.0f;
		Error += glm::all(glm::equal(G, glm::vec2(2, 3), glm::epsilon<float>())) ? 0 : 1;

		glm::vec2 H = B - 1.0f;
		Error += glm::all(glm::equal(H, glm::vec2(3, 4), glm::epsilon<float>())) ? 0 : 1;

		glm::vec2 I = A * 2.0f;
		Error += glm::all(glm::equal(I, glm::vec2(2, 4), glm::epsilon<float>())) ? 0 : 1;

		glm::vec2 J = B / 2.0f;
		Error += glm::all(glm::equal(J, glm::vec2(2, 2.5), glm::epsilon<float>())) ? 0 : 1;

		glm::vec2 K = 1.0f + A;
		Error += glm::all(glm::equal(K, glm::vec2(2, 3), glm::epsilon<float>())) ? 0 : 1;

		glm::vec2 L = 1.0f - B;
		Error += glm::all(glm::equal(L, glm::vec2(-3, -4), glm::epsilon<float>())) ? 0 : 1;

		glm::vec2 M = 2.0f * A;
		Error += glm::all(glm::equal(M, glm::vec2(2, 4), glm::epsilon<float>())) ? 0 : 1;

		glm::vec2 N = 2.0f / B;
		Error += glm::all(glm::equal(N, glm::vec2(0.5, 2.0 / 5.0), glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::vec2 A(1.0f, 2.0f);
		glm::vec2 B(4.0f, 5.0f);

		A += B;
		Error += glm::all(glm::equal(A, glm::vec2(5, 7), glm::epsilon<float>())) ? 0 : 1;

		A += 1.0f;
		Error += glm::all(glm::equal(A, glm::vec2(6, 8), glm::epsilon<float>())) ? 0 : 1;
	}
	{
		glm::ivec2 A(1.0f, 2.0f);
		glm::ivec2 B(4.0f, 5.0f);

		B -= A;
		Error += B == glm::ivec2(3, 3) ? 0 : 1;

		B -= 1.0f;
		Error += B == glm::ivec2(2, 2) ? 0 : 1;
	}
	{
		glm::ivec2 A(1.0f, 2.0f);
		glm::ivec2 B(4.0f, 5.0f);

		A *= B;
		Error += A == glm::ivec2(4, 10) ? 0 : 1;

		A *= 2;
		Error += A == glm::ivec2(8, 20) ? 0 : 1;
	}
	{
		glm::ivec2 A(1.0f, 2.0f);
		glm::ivec2 B(4.0f, 16.0f);

		B /= A;
		Error += B == glm::ivec2(4, 8) ? 0 : 1;

		B /= 2.0f;
		Error += B == glm::ivec2(2, 4) ? 0 : 1;
	}
	{
		glm::ivec2 B(2);

		B /= B.y;
		Error += B == glm::ivec2(1) ? 0 : 1;
	}

	{
		glm::ivec2 A(1.0f, 2.0f);
		glm::ivec2 B = -A;
		Error += B == glm::ivec2(-1.0f, -2.0f) ? 0 : 1;
	}

	{
		glm::ivec2 A(1.0f, 2.0f);
		glm::ivec2 B = --A;
		Error += B == glm::ivec2(0.0f, 1.0f) ? 0 : 1;
	}

	{
		glm::ivec2 A(1.0f, 2.0f);
		glm::ivec2 B = A--;
		Error += B == glm::ivec2(1.0f, 2.0f) ? 0 : 1;
		Error += A == glm::ivec2(0.0f, 1.0f) ? 0 : 1;
	}

	{
		glm::ivec2 A(1.0f, 2.0f);
		glm::ivec2 B = ++A;
		Error += B == glm::ivec2(2.0f, 3.0f) ? 0 : 1;
	}

	{
		glm::ivec2 A(1.0f, 2.0f);
		glm::ivec2 B = A++;
		Error += B == glm::ivec2(1.0f, 2.0f) ? 0 : 1;
		Error += A == glm::ivec2(2.0f, 3.0f) ? 0 : 1;
	}

	return Error;
}

static int test_ctor()
{
	int Error = 0;

	{
		glm::ivec2 A(1);
		glm::ivec2 B(A);
		Error += A == B ? 0 : 1;
	}

#	if GLM_HAS_TRIVIAL_QUERIES
	//	Error += std::is_trivially_default_constructible<glm::vec2>::value ? 0 : 1;
	//	Error += std::is_trivially_copy_assignable<glm::vec2>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::vec2>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::dvec2>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::ivec2>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::uvec2>::value ? 0 : 1;

		Error += std::is_copy_constructible<glm::vec2>::value ? 0 : 1;
#	endif

#if GLM_HAS_INITIALIZER_LISTS
	{
		glm::vec2 a{ 0, 1 };
		std::vector<glm::vec2> v = {
			{0, 1},
			{4, 5},
			{8, 9}};
	}

	{
		glm::dvec2 a{ 0, 1 };
		std::vector<glm::dvec2> v = {
			{0, 1},
			{4, 5},
			{8, 9}};
	}
#endif

	{
		glm::vec2 A = glm::vec2(2.0f);
		glm::vec2 B = glm::vec2(2.0f, 3.0f);
		glm::vec2 C = glm::vec2(2.0f, 3.0);
		//glm::vec2 D = glm::dvec2(2.0); // Build error TODO: What does the specification says?
		glm::vec2 E(glm::dvec2(2.0));
		glm::vec2 F(glm::ivec2(2));
	}

	{
		glm::vec1 const R(1.0f);
		glm::vec1 const S(2.0f);
		glm::vec2 const O(1.0f, 2.0f);

		glm::vec2 const A(R);
		glm::vec2 const B(1.0f);
		Error += glm::all(glm::equal(A, B, 0.0001f)) ? 0 : 1;

		glm::vec2 const C(R, S);
		Error += glm::all(glm::equal(C, O, 0.0001f)) ? 0 : 1;

		glm::vec2 const D(R, 2.0f);
		Error += glm::all(glm::equal(D, O, 0.0001f)) ? 0 : 1;

		glm::vec2 const E(1.0f, S);
		Error += glm::all(glm::equal(E, O, 0.0001f)) ? 0 : 1;
	}

	{
		glm::vec1 const R(1.0f);
		glm::dvec1 const S(2.0);
		glm::vec2 const O(1.0, 2.0);

		glm::vec2 const A(R);
		glm::vec2 const B(1.0);
		Error += glm::all(glm::equal(A, B, 0.0001f)) ? 0 : 1;

		glm::vec2 const C(R, S);
		Error += glm::all(glm::equal(C, O, 0.0001f)) ? 0 : 1;

		glm::vec2 const D(R, 2.0);
		Error += glm::all(glm::equal(D, O, 0.0001f)) ? 0 : 1;

		glm::vec2 const E(1.0, S);
		Error += glm::all(glm::equal(E, O, 0.0001f)) ? 0 : 1;
	}

	return Error;
}

static int test_size()
{
	int Error = 0;

	Error += sizeof(glm::vec2) == sizeof(glm::mediump_vec2) ? 0 : 1;
	Error += 8 == sizeof(glm::mediump_vec2) ? 0 : 1;
	Error += sizeof(glm::dvec2) == sizeof(glm::highp_dvec2) ? 0 : 1;
	Error += 16 == sizeof(glm::highp_dvec2) ? 0 : 1;
	Error += glm::vec2().length() == 2 ? 0 : 1;
	Error += glm::dvec2().length() == 2 ? 0 : 1;
	Error += glm::vec2::length() == 2 ? 0 : 1;
	Error += glm::dvec2::length() == 2 ? 0 : 1;

	GLM_CONSTEXPR std::size_t Length = glm::vec2::length();
	Error += Length == 2 ? 0 : 1;

	return Error;
}

static int test_operator_increment()
{
	int Error = 0;

	glm::ivec2 v0(1);
	glm::ivec2 v1(v0);
	glm::ivec2 v2(v0);
	glm::ivec2 v3 = ++v1;
	glm::ivec2 v4 = v2++;

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

static int test_constexpr()
{
#if GLM_HAS_CONSTEXPR
	static_assert(glm::vec2::length() == 2, "GLM: Failed constexpr");
	static_assert(glm::vec2(1.0f).x > 0.0f, "GLM: Failed constexpr");
	static_assert(glm::vec2(1.0f, -1.0f).x > 0.0f, "GLM: Failed constexpr");
	static_assert(glm::vec2(1.0f, -1.0f).y < 0.0f, "GLM: Failed constexpr");
#endif

	return 0;
}

static int test_swizzle()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	{
		glm::vec2 A = glm::vec2(1.0f, 2.0f);
		glm::vec2 B = A.xy;
		glm::vec2 C(A.xy);
		glm::vec2 D(A.xy());

		Error += glm::all(glm::equal(A, B, 0.0001f)) ? 0 : 1;
		Error += glm::all(glm::equal(A, C, 0.0001f)) ? 0 : 1;
		Error += glm::all(glm::equal(A, D, 0.0001f)) ? 0 : 1;
	}
#	endif//GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION
	{
		glm::vec2 A = glm::vec2(1.0f, 2.0f);
		glm::vec2 B = A.xy();
		glm::vec2 C(A.xy());

		Error += glm::all(glm::equal(A, B, 0.0001f)) ? 0 : 1;
		Error += glm::all(glm::equal(A, C, 0.0001f)) ? 0 : 1;
	}
#	endif//GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_size();
	Error += test_ctor();
	Error += test_operators();
	Error += test_operator_increment();
	Error += test_swizzle();
	Error += test_constexpr();

	return Error;
}
