#ifndef GLM_FORCE_PURE
#	define GLM_FORCE_PURE
#endif//GLM_FORCE_PURE
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_SWIZZLE
#include <glm/ext/vector_relational.hpp>
#include <glm/vector_relational.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <ctime>
#include <vector>

static int test_vec4_ctor()
{
	int Error = 0;

	{
		glm::ivec4 A(1, 2, 3, 4);
		glm::ivec4 B(A);
		Error += glm::all(glm::equal(A, B)) ? 0 : 1;
	}

#	if GLM_HAS_TRIVIAL_QUERIES
	//	Error += std::is_trivially_default_constructible<glm::vec4>::value ? 0 : 1;
	//	Error += std::is_trivially_copy_assignable<glm::vec4>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::vec4>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::dvec4>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::ivec4>::value ? 0 : 1;
		Error += std::is_trivially_copyable<glm::uvec4>::value ? 0 : 1;

		Error += std::is_copy_constructible<glm::vec4>::value ? 0 : 1;
#	endif

#if GLM_HAS_INITIALIZER_LISTS
	{
		glm::vec4 a{ 0, 1, 2, 3 };
		std::vector<glm::vec4> v = {
			{0, 1, 2, 3},
			{4, 5, 6, 7},
			{8, 9, 0, 1}};
	}

	{
		glm::dvec4 a{ 0, 1, 2, 3 };
		std::vector<glm::dvec4> v = {
			{0, 1, 2, 3},
			{4, 5, 6, 7},
			{8, 9, 0, 1}};
	}
#endif

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	{
		glm::ivec4 A = glm::vec4(1.0f, 2.0f, 3.0f, 4.0f);
		glm::ivec4 B = A.xyzw;
		glm::ivec4 C(A.xyzw);
		glm::ivec4 D(A.xyzw());
		glm::ivec4 E(A.x, A.yzw);
		glm::ivec4 F(A.x, A.yzw());
		glm::ivec4 G(A.xyz, A.w);
		glm::ivec4 H(A.xyz(), A.w);
		glm::ivec4 I(A.xy, A.zw);
		glm::ivec4 J(A.xy(), A.zw());
		glm::ivec4 K(A.x, A.y, A.zw);
		glm::ivec4 L(A.x, A.yz, A.w);
		glm::ivec4 M(A.xy, A.z, A.w);

		Error += glm::all(glm::equal(A, B)) ? 0 : 1;
		Error += glm::all(glm::equal(A, C)) ? 0 : 1;
		Error += glm::all(glm::equal(A, D)) ? 0 : 1;
		Error += glm::all(glm::equal(A, E)) ? 0 : 1;
		Error += glm::all(glm::equal(A, F)) ? 0 : 1;
		Error += glm::all(glm::equal(A, G)) ? 0 : 1;
		Error += glm::all(glm::equal(A, H)) ? 0 : 1;
		Error += glm::all(glm::equal(A, I)) ? 0 : 1;
		Error += glm::all(glm::equal(A, J)) ? 0 : 1;
		Error += glm::all(glm::equal(A, K)) ? 0 : 1;
		Error += glm::all(glm::equal(A, L)) ? 0 : 1;
		Error += glm::all(glm::equal(A, M)) ? 0 : 1;
	}
#	endif

#	if GLM_CONFIG_SWIZZLE
	{
		glm::ivec4 A = glm::vec4(1.0f, 2.0f, 3.0f, 4.0f);
		glm::ivec4 B = A.xyzw();
		glm::ivec4 C(A.xyzw());
		glm::ivec4 D(A.xyzw());
		glm::ivec4 E(A.x, A.yzw());
		glm::ivec4 F(A.x, A.yzw());
		glm::ivec4 G(A.xyz(), A.w);
		glm::ivec4 H(A.xyz(), A.w);
		glm::ivec4 I(A.xy(), A.zw());
		glm::ivec4 J(A.xy(), A.zw());
		glm::ivec4 K(A.x, A.y, A.zw());
		glm::ivec4 L(A.x, A.yz(), A.w);
		glm::ivec4 M(A.xy(), A.z, A.w);

		Error += glm::all(glm::equal(A, B)) ? 0 : 1;
		Error += glm::all(glm::equal(A, C)) ? 0 : 1;
		Error += glm::all(glm::equal(A, D)) ? 0 : 1;
		Error += glm::all(glm::equal(A, E)) ? 0 : 1;
		Error += glm::all(glm::equal(A, F)) ? 0 : 1;
		Error += glm::all(glm::equal(A, G)) ? 0 : 1;
		Error += glm::all(glm::equal(A, H)) ? 0 : 1;
		Error += glm::all(glm::equal(A, I)) ? 0 : 1;
		Error += glm::all(glm::equal(A, J)) ? 0 : 1;
		Error += glm::all(glm::equal(A, K)) ? 0 : 1;
		Error += glm::all(glm::equal(A, L)) ? 0 : 1;
		Error += glm::all(glm::equal(A, M)) ? 0 : 1;
	}
#	endif//GLM_CONFIG_SWIZZLE

	{
		glm::ivec4 A(1);
		glm::ivec4 B(1, 1, 1, 1);
		
		Error += A == B ? 0 : 1;
	}
	
	{
		std::vector<glm::ivec4> Tests;
		Tests.push_back(glm::ivec4(glm::ivec2(1, 2), 3, 4));
		Tests.push_back(glm::ivec4(1, glm::ivec2(2, 3), 4));
		Tests.push_back(glm::ivec4(1, 2, glm::ivec2(3, 4)));
		Tests.push_back(glm::ivec4(glm::ivec3(1, 2, 3), 4));
		Tests.push_back(glm::ivec4(1, glm::ivec3(2, 3, 4)));
		Tests.push_back(glm::ivec4(glm::ivec2(1, 2), glm::ivec2(3, 4)));
		Tests.push_back(glm::ivec4(1, 2, 3, 4));
		Tests.push_back(glm::ivec4(glm::ivec4(1, 2, 3, 4)));
		
		for(std::size_t i = 0; i < Tests.size(); ++i)
			Error += Tests[i] == glm::ivec4(1, 2, 3, 4) ? 0 : 1;
	}
	
	return Error;
}

static int test_bvec4_ctor()
{
	int Error = 0;

	glm::bvec4 const A(true);
	glm::bvec4 const B(true);
	glm::bvec4 const C(false);
	glm::bvec4 const D = A && B;
	glm::bvec4 const E = A && C;
	glm::bvec4 const F = A || C;

	Error += D == glm::bvec4(true) ? 0 : 1;
	Error += E == glm::bvec4(false) ? 0 : 1;
	Error += F == glm::bvec4(true) ? 0 : 1;

	bool const G = A == C;
	bool const H = A != C;

	Error += !G ? 0 : 1;
	Error += H ? 0 : 1;

	return Error;
}

static int test_vec4_operators()
{
	int Error = 0;
	
	{
		glm::ivec4 A(1);
		glm::ivec4 B(1);
		bool R = A != B;
		bool S = A == B;

		Error += (S && !R) ? 0 : 1;
	}

	{
		glm::vec4 const A(1.0f, 2.0f, 3.0f, 4.0f);
		glm::vec4 const B(4.0f, 5.0f, 6.0f, 7.0f);

		glm::vec4 const C = A + B;
		Error += glm::all(glm::equal(C, glm::vec4(5, 7, 9, 11), 0.001f)) ? 0 : 1;

		glm::vec4 D = B - A;
		Error += glm::all(glm::equal(D, glm::vec4(3, 3, 3, 3), 0.001f)) ? 0 : 1;

		glm::vec4 E = A * B;
		Error += glm::all(glm::equal(E, glm::vec4(4, 10, 18, 28), 0.001f)) ? 0 : 1;

		glm::vec4 F = B / A;
		Error += glm::all(glm::equal(F, glm::vec4(4, 2.5, 2, 7.0f / 4.0f), 0.001f)) ? 0 : 1;

		glm::vec4 G = A + 1.0f;
		Error += glm::all(glm::equal(G, glm::vec4(2, 3, 4, 5), 0.001f)) ? 0 : 1;

		glm::vec4 H = B - 1.0f;
		Error += glm::all(glm::equal(H, glm::vec4(3, 4, 5, 6), 0.001f)) ? 0 : 1;

		glm::vec4 I = A * 2.0f;
		Error += glm::all(glm::equal(I, glm::vec4(2, 4, 6, 8), 0.001f)) ? 0 : 1;

		glm::vec4 J = B / 2.0f;
		Error += glm::all(glm::equal(J, glm::vec4(2, 2.5, 3, 3.5), 0.001f)) ? 0 : 1;

		glm::vec4 K = 1.0f + A;
		Error += glm::all(glm::equal(K, glm::vec4(2, 3, 4, 5), 0.001f)) ? 0 : 1;

		glm::vec4 L = 1.0f - B;
		Error += glm::all(glm::equal(L, glm::vec4(-3, -4, -5, -6), 0.001f)) ? 0 : 1;

		glm::vec4 M = 2.0f * A;
		Error += glm::all(glm::equal(M, glm::vec4(2, 4, 6, 8), 0.001f)) ? 0 : 1;

		glm::vec4 const N = 2.0f / B;
		Error += glm::all(glm::equal(N, glm::vec4(0.5, 2.0 / 5.0, 2.0 / 6.0, 2.0 / 7.0), 0.0001f)) ? 0 : 1;
	}

	{
		glm::ivec4 A(1.0f, 2.0f, 3.0f, 4.0f);
		glm::ivec4 B(4.0f, 5.0f, 6.0f, 7.0f);

		A += B;
		Error += A == glm::ivec4(5, 7, 9, 11) ? 0 : 1;

		A += 1;
		Error += A == glm::ivec4(6, 8, 10, 12) ? 0 : 1;
	}
	{
		glm::ivec4 A(1.0f, 2.0f, 3.0f, 4.0f);
		glm::ivec4 B(4.0f, 5.0f, 6.0f, 7.0f);

		B -= A;
		Error += B == glm::ivec4(3, 3, 3, 3) ? 0 : 1;

		B -= 1;
		Error += B == glm::ivec4(2, 2, 2, 2) ? 0 : 1;
	}
	{
		glm::ivec4 A(1.0f, 2.0f, 3.0f, 4.0f);
		glm::ivec4 B(4.0f, 5.0f, 6.0f, 7.0f);

		A *= B;
		Error += A == glm::ivec4(4, 10, 18, 28) ? 0 : 1;

		A *= 2;
		Error += A == glm::ivec4(8, 20, 36, 56) ? 0 : 1;
	}
	{
		glm::ivec4 A(1.0f, 2.0f, 3.0f, 4.0f);
		glm::ivec4 B(4.0f, 4.0f, 6.0f, 8.0f);

		B /= A;
		Error += B == glm::ivec4(4, 2, 2, 2) ? 0 : 1;

		B /= 2;
		Error += B == glm::ivec4(2, 1, 1, 1) ? 0 : 1;
	}
	{
		glm::ivec4 B(2);

		B /= B.y;
		Error += B == glm::ivec4(1) ? 0 : 1;
	}

	{
		glm::ivec4 A(1.0f, 2.0f, 3.0f, 4.0f);
		glm::ivec4 B = -A;
		Error += B == glm::ivec4(-1.0f, -2.0f, -3.0f, -4.0f) ? 0 : 1;
	}

	{
		glm::ivec4 A(1.0f, 2.0f, 3.0f, 4.0f);
		glm::ivec4 B = --A;
		Error += B == glm::ivec4(0.0f, 1.0f, 2.0f, 3.0f) ? 0 : 1;
	}

	{
		glm::ivec4 A(1.0f, 2.0f, 3.0f, 4.0f);
		glm::ivec4 B = A--;
		Error += B == glm::ivec4(1.0f, 2.0f, 3.0f, 4.0f) ? 0 : 1;
		Error += A == glm::ivec4(0.0f, 1.0f, 2.0f, 3.0f) ? 0 : 1;
	}

	{
		glm::ivec4 A(1.0f, 2.0f, 3.0f, 4.0f);
		glm::ivec4 B = ++A;
		Error += B == glm::ivec4(2.0f, 3.0f, 4.0f, 5.0f) ? 0 : 1;
	}

	{
		glm::ivec4 A(1.0f, 2.0f, 3.0f, 4.0f);
		glm::ivec4 B = A++;
		Error += B == glm::ivec4(1.0f, 2.0f, 3.0f, 4.0f) ? 0 : 1;
		Error += A == glm::ivec4(2.0f, 3.0f, 4.0f, 5.0f) ? 0 : 1;
	}

	return Error;
}

static int test_vec4_equal()
{
	int Error = 0;

	{
		glm::uvec4 const A(1, 2, 3, 4);
		glm::uvec4 const B(1, 2, 3, 4);
		Error += A == B ? 0 : 1;
		Error += A != B ? 1 : 0;
	}

	{
		glm::ivec4 const A(1, 2, 3, 4);
		glm::ivec4 const B(1, 2, 3, 4);
		Error += A == B ? 0 : 1;
		Error += A != B ? 1 : 0;
	}

	return Error;
}

static int test_vec4_size()
{
	int Error = 0;

	Error += sizeof(glm::vec4) == sizeof(glm::lowp_vec4) ? 0 : 1;
	Error += sizeof(glm::vec4) == sizeof(glm::mediump_vec4) ? 0 : 1;
	Error += sizeof(glm::vec4) == sizeof(glm::highp_vec4) ? 0 : 1;
	Error += 16 == sizeof(glm::mediump_vec4) ? 0 : 1;
	Error += sizeof(glm::dvec4) == sizeof(glm::lowp_dvec4) ? 0 : 1;
	Error += sizeof(glm::dvec4) == sizeof(glm::mediump_dvec4) ? 0 : 1;
	Error += sizeof(glm::dvec4) == sizeof(glm::highp_dvec4) ? 0 : 1;
	Error += 32 == sizeof(glm::highp_dvec4) ? 0 : 1;
	Error += glm::vec4().length() == 4 ? 0 : 1;
	Error += glm::dvec4().length() == 4 ? 0 : 1;

	return Error;
}

static int test_vec4_swizzle_partial()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR

	glm::ivec4 A(1, 2, 3, 4);

	{
		glm::ivec4 B(A.xy, A.zw);
		Error += A == B ? 0 : 1;
	}
	{
		glm::ivec4 B(A.xy, 3, 4);
		Error += A == B ? 0 : 1;
	}
	{
		glm::ivec4 B(1, A.yz, 4);
		Error += A == B ? 0 : 1;
	}
	{
		glm::ivec4 B(1, 2, A.zw);
		Error += A == B ? 0 : 1;
	}

	{
		glm::ivec4 B(A.xyz, 4);
		Error += A == B ? 0 : 1;
	}
	{
		glm::ivec4 B(1, A.yzw);
		Error += A == B ? 0 : 1;
	}
#	endif

	return Error;
}

static int test_operator_increment()
{
	int Error(0);

	glm::ivec4 v0(1);
	glm::ivec4 v1(v0);
	glm::ivec4 v2(v0);
	glm::ivec4 v3 = ++v1;
	glm::ivec4 v4 = v2++;

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

static int test_vec4_simd()
{
	int Error = 0;

	glm::vec4 const a(std::clock(), std::clock(), std::clock(), std::clock());
	glm::vec4 const b(std::clock(), std::clock(), std::clock(), std::clock());

	glm::vec4 const c(b * a);
	glm::vec4 const d(a + c);

	Error += glm::all(glm::greaterThanEqual(d, glm::vec4(0))) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_vec4_ctor();
	Error += test_bvec4_ctor();
	Error += test_vec4_size();
	Error += test_vec4_operators();
	Error += test_vec4_equal();
	Error += test_vec4_swizzle_partial();
	Error += test_vec4_simd();
	Error += test_operator_increment();

	return Error;
}

