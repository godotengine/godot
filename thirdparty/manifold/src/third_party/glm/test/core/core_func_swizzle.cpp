#define GLM_FORCE_SWIZZLE
#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/glm.hpp>

static int test_ivec2_swizzle()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION
	{
		glm::ivec2 A(1, 2);
		glm::ivec2 B = A.yx();
		glm::ivec2 C = B.yx();

		Error += A != B ? 0 : 1;
		Error += A == C ? 0 : 1;
	}
#	endif//GLM_CONFIG_SWIZZLE

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	{
		glm::ivec2 A(1, 2);
		glm::ivec2 B = A.yx;
		glm::ivec2 C = A.yx;

		Error += A != B ? 0 : 1;
		Error += B == C ? 0 : 1;

		B.xy = B.yx;
		C.xy = C.yx;

		Error += B == C ? 0 : 1;

		glm::ivec2 D(0, 0);
		D.yx = A.xy;
		Error += A.yx() == D ? 0 : 1;

		glm::ivec2 E = A.yx;
		Error += E == D ? 0 : 1;
	}
#	endif//GLM_CONFIG_SWIZZLE

	return Error;
}

int test_ivec3_swizzle()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION
	{
		glm::ivec3 A(1, 2, 3);
		glm::ivec3 B = A.zyx();
		glm::ivec3 C = B.zyx();

		Error += A != B ? 0 : 1;
		Error += A == C ? 0 : 1;
	}
#	endif

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
	{
		glm::ivec3 const A(1, 2, 3);
		glm::ivec2 B = A.yx;
		glm::ivec2 C = A.yx;

		Error += A.yx() == B ? 0 : 1;
		Error += B == C ? 0 : 1;

		B.xy = B.yx;
		C.xy = C.yx;

		Error += B == C ? 0 : 1;

		glm::ivec2 D(0, 0);
		D.yx = A.xy;

		Error += A.yx() == D ? 0 : 1;

		glm::ivec2 E(0, 0);
		E.xy = A.xy();

		Error += E == A.xy() ? 0 : 1;
		Error += E.xy() == A.xy() ? 0 : 1;

		glm::ivec3 const F = A.xxx + A.xxx;
		Error += F == glm::ivec3(2) ? 0 : 1;

		glm::ivec3 const G = A.xxx - A.xxx;
		Error += G == glm::ivec3(0) ? 0 : 1;

		glm::ivec3 const H = A.xxx * A.xxx;
		Error += H == glm::ivec3(1) ? 0 : 1;

		glm::ivec3 const I = A.xxx / A.xxx;
		Error += I == glm::ivec3(1) ? 0 : 1;

		glm::ivec3 J(1, 2, 3);
		J.xyz += glm::ivec3(1);
		Error += J == glm::ivec3(2, 3, 4) ? 0 : 1;

		glm::ivec3 K(1, 2, 3);
		K.xyz += A.xyz;
		Error += K == glm::ivec3(2, 4, 6) ? 0 : 1;
	}
#	endif

	return Error;
}

int test_ivec4_swizzle()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION
	{
		glm::ivec4 A(1, 2, 3, 4);
		glm::ivec4 B = A.wzyx();
		glm::ivec4 C = B.wzyx();

		Error += A != B ? 0 : 1;
		Error += A == C ? 0 : 1;
	}
#	endif

	return Error;
}

int test_vec4_swizzle()
{
	int Error = 0;

#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR || GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION
	{
		glm::vec4 A(1, 2, 3, 4);
		glm::vec4 B = A.wzyx();
		glm::vec4 C = B.wzyx();

		Error += glm::any(glm::notEqual(A, B, 0.0001f)) ? 0 : 1;
		Error += glm::all(glm::equal(A, C, 0.0001f)) ? 0 : 1;

		float D = glm::dot(C.wzyx(), C.xyzw());
		Error += glm::equal(D, 20.f, 0.001f) ? 0 : 1;
	}
#	endif

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_ivec2_swizzle();
	Error += test_ivec3_swizzle();
	Error += test_ivec4_swizzle();
	Error += test_vec4_swizzle();

	return Error;
}



