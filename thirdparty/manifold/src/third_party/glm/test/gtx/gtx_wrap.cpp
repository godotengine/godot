#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/wrap.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/vector_relational.hpp>

namespace clamp
{
	int test()
	{
		int Error(0);

		float A = glm::clamp(0.5f);
		Error += glm::equal(A, 0.5f, 0.00001f) ? 0 : 1;

		float B = glm::clamp(0.0f);
		Error += glm::equal(B, 0.0f, 0.00001f) ? 0 : 1;

		float C = glm::clamp(1.0f);
		Error += glm::equal(C, 1.0f, 0.00001f) ? 0 : 1;

		float D = glm::clamp(-0.5f);
		Error += glm::equal(D, 0.0f, 0.00001f) ? 0 : 1;

		float E = glm::clamp(1.5f);
		Error += glm::equal(E, 1.0f, 0.00001f) ? 0 : 1;

		glm::vec2 K = glm::clamp(glm::vec2(0.5f));
		Error += glm::all(glm::equal(K, glm::vec2(0.5f), glm::vec2(0.00001f))) ? 0 : 1;

		glm::vec3 L = glm::clamp(glm::vec3(0.5f));
		Error += glm::all(glm::equal(L, glm::vec3(0.5f), glm::vec3(0.00001f))) ? 0 : 1;

		glm::vec4 M = glm::clamp(glm::vec4(0.5f));
		Error += glm::all(glm::equal(M, glm::vec4(0.5f), glm::vec4(0.00001f))) ? 0 : 1;

		glm::vec1 N = glm::clamp(glm::vec1(0.5f));
		Error += glm::all(glm::equal(N, glm::vec1(0.5f), glm::vec1(0.00001f))) ? 0 : 1;

		return Error;
	}
}//namespace clamp

namespace repeat
{
	int test()
	{
		int Error(0);

		float A = glm::repeat(0.5f);
		Error += glm::equal(A, 0.5f, 0.00001f) ? 0 : 1;

		float B = glm::repeat(0.0f);
		Error += glm::equal(B, 0.0f, 0.00001f) ? 0 : 1;

		float C = glm::repeat(1.0f);
		Error += glm::equal(C, 0.0f, 0.00001f) ? 0 : 1;

		float D = glm::repeat(-0.5f);
		Error += glm::equal(D, 0.5f, 0.00001f) ? 0 : 1;

		float E = glm::repeat(1.5f);
		Error += glm::equal(E, 0.5f, 0.00001f) ? 0 : 1;

		float F = glm::repeat(0.9f);
		Error += glm::equal(F, 0.9f, 0.00001f) ? 0 : 1;

		glm::vec2 K = glm::repeat(glm::vec2(0.5f));
		Error += glm::all(glm::equal(K, glm::vec2(0.5f), glm::vec2(0.00001f))) ? 0 : 1;

		glm::vec3 L = glm::repeat(glm::vec3(0.5f));
		Error += glm::all(glm::equal(L, glm::vec3(0.5f), glm::vec3(0.00001f))) ? 0 : 1;

		glm::vec4 M = glm::repeat(glm::vec4(0.5f));
		Error += glm::all(glm::equal(M, glm::vec4(0.5f), glm::vec4(0.00001f))) ? 0 : 1;

		glm::vec1 N = glm::repeat(glm::vec1(0.5f));
		Error += glm::all(glm::equal(N, glm::vec1(0.5f), glm::vec1(0.00001f))) ? 0 : 1;

		return Error;
	}
}//namespace repeat

namespace mirrorClamp
{
	int test()
	{
		int Error(0);

		float A = glm::mirrorClamp(0.5f);
		Error += glm::equal(A, 0.5f, 0.00001f) ? 0 : 1;

		float B = glm::mirrorClamp(0.0f);
		Error += glm::equal(B, 0.0f, 0.00001f) ? 0 : 1;

		float C = glm::mirrorClamp(1.1f);
		Error += glm::equal(C, 0.1f, 0.00001f) ? 0 : 1;

		float D = glm::mirrorClamp(-0.5f);
		Error += glm::equal(D, 0.5f, 0.00001f) ? 0 : 1;

		float E = glm::mirrorClamp(1.5f);
		Error += glm::equal(E, 0.5f, 0.00001f) ? 0 : 1;

		float F = glm::mirrorClamp(0.9f);
		Error += glm::equal(F, 0.9f, 0.00001f) ? 0 : 1;

		float G = glm::mirrorClamp(3.1f);
		Error += glm::equal(G, 0.1f, 0.00001f) ? 0 : 1;

		float H = glm::mirrorClamp(-3.1f);
		Error += glm::equal(H, 0.1f, 0.00001f) ? 0 : 1;

		float I = glm::mirrorClamp(-0.9f);
		Error += glm::equal(I, 0.9f, 0.00001f) ? 0 : 1;

		glm::vec2 K = glm::mirrorClamp(glm::vec2(0.5f));
		Error += glm::all(glm::equal(K, glm::vec2(0.5f), glm::vec2(0.00001f))) ? 0 : 1;

		glm::vec3 L = glm::mirrorClamp(glm::vec3(0.5f));
		Error += glm::all(glm::equal(L, glm::vec3(0.5f), glm::vec3(0.00001f))) ? 0 : 1;

		glm::vec4 M = glm::mirrorClamp(glm::vec4(0.5f));
		Error += glm::all(glm::equal(M, glm::vec4(0.5f), glm::vec4(0.00001f))) ? 0 : 1;

		glm::vec1 N = glm::mirrorClamp(glm::vec1(0.5f));
		Error += glm::all(glm::equal(N, glm::vec1(0.5f), glm::vec1(0.00001f))) ? 0 : 1;

		return Error;
	}
}//namespace mirrorClamp

namespace mirrorRepeat
{
	int test()
	{
		int Error(0);

		float A = glm::mirrorRepeat(0.5f);
		Error += glm::equal(A, 0.5f, 0.00001f) ? 0 : 1;

		float B = glm::mirrorRepeat(0.0f);
		Error += glm::equal(B, 0.0f, 0.00001f) ? 0 : 1;

		float C = glm::mirrorRepeat(1.0f);
		Error += glm::equal(C, 1.0f, 0.00001f) ? 0 : 1;

		float D = glm::mirrorRepeat(-0.5f);
		Error += glm::equal(D, 0.5f, 0.00001f) ? 0 : 1;

		float E = glm::mirrorRepeat(1.5f);
		Error += glm::equal(E, 0.5f, 0.00001f) ? 0 : 1;

		float F = glm::mirrorRepeat(0.9f);
		Error += glm::equal(F, 0.9f, 0.00001f) ? 0 : 1;

		float G = glm::mirrorRepeat(3.0f);
		Error += glm::equal(G, 1.0f, 0.00001f) ? 0 : 1;

		float H = glm::mirrorRepeat(-3.0f);
		Error += glm::equal(H, 1.0f, 0.00001f) ? 0 : 1;

		float I = glm::mirrorRepeat(-1.0f);
		Error += glm::equal(I, 1.0f, 0.00001f) ? 0 : 1;

		glm::vec2 K = glm::mirrorRepeat(glm::vec2(0.5f));
		Error += glm::all(glm::equal(K, glm::vec2(0.5f), glm::vec2(0.00001f))) ? 0 : 1;

		glm::vec3 L = glm::mirrorRepeat(glm::vec3(0.5f));
		Error += glm::all(glm::equal(L, glm::vec3(0.5f), glm::vec3(0.00001f))) ? 0 : 1;

		glm::vec4 M = glm::mirrorRepeat(glm::vec4(0.5f));
		Error += glm::all(glm::equal(M, glm::vec4(0.5f), glm::vec4(0.00001f))) ? 0 : 1;

		glm::vec1 N = glm::mirrorRepeat(glm::vec1(0.5f));
		Error += glm::all(glm::equal(N, glm::vec1(0.5f), glm::vec1(0.00001f))) ? 0 : 1;

		return Error;
	}
}//namespace mirrorRepeat

int main()
{
	int Error(0);

	Error += clamp::test();
	Error += repeat::test();
	Error += mirrorClamp::test();
	Error += mirrorRepeat::test();

	return Error;
}
