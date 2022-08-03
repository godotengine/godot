#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/epsilon.hpp>

int test_affine()
{
	int Error = 0;

	{
		glm::mat3 const M(
			2.f, 0.f, 0.f,
			0.f, 2.f, 0.f,
			0.f, 0.f, 1.f);
		glm::mat3 const A = glm::affineInverse(M);
		glm::mat3 const I = glm::inverse(M);
		glm::mat3 const R = glm::affineInverse(A);

		for(glm::length_t i = 0; i < A.length(); ++i)
		{
			Error += glm::all(glm::epsilonEqual(M[i], R[i], 0.01f)) ? 0 : 1;
			Error += glm::all(glm::epsilonEqual(A[i], I[i], 0.01f)) ? 0 : 1;
		}
	}

	{
		glm::mat4 const M(
			2.f, 0.f, 0.f, 0.f,
			0.f, 2.f, 0.f, 0.f,
			0.f, 0.f, 2.f, 0.f,
			0.f, 0.f, 0.f, 1.f);
		glm::mat4 const A = glm::affineInverse(M);
		glm::mat4 const I = glm::inverse(M);
		glm::mat4 const R = glm::affineInverse(A);

		for(glm::length_t i = 0; i < A.length(); ++i)
		{
			Error += glm::all(glm::epsilonEqual(M[i], R[i], 0.01f)) ? 0 : 1;
			Error += glm::all(glm::epsilonEqual(A[i], I[i], 0.01f)) ? 0 : 1;
		}
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_affine();

	return Error;
}
