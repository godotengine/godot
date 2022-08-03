#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/log_base.hpp>
#include <glm/gtc/vec1.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/exponential.hpp>

namespace test_log
{
	int run()
	{
		int Error = 0;

		{
			float A = glm::log(10.f, 2.0f);
			float B = glm::log2(10.f);
			Error += glm::epsilonEqual(A, B, 0.00001f) ? 0 : 1;
		}

		{
			glm::vec1 A = glm::log(glm::vec1(10.f), glm::vec1(2.0f));
			glm::vec1 B = glm::log2(glm::vec1(10.f));
			Error += glm::all(glm::epsilonEqual(A, B, glm::vec1(0.00001f))) ? 0 : 1;
		}

		{
			glm::vec2 A = glm::log(glm::vec2(10.f), glm::vec2(2.0f));
			glm::vec2 B = glm::log2(glm::vec2(10.f));
			Error += glm::all(glm::epsilonEqual(A, B, glm::vec2(0.00001f))) ? 0 : 1;
		}

		{
			glm::vec3 A = glm::log(glm::vec3(10.f), glm::vec3(2.0f));
			glm::vec3 B = glm::log2(glm::vec3(10.f));
			Error += glm::all(glm::epsilonEqual(A, B, glm::vec3(0.00001f))) ? 0 : 1;
		}

		{
			glm::vec4 A = glm::log(glm::vec4(10.f), glm::vec4(2.0f));
			glm::vec4 B = glm::log2(glm::vec4(10.f));
			Error += glm::all(glm::epsilonEqual(A, B, glm::vec4(0.00001f))) ? 0 : 1;
		}

		return Error;
	}
}//namespace test_log

int main()
{
	int Error(0);

	Error += test_log::run();

	return Error;
}
