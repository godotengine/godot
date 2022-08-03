#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/component_wise.hpp>
#include <glm/gtc/type_precision.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/constants.hpp>
#include <limits>

namespace compNormalize
{
	int run()
	{
		int Error(0);

		{
			glm::vec4 const A = glm::compNormalize<float>(glm::u8vec4(0, 127, 128, 255));

			Error += glm::epsilonEqual(A.x, 0.0f, glm::epsilon<float>()) ? 0 : 1;
			Error += A.y < 0.5f ? 0 : 1;
			Error += A.z > 0.5f ? 0 : 1;
			Error += glm::epsilonEqual(A.w, 1.0f, glm::epsilon<float>()) ? 0 : 1;
		}

		{
			glm::vec4 const A = glm::compNormalize<float>(glm::i8vec4(-128, -1, 0, 127));

			Error += glm::epsilonEqual(A.x,-1.0f, glm::epsilon<float>()) ? 0 : 1;
			Error += A.y < 0.0f ? 0 : 1;
			Error += A.z > 0.0f ? 0 : 1;
			Error += glm::epsilonEqual(A.w, 1.0f, glm::epsilon<float>()) ? 0 : 1;
		}

		{
			glm::vec4 const A = glm::compNormalize<float>(glm::u16vec4(
				std::numeric_limits<glm::u16>::min(),
				(std::numeric_limits<glm::u16>::max() >> 1) + 0,
				(std::numeric_limits<glm::u16>::max() >> 1) + 1,
				std::numeric_limits<glm::u16>::max()));

			Error += glm::epsilonEqual(A.x, 0.0f, glm::epsilon<float>()) ? 0 : 1;
			Error += A.y < 0.5f ? 0 : 1;
			Error += A.z > 0.5f ? 0 : 1;
			Error += glm::epsilonEqual(A.w, 1.0f, glm::epsilon<float>()) ? 0 : 1;
		}

		{
			glm::vec4 const A = glm::compNormalize<float>(glm::i16vec4(
				std::numeric_limits<glm::i16>::min(),
				static_cast<glm::i16>(-1),
				static_cast<glm::i16>(0),
				std::numeric_limits<glm::i16>::max()));

			Error += glm::epsilonEqual(A.x,-1.0f, glm::epsilon<float>()) ? 0 : 1;
			Error += A.y < 0.0f ? 0 : 1;
			Error += A.z > 0.0f ? 0 : 1;
			Error += glm::epsilonEqual(A.w, 1.0f, glm::epsilon<float>()) ? 0 : 1;
		}

		return Error;
	}
}//namespace compNormalize

namespace compScale
{
	int run()
	{
		int Error(0);

		{
			glm::u8vec4 const A = glm::compScale<glm::u8>(glm::vec4(0.0f, 0.2f, 0.5f, 1.0f));

			Error += A.x == std::numeric_limits<glm::u8>::min() ? 0 : 1;
			Error += A.y < (std::numeric_limits<glm::u8>::max() >> 2) ? 0 : 1;
			Error += A.z == 127 ? 0 : 1;
			Error += A.w == 255 ? 0 : 1;
		}

		{
			glm::i8vec4 const A = glm::compScale<glm::i8>(glm::vec4(0.0f,-1.0f, 0.5f, 1.0f));

			Error += A.x == 0 ? 0 : 1;
			Error += A.y == -128 ? 0 : 1;
			Error += A.z == 63 ? 0 : 1;
			Error += A.w == 127 ? 0 : 1;
		}

		{
			glm::u16vec4 const A = glm::compScale<glm::u16>(glm::vec4(0.0f, 0.2f, 0.5f, 1.0f));

			Error += A.x == std::numeric_limits<glm::u16>::min() ? 0 : 1;
			Error += A.y < (std::numeric_limits<glm::u16>::max() >> 2) ? 0 : 1;
			Error += A.z == 32767 ? 0 : 1;
			Error += A.w == 65535 ? 0 : 1;
		}

		{
			glm::i16vec4 const A = glm::compScale<glm::i16>(glm::vec4(0.0f,-1.0f, 0.5f, 1.0f));

			Error += A.x == 0 ? 0 : 1;
			Error += A.y == -32768 ? 0 : 1;
			Error += A.z == 16383 ? 0 : 1;
			Error += A.w == 32767 ? 0 : 1;
		}

		return Error;
	}
}// compScale

int main()
{
	int Error(0);

	Error += compNormalize::run();
	Error += compScale::run();

	return Error;
}
