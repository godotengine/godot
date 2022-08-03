#include <glm/gtx/color_encoding.hpp>
#include <glm/gtc/color_space.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/constants.hpp>

namespace srgb
{
	int test()
	{
		int Error(0);

		glm::vec3 const ColorSourceRGB(1.0, 0.5, 0.0);
/*
		{
			glm::vec3 const ColorSRGB = glm::convertLinearSRGBToD65XYZ(ColorSourceRGB);
			glm::vec3 const ColorRGB = glm::convertD65XYZToLinearSRGB(ColorSRGB);
			Error += glm::all(glm::epsilonEqual(ColorSourceRGB, ColorRGB, 0.00001f)) ? 0 : 1;
		}
*/
		{
			glm::vec3 const ColorSRGB = glm::convertLinearToSRGB(ColorSourceRGB, 2.8f);
			glm::vec3 const ColorRGB = glm::convertSRGBToLinear(ColorSRGB, 2.8f);
			Error += glm::all(glm::epsilonEqual(ColorSourceRGB, ColorRGB, 0.00001f)) ? 0 : 1;
		}

		glm::vec4 const ColorSourceRGBA(1.0, 0.5, 0.0, 1.0);

		{
			glm::vec4 const ColorSRGB = glm::convertLinearToSRGB(ColorSourceRGBA);
			glm::vec4 const ColorRGB = glm::convertSRGBToLinear(ColorSRGB);
			Error += glm::all(glm::epsilonEqual(ColorSourceRGBA, ColorRGB, 0.00001f)) ? 0 : 1;
		}

		{
			glm::vec4 const ColorSRGB = glm::convertLinearToSRGB(ColorSourceRGBA, 2.8f);
			glm::vec4 const ColorRGB = glm::convertSRGBToLinear(ColorSRGB, 2.8f);
			Error += glm::all(glm::epsilonEqual(ColorSourceRGBA, ColorRGB, 0.00001f)) ? 0 : 1;
		}

		return Error;
	}
}//namespace srgb

int main()
{
	int Error(0);

	Error += srgb::test();

	return Error;
}
