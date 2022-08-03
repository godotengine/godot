#include <glm/gtc/color_space.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/constants.hpp>

namespace srgb
{
	int test()
	{
		int Error(0);

		glm::vec3 const ColorSourceRGB(1.0, 0.5, 0.0);

		{
			glm::vec3 const ColorSRGB = glm::convertLinearToSRGB(ColorSourceRGB);
			glm::vec3 const ColorRGB = glm::convertSRGBToLinear(ColorSRGB);
			Error += glm::all(glm::epsilonEqual(ColorSourceRGB, ColorRGB, 0.00001f)) ? 0 : 1;
		}

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

		glm::vec4 const ColorSourceGNI = glm::vec4(107, 107, 104, 131) / glm::vec4(255);

		{
			glm::vec4 const ColorGNA = glm::convertSRGBToLinear(ColorSourceGNI) * glm::vec4(255);
			glm::vec4 const ColorGNE = glm::convertLinearToSRGB(ColorSourceGNI) * glm::vec4(255);
			glm::vec4 const ColorSRGB = glm::convertLinearToSRGB(ColorSourceGNI);
			glm::vec4 const ColorRGB = glm::convertSRGBToLinear(ColorSRGB);
			Error += glm::all(glm::epsilonEqual(ColorSourceGNI, ColorRGB, 0.00001f)) ? 0 : 1;
		}

		return Error;
	}
}//namespace srgb

namespace srgb_lowp
{
	int test()
	{
		int Error(0);

		for(float Color = 0.0f; Color < 1.0f; Color += 0.01f)
		{
			glm::highp_vec3 const HighpSRGB = glm::convertLinearToSRGB(glm::highp_vec3(Color));
			glm::lowp_vec3 const LowpSRGB = glm::convertLinearToSRGB(glm::lowp_vec3(Color));
			Error += glm::all(glm::epsilonEqual(glm::abs(HighpSRGB - glm::highp_vec3(LowpSRGB)), glm::highp_vec3(0), 0.1f)) ? 0 : 1;
		}

		return Error;
	}
}//namespace srgb_lowp

int main()
{
	int Error(0);

	Error += srgb::test();
	Error += srgb_lowp::test();

	return Error;
}
