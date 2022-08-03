#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/noise.hpp>
#include <glm/gtc/type_precision.hpp>
#include <glm/gtx/raw_data.hpp>

static int test_simplex_float()
{
	int Error = 0;

	glm::u8vec4 const PixelSimplex2D(glm::byte(glm::abs(glm::simplex(glm::vec2(0.f, 0.f))) * 255.f));
	glm::u8vec4 const PixelSimplex3D(glm::byte(glm::abs(glm::simplex(glm::vec3(0.f, 0.f, 0.f))) * 255.f));
	glm::u8vec4 const PixelSimplex4D(glm::byte(glm::abs(glm::simplex(glm::vec4(0.f, 0.f, 0.f, 0.f))) * 255.f));

	return Error;
}

static int test_simplex_double()
{
	int Error = 0;

	glm::u8vec4 const PixelSimplex2D(glm::byte(glm::abs(glm::simplex(glm::dvec2(0.f, 0.f))) * 255.));
	glm::u8vec4 const PixelSimplex3D(glm::byte(glm::abs(glm::simplex(glm::dvec3(0.f, 0.f, 0.f))) * 255.));
	glm::u8vec4 const PixelSimplex4D(glm::byte(glm::abs(glm::simplex(glm::dvec4(0.f, 0.f, 0.f, 0.f))) * 255.));

	return Error;
}

static int test_perlin_float()
{
	int Error = 0;

	glm::u8vec4 const PixelPerlin2D(glm::byte(glm::abs(glm::perlin(glm::vec2(0.f, 0.f))) * 255.f));
	glm::u8vec4 const PixelPerlin3D(glm::byte(glm::abs(glm::perlin(glm::vec3(0.f, 0.f, 0.f))) * 255.f));
	glm::u8vec4 const PixelPerlin4D(glm::byte(glm::abs(glm::perlin(glm::vec4(0.f, 0.f, 0.f, 0.f))) * 255.f));

	return Error;
}

static int test_perlin_double()
{
	int Error = 0;

	glm::u8vec4 const PixelPerlin2D(glm::byte(glm::abs(glm::perlin(glm::dvec2(0.f, 0.f))) * 255.));
	glm::u8vec4 const PixelPerlin3D(glm::byte(glm::abs(glm::perlin(glm::dvec3(0.f, 0.f, 0.f))) * 255.));
	glm::u8vec4 const PixelPerlin4D(glm::byte(glm::abs(glm::perlin(glm::dvec4(0.f, 0.f, 0.f, 0.f))) * 255.));

	return Error;
}

static int test_perlin_pedioric_float()
{
	int Error = 0;

	glm::u8vec4 const PixelPeriodic2D(glm::byte(glm::abs(glm::perlin(glm::vec2(0.f, 0.f), glm::vec2(2.0f))) * 255.f));
	glm::u8vec4 const PixelPeriodic3D(glm::byte(glm::abs(glm::perlin(glm::vec3(0.f, 0.f, 0.f), glm::vec3(2.0f))) * 255.f));
	glm::u8vec4 const PixelPeriodic4D(glm::byte(glm::abs(glm::perlin(glm::vec4(0.f, 0.f, 0.f, 0.f), glm::vec4(2.0f))) * 255.f));

	return Error;
}

static int test_perlin_pedioric_double()
{
	int Error = 0;

	glm::u8vec4 const PixelPeriodic2D(glm::byte(glm::abs(glm::perlin(glm::dvec2(0.f, 0.f), glm::dvec2(2.0))) * 255.));
	glm::u8vec4 const PixelPeriodic3D(glm::byte(glm::abs(glm::perlin(glm::dvec3(0.f, 0.f, 0.f), glm::dvec3(2.0))) * 255.));
	glm::u8vec4 const PixelPeriodic4D(glm::byte(glm::abs(glm::perlin(glm::dvec4(0.f, 0.f, 0.f, 0.f), glm::dvec4(2.0))) * 255.));

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_simplex_float();
	Error += test_simplex_double();

	Error += test_perlin_float();
	Error += test_perlin_double();

	Error += test_perlin_pedioric_float();
	Error += test_perlin_pedioric_double();

	return Error;
}
