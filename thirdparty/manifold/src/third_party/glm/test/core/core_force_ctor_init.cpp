#define GLM_FORCE_CTOR_INIT

#include <glm/glm.hpp>
#include <glm/ext.hpp>

static int test_vec()
{
	int Error = 0;

	glm::vec1 V1;
	Error += glm::all(glm::equal(V1, glm::vec1(0), glm::epsilon<float>())) ? 0 : 1;

	glm::dvec1 U1;
	Error += glm::all(glm::equal(U1, glm::dvec1(0), glm::epsilon<double>())) ? 0 : 1;

	glm::vec2 V2;
	Error += glm::all(glm::equal(V2, glm::vec2(0, 0), glm::epsilon<float>())) ? 0 : 1;

	glm::dvec2 U2;
	Error += glm::all(glm::equal(U2, glm::dvec2(0, 0), glm::epsilon<double>())) ? 0 : 1;

	glm::vec3 V3;
	Error += glm::all(glm::equal(V3, glm::vec3(0, 0, 0), glm::epsilon<float>())) ? 0 : 1;

	glm::dvec3 U3;
	Error += glm::all(glm::equal(U3, glm::dvec3(0, 0, 0), glm::epsilon<double>())) ? 0 : 1;

	glm::vec4 V4;
	Error += glm::all(glm::equal(V4, glm::vec4(0, 0, 0, 0), glm::epsilon<float>())) ? 0 : 1;

	glm::dvec4 U4;
	Error += glm::all(glm::equal(U4, glm::dvec4(0, 0, 0, 0), glm::epsilon<double>())) ? 0 : 1;

	return Error;
}

static int test_mat()
{
	int Error = 0;

	{
		glm::mat2x2 F;
		Error += glm::all(glm::equal(F, glm::mat2x2(1), glm::epsilon<float>())) ? 0 : 1;

		glm::dmat2x2 D;
		Error += glm::all(glm::equal(D, glm::dmat2x2(1), glm::epsilon<double>())) ? 0 : 1;
	}

	{
		glm::mat2x3 F;
		Error += glm::all(glm::equal(F, glm::mat2x3(1), glm::epsilon<float>())) ? 0 : 1;

		glm::dmat2x3 D;
		Error += glm::all(glm::equal(D, glm::dmat2x3(1), glm::epsilon<double>())) ? 0 : 1;
	}

	{
		glm::mat2x4 F;
		Error += glm::all(glm::equal(F, glm::mat2x4(1), glm::epsilon<float>())) ? 0 : 1;

		glm::dmat2x4 D;
		Error += glm::all(glm::equal(D, glm::dmat2x4(1), glm::epsilon<double>())) ? 0 : 1;
	}

	{
		glm::mat3x2 F;
		Error += glm::all(glm::equal(F, glm::mat3x2(1), glm::epsilon<float>())) ? 0 : 1;

		glm::dmat3x2 D;
		Error += glm::all(glm::equal(D, glm::dmat3x2(1), glm::epsilon<double>())) ? 0 : 1;
	}

	{
		glm::mat3x3 F;
		Error += glm::all(glm::equal(F, glm::mat3x3(1), glm::epsilon<float>())) ? 0 : 1;

		glm::dmat3x3 D;
		Error += glm::all(glm::equal(D, glm::dmat3x3(1), glm::epsilon<double>())) ? 0 : 1;
	}

	{
		glm::mat3x4 F;
		Error += glm::all(glm::equal(F, glm::mat3x4(1), glm::epsilon<float>())) ? 0 : 1;

		glm::dmat3x4 D;
		Error += glm::all(glm::equal(D, glm::dmat3x4(1), glm::epsilon<double>())) ? 0 : 1;
	}

	{
		glm::mat4x2 F;
		Error += glm::all(glm::equal(F, glm::mat4x2(1), glm::epsilon<float>())) ? 0 : 1;

		glm::dmat4x2 D;
		Error += glm::all(glm::equal(D, glm::dmat4x2(1), glm::epsilon<double>())) ? 0 : 1;
	}

	{
		glm::mat4x3 F;
		Error += glm::all(glm::equal(F, glm::mat4x3(1), glm::epsilon<float>())) ? 0 : 1;

		glm::dmat4x3 D;
		Error += glm::all(glm::equal(D, glm::dmat4x3(1), glm::epsilon<double>())) ? 0 : 1;
	}

	{
		glm::mat4x4 F;
		Error += glm::all(glm::equal(F, glm::mat4x4(1), glm::epsilon<float>())) ? 0 : 1;

		glm::dmat4x4 D;
		Error += glm::all(glm::equal(D, glm::dmat4x4(1), glm::epsilon<double>())) ? 0 : 1;
	}

	return Error;
}

static int test_qua()
{
	int Error = 0;

	glm::quat F;
	Error += glm::all(glm::equal(F, glm::quat(1, 0, 0, 0), glm::epsilon<float>())) ? 0 : 1;

	glm::dquat D;
	Error += glm::all(glm::equal(D, glm::dquat(1, 0, 0, 0), glm::epsilon<double>())) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_vec();
	Error += test_mat();
	Error += test_qua();

	return Error;
}

