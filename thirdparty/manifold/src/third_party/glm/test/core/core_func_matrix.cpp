#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/mat2x2.hpp>
#include <glm/mat2x3.hpp>
#include <glm/mat2x4.hpp>
#include <glm/mat3x2.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat3x4.hpp>
#include <glm/mat4x2.hpp>
#include <glm/mat4x3.hpp>
#include <glm/mat4x4.hpp>
#include <vector>
#include <ctime>
#include <cstdio>

using namespace glm;

int test_matrixCompMult()
{
	int Error(0);

	{
		mat2 m(0, 1, 2, 3);
		mat2 n = matrixCompMult(m, m);
		mat2 expected = mat2(0, 1, 4, 9);
		Error += all(equal(n, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat2x3 m(0, 1, 2, 3, 4, 5);
		mat2x3 n = matrixCompMult(m, m);
		mat2x3 expected = mat2x3(0, 1, 4, 9, 16, 25);
		Error += all(equal(n, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat2x4 m(0, 1, 2, 3, 4, 5, 6, 7);
		mat2x4 n = matrixCompMult(m, m);
		mat2x4 expected = mat2x4(0, 1, 4, 9, 16, 25, 36, 49);
		Error += all(equal(n, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat3 m(0, 1, 2, 3, 4, 5, 6, 7, 8);
		mat3 n = matrixCompMult(m, m);
		mat3 expected = mat3(0, 1, 4, 9, 16, 25, 36, 49, 64);
		Error += all(equal(n, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat3x2 m(0, 1, 2, 3, 4, 5);
		mat3x2 n = matrixCompMult(m, m);
		mat3x2 expected = mat3x2(0, 1, 4, 9, 16, 25);
		Error += all(equal(n, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat3x4 m(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
		mat3x4 n = matrixCompMult(m, m);
		mat3x4 expected = mat3x4(0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121);
		Error += all(equal(n, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat4 m(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		mat4 n = matrixCompMult(m, m);
		mat4 expected = mat4(0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225);
		Error += all(equal(n, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat4x2 m(0, 1, 2, 3, 4, 5, 6, 7);
		mat4x2 n = matrixCompMult(m, m);
		mat4x2 expected = mat4x2(0, 1, 4, 9, 16, 25, 36, 49);
		Error += all(equal(n, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat4x3 m(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
		mat4x3 n = matrixCompMult(m, m);
		mat4x3 expected = mat4x3(0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121);
		Error += all(equal(n, expected, epsilon<float>())) ? 0 : 1;
	}

	return Error;
}

int test_outerProduct()
{
	{ glm::mat2 m = glm::outerProduct(glm::vec2(1.0f), glm::vec2(1.0f)); }
	{ glm::mat3 m = glm::outerProduct(glm::vec3(1.0f), glm::vec3(1.0f)); }
	{ glm::mat4 m = glm::outerProduct(glm::vec4(1.0f), glm::vec4(1.0f)); }

	{ glm::mat2x3 m = glm::outerProduct(glm::vec3(1.0f), glm::vec2(1.0f)); }
	{ glm::mat2x4 m = glm::outerProduct(glm::vec4(1.0f), glm::vec2(1.0f)); }

	{ glm::mat3x2 m = glm::outerProduct(glm::vec2(1.0f), glm::vec3(1.0f)); }
	{ glm::mat3x4 m = glm::outerProduct(glm::vec4(1.0f), glm::vec3(1.0f)); }
  
	{ glm::mat4x2 m = glm::outerProduct(glm::vec2(1.0f), glm::vec4(1.0f)); }
	{ glm::mat4x3 m = glm::outerProduct(glm::vec3(1.0f), glm::vec4(1.0f)); }

	return 0;
}

int test_transpose()
{
	int Error(0);

	{
		mat2 const m(0, 1, 2, 3);
		mat2 const t = transpose(m);
		mat2 const expected = mat2(0, 2, 1, 3);
		Error += all(equal(t, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat2x3 m(0, 1, 2, 3, 4, 5);
		mat3x2 t = transpose(m);
		mat3x2 const expected = mat3x2(0, 3, 1, 4, 2, 5);
		Error += all(equal(t, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat2x4 m(0, 1, 2, 3, 4, 5, 6, 7);
		mat4x2 t = transpose(m);
		mat4x2 const expected = mat4x2(0, 4, 1, 5, 2, 6, 3, 7);
		Error += all(equal(t, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat3 m(0, 1, 2, 3, 4, 5, 6, 7, 8);
		mat3 t = transpose(m);
		mat3 const expected = mat3(0, 3, 6, 1, 4, 7, 2, 5, 8);
		Error += all(equal(t, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat3x2 m(0, 1, 2, 3, 4, 5);
		mat2x3 t = transpose(m);
		mat2x3 const expected = mat2x3(0, 2, 4, 1, 3, 5);
		Error += all(equal(t, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat3x4 m(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
		mat4x3 t = transpose(m);
		mat4x3 const expected = mat4x3(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11);
		Error += all(equal(t, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat4 m(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		mat4 t = transpose(m);
		mat4 const expected = mat4(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
		Error += all(equal(t, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat4x2 m(0, 1, 2, 3, 4, 5, 6, 7);
		mat2x4 t = transpose(m);
		mat2x4 const expected = mat2x4(0, 2, 4, 6, 1, 3, 5, 7);
		Error += all(equal(t, expected, epsilon<float>())) ? 0 : 1;
	}

	{
		mat4x3 m(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
		mat3x4 t = transpose(m);
		mat3x4 const expected = mat3x4(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11);
		Error += all(equal(t, expected, epsilon<float>())) ? 0 : 1;
	}

	return Error;
}

int test_determinant()
{


	return 0;
}

int test_inverse()
{
	int Error = 0;

	{
		glm::mat4x4 A4x4(
			glm::vec4(1, 0, 1, 0), 
			glm::vec4(0, 1, 0, 0), 
			glm::vec4(0, 0, 1, 0), 
			glm::vec4(0, 0, 0, 1));
		glm::mat4x4 B4x4 = inverse(A4x4);
		glm::mat4x4 I4x4 = A4x4 * B4x4;
		glm::mat4x4 Identity(1);
		Error += all(equal(I4x4, Identity, epsilon<float>())) ? 0 : 1;
	}

	{
		glm::mat3x3 A3x3(
			glm::vec3(1, 0, 1), 
			glm::vec3(0, 1, 0), 
			glm::vec3(0, 0, 1));
		glm::mat3x3 B3x3 = glm::inverse(A3x3);
		glm::mat3x3 I3x3 = A3x3 * B3x3;
		glm::mat3x3 Identity(1);
		Error += all(equal(I3x3, Identity, epsilon<float>())) ? 0 : 1;
	}

	{
		glm::mat2x2 A2x2(
			glm::vec2(1, 1), 
			glm::vec2(0, 1));
		glm::mat2x2 B2x2 = glm::inverse(A2x2);
		glm::mat2x2 I2x2 = A2x2 * B2x2;
		glm::mat2x2 Identity(1);
		Error += all(equal(I2x2, Identity, epsilon<float>())) ? 0 : 1;
	}

	return Error;
}

int test_inverse_simd()
{
	int Error = 0;

	glm::mat4x4 const Identity(1);

	glm::mat4x4 const A4x4(
		glm::vec4(1, 0, 1, 0),
		glm::vec4(0, 1, 0, 0),
		glm::vec4(0, 0, 1, 0),
		glm::vec4(0, 0, 0, 1));
	glm::mat4x4 const B4x4 = glm::inverse(A4x4);
	glm::mat4x4 const I4x4 = A4x4 * B4x4;

	Error += glm::all(glm::equal(I4x4, Identity, 0.001f)) ? 0 : 1;

	return Error;
}

int test_shearing()
{
    int Error = 0;

    {
        glm::vec3 const center(0, 0, 0);
        glm::vec2 const l_x(2, 0);
        glm::vec2 const l_y(0, 0);
        glm::vec2 const l_z(0, 0);
        glm::mat4x4 const A4x4(
                glm::vec4(0, 0, 1, 1),
                glm::vec4(0, 1, 1, 0),
                glm::vec4(1, 1, 1, 0),
                glm::vec4(1, 1, 0, 1));
        glm::mat4x4 const B4x4 = glm::shear(A4x4, center, l_x, l_y, l_z);
        glm::mat4x4 const expected(
                glm::vec4(0, 0, 1, 1),
                glm::vec4(2, 1, 1, 0),
                glm::vec4(3, 1, 1, 0),
                glm::vec4(3, 1, 0, 1));
        Error += all(equal(B4x4, expected, epsilon<float>())) ? 0 : 1;
    }

    {
        glm::vec3 const center(0, 0, 0);
        glm::vec2 const l_x(1, 0);
        glm::vec2 const l_y(0, 1);
        glm::vec2 const l_z(1, 0);
        glm::mat4x4 const A4x4(
                glm::vec4(0, 0, 1, 0),
                glm::vec4(0, 1, 1, 0),
                glm::vec4(1, 1, 1, 0),
                glm::vec4(1, 0, 0, 0));
        glm::mat4x4 const B4x4 = glm::shear(A4x4, center, l_x, l_y, l_z);
        glm::mat4x4 const expected(
                glm::vec4(0, 1, 1, 0),
                glm::vec4(1, 2, 1, 0),
                glm::vec4(2, 2, 2, 0),
                glm::vec4(1, 0, 1, 0));
        Error += all(equal(B4x4, expected, epsilon<float>())) ? 0 : 1;
    }

    {
        glm::vec3 const center(3, 2, 1);
        glm::vec2 const l_x(1, 2);
        glm::vec2 const l_y(3, 1);
        glm::vec2 const l_z(4, 5);
        glm::mat4x4 const A4x4(1);
        glm::mat4x4 const B4x4 = glm::shear(A4x4, center, l_x, l_y, l_z);
        glm::mat4x4 const expected(
                glm::vec4(1, 3, 4, 0),
                glm::vec4(1, 1, 5, 0),
                glm::vec4(2, 1, 1, 0),
                glm::vec4(-9, -8, -9, 1));
        Error += all(equal(B4x4, expected, epsilon<float>())) ? 0 : 1;
    }

    {
        glm::vec3 const center(3, 2, 1);
        glm::vec2 const l_x(1, 2);
        glm::vec2 const l_y(3, 1);
        glm::vec2 const l_z(4, 5);
        glm::mat4x4 const A4x4(
                glm::vec4(-3, 2, 1, 0),
                glm::vec4(3, 2, 1, 0),
                glm::vec4(4, -8, 0, 0),
                glm::vec4(7, 1, -2, 0));
        glm::mat4x4 const B4x4 = glm::shear(A4x4, center, l_x, l_y, l_z);
        glm::mat4x4 const expected(
                glm::vec4(1, -6, -1, 0),
                glm::vec4(7, 12, 23, 0),
                glm::vec4(-4, 4, -24, 0),
                glm::vec4(4, 20, 31, 0));
        Error += all(equal(B4x4, expected, epsilon<float>())) ? 0 : 1;
    }

    return Error;
}

template<typename VEC3, typename MAT4>
int test_inverse_perf(std::size_t Count, std::size_t Instance, char const * Message)
{
	std::vector<MAT4> TestInputs;
	TestInputs.resize(Count);
	std::vector<MAT4> TestOutputs;
	TestOutputs.resize(TestInputs.size());

	VEC3 Axis(glm::normalize(VEC3(1.0f, 2.0f, 3.0f)));

	for(std::size_t i = 0; i < TestInputs.size(); ++i)
	{
		typename MAT4::value_type f = static_cast<typename MAT4::value_type>(i + Instance) * typename MAT4::value_type(0.1) + typename MAT4::value_type(0.1);
		TestInputs[i] = glm::rotate(glm::translate(MAT4(1), Axis * f), f, Axis);
		//TestInputs[i] = glm::translate(MAT4(1), Axis * f);
	}

	std::clock_t StartTime = std::clock();

	for(std::size_t i = 0; i < TestInputs.size(); ++i)
		TestOutputs[i] = glm::inverse(TestInputs[i]);

	std::clock_t EndTime = std::clock();

	for(std::size_t i = 0; i < TestInputs.size(); ++i)
		TestOutputs[i] = TestOutputs[i] * TestInputs[i];

	typename MAT4::value_type Diff(0);
	for(std::size_t Entry = 0; Entry < TestOutputs.size(); ++Entry)
	{
		MAT4 i(1.0);
		MAT4 m(TestOutputs[Entry]);
		for(glm::length_t y = 0; y < m.length(); ++y)
		for(glm::length_t x = 0; x < m[y].length(); ++x)
			Diff = glm::max(m[y][x], i[y][x]);
	}

	//glm::uint Ulp = 0;
	//Ulp = glm::max(glm::float_distance(*Dst, *Src), Ulp);

	std::printf("inverse<%s>(%f): %lu\n", Message, static_cast<double>(Diff), EndTime - StartTime);

	return 0;
}

int main()
{
	int Error = 0;
	Error += test_matrixCompMult();
	Error += test_outerProduct();
	Error += test_transpose();
	Error += test_determinant();
	Error += test_inverse();
    Error += test_inverse_simd();
    Error += test_shearing();

#	ifdef NDEBUG
	std::size_t const Samples = 1000;
#	else
	std::size_t const Samples = 1;
#	endif//NDEBUG

	for(std::size_t i = 0; i < 1; ++i)
	{
		Error += test_inverse_perf<glm::vec3, glm::mat4>(Samples, i, "mat4");
		Error += test_inverse_perf<glm::dvec3, glm::dmat4>(Samples, i, "dmat4");
	}

	return Error;
}

