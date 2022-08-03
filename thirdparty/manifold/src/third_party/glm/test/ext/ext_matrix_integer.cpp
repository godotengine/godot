#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/matrix_integer.hpp>
#include <glm/ext/matrix_int2x2.hpp>
#include <glm/ext/matrix_int2x3.hpp>
#include <glm/ext/matrix_int2x4.hpp>
#include <glm/ext/matrix_int3x2.hpp>
#include <glm/ext/matrix_int3x3.hpp>
#include <glm/ext/matrix_int3x4.hpp>
#include <glm/ext/matrix_int4x2.hpp>
#include <glm/ext/matrix_int4x3.hpp>
#include <glm/ext/matrix_int4x4.hpp>

using namespace glm;

int test_matrixCompMult()
{
	int Error = 0;

	{
		imat2 m(0, 1, 2, 3);
		imat2 n = matrixCompMult(m, m);
		imat2 expected = imat2(0, 1, 4, 9);
		Error += all(equal(n, expected)) ? 0 : 1;
	}

	{
		imat2x3 m(0, 1, 2, 3, 4, 5);
		imat2x3 n = matrixCompMult(m, m);
		imat2x3 expected = imat2x3(0, 1, 4, 9, 16, 25);
		Error += all(equal(n, expected)) ? 0 : 1;
	}

	{
		imat2x4 m(0, 1, 2, 3, 4, 5, 6, 7);
		imat2x4 n = matrixCompMult(m, m);
		imat2x4 expected = imat2x4(0, 1, 4, 9, 16, 25, 36, 49);
		Error += all(equal(n, expected)) ? 0 : 1;
	}

	{
		imat3 m(0, 1, 2, 3, 4, 5, 6, 7, 8);
		imat3 n = matrixCompMult(m, m);
		imat3 expected = imat3(0, 1, 4, 9, 16, 25, 36, 49, 64);
		Error += all(equal(n, expected)) ? 0 : 1;
	}

	{
		imat3x2 m(0, 1, 2, 3, 4, 5);
		imat3x2 n = matrixCompMult(m, m);
		imat3x2 expected = imat3x2(0, 1, 4, 9, 16, 25);
		Error += all(equal(n, expected)) ? 0 : 1;
	}

	{
		imat3x4 m(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
		imat3x4 n = matrixCompMult(m, m);
		imat3x4 expected = imat3x4(0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121);
		Error += all(equal(n, expected)) ? 0 : 1;
	}

	{
		imat4 m(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		imat4 n = matrixCompMult(m, m);
		imat4 expected = imat4(0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225);
		Error += all(equal(n, expected)) ? 0 : 1;
	}

	{
		imat4x2 m(0, 1, 2, 3, 4, 5, 6, 7);
		imat4x2 n = matrixCompMult(m, m);
		imat4x2 expected = imat4x2(0, 1, 4, 9, 16, 25, 36, 49);
		Error += all(equal(n, expected)) ? 0 : 1;
	}

	{
		imat4x3 m(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
		imat4x3 n = matrixCompMult(m, m);
		imat4x3 expected = imat4x3(0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121);
		Error += all(equal(n, expected)) ? 0 : 1;
	}

	return Error;
}

int test_outerProduct()
{
	int Error = 0;

	{ 
		glm::imat2x2 const m = glm::outerProduct(glm::ivec2(1), glm::ivec2(1));
		Error += all(equal(m, glm::imat2x2(1, 1, 1, 1))) ? 0 : 1;
	}
	{ 
		glm::imat2x3 const m = glm::outerProduct(glm::ivec3(1), glm::ivec2(1)); 
		Error += all(equal(m, glm::imat2x3(1, 1, 1, 1, 1, 1))) ? 0 : 1;
	}
	{ 
		glm::imat2x4 const m = glm::outerProduct(glm::ivec4(1), glm::ivec2(1)); 
		Error += all(equal(m, glm::imat2x4(1, 1, 1, 1, 1, 1, 1, 1))) ? 0 : 1;
	}

	{
		glm::imat3x2 const m = glm::outerProduct(glm::ivec2(1), glm::ivec3(1));
		Error += all(equal(m, glm::imat3x2(1, 1, 1, 1, 1, 1))) ? 0 : 1;
	}
	{ 
		glm::imat3x3 const m = glm::outerProduct(glm::ivec3(1), glm::ivec3(1)); 
		Error += all(equal(m, glm::imat3x3(1, 1, 1, 1, 1, 1, 1, 1, 1))) ? 0 : 1;
	}
	{
		glm::imat3x4 const m = glm::outerProduct(glm::ivec4(1), glm::ivec3(1));
		Error += all(equal(m, glm::imat3x4(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))) ? 0 : 1;
	}

  
	{ 
		glm::imat4x2 const m = glm::outerProduct(glm::ivec2(1), glm::ivec4(1)); 
		Error += all(equal(m, glm::imat4x2(1, 1, 1, 1, 1, 1, 1, 1))) ? 0 : 1;
	}
	{ 
		glm::imat4x3 const m = glm::outerProduct(glm::ivec3(1), glm::ivec4(1));
		Error += all(equal(m, glm::imat4x3(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))) ? 0 : 1;
	}
	{ 
		glm::imat4x4 const m = glm::outerProduct(glm::ivec4(1), glm::ivec4(1)); 
		Error += all(equal(m, glm::imat4x4(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))) ? 0 : 1;
	}

	return Error;
}

int test_transpose()
{
	int Error = 0;

	{
		imat2 const m(0, 1, 2, 3);
		imat2 const t = transpose(m);
		imat2 const expected = imat2(0, 2, 1, 3);
		Error += all(equal(t, expected)) ? 0 : 1;
	}

	{
		imat2x3 m(0, 1, 2, 3, 4, 5);
		imat3x2 t = transpose(m);
		imat3x2 const expected = imat3x2(0, 3, 1, 4, 2, 5);
		Error += all(equal(t, expected)) ? 0 : 1;
	}

	{
		imat2x4 m(0, 1, 2, 3, 4, 5, 6, 7);
		imat4x2 t = transpose(m);
		imat4x2 const expected = imat4x2(0, 4, 1, 5, 2, 6, 3, 7);
		Error += all(equal(t, expected)) ? 0 : 1;
	}

	{
		imat3 m(0, 1, 2, 3, 4, 5, 6, 7, 8);
		imat3 t = transpose(m);
		imat3 const expected = imat3(0, 3, 6, 1, 4, 7, 2, 5, 8);
		Error += all(equal(t, expected)) ? 0 : 1;
	}

	{
		imat3x2 m(0, 1, 2, 3, 4, 5);
		imat2x3 t = transpose(m);
		imat2x3 const expected = imat2x3(0, 2, 4, 1, 3, 5);
		Error += all(equal(t, expected)) ? 0 : 1;
	}

	{
		imat3x4 m(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
		imat4x3 t = transpose(m);
		imat4x3 const expected = imat4x3(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11);
		Error += all(equal(t, expected)) ? 0 : 1;
	}

	{
		imat4 m(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		imat4 t = transpose(m);
		imat4 const expected = imat4(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
		Error += all(equal(t, expected)) ? 0 : 1;
	}

	{
		imat4x2 m(0, 1, 2, 3, 4, 5, 6, 7);
		imat2x4 t = transpose(m);
		imat2x4 const expected = imat2x4(0, 2, 4, 6, 1, 3, 5, 7);
		Error += all(equal(t, expected)) ? 0 : 1;
	}

	{
		imat4x3 m(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
		imat3x4 t = transpose(m);
		imat3x4 const expected = imat3x4(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11);
		Error += all(equal(t, expected)) ? 0 : 1;
	}

	return Error;
}

int test_determinant()
{
	int Error = 0;

	{
		imat2 const m(1, 1, 1, 1);
		int const t = determinant(m);
		Error += t == 0 ? 0 : 1;
	}

	{
		imat3 m(1, 1, 1, 1, 1, 1, 1, 1, 1);
		int t = determinant(m);
		Error += t == 0 ? 0 : 1;
	}

	{
		imat4 m(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
		int t = determinant(m);
		Error += t == 0 ? 0 : 1;
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_matrixCompMult();
	Error += test_outerProduct();
	Error += test_transpose();
	Error += test_determinant();

	return Error;
}
