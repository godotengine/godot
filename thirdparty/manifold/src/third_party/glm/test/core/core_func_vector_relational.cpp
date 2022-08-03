#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vector_relational.hpp>
#include <glm/gtc/vec1.hpp>

static int test_not()
{
	int Error = 0;

	{
		glm::bvec1 v(false);
		Error += glm::all(glm::not_(v)) ? 0 : 1;
	}

	{
		glm::bvec2 v(false);
		Error += glm::all(glm::not_(v)) ? 0 : 1;
	}

	{
		glm::bvec3 v(false);
		Error += glm::all(glm::not_(v)) ? 0 : 1;
	}
	
	{
		glm::bvec4 v(false);
		Error += glm::all(glm::not_(v)) ? 0 : 1;
	}

	return Error;
}

static int test_less()
{
	int Error = 0;

	{
		glm::vec2 const A(1, 2);
		glm::vec2 const B(2, 3);
		Error += glm::all(glm::lessThan(A, B)) ? 0: 1;
		Error += glm::all(glm::lessThanEqual(A, B)) ? 0: 1;
	}

	{
		glm::vec3 const A(1, 2, 3);
		glm::vec3 const B(2, 3, 4);
		Error += glm::all(glm::lessThan(A, B)) ? 0: 1;
		Error += glm::all(glm::lessThanEqual(A, B)) ? 0: 1;
	}

	{
		glm::vec4 const A(1, 2, 3, 4);
		glm::vec4 const B(2, 3, 4, 5);
		Error += glm::all(glm::lessThan(A, B)) ? 0: 1;
		Error += glm::all(glm::lessThanEqual(A, B)) ? 0: 1;
	}

	{
		glm::ivec2 const A(1, 2);
		glm::ivec2 const B(2, 3);
		Error += glm::all(glm::lessThan(A, B)) ? 0: 1;

		glm::ivec2 const C(1, 3);
		Error += glm::all(glm::lessThanEqual(A, C)) ? 0: 1;
	}

	{
		glm::ivec3 const A(1, 2, 3);
		glm::ivec3 const B(2, 3, 4);
		Error += glm::all(glm::lessThan(A, B)) ? 0: 1;

		glm::ivec3 const C(1, 3, 4);
		Error += glm::all(glm::lessThanEqual(A, C)) ? 0: 1;
	}

	{
		glm::ivec4 const A(1, 2, 3, 4);
		glm::ivec4 const B(2, 3, 4, 5);
		Error += glm::all(glm::lessThan(A, B)) ? 0: 1;

		glm::ivec4 const C(1, 3, 4, 5);
		Error += glm::all(glm::lessThanEqual(A, C)) ? 0: 1;
	}

	return Error;
}

static int test_greater()
{
	int Error = 0;

	{
		glm::vec2 const A(1, 2);
		glm::vec2 const B(2, 3);
		Error += glm::all(glm::greaterThan(B, A)) ? 0: 1;
		Error += glm::all(glm::greaterThanEqual(B, A)) ? 0: 1;
	}

	{
		glm::vec3 const A(1, 2, 3);
		glm::vec3 const B(2, 3, 4);
		Error += glm::all(glm::greaterThan(B, A)) ? 0: 1;
		Error += glm::all(glm::greaterThanEqual(B, A)) ? 0: 1;
	}

	{
		glm::vec4 const A(1, 2, 3, 4);
		glm::vec4 const B(2, 3, 4, 5);
		Error += glm::all(glm::greaterThan(B, A)) ? 0: 1;
		Error += glm::all(glm::greaterThanEqual(B, A)) ? 0: 1;
	}

	{
		glm::ivec2 const A(1, 2);
		glm::ivec2 const B(2, 3);
		Error += glm::all(glm::greaterThan(B, A)) ? 0: 1;

		glm::ivec2 const C(1, 3);
		Error += glm::all(glm::greaterThanEqual(C, A)) ? 0: 1;
	}

	{
		glm::ivec3 const A(1, 2, 3);
		glm::ivec3 const B(2, 3, 4);
		Error += glm::all(glm::greaterThan(B, A)) ? 0: 1;

		glm::ivec3 const C(1, 3, 4);
		Error += glm::all(glm::greaterThanEqual(C, A)) ? 0: 1;
	}

	{
		glm::ivec4 const A(1, 2, 3, 4);
		glm::ivec4 const B(2, 3, 4, 5);
		Error += glm::all(glm::greaterThan(B, A)) ? 0: 1;

		glm::ivec4 const C(1, 3, 4, 5);
		Error += glm::all(glm::greaterThanEqual(C, A)) ? 0: 1;
	}

	return Error;
}

static int test_equal()
{
	int Error = 0;

	{
		glm::ivec2 const A(1, 2);
		glm::ivec2 const B(1, 2);
		Error += glm::all(glm::equal(B, A)) ? 0: 1;
	}

	{
		glm::ivec3 const A(1, 2, 3);
		glm::ivec3 const B(1, 2, 3);
		Error += glm::all(glm::equal(B, A)) ? 0: 1;
	}

	{
		glm::ivec4 const A(1, 2, 3, 4);
		glm::ivec4 const B(1, 2, 3, 4);
		Error += glm::all(glm::equal(B, A)) ? 0: 1;
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_not();
	Error += test_less();
	Error += test_greater();
	Error += test_equal();

	return Error;
}

