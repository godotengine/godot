#include <glm/ext/vector_relational.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/mat2x2.hpp>
#include <glm/mat2x3.hpp>
#include <glm/mat2x4.hpp>
#include <glm/mat3x2.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat3x4.hpp>
#include <glm/mat4x2.hpp>
#include <glm/mat4x3.hpp>
#include <glm/mat4x4.hpp>

int test_mat2x2_row_set()
{
	int Error = 0;

	glm::mat2x2 m(1);

	m = glm::row(m, 0, glm::vec2( 0,  1));
	m = glm::row(m, 1, glm::vec2( 4,  5));

	Error += glm::all(glm::equal(glm::row(m, 0), glm::vec2( 0,  1), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 1), glm::vec2( 4,  5), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat2x2_col_set()
{
	int Error = 0;

	glm::mat2x2 m(1);

	m = glm::column(m, 0, glm::vec2( 0,  1));
	m = glm::column(m, 1, glm::vec2( 4,  5));

	Error += glm::all(glm::equal(glm::column(m, 0), glm::vec2( 0,  1), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 1), glm::vec2( 4,  5), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat2x3_row_set()
{
	int Error = 0;

	glm::mat2x3 m(1);

	m = glm::row(m, 0, glm::vec2( 0,  1));
	m = glm::row(m, 1, glm::vec2( 4,  5));
	m = glm::row(m, 2, glm::vec2( 8,  9));

	Error += glm::all(glm::equal(glm::row(m, 0), glm::vec2( 0,  1), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 1), glm::vec2( 4,  5), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 2), glm::vec2( 8,  9), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat2x3_col_set()
{
	int Error = 0;

	glm::mat2x3 m(1);

	m = glm::column(m, 0, glm::vec3( 0,  1,  2));
	m = glm::column(m, 1, glm::vec3( 4,  5,  6));

	Error += glm::all(glm::equal(glm::column(m, 0), glm::vec3( 0,  1,  2), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 1), glm::vec3( 4,  5,  6), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat2x4_row_set()
{
	int Error = 0;

	glm::mat2x4 m(1);

	m = glm::row(m, 0, glm::vec2( 0,  1));
	m = glm::row(m, 1, glm::vec2( 4,  5));
	m = glm::row(m, 2, glm::vec2( 8,  9));
	m = glm::row(m, 3, glm::vec2(12, 13));

	Error += glm::all(glm::equal(glm::row(m, 0), glm::vec2( 0,  1), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 1), glm::vec2( 4,  5), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 2), glm::vec2( 8,  9), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 3), glm::vec2(12, 13), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat2x4_col_set()
{
	int Error = 0;

	glm::mat2x4 m(1);

	m = glm::column(m, 0, glm::vec4( 0,  1,  2, 3));
	m = glm::column(m, 1, glm::vec4( 4,  5,  6, 7));

	Error += glm::all(glm::equal(glm::column(m, 0), glm::vec4( 0,  1,  2, 3), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 1), glm::vec4( 4,  5,  6, 7), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat3x2_row_set()
{
	int Error = 0;

	glm::mat3x2 m(1);

	m = glm::row(m, 0, glm::vec3( 0,  1,  2));
	m = glm::row(m, 1, glm::vec3( 4,  5,  6));

	Error += glm::all(glm::equal(glm::row(m, 0), glm::vec3( 0,  1,  2), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 1), glm::vec3( 4,  5,  6), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat3x2_col_set()
{
	int Error = 0;

	glm::mat3x2 m(1);

	m = glm::column(m, 0, glm::vec2( 0,  1));
	m = glm::column(m, 1, glm::vec2( 4,  5));
	m = glm::column(m, 2, glm::vec2( 8,  9));

	Error += glm::all(glm::equal(glm::column(m, 0), glm::vec2( 0,  1), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 1), glm::vec2( 4,  5), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 2), glm::vec2( 8,  9), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat3x3_row_set()
{
	int Error = 0;

	glm::mat3x3 m(1);

	m = glm::row(m, 0, glm::vec3( 0,  1,  2));
	m = glm::row(m, 1, glm::vec3( 4,  5,  6));
	m = glm::row(m, 2, glm::vec3( 8,  9, 10));

	Error += glm::all(glm::equal(glm::row(m, 0), glm::vec3( 0,  1,  2), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 1), glm::vec3( 4,  5,  6), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 2), glm::vec3( 8,  9, 10), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat3x3_col_set()
{
	int Error = 0;

	glm::mat3x3 m(1);

	m = glm::column(m, 0, glm::vec3( 0,  1,  2));
	m = glm::column(m, 1, glm::vec3( 4,  5,  6));
	m = glm::column(m, 2, glm::vec3( 8,  9, 10));

	Error += glm::all(glm::equal(glm::column(m, 0), glm::vec3( 0,  1,  2), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 1), glm::vec3( 4,  5,  6), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 2), glm::vec3( 8,  9, 10), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat3x4_row_set()
{
	int Error = 0;

	glm::mat3x4 m(1);

	m = glm::row(m, 0, glm::vec3( 0,  1,  2));
	m = glm::row(m, 1, glm::vec3( 4,  5,  6));
	m = glm::row(m, 2, glm::vec3( 8,  9, 10));
	m = glm::row(m, 3, glm::vec3(12, 13, 14));

	Error += glm::all(glm::equal(glm::row(m, 0), glm::vec3( 0,  1,  2), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 1), glm::vec3( 4,  5,  6), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 2), glm::vec3( 8,  9, 10), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 3), glm::vec3(12, 13, 14), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat3x4_col_set()
{
	int Error = 0;

	glm::mat3x4 m(1);

	m = glm::column(m, 0, glm::vec4( 0,  1,  2, 3));
	m = glm::column(m, 1, glm::vec4( 4,  5,  6, 7));
	m = glm::column(m, 2, glm::vec4( 8,  9, 10, 11));

	Error += glm::all(glm::equal(glm::column(m, 0), glm::vec4( 0,  1,  2, 3), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 1), glm::vec4( 4,  5,  6, 7), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 2), glm::vec4( 8,  9, 10, 11), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat4x2_row_set()
{
	int Error = 0;

	glm::mat4x2 m(1);

	m = glm::row(m, 0, glm::vec4( 0,  1,  2,  3));
	m = glm::row(m, 1, glm::vec4( 4,  5,  6,  7));

	Error += glm::all(glm::equal(glm::row(m, 0), glm::vec4( 0,  1,  2,  3), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 1), glm::vec4( 4,  5,  6,  7), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat4x2_col_set()
{
	int Error = 0;

	glm::mat4x2 m(1);

	m = glm::column(m, 0, glm::vec2( 0,  1));
	m = glm::column(m, 1, glm::vec2( 4,  5));
	m = glm::column(m, 2, glm::vec2( 8,  9));
	m = glm::column(m, 3, glm::vec2(12, 13));

	Error += glm::all(glm::equal(glm::column(m, 0), glm::vec2( 0,  1), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 1), glm::vec2( 4,  5), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 2), glm::vec2( 8,  9), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 3), glm::vec2(12, 13), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat4x3_row_set()
{
	int Error = 0;

	glm::mat4x3 m(1);

	m = glm::row(m, 0, glm::vec4( 0,  1,  2,  3));
	m = glm::row(m, 1, glm::vec4( 4,  5,  6,  7));
	m = glm::row(m, 2, glm::vec4( 8,  9, 10, 11));

	Error += glm::all(glm::equal(glm::row(m, 0), glm::vec4( 0,  1,  2,  3), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 1), glm::vec4( 4,  5,  6,  7), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 2), glm::vec4( 8,  9, 10, 11), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat4x3_col_set()
{
	int Error = 0;

	glm::mat4x3 m(1);

	m = glm::column(m, 0, glm::vec3( 0,  1,  2));
	m = glm::column(m, 1, glm::vec3( 4,  5,  6));
	m = glm::column(m, 2, glm::vec3( 8,  9, 10));
	m = glm::column(m, 3, glm::vec3(12, 13, 14));

	Error += glm::all(glm::equal(glm::column(m, 0), glm::vec3( 0,  1,  2), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 1), glm::vec3( 4,  5,  6), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 2), glm::vec3( 8,  9, 10), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 3), glm::vec3(12, 13, 14), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat4x4_row_set()
{
	int Error = 0;

	glm::mat4 m(1);

	m = glm::row(m, 0, glm::vec4( 0,  1,  2,  3));
	m = glm::row(m, 1, glm::vec4( 4,  5,  6,  7));
	m = glm::row(m, 2, glm::vec4( 8,  9, 10, 11));
	m = glm::row(m, 3, glm::vec4(12, 13, 14, 15));

	Error += glm::all(glm::equal(glm::row(m, 0), glm::vec4( 0,  1,  2,  3), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 1), glm::vec4( 4,  5,  6,  7), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 2), glm::vec4( 8,  9, 10, 11), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::row(m, 3), glm::vec4(12, 13, 14, 15), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat4x4_col_set()
{
	int Error = 0;

	glm::mat4 m(1);

	m = glm::column(m, 0, glm::vec4( 0,  1,  2,  3));
	m = glm::column(m, 1, glm::vec4( 4,  5,  6,  7));
	m = glm::column(m, 2, glm::vec4( 8,  9, 10, 11));
	m = glm::column(m, 3, glm::vec4(12, 13, 14, 15));

	Error += glm::all(glm::equal(glm::column(m, 0), glm::vec4( 0,  1,  2,  3), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 1), glm::vec4( 4,  5,  6,  7), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 2), glm::vec4( 8,  9, 10, 11), glm::epsilon<float>())) ? 0 : 1;
	Error += glm::all(glm::equal(glm::column(m, 3), glm::vec4(12, 13, 14, 15), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat4x4_row_get()
{
	int Error = 0;

	glm::mat4 m(1);

	glm::vec4 A = glm::row(m, 0);
	Error += glm::all(glm::equal(A, glm::vec4(1, 0, 0, 0), glm::epsilon<float>())) ? 0 : 1;
	glm::vec4 B = glm::row(m, 1);
	Error += glm::all(glm::equal(B, glm::vec4(0, 1, 0, 0), glm::epsilon<float>())) ? 0 : 1;
	glm::vec4 C = glm::row(m, 2);
	Error += glm::all(glm::equal(C, glm::vec4(0, 0, 1, 0), glm::epsilon<float>())) ? 0 : 1;
	glm::vec4 D = glm::row(m, 3);
	Error += glm::all(glm::equal(D, glm::vec4(0, 0, 0, 1), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int test_mat4x4_col_get()
{
	int Error = 0;

	glm::mat4 m(1);

	glm::vec4 A = glm::column(m, 0);
	Error += glm::all(glm::equal(A, glm::vec4(1, 0, 0, 0), glm::epsilon<float>())) ? 0 : 1;
	glm::vec4 B = glm::column(m, 1);
	Error += glm::all(glm::equal(B, glm::vec4(0, 1, 0, 0), glm::epsilon<float>())) ? 0 : 1;
	glm::vec4 C = glm::column(m, 2);
	Error += glm::all(glm::equal(C, glm::vec4(0, 0, 1, 0), glm::epsilon<float>())) ? 0 : 1;
	glm::vec4 D = glm::column(m, 3);
	Error += glm::all(glm::equal(D, glm::vec4(0, 0, 0, 1), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_mat2x2_row_set();
	Error += test_mat2x2_col_set();
	Error += test_mat2x3_row_set();
	Error += test_mat2x3_col_set();
	Error += test_mat2x4_row_set();
	Error += test_mat2x4_col_set();
	Error += test_mat3x2_row_set();
	Error += test_mat3x2_col_set();
	Error += test_mat3x3_row_set();
	Error += test_mat3x3_col_set();
	Error += test_mat3x4_row_set();
	Error += test_mat3x4_col_set();
	Error += test_mat4x2_row_set();
	Error += test_mat4x2_col_set();
	Error += test_mat4x3_row_set();
	Error += test_mat4x3_col_set();
	Error += test_mat4x4_row_set();
	Error += test_mat4x4_col_set();

	Error += test_mat4x4_row_get();
	Error += test_mat4x4_col_get();

	return Error;
}
