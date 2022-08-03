#include <glm/glm.hpp>

#if GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE
#include <glm/gtc/type_aligned.hpp>
#include <glm/gtc/type_precision.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/matrix_relational.hpp>

GLM_STATIC_ASSERT(glm::detail::is_aligned<glm::aligned_lowp>::value, "aligned_lowp is not aligned");
GLM_STATIC_ASSERT(glm::detail::is_aligned<glm::aligned_mediump>::value, "aligned_mediump is not aligned");
GLM_STATIC_ASSERT(glm::detail::is_aligned<glm::aligned_highp>::value, "aligned_highp is not aligned");
GLM_STATIC_ASSERT(!glm::detail::is_aligned<glm::packed_highp>::value, "packed_highp is aligned");
GLM_STATIC_ASSERT(!glm::detail::is_aligned<glm::packed_mediump>::value, "packed_mediump is aligned");
GLM_STATIC_ASSERT(!glm::detail::is_aligned<glm::packed_lowp>::value, "packed_lowp is aligned");

struct my_vec4_packed
{
	glm::uint32 a;
	glm::vec4 b;
};
GLM_STATIC_ASSERT(sizeof(my_vec4_packed) == sizeof(glm::uint32) + sizeof(glm::vec4), "glm::vec4 packed is not correct");

struct my_vec4_aligned
{
	glm::uint32 a;
	glm::aligned_vec4 b;
};
GLM_STATIC_ASSERT(sizeof(my_vec4_aligned) == sizeof(glm::aligned_vec4) * 2, "glm::vec4 aligned is not correct");

struct my_dvec4_packed
{
	glm::uint64 a;
	glm::dvec4 b;
};
GLM_STATIC_ASSERT(sizeof(my_dvec4_packed) == sizeof(glm::uint64) + sizeof(glm::dvec4), "glm::dvec4 packed is not correct");

struct my_dvec4_aligned
{
	glm::uint64 a;
	glm::aligned_dvec4 b;
};
//GLM_STATIC_ASSERT(sizeof(my_dvec4_aligned) == sizeof(glm::aligned_dvec4) * 2, "glm::dvec4 aligned is not correct");

struct my_ivec4_packed
{
	glm::uint32 a;
	glm::ivec4 b;
};
GLM_STATIC_ASSERT(sizeof(my_ivec4_packed) == sizeof(glm::uint32) + sizeof(glm::ivec4), "glm::ivec4 packed is not correct");

struct my_ivec4_aligned
{
	glm::uint32 a;
	glm::aligned_ivec4 b;
};
GLM_STATIC_ASSERT(sizeof(my_ivec4_aligned) == sizeof(glm::aligned_ivec4) * 2, "glm::ivec4 aligned is not correct");

struct my_u8vec4_packed
{
	glm::uint32 a;
	glm::u8vec4 b;
};
GLM_STATIC_ASSERT(sizeof(my_u8vec4_packed) == sizeof(glm::uint32) + sizeof(glm::u8vec4), "glm::u8vec4 packed is not correct");

static int test_copy()
{
	int Error = 0;

	{
		glm::aligned_ivec4 const a(1, 2, 3, 4);
		glm::ivec4 const u(a);

		Error += a.x == u.x ? 0 : 1;
		Error += a.y == u.y ? 0 : 1;
		Error += a.z == u.z ? 0 : 1;
		Error += a.w == u.w ? 0 : 1;
	}

	{
		my_ivec4_aligned a;
		a.b = glm::ivec4(1, 2, 3, 4);

		my_ivec4_packed u;
		u.b = a.b;

		Error += a.b.x == u.b.x ? 0 : 1;
		Error += a.b.y == u.b.y ? 0 : 1;
		Error += a.b.z == u.b.z ? 0 : 1;
		Error += a.b.w == u.b.w ? 0 : 1;
	}

	return Error;
}

static int test_ctor()
{
	int Error = 0;

#	if GLM_HAS_CONSTEXPR
	{
		constexpr glm::aligned_ivec4 v(1);

		Error += v.x == 1 ? 0 : 1;
		Error += v.y == 1 ? 0 : 1;
		Error += v.z == 1 ? 0 : 1;
		Error += v.w == 1 ? 0 : 1;
	}

	{
		constexpr glm::packed_ivec4 v(1);

		Error += v.x == 1 ? 0 : 1;
		Error += v.y == 1 ? 0 : 1;
		Error += v.z == 1 ? 0 : 1;
		Error += v.w == 1 ? 0 : 1;
	}

	{
		constexpr glm::ivec4 v(1);

		Error += v.x == 1 ? 0 : 1;
		Error += v.y == 1 ? 0 : 1;
		Error += v.z == 1 ? 0 : 1;
		Error += v.w == 1 ? 0 : 1;
	}
#	endif//GLM_HAS_CONSTEXPR

	return Error;
}

static int test_aligned_ivec4()
{
	int Error = 0;

	glm::aligned_ivec4 const v(1, 2, 3, 4);
	Error += glm::all(glm::equal(v, glm::aligned_ivec4(1, 2, 3, 4))) ? 0 : 1;

	glm::aligned_ivec4 const u = v * 2;
	Error += glm::all(glm::equal(u, glm::aligned_ivec4(2, 4, 6, 8))) ? 0 : 1;

	return Error;
}

static int test_aligned_mat4()
{
	int Error = 0;

	glm::aligned_vec4 const u(1.f, 2.f, 3.f, 4.f);
	Error += glm::all(glm::equal(u, glm::aligned_vec4(1.f, 2.f, 3.f, 4.f), 0.0001f)) ? 0 : 1;

	glm::aligned_vec4 const v(1, 2, 3, 4);
	Error += glm::all(glm::equal(v, glm::aligned_vec4(1.f, 2.f, 3.f, 4.f), 0.0001f)) ? 0 : 1;

	glm::aligned_mat4 const m(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	glm::aligned_mat4 const t = glm::transpose(m);
	glm::aligned_mat4 const expected = glm::mat4(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
	Error += glm::all(glm::equal(t, expected, 0.0001f)) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_ctor();
	Error += test_copy();
	Error += test_aligned_ivec4();
	Error += test_aligned_mat4();

	return Error;
}

#else

int main()
{
	return 0;
}

#endif
