#include <glm/gtc/vec1.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/glm.hpp>

static int test_vec1_ctor()
{
	int Error = 0;

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_ENABLE
	{
		union pack
		{
			glm::vec1 f;
			glm::ivec1 i;
		} A, B;

		A.f = glm::vec1(0);
		Error += glm::all(glm::equal(A.i, glm::ivec1(0))) ? 0 : 1;

		B.f = glm::vec1(1);
		Error += glm::all(glm::equal(B.i, glm::ivec1(1065353216))) ? 0 : 1;
	}
#	endif//GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_ENABLE

	return Error;
}

static int test_vec2_ctor()
{
	int Error = 0;

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_ENABLE
	{
		union pack
		{
			glm::vec2 f;
			glm::ivec2 i;
		} A, B;

		A.f = glm::vec2(0);
		Error += glm::all(glm::equal(A.i, glm::ivec2(0))) ? 0 : 1;

		B.f = glm::vec2(1);
		Error += glm::all(glm::equal(B.i, glm::ivec2(1065353216))) ? 0 : 1;
	}
#	endif

	return Error;
}

static int test_vec3_ctor()
{
	int Error = 0;

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_ENABLE
	{
		union pack
		{
			glm::vec3 f;
			glm::ivec3 i;
		} A, B;

		A.f = glm::vec3(0);
		Error += glm::all(glm::equal(A.i, glm::ivec3(0))) ? 0 : 1;

		B.f = glm::vec3(1);
		Error += glm::all(glm::equal(B.i, glm::ivec3(1065353216))) ? 0 : 1;
	}
#	endif

	return Error;
}

static int test_vec4_ctor()
{
	int Error = 0;

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_ENABLE
	{
		union pack
		{
			glm::vec4 f;
			glm::ivec4 i;
		} A, B;

		A.f = glm::vec4(0);
		Error += glm::all(glm::equal(A.i, glm::ivec4(0))) ? 0 : 1;

		B.f = glm::vec4(1);
		Error += glm::all(glm::equal(B.i, glm::ivec4(1065353216))) ? 0 : 1;
	}
#	endif

	return Error;
}

static int test_mat2x2_ctor()
{
	int Error = 0;

#	if GLM_LANG & GLM_LANG_CXX11_FLAG
	{
		union pack
		{
			glm::mat2x2 f;
			glm::mat2x2 i;
		} A, B;

		A.f = glm::mat2x2(0);
		Error += glm::all(glm::equal(A.i[0], glm::vec2(0), glm::epsilon<float>())) ? 0 : 1;

		B.f = glm::mat2x2(1);
		Error += glm::all(glm::equal(B.i[0], glm::vec2(1, 0), glm::epsilon<float>())) ? 0 : 1;
	}
#	endif//GLM_LANG & GLM_LANG_CXX11_FLAG

	return Error;
}

static int test_mat2x3_ctor()
{
	int Error = 0;

#	if GLM_LANG & GLM_LANG_CXX11_FLAG
	{
		union pack
		{
			glm::mat2x3 f;
			glm::mat2x3 i;
		} A, B;

		A.f = glm::mat2x3(0);
		Error += glm::all(glm::equal(A.i[0], glm::vec3(0), glm::epsilon<float>())) ? 0 : 1;

		B.f = glm::mat2x3(1);
		Error += glm::all(glm::equal(B.i[0], glm::vec3(1, 0, 0), glm::epsilon<float>())) ? 0 : 1;
	}
#	endif//GLM_LANG & GLM_LANG_CXX11_FLAG

	return Error;
}

static int test_mat2x4_ctor()
{
	int Error = 0;

#	if GLM_LANG & GLM_LANG_CXX11_FLAG
	{
		union pack
		{
			glm::mat2x4 f;
			glm::mat2x4 i;
		} A, B;

		A.f = glm::mat2x4(0);
		glm::vec4 const C(0, 0, 0, 0);
		Error += glm::all(glm::equal(A.i[0], C, glm::epsilon<float>())) ? 0 : 1;

		B.f = glm::mat2x4(1);
		glm::vec4 const D(1, 0, 0, 0);
		Error += glm::all(glm::equal(B.i[0], D, glm::epsilon<float>())) ? 0 : 1;
	}
#	endif//GLM_LANG & GLM_LANG_CXX11_FLAG

	return Error;
}

static int test_mat3x2_ctor()
{
	int Error = 0;

#	if GLM_LANG & GLM_LANG_CXX11_FLAG
	{
		union pack
		{
			glm::mat3x2 f;
			glm::mat3x2 i;
		} A, B;

		A.f = glm::mat3x2(0);
		Error += glm::all(glm::equal(A.i[0], glm::vec2(0), glm::epsilon<float>())) ? 0 : 1;

		B.f = glm::mat3x2(1);
		Error += glm::all(glm::equal(B.i[0], glm::vec2(1, 0), glm::epsilon<float>())) ? 0 : 1;
	}
#	endif//GLM_LANG & GLM_LANG_CXX11_FLAG

	return Error;
}

static int test_mat3x3_ctor()
{
	int Error = 0;

#	if GLM_LANG & GLM_LANG_CXX11_FLAG
	{
		union pack
		{
			glm::mat3x3 f;
			glm::mat3x3 i;
		} A, B;

		A.f = glm::mat3x3(0);
		Error += glm::all(glm::equal(A.i[0], glm::vec3(0), glm::epsilon<float>())) ? 0 : 1;

		B.f = glm::mat3x3(1);
		Error += glm::all(glm::equal(B.i[0], glm::vec3(1, 0, 0), glm::epsilon<float>())) ? 0 : 1;
	}
#	endif//GLM_LANG & GLM_LANG_CXX11_FLAG

	return Error;
}

static int test_mat3x4_ctor()
{
	int Error = 0;

#	if GLM_LANG & GLM_LANG_CXX11_FLAG
	{
		union pack
		{
			glm::mat3x4 f;
			glm::mat3x4 i;
		} A, B;

		A.f = glm::mat3x4(0);
		Error += glm::all(glm::equal(A.i[0], glm::vec4(0), glm::epsilon<float>())) ? 0 : 1;

		B.f = glm::mat3x4(1);
		Error += glm::all(glm::equal(B.i[0], glm::vec4(1, 0, 0, 0), glm::epsilon<float>())) ? 0 : 1;
	}
#	endif//GLM_LANG & GLM_LANG_CXX11_FLAG

	return Error;
}

static int test_mat4x2_ctor()
{
	int Error = 0;

#	if GLM_LANG & GLM_LANG_CXX11_FLAG
	{
		union pack
		{
			glm::mat4x2 f;
			glm::mat4x2 i;
		} A, B;

		A.f = glm::mat4x2(0);
		Error += glm::all(glm::equal(A.i[0], glm::vec2(0), glm::epsilon<float>())) ? 0 : 1;

		B.f = glm::mat4x2(1);
		Error += glm::all(glm::equal(B.i[0], glm::vec2(1, 0), glm::epsilon<float>())) ? 0 : 1;
	}
#	endif//GLM_LANG & GLM_LANG_CXX11_FLAG

	return Error;
}

static int test_mat4x3_ctor()
{
	int Error = 0;

#	if GLM_LANG & GLM_LANG_CXX11_FLAG
	{
		union pack
		{
			glm::mat4x3 f;
			glm::mat4x3 i;
		} A, B;

		A.f = glm::mat4x3(0);
		Error += glm::all(glm::equal(A.i[0], glm::vec3(0), glm::epsilon<float>())) ? 0 : 1;

		B.f = glm::mat4x3(1);
		Error += glm::all(glm::equal(B.i[0], glm::vec3(1, 0, 0), glm::epsilon<float>())) ? 0 : 1;
	}
#	endif//GLM_LANG & GLM_LANG_CXX11_FLAG

	return Error;
}

static int test_mat4x4_ctor()
{
	int Error = 0;

#	if GLM_LANG & GLM_LANG_CXX11_FLAG
	{
		union pack
		{
			glm::mat4 f;
			glm::mat4 i;
		} A, B;

		A.f = glm::mat4(0);
		Error += glm::all(glm::equal(A.i[0], glm::vec4(0), glm::epsilon<float>())) ? 0 : 1;

		B.f = glm::mat4(1);
		Error += glm::all(glm::equal(B.i[0], glm::vec4(1, 0, 0, 0), glm::epsilon<float>())) ? 0 : 1;
	}
#	endif//GLM_LANG & GLM_LANG_CXX11_FLAG

	return Error;
}

static int test_quat_ctor()
{
	int Error = 0;

#	if GLM_LANG & GLM_LANG_CXX11_FLAG
	{
		union pack
		{
			glm::quat f;
			glm::quat i;
		} A, B;

		A.f = glm::quat(0, 0, 0, 0);
		Error += glm::all(glm::equal(A.i, glm::quat(0, 0, 0, 0), glm::epsilon<float>())) ? 0 : 1;

		B.f = glm::quat(1, 1, 1, 1);
		Error += glm::all(glm::equal(B.i, glm::quat(1, 1, 1, 1), glm::epsilon<float>())) ? 0 : 1;
	}
#	endif//GLM_LANG & GLM_LANG_CXX11_FLAG

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_vec1_ctor();
	Error += test_vec2_ctor();
	Error += test_vec3_ctor();
	Error += test_vec4_ctor();
	Error += test_mat2x2_ctor();
	Error += test_mat2x3_ctor();
	Error += test_mat2x4_ctor();
	Error += test_mat3x2_ctor();
	Error += test_mat3x3_ctor();
	Error += test_mat3x4_ctor();
	Error += test_mat4x2_ctor();
	Error += test_mat4x3_ctor();
	Error += test_mat4x4_ctor();
	Error += test_quat_ctor();

	return Error;
}
