#include <glm/gtc/constants.hpp>
#include <glm/ext/quaternion_relational.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/quaternion_float_precision.hpp>
#include <glm/ext/quaternion_double.hpp>
#include <glm/ext/quaternion_double_precision.hpp>
#include <glm/ext/vector_float3.hpp>
#include <vector>

static int test_ctr()
{
	int Error(0);

#	if GLM_HAS_TRIVIAL_QUERIES
	//	Error += std::is_trivially_default_constructible<glm::quat>::value ? 0 : 1;
	//	Error += std::is_trivially_default_constructible<glm::dquat>::value ? 0 : 1;
	//	Error += std::is_trivially_copy_assignable<glm::quat>::value ? 0 : 1;
	//	Error += std::is_trivially_copy_assignable<glm::dquat>::value ? 0 : 1;
	Error += std::is_trivially_copyable<glm::quat>::value ? 0 : 1;
	Error += std::is_trivially_copyable<glm::dquat>::value ? 0 : 1;

	Error += std::is_copy_constructible<glm::quat>::value ? 0 : 1;
	Error += std::is_copy_constructible<glm::dquat>::value ? 0 : 1;
#	endif

#	if GLM_HAS_INITIALIZER_LISTS
	{
		glm::quat A{0, 1, 2, 3};

		std::vector<glm::quat> B{
			{0, 1, 2, 3},
		{0, 1, 2, 3}};
	}
#	endif//GLM_HAS_INITIALIZER_LISTS

	return Error;
}

static int test_two_axis_ctr()
{
	int Error = 0;

	glm::quat const q1(glm::vec3(1, 0, 0), glm::vec3(0, 1, 0));
	glm::vec3 const v1 = q1 * glm::vec3(1, 0, 0);
	Error += glm::all(glm::equal(v1, glm::vec3(0, 1, 0), 0.0001f)) ? 0 : 1;

	glm::quat const q2 = q1 * q1;
	glm::vec3 const v2 = q2 * glm::vec3(1, 0, 0);
	Error += glm::all(glm::equal(v2, glm::vec3(-1, 0, 0), 0.0001f)) ? 0 : 1;

	glm::quat const q3(glm::vec3(1, 0, 0), glm::vec3(-1, 0, 0));
	glm::vec3 const v3 = q3 * glm::vec3(1, 0, 0);
	Error += glm::all(glm::equal(v3, glm::vec3(-1, 0, 0), 0.0001f)) ? 0 : 1;

	glm::quat const q4(glm::vec3(0, 1, 0), glm::vec3(0, -1, 0));
	glm::vec3 const v4 = q4 * glm::vec3(0, 1, 0);
	Error += glm::all(glm::equal(v4, glm::vec3(0, -1, 0), 0.0001f)) ? 0 : 1;

	glm::quat const q5(glm::vec3(0, 0, 1), glm::vec3(0, 0, -1));
	glm::vec3 const v5 = q5 * glm::vec3(0, 0, 1);
	Error += glm::all(glm::equal(v5, glm::vec3(0, 0, -1), 0.0001f)) ? 0 : 1;

	return Error;
}

static int test_size()
{
	int Error = 0;

	std::size_t const A = sizeof(glm::quat);
	Error += 16 == A ? 0 : 1;
	std::size_t const B = sizeof(glm::dquat);
	Error += 32 == B ? 0 : 1;
	Error += glm::quat().length() == 4 ? 0 : 1;
	Error += glm::dquat().length() == 4 ? 0 : 1;
	Error += glm::quat::length() == 4 ? 0 : 1;
	Error += glm::dquat::length() == 4 ? 0 : 1;

	return Error;
}

static int test_precision()
{
	int Error = 0;

	Error += sizeof(glm::lowp_quat) <= sizeof(glm::mediump_quat) ? 0 : 1;
	Error += sizeof(glm::mediump_quat) <= sizeof(glm::highp_quat) ? 0 : 1;

	return Error;
}

static int test_constexpr()
{
#if GLM_HAS_CONSTEXPR
	static_assert(glm::quat::length() == 4, "GLM: Failed constexpr");
	static_assert(glm::quat(1.0f, glm::vec3(0.0f)).w > 0.0f, "GLM: Failed constexpr");
#endif

	return 0;
}

int main()
{
	int Error = 0;

	Error += test_ctr();
	Error += test_two_axis_ctr();
	Error += test_size();
	Error += test_precision();
	Error += test_constexpr();

	return Error;
}
