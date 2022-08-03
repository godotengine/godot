#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>

#if GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE
#include <type_traits>

static_assert(sizeof(glm::bvec4) > sizeof(glm::bvec2), "Invalid sizeof");
static_assert(sizeof(glm::ivec4) > sizeof(glm::uvec2), "Invalid sizeof");
static_assert(sizeof(glm::dvec4) > sizeof(glm::dvec2), "Invalid sizeof");

static_assert(sizeof(glm::bvec4) == sizeof(glm::bvec3), "Invalid sizeof");
static_assert(sizeof(glm::uvec4) == sizeof(glm::uvec3), "Invalid sizeof");
static_assert(sizeof(glm::dvec4) == sizeof(glm::dvec3), "Invalid sizeof");

static int test_storage_aligned()
{
	int Error = 0;

	size_t size1_aligned = sizeof(glm::detail::storage<1, int, true>::type);
	Error += size1_aligned == sizeof(int) * 1 ? 0 : 1;
	size_t size2_aligned = sizeof(glm::detail::storage<2, int, true>::type);
	Error += size2_aligned == sizeof(int) * 2 ? 0 : 1;
	size_t size4_aligned = sizeof(glm::detail::storage<4, int, true>::type);
	Error += size4_aligned == sizeof(int) * 4 ? 0 : 1;

	size_t align1_aligned = alignof(glm::detail::storage<1, int, true>::type);
	Error += align1_aligned == 4 ? 0 : 1;
	size_t align2_aligned = alignof(glm::detail::storage<2, int, true>::type);
	Error += align2_aligned == 8 ? 0 : 1;
	size_t align4_aligned = alignof(glm::detail::storage<4, int, true>::type);
	Error += align4_aligned == 16 ? 0 : 1;

	return Error;
}

static int test_storage_unaligned()
{
	int Error = 0;

	size_t align1_unaligned = alignof(glm::detail::storage<1, int, false>::type);
	Error += align1_unaligned == sizeof(int) ? 0 : 1;
	size_t align2_unaligned = alignof(glm::detail::storage<2, int, false>::type);
	Error += align2_unaligned == sizeof(int) ? 0 : 1;
	size_t align3_unaligned = alignof(glm::detail::storage<3, int, false>::type);
	Error += align3_unaligned == sizeof(int) ? 0 : 1;
	size_t align4_unaligned = alignof(glm::detail::storage<4, int, false>::type);
	Error += align4_unaligned == sizeof(int) ? 0 : 1;

	return Error;
}

static int test_vec3_aligned()
{
	int Error = 0;

	struct Struct1
	{
		glm::vec4 A;
		float B;
		glm::vec3 C;
	};

	std::size_t const Size1 = sizeof(Struct1);
	Error += Size1 == 48 ? 0 : 1;

	struct Struct2
	{
		glm::vec4 A;
		glm::vec3 B;
		float C;
	};

	std::size_t const Size2 = sizeof(Struct2);
	Error += Size2 == 48 ? 0 : 1;

	return Error;
}

#endif

int main()
{
	int Error = 0;

#	if GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE
		Error += test_storage_aligned();
		Error += test_storage_unaligned();
		Error += test_vec3_aligned();
#	endif

	return Error;
}
