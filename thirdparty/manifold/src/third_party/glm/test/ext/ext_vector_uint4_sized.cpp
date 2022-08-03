#include <glm/ext/vector_uint4_sized.hpp>

#if GLM_HAS_STATIC_ASSERT
static_assert(sizeof(glm::u8vec4) == 4, "int8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::u16vec4) == 8, "int16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::u32vec4) == 16, "int32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::u64vec4) == 32, "int64 size isn't 8 bytes on this platform");
#endif

static int test_size()
{
	int Error = 0;

	Error += sizeof(glm::u8vec4) == 4 ? 0 : 1;
	Error += sizeof(glm::u16vec4) == 8 ? 0 : 1;
	Error += sizeof(glm::u32vec4) == 16 ? 0 : 1;
	Error += sizeof(glm::u64vec4) == 32 ? 0 : 1;

	return Error;
}

static int test_comp()
{
	int Error = 0;

	Error += sizeof(glm::u8vec4) < sizeof(glm::u16vec4) ? 0 : 1;
	Error += sizeof(glm::u16vec4) < sizeof(glm::u32vec4) ? 0 : 1;
	Error += sizeof(glm::u32vec4) < sizeof(glm::u64vec4) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_size();
	Error += test_comp();

	return Error;
}
