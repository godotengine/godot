#include <glm/ext/matrix_uint4x4_sized.hpp>

#if GLM_HAS_STATIC_ASSERT
static_assert(sizeof(glm::u8mat4x4) == 16, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::u16mat4x4) == 32, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::u32mat4x4) == 64, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::u64mat4x4) == 128, "uint64 size isn't 8 bytes on this platform");
#endif

static int test_comp()
{
	int Error = 0;

	Error += sizeof(glm::u8mat4x4) < sizeof(glm::u16mat4x4) ? 0 : 1;
	Error += sizeof(glm::u16mat4x4) < sizeof(glm::u32mat4x4) ? 0 : 1;
	Error += sizeof(glm::u32mat4x4) < sizeof(glm::u64mat4x4) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_comp();

	return Error;
}
