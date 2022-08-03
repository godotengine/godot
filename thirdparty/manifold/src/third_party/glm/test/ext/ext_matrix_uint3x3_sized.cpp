#include <glm/ext/matrix_uint3x3_sized.hpp>

#if GLM_HAS_STATIC_ASSERT
static_assert(sizeof(glm::u8mat3x3) == 9, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::u16mat3x3) == 18, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::u32mat3x3) == 36, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::u64mat3x3) == 72, "uint64 size isn't 8 bytes on this platform");
#endif

static int test_comp()
{
	int Error = 0;

	Error += sizeof(glm::u8mat3x3) < sizeof(glm::u16mat3x3) ? 0 : 1;
	Error += sizeof(glm::u16mat3x3) < sizeof(glm::u32mat3x3) ? 0 : 1;
	Error += sizeof(glm::u32mat3x3) < sizeof(glm::u64mat3x3) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_comp();

	return Error;
}
