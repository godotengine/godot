#include <glm/ext/matrix_int3x3_sized.hpp>

#if GLM_HAS_STATIC_ASSERT
static_assert(sizeof(glm::i8mat3x3) == 9, "int8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::i16mat3x3) == 18, "int16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::i32mat3x3) == 36, "int32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::i64mat3x3) == 72, "int64 size isn't 8 bytes on this platform");
#endif

static int test_comp()
{
	int Error = 0;

	Error += sizeof(glm::i8mat3x3) < sizeof(glm::i16mat3x3) ? 0 : 1;
	Error += sizeof(glm::i16mat3x3) < sizeof(glm::i32mat3x3) ? 0 : 1;
	Error += sizeof(glm::i32mat3x3) < sizeof(glm::i64mat3x3) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_comp();

	return Error;
}
