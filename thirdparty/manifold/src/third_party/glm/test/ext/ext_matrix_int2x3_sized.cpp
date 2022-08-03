#include <glm/ext/matrix_int2x3_sized.hpp>

#if GLM_HAS_STATIC_ASSERT
static_assert(sizeof(glm::i8mat2x3) == 6, "int8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::i16mat2x3) == 12, "int16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::i32mat2x3) == 24, "int32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::i64mat2x3) == 48, "int64 size isn't 8 bytes on this platform");
#endif

static int test_comp()
{
	int Error = 0;

	Error += sizeof(glm::i8mat2x3) < sizeof(glm::i16mat2x3) ? 0 : 1;
	Error += sizeof(glm::i16mat2x3) < sizeof(glm::i32mat2x3) ? 0 : 1;
	Error += sizeof(glm::i32mat2x3) < sizeof(glm::i64mat2x3) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_comp();

	return Error;
}
