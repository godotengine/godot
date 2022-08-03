#include <glm/ext/vector_int2_sized.hpp>

#if GLM_HAS_STATIC_ASSERT
	static_assert(sizeof(glm::i8vec2) == 2, "int8 size isn't 1 byte on this platform");
	static_assert(sizeof(glm::i16vec2) == 4, "int16 size isn't 2 bytes on this platform");
	static_assert(sizeof(glm::i32vec2) == 8, "int32 size isn't 4 bytes on this platform");
	static_assert(sizeof(glm::i64vec2) == 16, "int64 size isn't 8 bytes on this platform");
#endif

static int test_size()
{
	int Error = 0;

	Error += sizeof(glm::i8vec2) == 2 ? 0 : 1;
	Error += sizeof(glm::i16vec2) == 4 ? 0 : 1;
	Error += sizeof(glm::i32vec2) == 8 ? 0 : 1;
	Error += sizeof(glm::i64vec2) == 16 ? 0 : 1;

	return Error;
}

static int test_comp()
{
	int Error = 0;

	Error += sizeof(glm::i8vec2) < sizeof(glm::i16vec2) ? 0 : 1;
	Error += sizeof(glm::i16vec2) < sizeof(glm::i32vec2) ? 0 : 1;
	Error += sizeof(glm::i32vec2) < sizeof(glm::i64vec2) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_size();
	Error += test_comp();

	return Error;
}
