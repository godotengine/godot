#include <glm/ext/scalar_int_sized.hpp>

#if GLM_HAS_STATIC_ASSERT
	static_assert(sizeof(glm::int8) == 1, "int8 size isn't 1 byte on this platform");
	static_assert(sizeof(glm::int16) == 2, "int16 size isn't 2 bytes on this platform");
	static_assert(sizeof(glm::int32) == 4, "int32 size isn't 4 bytes on this platform");
	static_assert(sizeof(glm::int64) == 8, "int64 size isn't 8 bytes on this platform");
	static_assert(sizeof(glm::int16) == sizeof(short), "signed short size isn't 4 bytes on this platform");
	static_assert(sizeof(glm::int32) == sizeof(int), "signed int size isn't 4 bytes on this platform");
#endif

static int test_size()
{
	int Error = 0;

	Error += sizeof(glm::int8) == 1 ? 0 : 1;
	Error += sizeof(glm::int16) == 2 ? 0 : 1;
	Error += sizeof(glm::int32) == 4 ? 0 : 1;
	Error += sizeof(glm::int64) == 8 ? 0 : 1;

	return Error;
}

static int test_comp()
{
	int Error = 0;

	Error += sizeof(glm::int8) < sizeof(glm::int16) ? 0 : 1;
	Error += sizeof(glm::int16) < sizeof(glm::int32) ? 0 : 1;
	Error += sizeof(glm::int32) < sizeof(glm::int64) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_size();
	Error += test_comp();

	return Error;
}
