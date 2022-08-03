#include <glm/ext/vector_uint1_sized.hpp>

#if GLM_HAS_STATIC_ASSERT
	static_assert(sizeof(glm::u8vec1) == 1, "uint8 size isn't 1 byte on this platform");
	static_assert(sizeof(glm::u16vec1) == 2, "uint16 size isn't 2 bytes on this platform");
	static_assert(sizeof(glm::u32vec1) == 4, "uint32 size isn't 4 bytes on this platform");
	static_assert(sizeof(glm::u64vec1) == 8, "uint64 size isn't 8 bytes on this platform");
#endif

static int test_size()
{
	int Error = 0;

	Error += sizeof(glm::u8vec1) == 1 ? 0 : 1;
	Error += sizeof(glm::u16vec1) == 2 ? 0 : 1;
	Error += sizeof(glm::u32vec1) == 4 ? 0 : 1;
	Error += sizeof(glm::u64vec1) == 8 ? 0 : 1;

	return Error;
}

static int test_comp()
{
	int Error = 0;

	Error += sizeof(glm::u8vec1) < sizeof(glm::u16vec1) ? 0 : 1;
	Error += sizeof(glm::u16vec1) < sizeof(glm::u32vec1) ? 0 : 1;
	Error += sizeof(glm::u32vec1) < sizeof(glm::u64vec1) ? 0 : 1;

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_size();
	Error += test_comp();

	return Error;
}
