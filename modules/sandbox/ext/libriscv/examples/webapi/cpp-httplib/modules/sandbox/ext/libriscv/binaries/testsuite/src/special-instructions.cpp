#include <cstdint>

inline uint32_t rdcycle()
{
	union {
		uint64_t whole;
		uint32_t word[2];
	};
	asm ("rdcycleh %0\n rdcycle %1\n" : "=r"(word[1]), "=r"(word[0]));
	return whole;
}
inline uint64_t rdtime()
{
	union {
		uint64_t whole;
		uint32_t word[2];
	};
	asm ("rdtimeh %0\n rdtime %1\n" : "=r"(word[1]), "=r"(word[0]));
	return whole;
}
