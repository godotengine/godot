#include <cassert>
#include <cstdint>
#include <cstdio>

static inline int64_t clmul(int64_t x1, int64_t x2) {
	int64_t result;
	asm("clmul %0, %1, %2" : "=r"(result) : "r"(x1), "r"(x2));
	return result;
}
static inline int64_t clmulh(int64_t x1, int64_t x2) {
	int64_t result;
	asm("clmulh %0, %1, %2" : "=r"(result) : "r"(x1), "r"(x2));
	return result;
}
static inline int64_t clmulr(int64_t x1, int64_t x2) {
	int64_t result;
	asm("clmulr %0, %1, %2" : "=r"(result) : "r"(x1), "r"(x2));
	return result;
}

int main()
{
	printf("clmul(2, 25) = %ld\n", clmul(2, 25));
	printf("clmul(-1284, 65535) = %ld\n", clmul(-1284, 65535));
	printf("clmul(-1284, 25) = %ld\n", clmul(-1284, 25));

	assert(clmul(2, 25) == 50);
	assert(clmul(-1284, 65535) == 50419284);
	assert(clmul(-1284, 25) == -32036);

	printf("clmulh(2, 25) = %ld\n", clmulh(2, 25));
	printf("clmulh(-1284, 65535) = %ld\n", clmulh(-1284, 65535));
	printf("clmulh(-1284, 25) = %ld\n", clmulh(-1284, 25));

	assert(clmulh(2, 25) == 0);

	printf("clmulr(2, 25) = %ld\n", clmulr(2, 25));
	printf("clmulr(-1284, 65535) = %ld\n", clmulr(-1284, 65535));
	printf("clmulr(-1284, 25) = %ld\n", clmulr(-1284, 25));

	assert(clmulr(2, 25) == 0);

	printf("Tests passed!\n");
}
