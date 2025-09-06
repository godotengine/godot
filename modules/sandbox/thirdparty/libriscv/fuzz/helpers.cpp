#include <climits>

/* It is necessary to link with libgcc when fuzzing.
   See llvm.org/PR30643 for details. */
__attribute__((weak, no_sanitize("undefined")))
extern "C" __int128_t
__muloti4(__int128_t a, __int128_t b, int* overflow) {
	const int N = (int)(sizeof(__int128_t) * CHAR_BIT);
	const __int128_t MIN = (__int128_t)1 << (N - 1);
	const __int128_t MAX = ~MIN;
	*overflow = 0;
	__int128_t result = a * b;
	if (a == MIN) {
	if (b != 0 && b != 1)
	  *overflow = 1;
	return result;
	}
	if (b == MIN) {
	if (a != 0 && a != 1)
	  *overflow = 1;
	return result;
	}
	__int128_t sa = a >> (N - 1);
	__int128_t abs_a = (a ^ sa) - sa;
	__int128_t sb = b >> (N - 1);
	__int128_t abs_b = (b ^ sb) - sb;
	if (abs_a < 2 || abs_b < 2)
	return result;
	if (sa == sb) {
	if (abs_a > MAX / abs_b)
	  *overflow = 1;
	} else {
	if (abs_a > MIN / -abs_b)
	  *overflow = 1;
	}
	return result;
}
