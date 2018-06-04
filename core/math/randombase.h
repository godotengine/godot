#ifndef RANDBASE_CORE_H
#define RANDBASE_CORE_H

#include "math.h"
#include "math_defs.h"
#include "thirdparty/misc/pcg.h"

class RandomBase {
	pcg32_random_t pcg;

public:
	static const uint64_t DEFAULT_SEED = 12047754176567800795ULL;
	static const uint64_t DEFAULT_INC = PCG_DEFAULT_INC_64;
	static const uint64_t RANDOM_MAX = 4294967295;

	RandomBase(uint64_t seed = DEFAULT_SEED, uint64_t inc = PCG_DEFAULT_INC_64);

	void seed(uint64_t x);
	void randomize();
	uint32_t rand();
	_ALWAYS_INLINE_ double randf() { return (double)rand() / (double)RANDOM_MAX; }
	_ALWAYS_INLINE_ float randd() { return (float)rand() / (float)RANDOM_MAX; }

	double random(double from, double to);
	float random(float from, float to);
	real_t random(int from, int to) { return (real_t)random((real_t)from, (real_t)to); }
};

#endif // RANDBASE_CORE_H
