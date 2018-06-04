#include "randombase.h"

#include "core/os/os.h"

RandomBase::RandomBase(uint64_t seed, uint64_t inc) :
		pcg() {
	pcg.state = seed;
	pcg.inc = inc;
}

void RandomBase::seed(uint64_t seed) {
	pcg.state = seed;
}

void RandomBase::randomize() {
	seed(OS::get_singleton()->get_ticks_usec() * pcg.state + PCG_DEFAULT_INC_64);
}

uint32_t RandomBase::rand() {
	return pcg32_random_r(&pcg);
}

double RandomBase::random(double from, double to) {
	unsigned int r = rand();
	double ret = (double)r / (double)RANDOM_MAX;
	return (ret) * (to - from) + from;
}

float RandomBase::random(float from, float to) {
	unsigned int r = rand();
	float ret = (float)r / (float)RANDOM_MAX;
	return (ret) * (to - from) + from;
}
