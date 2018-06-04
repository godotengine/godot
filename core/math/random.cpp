#include "random.h"

Random::Random() :
		randbase() {}

void Random::seed(uint64_t seed) {
	return randbase.seed(seed);
}

void Random::randomize() {
	return randbase.randomize();
}

uint32_t Random::randi() {
	return randbase.rand();
}

double Random::randd(double from, double to) {
	if (isnan(to)) {
		if (isnan(from)) {
			// randd()
			from = (double)0;
			to = (double)1;
		} else {
			// randd(to)
			to = from;
			from = (double)0;
		}
	}

	return randbase.random(from, to);
}

void Random::_bind_methods() {

	ClassDB::bind_method(D_METHOD("seed", "seed"), &Random::seed);
	ClassDB::bind_method(D_METHOD("randi"), &Random::randi);
	ClassDB::bind_method(D_METHOD("randd", "from", "to"), &Random::randd, DEFVAL(NAN), DEFVAL(NAN));
	ClassDB::bind_method(D_METHOD("randomize"), &Random::randomize);
}
