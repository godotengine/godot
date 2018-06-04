#ifndef RAND_CORE_H
#define RAND_CORE_H

#include "randombase.h"
#include "reference.h"

class Random : public Reference {
	GDCLASS(Random, Reference);

	RandomBase randbase;

protected:
	static void _bind_methods();

public:
	void seed(uint64_t seed);
	void randomize();

	uint32_t randi();
	double randd(double from = NAN, double to = NAN);

	Random();
};

#endif // RAND_CORE_H
