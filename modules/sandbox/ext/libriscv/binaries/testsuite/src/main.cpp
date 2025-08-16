#include <cassert>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <errno.h>
#include "floating-point.hpp"
static constexpr double PI = std::atan(1.0) * 4.0;

inline bool kinda32(float val, float expectation) {
	return val >= expectation-FLT_EPSILON
		&& val < expectation+FLT_EPSILON;
}
inline bool kinda64(float val, double expectation) {
	return val >= expectation-FLT_EPSILON
		&& val < expectation+FLT_EPSILON;
}
inline bool kinda(double val, double expectation) {
	return val >= expectation-FLT_EPSILON
		&& val < expectation+FLT_EPSILON;
}

static struct {
	double sum = 0.0;
	int counter = 0;
	int sign = 1;
} pi;

static double compute_more_pi()
{
    pi.sum += pi.sign / (2.0 * pi.counter + 1.0);
	pi.counter ++;
	pi.sign = -pi.sign;
    return 4.0 * pi.sum;
}

int main()
{
	assert((int32_t) -44.0f == -44);
	assert((int32_t) -44.0 == -44);
	assert((uint32_t) 44.0f == 44);
	assert((uint32_t) 44.0 == 44);
	const float cf = 5.0;
	const float* pf = &cf;
	assert(*pf == cf);
	const double cd = 5.0;
	const double* pd = &cd;
	assert(*pd == cd);

	assert(test_fadd(1.0f, 1.0f) == 2.0f);
	assert(test_fadd(2.0f, 2.0f) == 4.0f);

	assert(test_fsub(2.0f, 1.0f) == 1.0f);
	assert(test_fsub(4.0f, 2.0f) == 2.0f);

	assert(test_fmul(2.0f, 2.0f) == 4.0f);
	assert(test_fmul(4.0f, 4.0f) == 16.0f);
	assert(test_fmul(2.5f, 2.5f) == (2.5f * 2.5f));

	assert(test_fdiv(16.0f, 4.0f) == 4.0f);
	assert(test_fdiv(4.0f, 2.0f) == 2.0f);
	assert(test_fdiv(4.0f, 3.0f) == (4.0f / 3.0f));

	assert(test_fmax(4.0f, 3.0f) == 4.0f);
	assert(test_fmax(0.999f, 0.998f) == 0.999f);

	assert(test_fmin(4.0f, 3.0f) == 3.0f);
	assert(test_fmin(0.999f, 0.998f) == 0.998f);

	assert(test_ftod(4.0f) == 4.0);
	assert(test_ftod(1.5f) == 1.5);
	assert(kinda64(test_ftod(0.999f), 0.999));

	assert(test_dtof(4.0) == 4.0f);
	assert(test_dtof(1.5) == 1.5f);
	assert(kinda32(test_dtof(0.999), 0.999f));

	assert(test_fneg(1.0f) == -1.0f);
	assert(test_fneg(-1.0f) == 1.0f);
	assert(test_dneg(16.0) == -16.0);
	assert(test_dneg(-16.0) == 16.0);

	assert(test_fmadd(4.0f, 4.0f, 16.0f) == 32.0f);
	assert(test_fmsub(4.0f, 4.0f, 16.0f) == 0.0f);
	//assert(test_fnmadd(4.0f, 4.0f, 0.0f) == -16.0f);
	//assert(test_fnmsub(4.0f, 4.0f, -16.0f) == 0.0f);

	assert(test_fsqrt(4.0f) == 2.0f);
	assert(test_fsqrt(2.0f) > 1.41f);
	assert(test_fsqrt(2.0f) < 1.42f);
	assert(test_fsqrt(1.0f) == 1.0f);
	assert(test_dsqrt(1.0) == 1.0);
	assert(test_dsqrt(4.0) == 2.0);

	assert(test_fpow(2.0f, 2.0f) == 4.0f);
	assert(test_fpow(3.0f, 3.0f) == 27.0f);
	assert(test_dpow(2.0, 2.0) == 4.0);
	assert(test_dpow(3.0, 3.0) == 27.0);

	assert(test_sinf(0.0f) == 0.0f);
	assert(test_cosf(0.0f) == 1.0f);
	assert(test_tanf(0.0f) == 0.0f);

	printf("sin(0.0pi) = %f\n", test_sinf(0.0*PI)); // ~0.0
	printf("sin(0.5pi) = %f\n", test_sinf(0.5*PI)); // 1.0
	printf("sin(1.0pi) = %f\n", test_sinf(1.0*PI)); // ~0.0
	printf("sin(1.5pi) = %f\n", test_sinf(1.5*PI)); // -1.0
	printf("sin(2.0pi) = %f\n", test_sinf(2.0*PI)); // ~0.0
	//assert(kinda32(test_sinf(0.0), test_sinf(2.0*PI)));
	assert(kinda32(test_sinf(PI), 0.0f));
	assert(test_cosf(PI) == -1.0f);
	assert(test_tanf(PI) < 0.001f);

	assert(kinda(compute_more_pi(), 4.0));
	assert(kinda(compute_more_pi(), 2.66666666666));
	assert(kinda(compute_more_pi(), 3.46666666666));
	// ...
}
