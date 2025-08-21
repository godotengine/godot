#include <array>
#include <cassert>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <errno.h>
#include "floating-point.hpp"
static const double PI = std::atan(1.0) * 4.0;

// These are largely arbitrary, but so is the exactness
// of certain algorithms...
inline bool kinda32(float val, float expectation) {
	return val >= expectation-1e-4
		&& val < expectation+1e-4;
}
inline bool kinda64(float val, double expectation) {
	return val >= expectation-1e-7
		&& val < expectation+1e-7;
}

int main()
{
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
	assert(test_ftod(2.0) == 2.0f);
	assert(test_ftod(1.5f) == 1.5);
	assert(kinda64(test_ftod(0.999f), 0.999));

	assert(test_dtof(4.0) == 4.0f);
	assert(test_dtof(2.0) == 2.0f);
	assert(test_dtof(1.5) == 1.5f);
	assert(kinda32(test_dtof(0.999), 0.999f));

	assert(test_fneg(1.0f) == -1.0f);
	assert(test_fneg(-1.0f) == 1.0f);
	assert(test_dneg(16.0) == -16.0);
	assert(test_dneg(-16.0) == 16.0);

	assert(test_fmadd(4.0f, 4.0f, 16.0f) == 32.0f);
	assert(test_fmadd(4.0f, 2.0f, 0.0f) == 8.0f);
	assert(test_fmadd(1.0f, 1.0f, 31.0f) == 32.0f);
	assert(test_fmsub(4.0f, 4.0f, 16.0f) == 0.0f);
	assert(test_fnmadd(4.0f, 4.0f, 0.0f) == -16.0f);
	assert(test_fnmsub(4.0f, 4.0f, -16.0f) == 0.0f);

	std::array<float, 8> a = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	std::array<float, 8> b = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	assert(test_dotp(a.data(), b.data(), a.size()) == 8.0f);

	assert(test_fsqrt(4.0f) == 2.0f);
	assert(test_fsqrt(2.0f) > 1.41f);
	assert(test_fsqrt(2.0f) < 1.42f);
	assert(test_fsqrt(1.0f) == 1.0f);
	assert(test_dsqrt(1.0) == 1.0);
	assert(test_dsqrt(4.0) == 2.0);

	assert(test_fpow(2.0f, 2.0f) == 4.0f);
	assert(kinda32(test_fpow(3.0f, 3.0f), 27.0f));
	assert(test_dpow(2.0, 2.0) == 4.0);
	//printf("dpow(3.0, 3.0) = %f\n", test_dpow(3.0, 3.0));
	assert(kinda64(test_dpow(3.0, 3.0), 27.0));

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
}
