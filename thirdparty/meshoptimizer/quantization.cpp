// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <float.h>
#include <math.h>

union FloatBits
{
	float f;
	unsigned int ui;
};

unsigned short meshopt_quantizeHalf(float v)
{
	FloatBits u = {v};
	unsigned int ui = u.ui;

	int s = (ui >> 16) & 0x8000;
	int em = ui & 0x7fffffff;

	// bias exponent and round to nearest; 112 is relative exponent bias (127-15)
	int h = (em - (112 << 23) + (1 << 12)) >> 13;

	// underflow: flush to zero; 113 encodes exponent -14
	h = (em < (113 << 23)) ? 0 : h;

	// overflow: infinity; 143 encodes exponent 16
	h = (em >= (143 << 23)) ? 0x7c00 : h;

	// NaN; note that we convert all types of NaN to qNaN
	h = (em > (255 << 23)) ? 0x7e00 : h;

	return (unsigned short)(s | h);
}

float meshopt_quantizeFloat(float v, int N)
{
	assert(N >= 0 && N <= 23);

	FloatBits u = {v};
	unsigned int ui = u.ui;

	const int mask = (1 << (23 - N)) - 1;
	const int round = (1 << (23 - N)) >> 1;

	int e = ui & 0x7f800000;
	unsigned int rui = (ui + round) & ~mask;

	// round all numbers except inf/nan; this is important to make sure nan doesn't overflow into -0
	ui = e == 0x7f800000 ? ui : rui;

	// flush denormals to zero
	ui = e == 0 ? 0 : ui;

	u.ui = ui;
	return u.f;
}

float meshopt_dequantizeHalf(unsigned short h)
{
	unsigned int s = unsigned(h & 0x8000) << 16;
	int em = h & 0x7fff;

	// bias exponent and pad mantissa with 0; 112 is relative exponent bias (127-15)
	int r = (em + (112 << 10)) << 13;

	// denormal: flush to zero
	r = (em < (1 << 10)) ? 0 : r;

	// infinity/NaN; note that we preserve NaN payload as a byproduct of unifying inf/nan cases
	// 112 is an exponent bias fixup; since we already applied it once, applying it twice converts 31 to 255
	r += (em >= (31 << 10)) ? (112 << 23) : 0;

	FloatBits u;
	u.ui = s | r;
	return u.f;
}

int meshopt_computePositionExponent(const float* minv, const float* maxv, int min_exp, int max_bits)
{
	assert(min_exp >= -126);
	assert(max_bits >= 2 && max_bits <= 24);

	int exp = min_exp;

	// compute max absolute component to ensure that individual endpoints fit on a 24-bit signed grid
	float maxc = 0.f;

	for (int k = 0; k < 3; ++k)
	{
		maxc = maxc < fabsf(minv[k]) ? fabsf(minv[k]) : maxc;
		maxc = maxc < fabsf(maxv[k]) ? fabsf(maxv[k]) : maxc;
	}

	int maxc_exp = 0;
	float maxc_fr = frexpf(maxc, &maxc_exp);

	// maxc is representable as 2^(maxc_exp-24) * 24-bit *unsigned* integer
	// we have to use a 24-bit *signed* grid, so we have to chop off the last bit
	// however, rounding in the corner case (mantissa is 1.111...) may increase the exponent by 1 so we need to offset it
	int maxc_off = 23 - (maxc_fr >= 1.f - FLT_EPSILON / 2);

	exp = exp < maxc_exp - maxc_off ? maxc_exp - maxc_off : exp;

	// compute effective range with conservative rounding, to allow for some ambiguity in the caller's rounding direction
	float scale = ldexpf(1.f, -exp);
	float range = 0.f;

	for (int k = 0; k < 3; ++k)
	{
		float a = floorf(minv[k] * scale);
		float v = ceilf(maxv[k] * scale);
		range = range < v - a ? v - a : range;
	}

	// range must be representable as max_bits unsigned integer
	int range_exp = 0;
	float range_fr = frexpf(range, &range_exp);

	exp += (range_exp > max_bits) ? range_exp - max_bits : 0;

	// correct range if we are at the rounding boundary that would push us to overflow max_bits
	exp += range_exp > max_bits && range_fr >= 1.f - 1.f / float((1 << max_bits) - 1);

	return exp;
}
