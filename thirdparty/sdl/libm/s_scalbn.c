#include "SDL_internal.h"
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/*
 * scalbln(double x, long n)
 * scalbln(x,n) returns x * 2**n computed by exponent
 * manipulation rather than by actually performing an
 * exponentiation or a multiplication.
 */

#include "math_libm.h"
#include "math_private.h"
#include <limits.h>

#ifdef __WATCOMC__ /* Watcom defines huge=__huge */
#undef huge
#endif

static const double
two54  = 1.80143985094819840000e+16, /* 0x43500000, 0x00000000 */
twom54 = 5.55111512312578270212e-17, /* 0x3C900000, 0x00000000 */
huge   = 1.0e+300,
tiny   = 1.0e-300;

double scalbln(double x, long n)
{
	int32_t k, hx, lx;

	EXTRACT_WORDS(hx, lx, x);
	k = (hx & 0x7ff00000) >> 20; /* extract exponent */
	if (k == 0) { /* 0 or subnormal x */
		if ((lx | (hx & 0x7fffffff)) == 0)
			return x; /* +-0 */
		x *= two54;
		GET_HIGH_WORD(hx, x);
		k = ((hx & 0x7ff00000) >> 20) - 54;
	}
	if (k == 0x7ff)
		return x + x; /* NaN or Inf */
	k = (int32_t)(k + n);
	if (k > 0x7fe)
		return huge * copysign(huge, x); /* overflow */
	if (n < -50000)
		return tiny * copysign(tiny, x); /* underflow */
	if (k > 0) { /* normal result */
		SET_HIGH_WORD(x, (hx & 0x800fffff) | (k << 20));
		return x;
	}
	if (k <= -54) {
		if (n > 50000) /* in case integer overflow in n+k */
			return huge * copysign(huge, x); /* overflow */
		return tiny * copysign(tiny, x); /* underflow */
	}
	k += 54; /* subnormal result */
	SET_HIGH_WORD(x, (hx & 0x800fffff) | (k << 20));
	return x * twom54;
}
libm_hidden_def(scalbln)


double scalbn(double x, int n)
{
	return scalbln(x, n);
}
libm_hidden_def(scalbn)
