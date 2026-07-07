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
 * modf(double x, double *iptr)
 * return fraction part of x, and return x's integral part in *iptr.
 * Method:
 *	Bit twiddling.
 *
 * Exception:
 *	No exception.
 */

#include "math_libm.h"
#include "math_private.h"

static const double one = 1.0;

double modf(double x, double *iptr)
{
	int32_t i0,i1,_j0;
	u_int32_t i;
	EXTRACT_WORDS(i0,i1,x);
	_j0 = ((i0>>20)&0x7ff)-0x3ff;	/* exponent of x */
	if(_j0<20) {			/* integer part in high x */
	    if(_j0<0) {			/* |x|<1 */
	        INSERT_WORDS(*iptr,i0&0x80000000,0);	/* *iptr = +-0 */
		return x;
	    } else {
		i = (0x000fffff)>>_j0;
		if(((i0&i)|i1)==0) {		/* x is integral */
		    *iptr = x;
		    INSERT_WORDS(x,i0&0x80000000,0);	/* return +-0 */
		    return x;
		} else {
		    INSERT_WORDS(*iptr,i0&(~i),0);
		    return x - *iptr;
		}
	    }
	} else if (_j0>51) {		/* no fraction part */
	    *iptr = x*one;
	    /* We must handle NaNs separately.  */
	    if (_j0 == 0x400 && ((i0 & 0xfffff) | i1))
	      return x*one;
	    INSERT_WORDS(x,i0&0x80000000,0);	/* return +-0 */
	    return x;
	} else {			/* fraction part in low x */
	    i = ((u_int32_t)(0xffffffff))>>(_j0-20);
	    if((i1&i)==0) { 		/* x is integral */
		*iptr = x;
		INSERT_WORDS(x,i0&0x80000000,0);	/* return +-0 */
		return x;
	    } else {
	        INSERT_WORDS(*iptr,i0,i1&(~i));
		return x - *iptr;
	    }
	}
}
libm_hidden_def(modf)
