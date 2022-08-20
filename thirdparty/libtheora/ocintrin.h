/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation and contributors http://www.xiph.org/ *
 *                                                                  *
 ********************************************************************

  function:
    last mod: $Id: ocintrin.h 16503 2009-08-22 18:14:02Z giles $

 ********************************************************************/

/*Some common macros for potential platform-specific optimization.*/
#include <math.h>
#if !defined(_ocintrin_H)
# define _ocintrin_H (1)

/*Some specific platforms may have optimized intrinsic or inline assembly
   versions of these functions which can substantially improve performance.
  We define macros for them to allow easy incorporation of these non-ANSI
   features.*/

/*Note that we do not provide a macro for abs(), because it is provided as a
   library function, which we assume is translated into an intrinsic to avoid
   the function call overhead and then implemented in the smartest way for the
   target platform.
  With modern gcc (4.x), this is true: it uses cmov instructions if the
   architecture supports it and branchless bit-twiddling if it does not (the
   speed difference between the two approaches is not measurable).
  Interestingly, the bit-twiddling method was patented in 2000 (US 6,073,150)
   by Sun Microsystems, despite prior art dating back to at least 1996:
   http://web.archive.org/web/19961201174141/www.x86.org/ftp/articles/pentopt/PENTOPT.TXT
  On gcc 3.x, however, our assumption is not true, as abs() is translated to a
   conditional jump, which is horrible on deeply piplined architectures (e.g.,
   all consumer architectures for the past decade or more).
  Also be warned that -C*abs(x) where C is a constant is mis-optimized as
   abs(C*x) on every gcc release before 4.2.3.
  See bug http://gcc.gnu.org/bugzilla/show_bug.cgi?id=34130 */

/*Modern gcc (4.x) can compile the naive versions of min and max with cmov if
   given an appropriate architecture, but the branchless bit-twiddling versions
   are just as fast, and do not require any special target architecture.
  Earlier gcc versions (3.x) compiled both code to the same assembly
   instructions, because of the way they represented ((_b)>(_a)) internally.*/
#define OC_MAXI(_a,_b)      ((_a)-((_a)-(_b)&-((_b)>(_a))))
#define OC_MINI(_a,_b)      ((_a)+((_b)-(_a)&-((_b)<(_a))))
/*Clamps an integer into the given range.
  If _a>_c, then the lower bound _a is respected over the upper bound _c (this
   behavior is required to meet our documented API behavior).
  _a: The lower bound.
  _b: The value to clamp.
  _c: The upper boud.*/
#define OC_CLAMPI(_a,_b,_c) (OC_MAXI(_a,OC_MINI(_b,_c)))
#define OC_CLAMP255(_x)     ((unsigned char)((((_x)<0)-1)&((_x)|-((_x)>255))))
/*This has a chance of compiling branchless, and is just as fast as the
   bit-twiddling method, which is slightly less portable, since it relies on a
   sign-extended rightshift, which is not guaranteed by ANSI (but present on
   every relevant platform).*/
#define OC_SIGNI(_a)        (((_a)>0)-((_a)<0))
/*Slightly more portable than relying on a sign-extended right-shift (which is
   not guaranteed by ANSI), and just as fast, since gcc (3.x and 4.x both)
   compile it into the right-shift anyway.*/
#define OC_SIGNMASK(_a)     (-((_a)<0))
/*Divides an integer by a power of two, truncating towards 0.
  _dividend: The integer to divide.
  _shift:    The non-negative power of two to divide by.
  _rmask:    (1<<_shift)-1*/
#define OC_DIV_POW2(_dividend,_shift,_rmask)\
  ((_dividend)+(OC_SIGNMASK(_dividend)&(_rmask))>>(_shift))
/*Divides _x by 65536, truncating towards 0.*/
#define OC_DIV2_16(_x) OC_DIV_POW2(_x,16,0xFFFF)
/*Divides _x by 2, truncating towards 0.*/
#define OC_DIV2(_x) OC_DIV_POW2(_x,1,0x1)
/*Divides _x by 8, truncating towards 0.*/
#define OC_DIV8(_x) OC_DIV_POW2(_x,3,0x7)
/*Divides _x by 16, truncating towards 0.*/
#define OC_DIV16(_x) OC_DIV_POW2(_x,4,0xF)
/*Right shifts _dividend by _shift, adding _rval, and subtracting one for
   negative dividends first.
  When _rval is (1<<_shift-1), this is equivalent to division with rounding
   ties away from zero.*/
#define OC_DIV_ROUND_POW2(_dividend,_shift,_rval)\
  ((_dividend)+OC_SIGNMASK(_dividend)+(_rval)>>(_shift))
/*Divides a _x by 2, rounding towards even numbers.*/
#define OC_DIV2_RE(_x) ((_x)+((_x)>>1&1)>>1)
/*Divides a _x by (1<<(_shift)), rounding towards even numbers.*/
#define OC_DIV_POW2_RE(_x,_shift) \
  ((_x)+((_x)>>(_shift)&1)+((1<<(_shift))-1>>1)>>(_shift))
/*Swaps two integers _a and _b if _a>_b.*/
#define OC_SORT2I(_a,_b) \
  do{ \
    int t__; \
    t__=((_a)^(_b))&-((_b)<(_a)); \
    (_a)^=t__; \
    (_b)^=t__; \
  } \
  while(0)

/*Accesses one of four (signed) bytes given an index.
  This can be used to avoid small lookup tables.*/
#define OC_BYTE_TABLE32(_a,_b,_c,_d,_i) \
  ((signed char) \
   (((_a)&0xFF|((_b)&0xFF)<<8|((_c)&0xFF)<<16|((_d)&0xFF)<<24)>>(_i)*8))
/*Accesses one of eight (unsigned) nibbles given an index.
  This can be used to avoid small lookup tables.*/
#define OC_UNIBBLE_TABLE32(_a,_b,_c,_d,_e,_f,_g,_h,_i) \
  ((((_a)&0xF|((_b)&0xF)<<4|((_c)&0xF)<<8|((_d)&0xF)<<12| \
   ((_e)&0xF)<<16|((_f)&0xF)<<20|((_g)&0xF)<<24|((_h)&0xF)<<28)>>(_i)*4)&0xF)



/*All of these macros should expect floats as arguments.*/
#define OC_MAXF(_a,_b)      ((_a)<(_b)?(_b):(_a))
#define OC_MINF(_a,_b)      ((_a)>(_b)?(_b):(_a))
#define OC_CLAMPF(_a,_b,_c) (OC_MINF(_a,OC_MAXF(_b,_c)))
#define OC_FABSF(_f)        ((float)fabs(_f))
#define OC_SQRTF(_f)        ((float)sqrt(_f))
#define OC_POWF(_b,_e)      ((float)pow(_b,_e))
#define OC_LOGF(_f)         ((float)log(_f))
#define OC_IFLOORF(_f)      ((int)floor(_f))
#define OC_ICEILF(_f)       ((int)ceil(_f))

#endif
