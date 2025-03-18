/*
 * jaricom.c
 *
 * Developed 1997-2011 by Guido Vollbeding.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains probability estimation tables for common use in
 * arithmetic entropy encoding and decoding routines.
 *
 * This data represents Table D.3 in the JPEG spec (D.2 in the draft),
 * ISO/IEC IS 10918-1 and CCITT Recommendation ITU-T T.81, and Table 24
 * in the JBIG spec, ISO/IEC IS 11544 and CCITT Recommendation ITU-T T.82.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"

/* The following #define specifies the packing of the four components
 * into the compact INT32 representation.
 * Note that this formula must match the actual arithmetic encoder
 * and decoder implementation.  The implementation has to be changed
 * if this formula is changed.
 * The current organization is leaned on Markus Kuhn's JBIG
 * implementation (jbig_tab.c).
 */

#define V(i,a,b,c,d) (((INT32)a << 16) | ((INT32)c << 8) | ((INT32)d << 7) | b)

const INT32 jpeg_aritab[113+1] = {
/*
 * Index, Qe_Value, Next_Index_LPS, Next_Index_MPS, Switch_MPS
 */
  V(   0, 0x5a1d,   1,   1, 1 ),
  V(   1, 0x2586,  14,   2, 0 ),
  V(   2, 0x1114,  16,   3, 0 ),
  V(   3, 0x080b,  18,   4, 0 ),
  V(   4, 0x03d8,  20,   5, 0 ),
  V(   5, 0x01da,  23,   6, 0 ),
  V(   6, 0x00e5,  25,   7, 0 ),
  V(   7, 0x006f,  28,   8, 0 ),
  V(   8, 0x0036,  30,   9, 0 ),
  V(   9, 0x001a,  33,  10, 0 ),
  V(  10, 0x000d,  35,  11, 0 ),
  V(  11, 0x0006,   9,  12, 0 ),
  V(  12, 0x0003,  10,  13, 0 ),
  V(  13, 0x0001,  12,  13, 0 ),
  V(  14, 0x5a7f,  15,  15, 1 ),
  V(  15, 0x3f25,  36,  16, 0 ),
  V(  16, 0x2cf2,  38,  17, 0 ),
  V(  17, 0x207c,  39,  18, 0 ),
  V(  18, 0x17b9,  40,  19, 0 ),
  V(  19, 0x1182,  42,  20, 0 ),
  V(  20, 0x0cef,  43,  21, 0 ),
  V(  21, 0x09a1,  45,  22, 0 ),
  V(  22, 0x072f,  46,  23, 0 ),
  V(  23, 0x055c,  48,  24, 0 ),
  V(  24, 0x0406,  49,  25, 0 ),
  V(  25, 0x0303,  51,  26, 0 ),
  V(  26, 0x0240,  52,  27, 0 ),
  V(  27, 0x01b1,  54,  28, 0 ),
  V(  28, 0x0144,  56,  29, 0 ),
  V(  29, 0x00f5,  57,  30, 0 ),
  V(  30, 0x00b7,  59,  31, 0 ),
  V(  31, 0x008a,  60,  32, 0 ),
  V(  32, 0x0068,  62,  33, 0 ),
  V(  33, 0x004e,  63,  34, 0 ),
  V(  34, 0x003b,  32,  35, 0 ),
  V(  35, 0x002c,  33,   9, 0 ),
  V(  36, 0x5ae1,  37,  37, 1 ),
  V(  37, 0x484c,  64,  38, 0 ),
  V(  38, 0x3a0d,  65,  39, 0 ),
  V(  39, 0x2ef1,  67,  40, 0 ),
  V(  40, 0x261f,  68,  41, 0 ),
  V(  41, 0x1f33,  69,  42, 0 ),
  V(  42, 0x19a8,  70,  43, 0 ),
  V(  43, 0x1518,  72,  44, 0 ),
  V(  44, 0x1177,  73,  45, 0 ),
  V(  45, 0x0e74,  74,  46, 0 ),
  V(  46, 0x0bfb,  75,  47, 0 ),
  V(  47, 0x09f8,  77,  48, 0 ),
  V(  48, 0x0861,  78,  49, 0 ),
  V(  49, 0x0706,  79,  50, 0 ),
  V(  50, 0x05cd,  48,  51, 0 ),
  V(  51, 0x04de,  50,  52, 0 ),
  V(  52, 0x040f,  50,  53, 0 ),
  V(  53, 0x0363,  51,  54, 0 ),
  V(  54, 0x02d4,  52,  55, 0 ),
  V(  55, 0x025c,  53,  56, 0 ),
  V(  56, 0x01f8,  54,  57, 0 ),
  V(  57, 0x01a4,  55,  58, 0 ),
  V(  58, 0x0160,  56,  59, 0 ),
  V(  59, 0x0125,  57,  60, 0 ),
  V(  60, 0x00f6,  58,  61, 0 ),
  V(  61, 0x00cb,  59,  62, 0 ),
  V(  62, 0x00ab,  61,  63, 0 ),
  V(  63, 0x008f,  61,  32, 0 ),
  V(  64, 0x5b12,  65,  65, 1 ),
  V(  65, 0x4d04,  80,  66, 0 ),
  V(  66, 0x412c,  81,  67, 0 ),
  V(  67, 0x37d8,  82,  68, 0 ),
  V(  68, 0x2fe8,  83,  69, 0 ),
  V(  69, 0x293c,  84,  70, 0 ),
  V(  70, 0x2379,  86,  71, 0 ),
  V(  71, 0x1edf,  87,  72, 0 ),
  V(  72, 0x1aa9,  87,  73, 0 ),
  V(  73, 0x174e,  72,  74, 0 ),
  V(  74, 0x1424,  72,  75, 0 ),
  V(  75, 0x119c,  74,  76, 0 ),
  V(  76, 0x0f6b,  74,  77, 0 ),
  V(  77, 0x0d51,  75,  78, 0 ),
  V(  78, 0x0bb6,  77,  79, 0 ),
  V(  79, 0x0a40,  77,  48, 0 ),
  V(  80, 0x5832,  80,  81, 1 ),
  V(  81, 0x4d1c,  88,  82, 0 ),
  V(  82, 0x438e,  89,  83, 0 ),
  V(  83, 0x3bdd,  90,  84, 0 ),
  V(  84, 0x34ee,  91,  85, 0 ),
  V(  85, 0x2eae,  92,  86, 0 ),
  V(  86, 0x299a,  93,  87, 0 ),
  V(  87, 0x2516,  86,  71, 0 ),
  V(  88, 0x5570,  88,  89, 1 ),
  V(  89, 0x4ca9,  95,  90, 0 ),
  V(  90, 0x44d9,  96,  91, 0 ),
  V(  91, 0x3e22,  97,  92, 0 ),
  V(  92, 0x3824,  99,  93, 0 ),
  V(  93, 0x32b4,  99,  94, 0 ),
  V(  94, 0x2e17,  93,  86, 0 ),
  V(  95, 0x56a8,  95,  96, 1 ),
  V(  96, 0x4f46, 101,  97, 0 ),
  V(  97, 0x47e5, 102,  98, 0 ),
  V(  98, 0x41cf, 103,  99, 0 ),
  V(  99, 0x3c3d, 104, 100, 0 ),
  V( 100, 0x375e,  99,  93, 0 ),
  V( 101, 0x5231, 105, 102, 0 ),
  V( 102, 0x4c0f, 106, 103, 0 ),
  V( 103, 0x4639, 107, 104, 0 ),
  V( 104, 0x415e, 103,  99, 0 ),
  V( 105, 0x5627, 105, 106, 1 ),
  V( 106, 0x50e7, 108, 107, 0 ),
  V( 107, 0x4b85, 109, 103, 0 ),
  V( 108, 0x5597, 110, 109, 0 ),
  V( 109, 0x504f, 111, 107, 0 ),
  V( 110, 0x5a10, 110, 111, 1 ),
  V( 111, 0x5522, 112, 109, 0 ),
  V( 112, 0x59eb, 112, 111, 1 ),
/*
 * This last entry is used for fixed probability estimate of 0.5
 * as suggested in Section 10.3 Table 5 of ITU-T Rec. T.851.
 */
  V( 113, 0x5a1d, 113, 113, 0 )
};
