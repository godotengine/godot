/*
 * jutils.c
 *
 * Copyright (C) 1991-1996, Thomas G. Lane.
 * Modified 2009-2020 by Guido Vollbeding.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains tables and miscellaneous utility routines needed
 * for both compression and decompression.
 * Note we prefix all global names with "j" to minimize conflicts with
 * a surrounding application.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"


/*
 * jpeg_zigzag_order[i] is the zigzag-order position of the i'th element
 * of a DCT block read in natural order (left to right, top to bottom).
 */

#if 0				/* This table is not actually needed in v6a */

const int jpeg_zigzag_order[DCTSIZE2] = {
   0,  1,  5,  6, 14, 15, 27, 28,
   2,  4,  7, 13, 16, 26, 29, 42,
   3,  8, 12, 17, 25, 30, 41, 43,
   9, 11, 18, 24, 31, 40, 44, 53,
  10, 19, 23, 32, 39, 45, 52, 54,
  20, 22, 33, 38, 46, 51, 55, 60,
  21, 34, 37, 47, 50, 56, 59, 61,
  35, 36, 48, 49, 57, 58, 62, 63
};

#endif

/*
 * jpeg_natural_order[i] is the natural-order position of the i'th element
 * of zigzag order.
 *
 * When reading corrupted data, the Huffman decoders could attempt
 * to reference an entry beyond the end of this array (if the decoded
 * zero run length reaches past the end of the block).  To prevent
 * wild stores without adding an inner-loop test, we put some extra
 * "63"s after the real entries.  This will cause the extra coefficient
 * to be stored in location 63 of the block, not somewhere random.
 * The worst case would be a run-length of 15, which means we need 16
 * fake entries.
 */

const int jpeg_natural_order[DCTSIZE2+16] = {
   0,  1,  8, 16,  9,  2,  3, 10,
  17, 24, 32, 25, 18, 11,  4,  5,
  12, 19, 26, 33, 40, 48, 41, 34,
  27, 20, 13,  6,  7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36,
  29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46,
  53, 60, 61, 54, 47, 55, 62, 63,
  63, 63, 63, 63, 63, 63, 63, 63, /* extra entries for safety in decoder */
  63, 63, 63, 63, 63, 63, 63, 63
};

const int jpeg_natural_order7[7*7+16] = {
   0,  1,  8, 16,  9,  2,  3, 10,
  17, 24, 32, 25, 18, 11,  4,  5,
  12, 19, 26, 33, 40, 48, 41, 34,
  27, 20, 13,  6, 14, 21, 28, 35,
  42, 49, 50, 43, 36, 29, 22, 30,
  37, 44, 51, 52, 45, 38, 46, 53,
  54,
  63, 63, 63, 63, 63, 63, 63, 63, /* extra entries for safety in decoder */
  63, 63, 63, 63, 63, 63, 63, 63
};

const int jpeg_natural_order6[6*6+16] = {
   0,  1,  8, 16,  9,  2,  3, 10,
  17, 24, 32, 25, 18, 11,  4,  5,
  12, 19, 26, 33, 40, 41, 34, 27,
  20, 13, 21, 28, 35, 42, 43, 36,
  29, 37, 44, 45,
  63, 63, 63, 63, 63, 63, 63, 63, /* extra entries for safety in decoder */
  63, 63, 63, 63, 63, 63, 63, 63
};

const int jpeg_natural_order5[5*5+16] = {
   0,  1,  8, 16,  9,  2,  3, 10,
  17, 24, 32, 25, 18, 11,  4, 12,
  19, 26, 33, 34, 27, 20, 28, 35,
  36,
  63, 63, 63, 63, 63, 63, 63, 63, /* extra entries for safety in decoder */
  63, 63, 63, 63, 63, 63, 63, 63
};

const int jpeg_natural_order4[4*4+16] = {
   0,  1,  8, 16,  9,  2,  3, 10,
  17, 24, 25, 18, 11, 19, 26, 27,
  63, 63, 63, 63, 63, 63, 63, 63, /* extra entries for safety in decoder */
  63, 63, 63, 63, 63, 63, 63, 63
};

const int jpeg_natural_order3[3*3+16] = {
   0,  1,  8, 16,  9,  2, 10, 17,
  18,
  63, 63, 63, 63, 63, 63, 63, 63, /* extra entries for safety in decoder */
  63, 63, 63, 63, 63, 63, 63, 63
};

const int jpeg_natural_order2[2*2+16] = {
   0,  1,  8,  9,
  63, 63, 63, 63, 63, 63, 63, 63, /* extra entries for safety in decoder */
  63, 63, 63, 63, 63, 63, 63, 63
};


/*
 * Arithmetic utilities
 */

GLOBAL(long)
jdiv_round_up (long a, long b)
/* Compute a/b rounded up to next integer, ie, ceil(a/b) */
/* Assumes a >= 0, b > 0 */
{
  return (a + b - 1L) / b;
}


GLOBAL(long)
jround_up (long a, long b)
/* Compute a rounded up to next multiple of b, ie, ceil(a/b)*b */
/* Assumes a >= 0, b > 0 */
{
  a += b - 1L;
  return a - (a % b);
}


/* On normal machines we can apply MEMCOPY() and MEMZERO() to sample arrays
 * and coefficient-block arrays.  This won't work on 80x86 because the arrays
 * are FAR and we're assuming a small-pointer memory model.  However, some
 * DOS compilers provide far-pointer versions of memcpy() and memset() even
 * in the small-model libraries.  These will be used if USE_FMEM is defined.
 * Otherwise, the routines below do it the hard way.  (The performance cost
 * is not all that great, because these routines aren't very heavily used.)
 */

#ifndef NEED_FAR_POINTERS	/* normal case, same as regular macro */
#define FMEMCOPY(dest,src,size)	MEMCOPY(dest,src,size)
#else				/* 80x86 case, define if we can */
#ifdef USE_FMEM
#define FMEMCOPY(dest,src,size)	_fmemcpy((void FAR *)(dest), (const void FAR *)(src), (size_t)(size))
#else
/* This function is for use by the FMEMZERO macro defined in jpegint.h.
 * Do not call this function directly, use the FMEMZERO macro instead.
 */
GLOBAL(void)
jzero_far (void FAR * target, size_t bytestozero)
/* Zero out a chunk of FAR memory. */
/* This might be sample-array data, block-array data, or alloc_large data. */
{
  register char FAR * ptr = (char FAR *) target;
  register size_t count;

  for (count = bytestozero; count > 0; count--) {
    *ptr++ = 0;
  }
}
#endif
#endif


GLOBAL(void)
jcopy_sample_rows (JSAMPARRAY input_array,
		   JSAMPARRAY output_array,
		   int num_rows, JDIMENSION num_cols)
/* Copy some rows of samples from one place to another.
 * num_rows rows are copied from *input_array++ to *output_array++;
 * these areas may overlap for duplication.
 * The source and destination arrays must be at least as wide as num_cols.
 */
{
  register JSAMPROW inptr, outptr;
#ifdef FMEMCOPY
  register size_t count = (size_t) num_cols * SIZEOF(JSAMPLE);
#else
  register JDIMENSION count;
#endif
  register int row;

  for (row = num_rows; row > 0; row--) {
    inptr = *input_array++;
    outptr = *output_array++;
#ifdef FMEMCOPY
    FMEMCOPY(outptr, inptr, count);
#else
    for (count = num_cols; count > 0; count--)
      *outptr++ = *inptr++;	/* needn't bother with GETJSAMPLE() here */
#endif
  }
}


GLOBAL(void)
jcopy_block_row (JBLOCKROW input_row, JBLOCKROW output_row,
		 JDIMENSION num_blocks)
/* Copy a row of coefficient blocks from one place to another. */
{
#ifdef FMEMCOPY
  FMEMCOPY(output_row, input_row, (size_t) num_blocks * (DCTSIZE2 * SIZEOF(JCOEF)));
#else
  register JCOEFPTR inptr, outptr;
  register long count;

  inptr = (JCOEFPTR) input_row;
  outptr = (JCOEFPTR) output_row;
  for (count = (long) num_blocks * DCTSIZE2; count > 0; count--) {
    *outptr++ = *inptr++;
  }
#endif
}
