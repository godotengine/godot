/*
 * jchuff.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains declarations for Huffman entropy encoding routines
 * that are shared between the sequential encoder (jchuff.c) and the
 * progressive encoder (jcphuff.c).  No other modules need to see these.
 */

/* The legal range of a DCT coefficient is
 *  -1024 .. +1023  for 8-bit data;
 * -16384 .. +16383 for 12-bit data.
 * Hence the magnitude should always fit in 10 or 14 bits respectively.
 */

/* The progressive Huffman encoder uses an unsigned 16-bit data type to store
 * absolute values of coefficients, because it is possible to inject a
 * coefficient value of -32768 into the encoder by attempting to transform a
 * malformed 12-bit JPEG image, and the absolute value of -32768 would overflow
 * a signed 16-bit integer.
 */
typedef unsigned short UJCOEF;

/* Derived data constructed for each Huffman table */

typedef struct {
  unsigned int ehufco[256];     /* code for each symbol */
  char ehufsi[256];             /* length of code for each symbol */
  /* If no code has been allocated for a symbol S, ehufsi[S] contains 0 */
} c_derived_tbl;

/* Expand a Huffman table definition into the derived format */
EXTERN(void) jpeg_make_c_derived_tbl(j_compress_ptr cinfo, boolean isDC,
                                     int tblno, c_derived_tbl **pdtbl);

/* Generate an optimal table definition given the specified counts */
EXTERN(void) jpeg_gen_optimal_table(j_compress_ptr cinfo, JHUFF_TBL *htbl,
                                    long freq[]);
