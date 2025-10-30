/*
 * jcdctmgr.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 1999-2006, MIYASAKA Masaru.
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2011, 2014-2015, 2022, 2024, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains the forward-DCT management logic.
 * This code selects a particular DCT implementation to be used,
 * and it performs related housekeeping chores including coefficient
 * quantization.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jdct.h"               /* Private declarations for DCT subsystem */
#include "jsimddct.h"


/* Private subobject for this module */

typedef void (*forward_DCT_method_ptr) (DCTELEM *data);
typedef void (*float_DCT_method_ptr) (FAST_FLOAT *data);

typedef void (*convsamp_method_ptr) (_JSAMPARRAY sample_data,
                                     JDIMENSION start_col,
                                     DCTELEM *workspace);
typedef void (*float_convsamp_method_ptr) (_JSAMPARRAY sample_data,
                                           JDIMENSION start_col,
                                           FAST_FLOAT *workspace);

typedef void (*quantize_method_ptr) (JCOEFPTR coef_block, DCTELEM *divisors,
                                     DCTELEM *workspace);
typedef void (*float_quantize_method_ptr) (JCOEFPTR coef_block,
                                           FAST_FLOAT *divisors,
                                           FAST_FLOAT *workspace);

METHODDEF(void) quantize(JCOEFPTR, DCTELEM *, DCTELEM *);

typedef struct {
  struct jpeg_forward_dct pub;  /* public fields */

  /* Pointer to the DCT routine actually in use */
  forward_DCT_method_ptr dct;
  convsamp_method_ptr convsamp;
  quantize_method_ptr quantize;

  /* The actual post-DCT divisors --- not identical to the quant table
   * entries, because of scaling (especially for an unnormalized DCT).
   * Each table is given in normal array order.
   */
  DCTELEM *divisors[NUM_QUANT_TBLS];

  /* work area for FDCT subroutine */
  DCTELEM *workspace;

#ifdef DCT_FLOAT_SUPPORTED
  /* Same as above for the floating-point case. */
  float_DCT_method_ptr float_dct;
  float_convsamp_method_ptr float_convsamp;
  float_quantize_method_ptr float_quantize;
  FAST_FLOAT *float_divisors[NUM_QUANT_TBLS];
  FAST_FLOAT *float_workspace;
#endif
} my_fdct_controller;

typedef my_fdct_controller *my_fdct_ptr;


#if BITS_IN_JSAMPLE == 8

/*
 * Find the highest bit in an integer through binary search.
 */

LOCAL(int)
flss(UINT16 val)
{
  int bit;

  bit = 16;

  if (!val)
    return 0;

  if (!(val & 0xff00)) {
    bit -= 8;
    val <<= 8;
  }
  if (!(val & 0xf000)) {
    bit -= 4;
    val <<= 4;
  }
  if (!(val & 0xc000)) {
    bit -= 2;
    val <<= 2;
  }
  if (!(val & 0x8000)) {
    bit -= 1;
    val <<= 1;
  }

  return bit;
}


/*
 * Compute values to do a division using reciprocal.
 *
 * This implementation is based on an algorithm described in
 *   "Optimizing subroutines in assembly language:
 *   An optimization guide for x86 platforms" (https://agner.org/optimize).
 * More information about the basic algorithm can be found in
 * the paper "Integer Division Using Reciprocals" by Robert Alverson.
 *
 * The basic idea is to replace x/d by x * d^-1. In order to store
 * d^-1 with enough precision we shift it left a few places. It turns
 * out that this algoright gives just enough precision, and also fits
 * into DCTELEM:
 *
 *   b = (the number of significant bits in divisor) - 1
 *   r = (word size) + b
 *   f = 2^r / divisor
 *
 * f will not be an integer for most cases, so we need to compensate
 * for the rounding error introduced:
 *
 *   no fractional part:
 *
 *       result = input >> r
 *
 *   fractional part of f < 0.5:
 *
 *       round f down to nearest integer
 *       result = ((input + 1) * f) >> r
 *
 *   fractional part of f > 0.5:
 *
 *       round f up to nearest integer
 *       result = (input * f) >> r
 *
 * This is the original algorithm that gives truncated results. But we
 * want properly rounded results, so we replace "input" with
 * "input + divisor/2".
 *
 * In order to allow SIMD implementations we also tweak the values to
 * allow the same calculation to be made at all times:
 *
 *   dctbl[0] = f rounded to nearest integer
 *   dctbl[1] = divisor / 2 (+ 1 if fractional part of f < 0.5)
 *   dctbl[2] = 1 << ((word size) * 2 - r)
 *   dctbl[3] = r - (word size)
 *
 * dctbl[2] is for stupid instruction sets where the shift operation
 * isn't member wise (e.g. MMX).
 *
 * The reason dctbl[2] and dctbl[3] reduce the shift with (word size)
 * is that most SIMD implementations have a "multiply and store top
 * half" operation.
 *
 * Lastly, we store each of the values in their own table instead
 * of in a consecutive manner, yet again in order to allow SIMD
 * routines.
 */

LOCAL(int)
compute_reciprocal(UINT16 divisor, DCTELEM *dtbl)
{
  UDCTELEM2 fq, fr;
  UDCTELEM c;
  int b, r;

  if (divisor == 1) {
    /* divisor == 1 means unquantized, so these reciprocal/correction/shift
     * values will cause the C quantization algorithm to act like the
     * identity function.  Since only the C quantization algorithm is used in
     * these cases, the scale value is irrelevant.
     */
    dtbl[DCTSIZE2 * 0] = (DCTELEM)1;                        /* reciprocal */
    dtbl[DCTSIZE2 * 1] = (DCTELEM)0;                        /* correction */
    dtbl[DCTSIZE2 * 2] = (DCTELEM)1;                        /* scale */
    dtbl[DCTSIZE2 * 3] = -(DCTELEM)(sizeof(DCTELEM) * 8);   /* shift */
    return 0;
  }

  b = flss(divisor) - 1;
  r  = sizeof(DCTELEM) * 8 + b;

  fq = ((UDCTELEM2)1 << r) / divisor;
  fr = ((UDCTELEM2)1 << r) % divisor;

  c = divisor / 2;                      /* for rounding */

  if (fr == 0) {                        /* divisor is power of two */
    /* fq will be one bit too large to fit in DCTELEM, so adjust */
    fq >>= 1;
    r--;
  } else if (fr <= (divisor / 2U)) {    /* fractional part is < 0.5 */
    c++;
  } else {                              /* fractional part is > 0.5 */
    fq++;
  }

  dtbl[DCTSIZE2 * 0] = (DCTELEM)fq;     /* reciprocal */
  dtbl[DCTSIZE2 * 1] = (DCTELEM)c;      /* correction + roundfactor */
#ifdef WITH_SIMD
  dtbl[DCTSIZE2 * 2] = (DCTELEM)(1 << (sizeof(DCTELEM) * 8 * 2 - r)); /* scale */
#else
  dtbl[DCTSIZE2 * 2] = 1;
#endif
  dtbl[DCTSIZE2 * 3] = (DCTELEM)r - sizeof(DCTELEM) * 8; /* shift */

  if (r <= 16) return 0;
  else return 1;
}

#endif


/*
 * Initialize for a processing pass.
 * Verify that all referenced Q-tables are present, and set up
 * the divisor table for each one.
 * In the current implementation, DCT of all components is done during
 * the first pass, even if only some components will be output in the
 * first scan.  Hence all components should be examined here.
 */

METHODDEF(void)
start_pass_fdctmgr(j_compress_ptr cinfo)
{
  my_fdct_ptr fdct = (my_fdct_ptr)cinfo->fdct;
  int ci, qtblno, i;
  jpeg_component_info *compptr;
  JQUANT_TBL *qtbl;
  DCTELEM *dtbl;

  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    qtblno = compptr->quant_tbl_no;
    /* Make sure specified quantization table is present */
    if (qtblno < 0 || qtblno >= NUM_QUANT_TBLS ||
        cinfo->quant_tbl_ptrs[qtblno] == NULL)
      ERREXIT1(cinfo, JERR_NO_QUANT_TABLE, qtblno);
    qtbl = cinfo->quant_tbl_ptrs[qtblno];
    /* Compute divisors for this quant table */
    /* We may do this more than once for same table, but it's not a big deal */
    switch (cinfo->dct_method) {
#ifdef DCT_ISLOW_SUPPORTED
    case JDCT_ISLOW:
      /* For LL&M IDCT method, divisors are equal to raw quantization
       * coefficients multiplied by 8 (to counteract scaling).
       */
      if (fdct->divisors[qtblno] == NULL) {
        fdct->divisors[qtblno] = (DCTELEM *)
          (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                      (DCTSIZE2 * 4) * sizeof(DCTELEM));
      }
      dtbl = fdct->divisors[qtblno];
      for (i = 0; i < DCTSIZE2; i++) {
#if BITS_IN_JSAMPLE == 8
#ifdef WITH_SIMD
        if (!compute_reciprocal(qtbl->quantval[i] << 3, &dtbl[i]) &&
            fdct->quantize == jsimd_quantize)
          fdct->quantize = quantize;
#else
        compute_reciprocal(qtbl->quantval[i] << 3, &dtbl[i]);
#endif
#else
        dtbl[i] = ((DCTELEM)qtbl->quantval[i]) << 3;
#endif
      }
      break;
#endif
#ifdef DCT_IFAST_SUPPORTED
    case JDCT_IFAST:
      {
        /* For AA&N IDCT method, divisors are equal to quantization
         * coefficients scaled by scalefactor[row]*scalefactor[col], where
         *   scalefactor[0] = 1
         *   scalefactor[k] = cos(k*PI/16) * sqrt(2)    for k=1..7
         * We apply a further scale factor of 8.
         */
#define CONST_BITS  14
        static const INT16 aanscales[DCTSIZE2] = {
          /* precomputed values scaled up by 14 bits */
          16384, 22725, 21407, 19266, 16384, 12873,  8867,  4520,
          22725, 31521, 29692, 26722, 22725, 17855, 12299,  6270,
          21407, 29692, 27969, 25172, 21407, 16819, 11585,  5906,
          19266, 26722, 25172, 22654, 19266, 15137, 10426,  5315,
          16384, 22725, 21407, 19266, 16384, 12873,  8867,  4520,
          12873, 17855, 16819, 15137, 12873, 10114,  6967,  3552,
           8867, 12299, 11585, 10426,  8867,  6967,  4799,  2446,
           4520,  6270,  5906,  5315,  4520,  3552,  2446,  1247
        };
        SHIFT_TEMPS

        if (fdct->divisors[qtblno] == NULL) {
          fdct->divisors[qtblno] = (DCTELEM *)
            (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                        (DCTSIZE2 * 4) * sizeof(DCTELEM));
        }
        dtbl = fdct->divisors[qtblno];
        for (i = 0; i < DCTSIZE2; i++) {
#if BITS_IN_JSAMPLE == 8
#ifdef WITH_SIMD
          if (!compute_reciprocal(
                DESCALE(MULTIPLY16V16((JLONG)qtbl->quantval[i],
                                      (JLONG)aanscales[i]),
                        CONST_BITS - 3), &dtbl[i]) &&
              fdct->quantize == jsimd_quantize)
            fdct->quantize = quantize;
#else
          compute_reciprocal(
            DESCALE(MULTIPLY16V16((JLONG)qtbl->quantval[i],
                                  (JLONG)aanscales[i]),
                    CONST_BITS-3), &dtbl[i]);
#endif
#else
          dtbl[i] = (DCTELEM)
            DESCALE(MULTIPLY16V16((JLONG)qtbl->quantval[i],
                                  (JLONG)aanscales[i]),
                    CONST_BITS - 3);
#endif
        }
      }
      break;
#endif
#ifdef DCT_FLOAT_SUPPORTED
    case JDCT_FLOAT:
      {
        /* For float AA&N IDCT method, divisors are equal to quantization
         * coefficients scaled by scalefactor[row]*scalefactor[col], where
         *   scalefactor[0] = 1
         *   scalefactor[k] = cos(k*PI/16) * sqrt(2)    for k=1..7
         * We apply a further scale factor of 8.
         * What's actually stored is 1/divisor so that the inner loop can
         * use a multiplication rather than a division.
         */
        FAST_FLOAT *fdtbl;
        int row, col;
        static const double aanscalefactor[DCTSIZE] = {
          1.0, 1.387039845, 1.306562965, 1.175875602,
          1.0, 0.785694958, 0.541196100, 0.275899379
        };

        if (fdct->float_divisors[qtblno] == NULL) {
          fdct->float_divisors[qtblno] = (FAST_FLOAT *)
            (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                        DCTSIZE2 * sizeof(FAST_FLOAT));
        }
        fdtbl = fdct->float_divisors[qtblno];
        i = 0;
        for (row = 0; row < DCTSIZE; row++) {
          for (col = 0; col < DCTSIZE; col++) {
            fdtbl[i] = (FAST_FLOAT)
              (1.0 / (((double)qtbl->quantval[i] *
                       aanscalefactor[row] * aanscalefactor[col] * 8.0)));
            i++;
          }
        }
      }
      break;
#endif
    default:
      ERREXIT(cinfo, JERR_NOT_COMPILED);
      break;
    }
  }
}


/*
 * Load data into workspace, applying unsigned->signed conversion.
 */

METHODDEF(void)
convsamp(_JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace)
{
  register DCTELEM *workspaceptr;
  register _JSAMPROW elemptr;
  register int elemr;

  workspaceptr = workspace;
  for (elemr = 0; elemr < DCTSIZE; elemr++) {
    elemptr = sample_data[elemr] + start_col;

#if DCTSIZE == 8                /* unroll the inner loop */
    *workspaceptr++ = (*elemptr++) - _CENTERJSAMPLE;
    *workspaceptr++ = (*elemptr++) - _CENTERJSAMPLE;
    *workspaceptr++ = (*elemptr++) - _CENTERJSAMPLE;
    *workspaceptr++ = (*elemptr++) - _CENTERJSAMPLE;
    *workspaceptr++ = (*elemptr++) - _CENTERJSAMPLE;
    *workspaceptr++ = (*elemptr++) - _CENTERJSAMPLE;
    *workspaceptr++ = (*elemptr++) - _CENTERJSAMPLE;
    *workspaceptr++ = (*elemptr++) - _CENTERJSAMPLE;
#else
    {
      register int elemc;
      for (elemc = DCTSIZE; elemc > 0; elemc--)
        *workspaceptr++ = (*elemptr++) - _CENTERJSAMPLE;
    }
#endif
  }
}


/*
 * Quantize/descale the coefficients, and store into coef_blocks[].
 */

METHODDEF(void)
quantize(JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace)
{
  int i;
  DCTELEM temp;
  JCOEFPTR output_ptr = coef_block;

#if BITS_IN_JSAMPLE == 8

  UDCTELEM recip, corr;
  int shift;
  UDCTELEM2 product;

  for (i = 0; i < DCTSIZE2; i++) {
    temp = workspace[i];
    recip = divisors[i + DCTSIZE2 * 0];
    corr =  divisors[i + DCTSIZE2 * 1];
    shift = divisors[i + DCTSIZE2 * 3];

    if (temp < 0) {
      temp = -temp;
      product = (UDCTELEM2)(temp + corr) * recip;
      product >>= shift + sizeof(DCTELEM) * 8;
      temp = (DCTELEM)product;
      temp = -temp;
    } else {
      product = (UDCTELEM2)(temp + corr) * recip;
      product >>= shift + sizeof(DCTELEM) * 8;
      temp = (DCTELEM)product;
    }
    output_ptr[i] = (JCOEF)temp;
  }

#else

  register DCTELEM qval;

  for (i = 0; i < DCTSIZE2; i++) {
    qval = divisors[i];
    temp = workspace[i];
    /* Divide the coefficient value by qval, ensuring proper rounding.
     * Since C does not specify the direction of rounding for negative
     * quotients, we have to force the dividend positive for portability.
     *
     * In most files, at least half of the output values will be zero
     * (at default quantization settings, more like three-quarters...)
     * so we should ensure that this case is fast.  On many machines,
     * a comparison is enough cheaper than a divide to make a special test
     * a win.  Since both inputs will be nonnegative, we need only test
     * for a < b to discover whether a/b is 0.
     * If your machine's division is fast enough, define FAST_DIVIDE.
     */
#ifdef FAST_DIVIDE
#define DIVIDE_BY(a, b)  a /= b
#else
#define DIVIDE_BY(a, b)  if (a >= b) a /= b;  else a = 0
#endif
    if (temp < 0) {
      temp = -temp;
      temp += qval >> 1;        /* for rounding */
      DIVIDE_BY(temp, qval);
      temp = -temp;
    } else {
      temp += qval >> 1;        /* for rounding */
      DIVIDE_BY(temp, qval);
    }
    output_ptr[i] = (JCOEF)temp;
  }

#endif

}


/*
 * Perform forward DCT on one or more blocks of a component.
 *
 * The input samples are taken from the sample_data[] array starting at
 * position start_row/start_col, and moving to the right for any additional
 * blocks. The quantized coefficients are returned in coef_blocks[].
 */

METHODDEF(void)
forward_DCT(j_compress_ptr cinfo, jpeg_component_info *compptr,
            _JSAMPARRAY sample_data, JBLOCKROW coef_blocks,
            JDIMENSION start_row, JDIMENSION start_col, JDIMENSION num_blocks)
/* This version is used for integer DCT implementations. */
{
  /* This routine is heavily used, so it's worth coding it tightly. */
  my_fdct_ptr fdct = (my_fdct_ptr)cinfo->fdct;
  DCTELEM *divisors = fdct->divisors[compptr->quant_tbl_no];
  DCTELEM *workspace;
  JDIMENSION bi;

  /* Make sure the compiler doesn't look up these every pass */
  forward_DCT_method_ptr do_dct = fdct->dct;
  convsamp_method_ptr do_convsamp = fdct->convsamp;
  quantize_method_ptr do_quantize = fdct->quantize;
  workspace = fdct->workspace;

  sample_data += start_row;     /* fold in the vertical offset once */

  for (bi = 0; bi < num_blocks; bi++, start_col += DCTSIZE) {
    /* Load data into workspace, applying unsigned->signed conversion */
    (*do_convsamp) (sample_data, start_col, workspace);

    /* Perform the DCT */
    (*do_dct) (workspace);

    /* Quantize/descale the coefficients, and store into coef_blocks[] */
    (*do_quantize) (coef_blocks[bi], divisors, workspace);
  }
}


#ifdef DCT_FLOAT_SUPPORTED

METHODDEF(void)
convsamp_float(_JSAMPARRAY sample_data, JDIMENSION start_col,
               FAST_FLOAT *workspace)
{
  register FAST_FLOAT *workspaceptr;
  register _JSAMPROW elemptr;
  register int elemr;

  workspaceptr = workspace;
  for (elemr = 0; elemr < DCTSIZE; elemr++) {
    elemptr = sample_data[elemr] + start_col;
#if DCTSIZE == 8                /* unroll the inner loop */
    *workspaceptr++ = (FAST_FLOAT)((*elemptr++) - _CENTERJSAMPLE);
    *workspaceptr++ = (FAST_FLOAT)((*elemptr++) - _CENTERJSAMPLE);
    *workspaceptr++ = (FAST_FLOAT)((*elemptr++) - _CENTERJSAMPLE);
    *workspaceptr++ = (FAST_FLOAT)((*elemptr++) - _CENTERJSAMPLE);
    *workspaceptr++ = (FAST_FLOAT)((*elemptr++) - _CENTERJSAMPLE);
    *workspaceptr++ = (FAST_FLOAT)((*elemptr++) - _CENTERJSAMPLE);
    *workspaceptr++ = (FAST_FLOAT)((*elemptr++) - _CENTERJSAMPLE);
    *workspaceptr++ = (FAST_FLOAT)((*elemptr++) - _CENTERJSAMPLE);
#else
    {
      register int elemc;
      for (elemc = DCTSIZE; elemc > 0; elemc--)
        *workspaceptr++ = (FAST_FLOAT)((*elemptr++) - _CENTERJSAMPLE);
    }
#endif
  }
}


METHODDEF(void)
quantize_float(JCOEFPTR coef_block, FAST_FLOAT *divisors,
               FAST_FLOAT *workspace)
{
  register FAST_FLOAT temp;
  register int i;
  register JCOEFPTR output_ptr = coef_block;

  for (i = 0; i < DCTSIZE2; i++) {
    /* Apply the quantization and scaling factor */
    temp = workspace[i] * divisors[i];

    /* Round to nearest integer.
     * Since C does not specify the direction of rounding for negative
     * quotients, we have to force the dividend positive for portability.
     * The maximum coefficient size is +-16K (for 12-bit data), so this
     * code should work for either 16-bit or 32-bit ints.
     */
    output_ptr[i] = (JCOEF)((int)(temp + (FAST_FLOAT)16384.5) - 16384);
  }
}


METHODDEF(void)
forward_DCT_float(j_compress_ptr cinfo, jpeg_component_info *compptr,
                  _JSAMPARRAY sample_data, JBLOCKROW coef_blocks,
                  JDIMENSION start_row, JDIMENSION start_col,
                  JDIMENSION num_blocks)
/* This version is used for floating-point DCT implementations. */
{
  /* This routine is heavily used, so it's worth coding it tightly. */
  my_fdct_ptr fdct = (my_fdct_ptr)cinfo->fdct;
  FAST_FLOAT *divisors = fdct->float_divisors[compptr->quant_tbl_no];
  FAST_FLOAT *workspace;
  JDIMENSION bi;


  /* Make sure the compiler doesn't look up these every pass */
  float_DCT_method_ptr do_dct = fdct->float_dct;
  float_convsamp_method_ptr do_convsamp = fdct->float_convsamp;
  float_quantize_method_ptr do_quantize = fdct->float_quantize;
  workspace = fdct->float_workspace;

  sample_data += start_row;     /* fold in the vertical offset once */

  for (bi = 0; bi < num_blocks; bi++, start_col += DCTSIZE) {
    /* Load data into workspace, applying unsigned->signed conversion */
    (*do_convsamp) (sample_data, start_col, workspace);

    /* Perform the DCT */
    (*do_dct) (workspace);

    /* Quantize/descale the coefficients, and store into coef_blocks[] */
    (*do_quantize) (coef_blocks[bi], divisors, workspace);
  }
}

#endif /* DCT_FLOAT_SUPPORTED */


/*
 * Initialize FDCT manager.
 */

GLOBAL(void)
_jinit_forward_dct(j_compress_ptr cinfo)
{
  my_fdct_ptr fdct;
  int i;

  if (cinfo->data_precision != BITS_IN_JSAMPLE)
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  fdct = (my_fdct_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(my_fdct_controller));
  cinfo->fdct = (struct jpeg_forward_dct *)fdct;
  fdct->pub.start_pass = start_pass_fdctmgr;

  /* First determine the DCT... */
  switch (cinfo->dct_method) {
#ifdef DCT_ISLOW_SUPPORTED
  case JDCT_ISLOW:
    fdct->pub._forward_DCT = forward_DCT;
#ifdef WITH_SIMD
    if (jsimd_can_fdct_islow())
      fdct->dct = jsimd_fdct_islow;
    else
#endif
      fdct->dct = _jpeg_fdct_islow;
    break;
#endif
#ifdef DCT_IFAST_SUPPORTED
  case JDCT_IFAST:
    fdct->pub._forward_DCT = forward_DCT;
#ifdef WITH_SIMD
    if (jsimd_can_fdct_ifast())
      fdct->dct = jsimd_fdct_ifast;
    else
#endif
      fdct->dct = _jpeg_fdct_ifast;
    break;
#endif
#ifdef DCT_FLOAT_SUPPORTED
  case JDCT_FLOAT:
    fdct->pub._forward_DCT = forward_DCT_float;
#ifdef WITH_SIMD
    if (jsimd_can_fdct_float())
      fdct->float_dct = jsimd_fdct_float;
    else
#endif
      fdct->float_dct = jpeg_fdct_float;
    break;
#endif
  default:
    ERREXIT(cinfo, JERR_NOT_COMPILED);
    break;
  }

  /* ...then the supporting stages. */
  switch (cinfo->dct_method) {
#ifdef DCT_ISLOW_SUPPORTED
  case JDCT_ISLOW:
#endif
#ifdef DCT_IFAST_SUPPORTED
  case JDCT_IFAST:
#endif
#if defined(DCT_ISLOW_SUPPORTED) || defined(DCT_IFAST_SUPPORTED)
#ifdef WITH_SIMD
    if (jsimd_can_convsamp())
      fdct->convsamp = jsimd_convsamp;
    else
#endif
      fdct->convsamp = convsamp;
#ifdef WITH_SIMD
    if (jsimd_can_quantize())
      fdct->quantize = jsimd_quantize;
    else
#endif
      fdct->quantize = quantize;
    break;
#endif
#ifdef DCT_FLOAT_SUPPORTED
  case JDCT_FLOAT:
#ifdef WITH_SIMD
    if (jsimd_can_convsamp_float())
      fdct->float_convsamp = jsimd_convsamp_float;
    else
#endif
      fdct->float_convsamp = convsamp_float;
#ifdef WITH_SIMD
    if (jsimd_can_quantize_float())
      fdct->float_quantize = jsimd_quantize_float;
    else
#endif
      fdct->float_quantize = quantize_float;
    break;
#endif
  default:
    ERREXIT(cinfo, JERR_NOT_COMPILED);
    break;
  }

  /* Allocate workspace memory */
#ifdef DCT_FLOAT_SUPPORTED
  if (cinfo->dct_method == JDCT_FLOAT)
    fdct->float_workspace = (FAST_FLOAT *)
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                  sizeof(FAST_FLOAT) * DCTSIZE2);
  else
#endif
    fdct->workspace = (DCTELEM *)
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                  sizeof(DCTELEM) * DCTSIZE2);

  /* Mark divisor tables unallocated */
  for (i = 0; i < NUM_QUANT_TBLS; i++) {
    fdct->divisors[i] = NULL;
#ifdef DCT_FLOAT_SUPPORTED
    fdct->float_divisors[i] = NULL;
#endif
  }
}
