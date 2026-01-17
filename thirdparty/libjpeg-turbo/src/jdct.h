/*
 * jdct.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2015, 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This include file contains common declarations for the forward and
 * inverse DCT modules.  These declarations are private to the DCT managers
 * (jcdctmgr.c, jddctmgr.c) and the individual DCT algorithms.
 * The individual DCT algorithms are kept in separate files to ease
 * machine-dependent tuning (e.g., assembly coding).
 */

#include "jsamplecomp.h"


/*
 * A forward DCT routine is given a pointer to a work area of type DCTELEM[];
 * the DCT is to be performed in-place in that buffer.  Type DCTELEM is int
 * for 8-bit samples, JLONG for 12-bit samples.  (NOTE: Floating-point DCT
 * implementations use an array of type FAST_FLOAT, instead.)
 * The DCT inputs are expected to be signed (range +-_CENTERJSAMPLE).
 * The DCT outputs are returned scaled up by a factor of 8; they therefore
 * have a range of +-8K for 8-bit data, +-128K for 12-bit data.  This
 * convention improves accuracy in integer implementations and saves some
 * work in floating-point ones.
 * Quantization of the output coefficients is done by jcdctmgr.c. This
 * step requires an unsigned type and also one with twice the bits.
 */

#if BITS_IN_JSAMPLE == 8
#ifndef WITH_SIMD
typedef int DCTELEM;            /* 16 or 32 bits is fine */
typedef unsigned int UDCTELEM;
typedef unsigned long long UDCTELEM2;
#else
typedef short DCTELEM;          /* prefer 16 bit with SIMD for parellelism */
typedef unsigned short UDCTELEM;
typedef unsigned int UDCTELEM2;
#endif
#else
typedef JLONG DCTELEM;          /* must have 32 bits */
typedef unsigned long long UDCTELEM2;
#endif


/*
 * An inverse DCT routine is given a pointer to the input JBLOCK and a pointer
 * to an output sample array.  The routine must dequantize the input data as
 * well as perform the IDCT; for dequantization, it uses the multiplier table
 * pointed to by compptr->dct_table.  The output data is to be placed into the
 * sample array starting at a specified column.  (Any row offset needed will
 * be applied to the array pointer before it is passed to the IDCT code.)
 * Note that the number of samples emitted by the IDCT routine is
 * DCT_scaled_size * DCT_scaled_size.
 */

/* typedef inverse_DCT_method_ptr is declared in jpegint.h */

/*
 * Each IDCT routine has its own ideas about the best dct_table element type.
 */

typedef MULTIPLIER ISLOW_MULT_TYPE;  /* short or int, whichever is faster */
#if BITS_IN_JSAMPLE == 8
typedef MULTIPLIER IFAST_MULT_TYPE;  /* 16 bits is OK, use short if faster */
#define IFAST_SCALE_BITS  2          /* fractional bits in scale factors */
#else
typedef JLONG IFAST_MULT_TYPE;       /* need 32 bits for scaled quantizers */
#define IFAST_SCALE_BITS  13         /* fractional bits in scale factors */
#endif
typedef FAST_FLOAT FLOAT_MULT_TYPE;  /* preferred floating type */


/*
 * Each IDCT routine is responsible for range-limiting its results and
 * converting them to unsigned form (0.._MAXJSAMPLE).  The raw outputs could
 * be quite far out of range if the input data is corrupt, so a bulletproof
 * range-limiting step is required.  We use a mask-and-table-lookup method
 * to do the combined operations quickly.  See the comments with
 * prepare_range_limit_table (in jdmaster.c) for more info.
 */

#define IDCT_range_limit(cinfo) \
  ((_JSAMPLE *)((cinfo)->sample_range_limit) + _CENTERJSAMPLE)

#define RANGE_MASK  (_MAXJSAMPLE * 4 + 3) /* 2 bits wider than legal samples */


/* Extern declarations for the forward and inverse DCT routines. */

EXTERN(void) _jpeg_fdct_islow(DCTELEM *data);
EXTERN(void) _jpeg_fdct_ifast(DCTELEM *data);
EXTERN(void) jpeg_fdct_float(FAST_FLOAT *data);

EXTERN(void) _jpeg_idct_islow(j_decompress_ptr cinfo,
                              jpeg_component_info *compptr,
                              JCOEFPTR coef_block, _JSAMPARRAY output_buf,
                              JDIMENSION output_col);
EXTERN(void) _jpeg_idct_ifast(j_decompress_ptr cinfo,
                              jpeg_component_info *compptr,
                              JCOEFPTR coef_block, _JSAMPARRAY output_buf,
                              JDIMENSION output_col);
EXTERN(void) _jpeg_idct_float(j_decompress_ptr cinfo,
                              jpeg_component_info *compptr,
                              JCOEFPTR coef_block, _JSAMPARRAY output_buf,
                              JDIMENSION output_col);
EXTERN(void) _jpeg_idct_7x7(j_decompress_ptr cinfo,
                            jpeg_component_info *compptr, JCOEFPTR coef_block,
                            _JSAMPARRAY output_buf, JDIMENSION output_col);
EXTERN(void) _jpeg_idct_6x6(j_decompress_ptr cinfo,
                            jpeg_component_info *compptr, JCOEFPTR coef_block,
                            _JSAMPARRAY output_buf, JDIMENSION output_col);
EXTERN(void) _jpeg_idct_5x5(j_decompress_ptr cinfo,
                            jpeg_component_info *compptr, JCOEFPTR coef_block,
                            _JSAMPARRAY output_buf, JDIMENSION output_col);
EXTERN(void) _jpeg_idct_4x4(j_decompress_ptr cinfo,
                            jpeg_component_info *compptr, JCOEFPTR coef_block,
                            _JSAMPARRAY output_buf, JDIMENSION output_col);
EXTERN(void) _jpeg_idct_3x3(j_decompress_ptr cinfo,
                            jpeg_component_info *compptr, JCOEFPTR coef_block,
                            _JSAMPARRAY output_buf, JDIMENSION output_col);
EXTERN(void) _jpeg_idct_2x2(j_decompress_ptr cinfo,
                            jpeg_component_info *compptr, JCOEFPTR coef_block,
                            _JSAMPARRAY output_buf, JDIMENSION output_col);
EXTERN(void) _jpeg_idct_1x1(j_decompress_ptr cinfo,
                            jpeg_component_info *compptr, JCOEFPTR coef_block,
                            _JSAMPARRAY output_buf, JDIMENSION output_col);
EXTERN(void) _jpeg_idct_9x9(j_decompress_ptr cinfo,
                            jpeg_component_info *compptr, JCOEFPTR coef_block,
                            _JSAMPARRAY output_buf, JDIMENSION output_col);
EXTERN(void) _jpeg_idct_10x10(j_decompress_ptr cinfo,
                              jpeg_component_info *compptr,
                              JCOEFPTR coef_block, _JSAMPARRAY output_buf,
                              JDIMENSION output_col);
EXTERN(void) _jpeg_idct_11x11(j_decompress_ptr cinfo,
                              jpeg_component_info *compptr,
                              JCOEFPTR coef_block, _JSAMPARRAY output_buf,
                              JDIMENSION output_col);
EXTERN(void) _jpeg_idct_12x12(j_decompress_ptr cinfo,
                              jpeg_component_info *compptr,
                              JCOEFPTR coef_block, _JSAMPARRAY output_buf,
                              JDIMENSION output_col);
EXTERN(void) _jpeg_idct_13x13(j_decompress_ptr cinfo,
                              jpeg_component_info *compptr,
                              JCOEFPTR coef_block, _JSAMPARRAY output_buf,
                              JDIMENSION output_col);
EXTERN(void) _jpeg_idct_14x14(j_decompress_ptr cinfo,
                              jpeg_component_info *compptr,
                              JCOEFPTR coef_block, _JSAMPARRAY output_buf,
                              JDIMENSION output_col);
EXTERN(void) _jpeg_idct_15x15(j_decompress_ptr cinfo,
                              jpeg_component_info *compptr,
                              JCOEFPTR coef_block, _JSAMPARRAY output_buf,
                              JDIMENSION output_col);
EXTERN(void) _jpeg_idct_16x16(j_decompress_ptr cinfo,
                              jpeg_component_info *compptr,
                              JCOEFPTR coef_block, _JSAMPARRAY output_buf,
                              JDIMENSION output_col);


/*
 * Macros for handling fixed-point arithmetic; these are used by many
 * but not all of the DCT/IDCT modules.
 *
 * All values are expected to be of type JLONG.
 * Fractional constants are scaled left by CONST_BITS bits.
 * CONST_BITS is defined within each module using these macros,
 * and may differ from one module to the next.
 */

#define ONE          ((JLONG)1)
#define CONST_SCALE  (ONE << CONST_BITS)

/* Convert a positive real constant to an integer scaled by CONST_SCALE.
 * Caution: some C compilers fail to reduce "FIX(constant)" at compile time,
 * thus causing a lot of useless floating-point operations at run time.
 */

#define FIX(x)  ((JLONG)((x) * CONST_SCALE + 0.5))

/* Descale and correctly round a JLONG value that's scaled by N bits.
 * We assume RIGHT_SHIFT rounds towards minus infinity, so adding
 * the fudge factor is correct for either sign of X.
 */

#define DESCALE(x, n)  RIGHT_SHIFT((x) + (ONE << ((n) - 1)), n)

/* Multiply a JLONG variable by a JLONG constant to yield a JLONG result.
 * This macro is used only when the two inputs will actually be no more than
 * 16 bits wide, so that a 16x16->32 bit multiply can be used instead of a
 * full 32x32 multiply.  This provides a useful speedup on many machines.
 * Unfortunately there is no way to specify a 16x16->32 multiply portably
 * in C, but some C compilers will do the right thing if you provide the
 * correct combination of casts.
 */

#ifdef SHORTxSHORT_32           /* may work if 'int' is 32 bits */
#define MULTIPLY16C16(var, const)  (((INT16)(var)) * ((INT16)(const)))
#endif
#ifdef SHORTxLCONST_32          /* known to work with Microsoft C 6.0 */
#define MULTIPLY16C16(var, const)  (((INT16)(var)) * ((JLONG)(const)))
#endif

#ifndef MULTIPLY16C16           /* default definition */
#define MULTIPLY16C16(var, const)  ((var) * (const))
#endif

/* Same except both inputs are variables. */

#ifdef SHORTxSHORT_32           /* may work if 'int' is 32 bits */
#define MULTIPLY16V16(var1, var2)  (((INT16)(var1)) * ((INT16)(var2)))
#endif

#ifndef MULTIPLY16V16           /* default definition */
#define MULTIPLY16V16(var1, var2)  ((var1) * (var2))
#endif
