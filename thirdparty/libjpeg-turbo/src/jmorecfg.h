/*
 * jmorecfg.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Modified 1997-2009 by Guido Vollbeding.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2009, 2011, 2014-2015, 2018, 2020, 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains additional configuration options that customize the
 * JPEG software for special applications or support machine-dependent
 * optimizations.  Most users will not need to touch this file.
 */


/*
 * Maximum number of components (color channels) allowed in JPEG image.
 * To meet the letter of Rec. ITU-T T.81 | ISO/IEC 10918-1, set this to 255.
 * However, darn few applications need more than 4 channels (maybe 5 for CMYK +
 * alpha mask).  We recommend 10 as a reasonable compromise; use 4 if you are
 * really short on memory.  (Each allowed component costs a hundred or so
 * bytes of storage, whether actually used in an image or not.)
 */

#define MAX_COMPONENTS  10      /* maximum number of image components */


/*
 * Basic data types.
 * You may need to change these if you have a machine with unusual data
 * type sizes; for example, "char" not 8 bits, "short" not 16 bits,
 * or "long" not 32 bits.  We don't care whether "int" is 16 or 32 bits,
 * but it had better be at least 16.
 */

/* Representation of a single sample (pixel element value).
 * We frequently allocate large arrays of these, so it's important to keep
 * them small.  But if you have memory to burn and access to char or short
 * arrays is very slow on your hardware, you might want to change these.
 */

/* JSAMPLE should be the smallest type that will hold the values 0..255. */

typedef unsigned char JSAMPLE;
#define GETJSAMPLE(value)  ((int)(value))

#define MAXJSAMPLE       255
#define CENTERJSAMPLE    128


/* J12SAMPLE should be the smallest type that will hold the values 0..4095. */

typedef short J12SAMPLE;

#define MAXJ12SAMPLE     4095
#define CENTERJ12SAMPLE  2048


/* J16SAMPLE should be the smallest type that will hold the values 0..65535. */

typedef unsigned short J16SAMPLE;

#define MAXJ16SAMPLE     65535
#define CENTERJ16SAMPLE  32768


/* Representation of a DCT frequency coefficient.
 * This should be a signed value of at least 16 bits; "short" is usually OK.
 * Again, we allocate large arrays of these, but you can change to int
 * if you have memory to burn and "short" is really slow.
 */

typedef short JCOEF;


/* Compressed datastreams are represented as arrays of JOCTET.
 * These must be EXACTLY 8 bits wide, at least once they are written to
 * external storage.  Note that when using the stdio data source/destination
 * managers, this is also the data type passed to fread/fwrite.
 */

typedef unsigned char JOCTET;
#define GETJOCTET(value)  (value)


/* These typedefs are used for various table entries and so forth.
 * They must be at least as wide as specified; but making them too big
 * won't cost a huge amount of memory, so we don't provide special
 * extraction code like we did for JSAMPLE.  (In other words, these
 * typedefs live at a different point on the speed/space tradeoff curve.)
 */

/* UINT8 must hold at least the values 0..255. */

typedef unsigned char UINT8;

/* UINT16 must hold at least the values 0..65535. */

typedef unsigned short UINT16;

/* INT16 must hold at least the values -32768..32767. */

#ifndef XMD_H                   /* X11/xmd.h correctly defines INT16 */
typedef short INT16;
#endif

/* INT32 must hold at least signed 32-bit values.
 *
 * NOTE: The INT32 typedef dates back to libjpeg v5 (1994.)  Integers were
 * sometimes 16-bit back then (MS-DOS), which is why INT32 is typedef'd to
 * long.  It also wasn't common (or at least as common) in 1994 for INT32 to be
 * defined by platform headers.  Since then, however, INT32 is defined in
 * several other common places:
 *
 * Xmd.h (X11 header) typedefs INT32 to int on 64-bit platforms and long on
 * 32-bit platforms (i.e always a 32-bit signed type.)
 *
 * basetsd.h (Win32 header) typedefs INT32 to int (always a 32-bit signed type
 * on modern platforms.)
 *
 * qglobal.h (Qt header) typedefs INT32 to int (always a 32-bit signed type on
 * modern platforms.)
 *
 * This is a recipe for conflict, since "long" and "int" aren't always
 * compatible types.  Since the definition of INT32 has technically been part
 * of the libjpeg API for more than 20 years, we can't remove it, but we do not
 * use it internally any longer.  We instead define a separate type (JLONG)
 * for internal use, which ensures that internal behavior will always be the
 * same regardless of any external headers that may be included.
 */

#ifndef XMD_H                   /* X11/xmd.h correctly defines INT32 */
#ifndef _BASETSD_H_             /* Microsoft defines it in basetsd.h */
#ifndef _BASETSD_H              /* MinGW is slightly different */
#ifndef QGLOBAL_H               /* Qt defines it in qglobal.h */
typedef long INT32;
#endif
#endif
#endif
#endif

/* Datatype used for image dimensions.  The JPEG standard only supports
 * images up to 64K*64K due to 16-bit fields in SOF markers.  Therefore
 * "unsigned int" is sufficient on all machines.  However, if you need to
 * handle larger images and you don't mind deviating from the spec, you
 * can change this datatype.  (Note that changing this datatype will
 * potentially require modifying the SIMD code.  The x86-64 SIMD extensions,
 * in particular, assume a 32-bit JDIMENSION.)
 */

typedef unsigned int JDIMENSION;

#define JPEG_MAX_DIMENSION  65500L  /* a tad under 64K to prevent overflows */


/* These macros are used in all function definitions and extern declarations.
 * You could modify them if you need to change function linkage conventions;
 * in particular, you'll need to do that to make the library a Windows DLL.
 * Another application is to make all functions global for use with debuggers
 * or code profilers that require it.
 */

/* a function called through method pointers: */
#define METHODDEF(type)         static type
/* a function used only in its module: */
#define LOCAL(type)             static type
/* a function referenced thru EXTERNs: */
#define GLOBAL(type)            type
/* a reference to a GLOBAL function: */
#define EXTERN(type)            extern type


/* Originally, this macro was used as a way of defining function prototypes
 * for both modern compilers as well as older compilers that did not support
 * prototype parameters.  libjpeg-turbo has never supported these older,
 * non-ANSI compilers, but the macro is still included because there is some
 * software out there that uses it.
 */

#define JMETHOD(type, methodname, arglist)  type (*methodname) arglist


/* libjpeg-turbo no longer supports platforms that have far symbols (MS-DOS),
 * but again, some software relies on this macro.
 */

#undef FAR
#define FAR


/*
 * On a few systems, type boolean and/or its values FALSE, TRUE may appear
 * in standard header files.  Or you may have conflicts with application-
 * specific header files that you want to include together with these files.
 * Defining HAVE_BOOLEAN before including jpeglib.h should make it work.
 */

#ifndef HAVE_BOOLEAN
typedef int boolean;
#endif
#ifndef FALSE                   /* in case these macros already exist */
#define FALSE   0               /* values of boolean */
#endif
#ifndef TRUE
#define TRUE    1
#endif


/*
 * The remaining options affect code selection within the JPEG library,
 * but they don't need to be visible to most applications using the library.
 * To minimize application namespace pollution, the symbols won't be
 * defined unless JPEG_INTERNALS or JPEG_INTERNAL_OPTIONS has been defined.
 */

#ifdef JPEG_INTERNALS
#define JPEG_INTERNAL_OPTIONS
#endif

#ifdef JPEG_INTERNAL_OPTIONS


/*
 * These defines indicate whether to include various optional functions.
 * Undefining some of these symbols will produce a smaller but less capable
 * library.  Note that you can leave certain source files out of the
 * compilation/linking process if you've #undef'd the corresponding symbols.
 * (You may HAVE to do that if your compiler doesn't like null source files.)
 */

/* Capability options common to encoder and decoder: */

#define DCT_ISLOW_SUPPORTED     /* accurate integer method */
#define DCT_IFAST_SUPPORTED     /* less accurate int method [legacy feature] */
#define DCT_FLOAT_SUPPORTED     /* floating-point method [legacy feature] */

/* Encoder capability options: */

#define C_MULTISCAN_FILES_SUPPORTED /* Multiple-scan JPEG files? */
#define C_PROGRESSIVE_SUPPORTED     /* Progressive JPEG? (Requires MULTISCAN)*/
#define C_LOSSLESS_SUPPORTED        /* Lossless JPEG? */
#define ENTROPY_OPT_SUPPORTED       /* Optimization of entropy coding parms? */
/* Note: if you selected 12-bit data precision, it is dangerous to turn off
 * ENTROPY_OPT_SUPPORTED.  The standard Huffman tables are only good for 8-bit
 * precision, so jchuff.c normally uses entropy optimization to compute
 * usable tables for higher precision.  If you don't want to do optimization,
 * you'll have to supply different default Huffman tables.
 * The exact same statements apply for progressive and lossless JPEG:
 * the default tables don't work for progressive mode or lossless mode.
 * (This may get fixed, however.)
 */
#define INPUT_SMOOTHING_SUPPORTED   /* Input image smoothing option? */

/* Decoder capability options: */

#define D_MULTISCAN_FILES_SUPPORTED /* Multiple-scan JPEG files? */
#define D_PROGRESSIVE_SUPPORTED     /* Progressive JPEG? (Requires MULTISCAN)*/
#define D_LOSSLESS_SUPPORTED        /* Lossless JPEG? */
#define SAVE_MARKERS_SUPPORTED      /* jpeg_save_markers() needed? */
#define BLOCK_SMOOTHING_SUPPORTED   /* Block smoothing? (Progressive only) */
#define IDCT_SCALING_SUPPORTED      /* Output rescaling via IDCT? */
#undef  UPSAMPLE_SCALING_SUPPORTED  /* Output rescaling at upsample stage? */
#define UPSAMPLE_MERGING_SUPPORTED  /* Fast path for sloppy upsampling? */
#define QUANT_1PASS_SUPPORTED       /* 1-pass color quantization? */
#define QUANT_2PASS_SUPPORTED       /* 2-pass color quantization? */

/* more capability options later, no doubt */


/*
 * The RGB_RED, RGB_GREEN, RGB_BLUE, and RGB_PIXELSIZE macros are a vestigial
 * feature of libjpeg.  The idea was that, if an application developer needed
 * to compress from/decompress to a BGR/BGRX/RGBX/XBGR/XRGB buffer, they could
 * change these macros, rebuild libjpeg, and link their application statically
 * with it.  In reality, few people ever did this, because there were some
 * severe restrictions involved (cjpeg and djpeg no longer worked properly,
 * compressing/decompressing RGB JPEGs no longer worked properly, and the color
 * quantizer wouldn't work with pixel sizes other than 3.)  Furthermore, since
 * all of the O/S-supplied versions of libjpeg were built with the default
 * values of RGB_RED, RGB_GREEN, RGB_BLUE, and RGB_PIXELSIZE, many applications
 * have come to regard these values as immutable.
 *
 * The libjpeg-turbo colorspace extensions provide a much cleaner way of
 * compressing from/decompressing to buffers with arbitrary component orders
 * and pixel sizes.  Thus, we do not support changing the values of RGB_RED,
 * RGB_GREEN, RGB_BLUE, or RGB_PIXELSIZE.  In addition to the restrictions
 * listed above, changing these values will also break the SIMD extensions and
 * the regression tests.
 */

#define RGB_RED         0       /* Offset of Red in an RGB scanline element */
#define RGB_GREEN       1       /* Offset of Green */
#define RGB_BLUE        2       /* Offset of Blue */
#define RGB_PIXELSIZE   3       /* JSAMPLEs per RGB scanline element */

#define JPEG_NUMCS  17

#define EXT_RGB_RED         0
#define EXT_RGB_GREEN       1
#define EXT_RGB_BLUE        2
#define EXT_RGB_PIXELSIZE   3

#define EXT_RGBX_RED        0
#define EXT_RGBX_GREEN      1
#define EXT_RGBX_BLUE       2
#define EXT_RGBX_PIXELSIZE  4

#define EXT_BGR_RED         2
#define EXT_BGR_GREEN       1
#define EXT_BGR_BLUE        0
#define EXT_BGR_PIXELSIZE   3

#define EXT_BGRX_RED        2
#define EXT_BGRX_GREEN      1
#define EXT_BGRX_BLUE       0
#define EXT_BGRX_PIXELSIZE  4

#define EXT_XBGR_RED        3
#define EXT_XBGR_GREEN      2
#define EXT_XBGR_BLUE       1
#define EXT_XBGR_PIXELSIZE  4

#define EXT_XRGB_RED        1
#define EXT_XRGB_GREEN      2
#define EXT_XRGB_BLUE       3
#define EXT_XRGB_PIXELSIZE  4

static const int rgb_red[JPEG_NUMCS] = {
  -1, -1, RGB_RED, -1, -1, -1, EXT_RGB_RED, EXT_RGBX_RED,
  EXT_BGR_RED, EXT_BGRX_RED, EXT_XBGR_RED, EXT_XRGB_RED,
  EXT_RGBX_RED, EXT_BGRX_RED, EXT_XBGR_RED, EXT_XRGB_RED,
  -1
};

static const int rgb_green[JPEG_NUMCS] = {
  -1, -1, RGB_GREEN, -1, -1, -1, EXT_RGB_GREEN, EXT_RGBX_GREEN,
  EXT_BGR_GREEN, EXT_BGRX_GREEN, EXT_XBGR_GREEN, EXT_XRGB_GREEN,
  EXT_RGBX_GREEN, EXT_BGRX_GREEN, EXT_XBGR_GREEN, EXT_XRGB_GREEN,
  -1
};

static const int rgb_blue[JPEG_NUMCS] = {
  -1, -1, RGB_BLUE, -1, -1, -1, EXT_RGB_BLUE, EXT_RGBX_BLUE,
  EXT_BGR_BLUE, EXT_BGRX_BLUE, EXT_XBGR_BLUE, EXT_XRGB_BLUE,
  EXT_RGBX_BLUE, EXT_BGRX_BLUE, EXT_XBGR_BLUE, EXT_XRGB_BLUE,
  -1
};

static const int rgb_pixelsize[JPEG_NUMCS] = {
  -1, -1, RGB_PIXELSIZE, -1, -1, -1, EXT_RGB_PIXELSIZE, EXT_RGBX_PIXELSIZE,
  EXT_BGR_PIXELSIZE, EXT_BGRX_PIXELSIZE, EXT_XBGR_PIXELSIZE, EXT_XRGB_PIXELSIZE,
  EXT_RGBX_PIXELSIZE, EXT_BGRX_PIXELSIZE, EXT_XBGR_PIXELSIZE, EXT_XRGB_PIXELSIZE,
  -1
};

/* Definitions for speed-related optimizations. */

/* On some machines (notably 68000 series) "int" is 32 bits, but multiplying
 * two 16-bit shorts is faster than multiplying two ints.  Define MULTIPLIER
 * as short on such a machine.  MULTIPLIER must be at least 16 bits wide.
 */

#ifndef MULTIPLIER
#ifndef WITH_SIMD
#define MULTIPLIER  int         /* type for fastest integer multiply */
#else
#define MULTIPLIER  short       /* prefer 16-bit with SIMD for parellelism */
#endif
#endif


/* FAST_FLOAT should be either float or double, whichever is done faster
 * by your compiler.  (Note that this type is only used in the floating point
 * DCT routines, so it only matters if you've defined DCT_FLOAT_SUPPORTED.)
 */

#ifndef FAST_FLOAT
#define FAST_FLOAT  float
#endif

#endif /* JPEG_INTERNAL_OPTIONS */
