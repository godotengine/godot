/*
 * rdppm.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Modified 2009 by Bill Allombert, Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2015-2017, 2020-2024, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains routines to read input images in PPM/PGM format.
 * The extended 2-byte-per-sample raw PPM/PGM formats are supported.
 * The PBMPLUS library is NOT required to compile this software
 * (but it is highly useful as a set of PPM image manipulation programs).
 *
 * These routines may need modification for non-Unix environments or
 * specialized applications.  As they stand, they assume input from
 * an ordinary stdio stream.  They further assume that reading begins
 * at the start of the file; start_input may need work if the
 * user interface has already read some data (e.g., to determine that
 * the file is indeed PPM format).
 */

#include "cmyk.h"
#include "cdjpeg.h"             /* Common decls for cjpeg/djpeg applications */

#if defined(PPM_SUPPORTED) && \
    (BITS_IN_JSAMPLE != 16 || defined(C_LOSSLESS_SUPPORTED))


/* Portions of this code are based on the PBMPLUS library, which is:
**
** Copyright (C) 1988 by Jef Poskanzer.
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
*/


/* Macros to deal with unsigned chars as efficiently as compiler allows */

typedef unsigned char U_CHAR;
#define UCH(x)  ((int)(x))


#define ReadOK(file, buffer, len) \
  (fread(buffer, 1, len, file) == ((size_t)(len)))

static int alpha_index[JPEG_NUMCS] = {
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 3, 0, 0, -1
};


/* Private version of data source object */

typedef struct {
  struct cjpeg_source_struct pub; /* public fields */

  /* Usually these two pointers point to the same place: */
  U_CHAR *iobuffer;             /* fread's I/O buffer */
  _JSAMPROW pixrow;             /* compressor input buffer */
  size_t buffer_width;          /* width of I/O buffer */
  _JSAMPLE *rescale;            /* => maxval-remapping array, or NULL */
  unsigned int maxval;
} ppm_source_struct;

typedef ppm_source_struct *ppm_source_ptr;


LOCAL(int)
pbm_getc(FILE *infile)
/* Read next char, skipping over any comments */
/* A comment/newline sequence is returned as a newline */
{
  register int ch;

  ch = getc(infile);
  if (ch == '#') {
    do {
      ch = getc(infile);
    } while (ch != '\n' && ch != EOF);
  }
  return ch;
}


LOCAL(unsigned int)
read_pbm_integer(j_compress_ptr cinfo, FILE *infile, unsigned int maxval)
/* Read an unsigned decimal integer from the PPM file */
/* Swallows one trailing character after the integer */
/* Note that on a 16-bit-int machine, only values up to 64k can be read. */
/* This should not be a problem in practice. */
{
  register int ch;
  register unsigned int val;

  /* Skip any leading whitespace */
  do {
    ch = pbm_getc(infile);
    if (ch == EOF)
      ERREXIT(cinfo, JERR_INPUT_EOF);
  } while (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r');

  if (ch < '0' || ch > '9')
    ERREXIT(cinfo, JERR_PPM_NONNUMERIC);

  val = ch - '0';
  while ((ch = pbm_getc(infile)) >= '0' && ch <= '9') {
    val *= 10;
    val += ch - '0';
    if (val > maxval)
      ERREXIT(cinfo, JERR_PPM_OUTOFRANGE);
  }

  return val;
}


/*
 * Read one row of pixels.
 *
 * We provide several different versions depending on input file format.
 * In all cases, input is scaled to cinfo->data_precision.
 *
 * A really fast path is provided for reading byte/sample raw files with
 * maxval <= _MAXJSAMPLE and maxval == (1U << cinfo->data_precision) - 1U,
 * which is the normal case for 8-bit data.
 */


METHODDEF(JDIMENSION)
get_text_gray_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading text-format PGM files with any maxval */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  FILE *infile = source->pub.input_file;
  register _JSAMPROW ptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;

  ptr = source->pub._buffer[0];
  for (col = cinfo->image_width; col > 0; col--) {
    *ptr++ = rescale[read_pbm_integer(cinfo, infile, maxval)];
  }
  return 1;
}


#define GRAY_RGB_READ_LOOP(read_op, alpha_set_op) { \
  for (col = cinfo->image_width; col > 0; col--) { \
    ptr[rindex] = ptr[gindex] = ptr[bindex] = read_op; \
    alpha_set_op \
    ptr += ps; \
  } \
}

METHODDEF(JDIMENSION)
get_text_gray_rgb_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading text-format PGM files with any maxval and
   converting to extended RGB */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  FILE *infile = source->pub.input_file;
  register _JSAMPROW ptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;
  register int rindex = rgb_red[cinfo->in_color_space];
  register int gindex = rgb_green[cinfo->in_color_space];
  register int bindex = rgb_blue[cinfo->in_color_space];
  register int aindex = alpha_index[cinfo->in_color_space];
  register int ps = rgb_pixelsize[cinfo->in_color_space];

  ptr = source->pub._buffer[0];
  if (maxval == (1U << cinfo->data_precision) - 1U) {
    if (aindex >= 0)
      GRAY_RGB_READ_LOOP((_JSAMPLE)read_pbm_integer(cinfo, infile, maxval),
                         ptr[aindex] = (_JSAMPLE)maxval;)
    else
      GRAY_RGB_READ_LOOP((_JSAMPLE)read_pbm_integer(cinfo, infile, maxval), {})
  } else {
    if (aindex >= 0)
      GRAY_RGB_READ_LOOP(rescale[read_pbm_integer(cinfo, infile, maxval)],
                         ptr[aindex] = (1 << cinfo->data_precision) - 1;)
    else
      GRAY_RGB_READ_LOOP(rescale[read_pbm_integer(cinfo, infile, maxval)], {})
  }
  return 1;
}


METHODDEF(JDIMENSION)
get_text_gray_cmyk_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading text-format PGM files with any maxval and
   converting to CMYK */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  FILE *infile = source->pub.input_file;
  register _JSAMPROW ptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;

  ptr = source->pub._buffer[0];
  if (maxval == (1U << cinfo->data_precision) - 1U) {
    for (col = cinfo->image_width; col > 0; col--) {
      _JSAMPLE gray = (_JSAMPLE)read_pbm_integer(cinfo, infile, maxval);
      rgb_to_cmyk(maxval, gray, gray, gray, ptr, ptr + 1, ptr + 2, ptr + 3);
      ptr += 4;
    }
  } else {
    for (col = cinfo->image_width; col > 0; col--) {
      _JSAMPLE gray = rescale[read_pbm_integer(cinfo, infile, maxval)];
      rgb_to_cmyk(maxval, gray, gray, gray, ptr, ptr + 1, ptr + 2, ptr + 3);
      ptr += 4;
    }
  }
  return 1;
}


#define RGB_READ_LOOP(read_op, alpha_set_op) { \
  for (col = cinfo->image_width; col > 0; col--) { \
    ptr[rindex] = read_op; \
    ptr[gindex] = read_op; \
    ptr[bindex] = read_op; \
    alpha_set_op \
    ptr += ps; \
  } \
}

METHODDEF(JDIMENSION)
get_text_rgb_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading text-format PPM files with any maxval */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  FILE *infile = source->pub.input_file;
  register _JSAMPROW ptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;
  register int rindex = rgb_red[cinfo->in_color_space];
  register int gindex = rgb_green[cinfo->in_color_space];
  register int bindex = rgb_blue[cinfo->in_color_space];
  register int aindex = alpha_index[cinfo->in_color_space];
  register int ps = rgb_pixelsize[cinfo->in_color_space];

  ptr = source->pub._buffer[0];
  if (maxval == (1U << cinfo->data_precision) - 1U) {
    if (aindex >= 0)
      RGB_READ_LOOP((_JSAMPLE)read_pbm_integer(cinfo, infile, maxval),
                    ptr[aindex] = (_JSAMPLE)maxval;)
    else
      RGB_READ_LOOP((_JSAMPLE)read_pbm_integer(cinfo, infile, maxval), {})
  } else {
    if (aindex >= 0)
      RGB_READ_LOOP(rescale[read_pbm_integer(cinfo, infile, maxval)],
                    ptr[aindex] = (1 << cinfo->data_precision) - 1;)
    else
      RGB_READ_LOOP(rescale[read_pbm_integer(cinfo, infile, maxval)], {})
  }
  return 1;
}


METHODDEF(JDIMENSION)
get_text_rgb_cmyk_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading text-format PPM files with any maxval and
   converting to CMYK */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  FILE *infile = source->pub.input_file;
  register _JSAMPROW ptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;

  ptr = source->pub._buffer[0];
  if (maxval == (1U << cinfo->data_precision) - 1U) {
    for (col = cinfo->image_width; col > 0; col--) {
      _JSAMPLE r = (_JSAMPLE)read_pbm_integer(cinfo, infile, maxval);
      _JSAMPLE g = (_JSAMPLE)read_pbm_integer(cinfo, infile, maxval);
      _JSAMPLE b = (_JSAMPLE)read_pbm_integer(cinfo, infile, maxval);
      rgb_to_cmyk(maxval, r, g, b, ptr, ptr + 1, ptr + 2, ptr + 3);
      ptr += 4;
    }
  } else {
    for (col = cinfo->image_width; col > 0; col--) {
      _JSAMPLE r = rescale[read_pbm_integer(cinfo, infile, maxval)];
      _JSAMPLE g = rescale[read_pbm_integer(cinfo, infile, maxval)];
      _JSAMPLE b = rescale[read_pbm_integer(cinfo, infile, maxval)];
      rgb_to_cmyk(maxval, r, g, b, ptr, ptr + 1, ptr + 2, ptr + 3);
      ptr += 4;
    }
  }
  return 1;
}


METHODDEF(JDIMENSION)
get_scaled_gray_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading raw-byte-format PGM files with any maxval */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  register _JSAMPROW ptr;
  register U_CHAR *bufferptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;

  if (!ReadOK(source->pub.input_file, source->iobuffer, source->buffer_width))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  ptr = source->pub._buffer[0];
  bufferptr = source->iobuffer;
  for (col = cinfo->image_width; col > 0; col--) {
    *ptr++ = rescale[UCH(*bufferptr++)];
  }
  return 1;
}


METHODDEF(JDIMENSION)
get_gray_rgb_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading raw-byte-format PGM files with any maxval
   and converting to extended RGB */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  register _JSAMPROW ptr;
  register U_CHAR *bufferptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;
  register int rindex = rgb_red[cinfo->in_color_space];
  register int gindex = rgb_green[cinfo->in_color_space];
  register int bindex = rgb_blue[cinfo->in_color_space];
  register int aindex = alpha_index[cinfo->in_color_space];
  register int ps = rgb_pixelsize[cinfo->in_color_space];

  if (!ReadOK(source->pub.input_file, source->iobuffer, source->buffer_width))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  ptr = source->pub._buffer[0];
  bufferptr = source->iobuffer;
  if (maxval == (1U << cinfo->data_precision) - 1U) {
    if (aindex >= 0)
      GRAY_RGB_READ_LOOP(*bufferptr++, ptr[aindex] = (_JSAMPLE)maxval;)
    else
      GRAY_RGB_READ_LOOP(*bufferptr++, {})
  } else {
    if (aindex >= 0)
      GRAY_RGB_READ_LOOP(rescale[UCH(*bufferptr++)],
                         ptr[aindex] = (1 << cinfo->data_precision) - 1;)
    else
      GRAY_RGB_READ_LOOP(rescale[UCH(*bufferptr++)], {})
  }
  return 1;
}


METHODDEF(JDIMENSION)
get_gray_cmyk_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading raw-byte-format PGM files with any maxval
   and converting to CMYK */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  register _JSAMPROW ptr;
  register U_CHAR *bufferptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;

  if (!ReadOK(source->pub.input_file, source->iobuffer, source->buffer_width))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  ptr = source->pub._buffer[0];
  bufferptr = source->iobuffer;
  if (maxval == (1U << cinfo->data_precision) - 1U) {
    for (col = cinfo->image_width; col > 0; col--) {
      _JSAMPLE gray = *bufferptr++;
      rgb_to_cmyk(maxval, gray, gray, gray, ptr, ptr + 1, ptr + 2, ptr + 3);
      ptr += 4;
    }
  } else {
    for (col = cinfo->image_width; col > 0; col--) {
      _JSAMPLE gray = rescale[UCH(*bufferptr++)];
      rgb_to_cmyk(maxval, gray, gray, gray, ptr, ptr + 1, ptr + 2, ptr + 3);
      ptr += 4;
    }
  }
  return 1;
}


METHODDEF(JDIMENSION)
get_rgb_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading raw-byte-format PPM files with any maxval */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  register _JSAMPROW ptr;
  register U_CHAR *bufferptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;
  register int rindex = rgb_red[cinfo->in_color_space];
  register int gindex = rgb_green[cinfo->in_color_space];
  register int bindex = rgb_blue[cinfo->in_color_space];
  register int aindex = alpha_index[cinfo->in_color_space];
  register int ps = rgb_pixelsize[cinfo->in_color_space];

  if (!ReadOK(source->pub.input_file, source->iobuffer, source->buffer_width))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  ptr = source->pub._buffer[0];
  bufferptr = source->iobuffer;
  if (maxval == (1U << cinfo->data_precision) - 1U) {
    if (aindex >= 0)
      RGB_READ_LOOP(*bufferptr++, ptr[aindex] = (_JSAMPLE)maxval;)
    else
      RGB_READ_LOOP(*bufferptr++, {})
  } else {
    if (aindex >= 0)
      RGB_READ_LOOP(rescale[UCH(*bufferptr++)],
                    ptr[aindex] = (1 << cinfo->data_precision) - 1;)
    else
      RGB_READ_LOOP(rescale[UCH(*bufferptr++)], {})
  }
  return 1;
}


METHODDEF(JDIMENSION)
get_rgb_cmyk_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading raw-byte-format PPM files with any maxval and
   converting to CMYK */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  register _JSAMPROW ptr;
  register U_CHAR *bufferptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;

  if (!ReadOK(source->pub.input_file, source->iobuffer, source->buffer_width))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  ptr = source->pub._buffer[0];
  bufferptr = source->iobuffer;
  if (maxval == (1U << cinfo->data_precision) - 1U) {
    for (col = cinfo->image_width; col > 0; col--) {
      _JSAMPLE r = *bufferptr++;
      _JSAMPLE g = *bufferptr++;
      _JSAMPLE b = *bufferptr++;
      rgb_to_cmyk(maxval, r, g, b, ptr, ptr + 1, ptr + 2, ptr + 3);
      ptr += 4;
    }
  } else {
    for (col = cinfo->image_width; col > 0; col--) {
      _JSAMPLE r = rescale[UCH(*bufferptr++)];
      _JSAMPLE g = rescale[UCH(*bufferptr++)];
      _JSAMPLE b = rescale[UCH(*bufferptr++)];
      rgb_to_cmyk(maxval, r, g, b, ptr, ptr + 1, ptr + 2, ptr + 3);
      ptr += 4;
    }
  }
  return 1;
}


METHODDEF(JDIMENSION)
get_raw_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading raw-byte-format files with
 * maxval <= _MAXJSAMPLE and maxval == (1U << cinfo->data_precision) - 1U.
 * In this case we just read right into the _JSAMPLE buffer!
 * Note that same code works for PPM and PGM files.
 */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;

  if (!ReadOK(source->pub.input_file, source->iobuffer, source->buffer_width))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  return 1;
}


METHODDEF(JDIMENSION)
get_word_gray_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading raw-word-format PGM files with any maxval */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  register _JSAMPROW ptr;
  register U_CHAR *bufferptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;

  if (!ReadOK(source->pub.input_file, source->iobuffer, source->buffer_width))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  ptr = source->pub._buffer[0];
  bufferptr = source->iobuffer;
  for (col = cinfo->image_width; col > 0; col--) {
    register unsigned int temp;
    temp  = UCH(*bufferptr++) << 8;
    temp |= UCH(*bufferptr++);
    if (temp > maxval)
      ERREXIT(cinfo, JERR_PPM_OUTOFRANGE);
    *ptr++ = rescale[temp];
  }
  return 1;
}


METHODDEF(JDIMENSION)
get_word_gray_rgb_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading raw-word-format PGM files with any maxval */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  register _JSAMPROW ptr;
  register U_CHAR *bufferptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;
  register int rindex = rgb_red[cinfo->in_color_space];
  register int gindex = rgb_green[cinfo->in_color_space];
  register int bindex = rgb_blue[cinfo->in_color_space];
  register int aindex = alpha_index[cinfo->in_color_space];
  register int ps = rgb_pixelsize[cinfo->in_color_space];

  if (!ReadOK(source->pub.input_file, source->iobuffer, source->buffer_width))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  ptr = source->pub._buffer[0];
  bufferptr = source->iobuffer;
  for (col = cinfo->image_width; col > 0; col--) {
    register unsigned int temp;
    temp  = UCH(*bufferptr++) << 8;
    temp |= UCH(*bufferptr++);
    if (temp > maxval)
      ERREXIT(cinfo, JERR_PPM_OUTOFRANGE);
    ptr[rindex] = ptr[gindex] = ptr[bindex] = rescale[temp];
    if (aindex >= 0)
      ptr[aindex] = (1 << cinfo->data_precision) - 1;
    ptr += ps;
  }
  return 1;
}


METHODDEF(JDIMENSION)
get_word_gray_cmyk_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading raw-word-format PGM files with any maxval */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  register _JSAMPROW ptr;
  register U_CHAR *bufferptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;

  if (!ReadOK(source->pub.input_file, source->iobuffer, source->buffer_width))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  ptr = source->pub._buffer[0];
  bufferptr = source->iobuffer;
  for (col = cinfo->image_width; col > 0; col--) {
    register unsigned int gray;
    gray  = UCH(*bufferptr++) << 8;
    gray |= UCH(*bufferptr++);
    if (gray > maxval)
      ERREXIT(cinfo, JERR_PPM_OUTOFRANGE);
    rgb_to_cmyk(maxval, rescale[gray], rescale[gray], rescale[gray], ptr,
                ptr + 1, ptr + 2, ptr + 3);
    ptr += 4;
  }
  return 1;
}


METHODDEF(JDIMENSION)
get_word_rgb_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading raw-word-format PPM files with any maxval */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  register _JSAMPROW ptr;
  register U_CHAR *bufferptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;
  register int rindex = rgb_red[cinfo->in_color_space];
  register int gindex = rgb_green[cinfo->in_color_space];
  register int bindex = rgb_blue[cinfo->in_color_space];
  register int aindex = alpha_index[cinfo->in_color_space];
  register int ps = rgb_pixelsize[cinfo->in_color_space];

  if (!ReadOK(source->pub.input_file, source->iobuffer, source->buffer_width))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  ptr = source->pub._buffer[0];
  bufferptr = source->iobuffer;
  for (col = cinfo->image_width; col > 0; col--) {
    register unsigned int temp;
    temp  = UCH(*bufferptr++) << 8;
    temp |= UCH(*bufferptr++);
    if (temp > maxval)
      ERREXIT(cinfo, JERR_PPM_OUTOFRANGE);
    ptr[rindex] = rescale[temp];
    temp  = UCH(*bufferptr++) << 8;
    temp |= UCH(*bufferptr++);
    if (temp > maxval)
      ERREXIT(cinfo, JERR_PPM_OUTOFRANGE);
    ptr[gindex] = rescale[temp];
    temp  = UCH(*bufferptr++) << 8;
    temp |= UCH(*bufferptr++);
    if (temp > maxval)
      ERREXIT(cinfo, JERR_PPM_OUTOFRANGE);
    ptr[bindex] = rescale[temp];
    if (aindex >= 0)
      ptr[aindex] = (1 << cinfo->data_precision) - 1;
    ptr += ps;
  }
  return 1;
}


METHODDEF(JDIMENSION)
get_word_rgb_cmyk_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading raw-word-format PPM files with any maxval */
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  register _JSAMPROW ptr;
  register U_CHAR *bufferptr;
  register _JSAMPLE *rescale = source->rescale;
  JDIMENSION col;
  unsigned int maxval = source->maxval;

  if (!ReadOK(source->pub.input_file, source->iobuffer, source->buffer_width))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  ptr = source->pub._buffer[0];
  bufferptr = source->iobuffer;
  for (col = cinfo->image_width; col > 0; col--) {
    register unsigned int r, g, b;
    r  = UCH(*bufferptr++) << 8;
    r |= UCH(*bufferptr++);
    if (r > maxval)
      ERREXIT(cinfo, JERR_PPM_OUTOFRANGE);
    g  = UCH(*bufferptr++) << 8;
    g |= UCH(*bufferptr++);
    if (g > maxval)
      ERREXIT(cinfo, JERR_PPM_OUTOFRANGE);
    b  = UCH(*bufferptr++) << 8;
    b |= UCH(*bufferptr++);
    if (b > maxval)
      ERREXIT(cinfo, JERR_PPM_OUTOFRANGE);
    rgb_to_cmyk(maxval, rescale[r], rescale[g], rescale[b], ptr, ptr + 1,
                ptr + 2, ptr + 3);
    ptr += 4;
  }
  return 1;
}


/*
 * Read the file header; return image size and component count.
 */

METHODDEF(void)
start_input_ppm(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  ppm_source_ptr source = (ppm_source_ptr)sinfo;
  int c;
  unsigned int w, h, maxval;
  boolean need_iobuffer, use_raw_buffer, need_rescale;

  if (getc(source->pub.input_file) != 'P')
    ERREXIT(cinfo, JERR_PPM_NOT);

  c = getc(source->pub.input_file); /* subformat discriminator character */

  /* detect unsupported variants (ie, PBM) before trying to read header */
  switch (c) {
  case '2':                     /* it's a text-format PGM file */
  case '3':                     /* it's a text-format PPM file */
  case '5':                     /* it's a raw-format PGM file */
  case '6':                     /* it's a raw-format PPM file */
    break;
  default:
    ERREXIT(cinfo, JERR_PPM_NOT);
    break;
  }

  /* fetch the remaining header info */
  w = read_pbm_integer(cinfo, source->pub.input_file, 65535);
  h = read_pbm_integer(cinfo, source->pub.input_file, 65535);
  maxval = read_pbm_integer(cinfo, source->pub.input_file, 65535);

  if (w <= 0 || h <= 0 || maxval <= 0) /* error check */
    ERREXIT(cinfo, JERR_PPM_NOT);
  if (sinfo->max_pixels && (unsigned long long)w * h > sinfo->max_pixels)
    ERREXIT1(cinfo, JERR_IMAGE_TOO_BIG, sinfo->max_pixels);

  cinfo->image_width = (JDIMENSION)w;
  cinfo->image_height = (JDIMENSION)h;
  source->maxval = maxval;

  /* initialize flags to most common settings */
  need_iobuffer = TRUE;         /* do we need an I/O buffer? */
  use_raw_buffer = FALSE;       /* do we map input buffer onto I/O buffer? */
  need_rescale = TRUE;          /* do we need a rescale array? */

  switch (c) {
  case '2':                     /* it's a text-format PGM file */
    if (cinfo->in_color_space == JCS_UNKNOWN ||
        cinfo->in_color_space == JCS_RGB)
      cinfo->in_color_space = JCS_GRAYSCALE;
    TRACEMS3(cinfo, 1, JTRC_PGM_TEXT, w, h, maxval);
    if (cinfo->in_color_space == JCS_GRAYSCALE)
      source->pub.get_pixel_rows = get_text_gray_row;
    else if (IsExtRGB(cinfo->in_color_space))
      source->pub.get_pixel_rows = get_text_gray_rgb_row;
    else if (cinfo->in_color_space == JCS_CMYK)
      source->pub.get_pixel_rows = get_text_gray_cmyk_row;
    else
      ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    need_iobuffer = FALSE;
    break;

  case '3':                     /* it's a text-format PPM file */
    if (cinfo->in_color_space == JCS_UNKNOWN)
      cinfo->in_color_space = JCS_EXT_RGB;
    TRACEMS3(cinfo, 1, JTRC_PPM_TEXT, w, h, maxval);
    if (IsExtRGB(cinfo->in_color_space))
      source->pub.get_pixel_rows = get_text_rgb_row;
    else if (cinfo->in_color_space == JCS_CMYK)
      source->pub.get_pixel_rows = get_text_rgb_cmyk_row;
    else
      ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    need_iobuffer = FALSE;
    break;

  case '5':                     /* it's a raw-format PGM file */
    if (cinfo->in_color_space == JCS_UNKNOWN ||
        cinfo->in_color_space == JCS_RGB)
      cinfo->in_color_space = JCS_GRAYSCALE;
    TRACEMS3(cinfo, 1, JTRC_PGM, w, h, maxval);
    if (maxval > 255) {
      if (cinfo->in_color_space == JCS_GRAYSCALE)
        source->pub.get_pixel_rows = get_word_gray_row;
      else if (IsExtRGB(cinfo->in_color_space))
        source->pub.get_pixel_rows = get_word_gray_rgb_row;
      else if (cinfo->in_color_space == JCS_CMYK)
        source->pub.get_pixel_rows = get_word_gray_cmyk_row;
      else
        ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    } else if (maxval <= _MAXJSAMPLE && sizeof(_JSAMPLE) == sizeof(U_CHAR) &&
               maxval == ((1U << cinfo->data_precision) - 1U) &&
               cinfo->in_color_space == JCS_GRAYSCALE) {
      source->pub.get_pixel_rows = get_raw_row;
      use_raw_buffer = TRUE;
      need_rescale = FALSE;
    } else {
      if (cinfo->in_color_space == JCS_GRAYSCALE)
        source->pub.get_pixel_rows = get_scaled_gray_row;
      else if (IsExtRGB(cinfo->in_color_space))
        source->pub.get_pixel_rows = get_gray_rgb_row;
      else if (cinfo->in_color_space == JCS_CMYK)
        source->pub.get_pixel_rows = get_gray_cmyk_row;
      else
        ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    }
    break;

  case '6':                     /* it's a raw-format PPM file */
    if (cinfo->in_color_space == JCS_UNKNOWN)
      cinfo->in_color_space = JCS_EXT_RGB;
    TRACEMS3(cinfo, 1, JTRC_PPM, w, h, maxval);
    if (maxval > 255) {
      if (IsExtRGB(cinfo->in_color_space))
        source->pub.get_pixel_rows = get_word_rgb_row;
      else if (cinfo->in_color_space == JCS_CMYK)
        source->pub.get_pixel_rows = get_word_rgb_cmyk_row;
      else
        ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    } else if (maxval <= _MAXJSAMPLE && sizeof(_JSAMPLE) == sizeof(U_CHAR) &&
               maxval == ((1U << cinfo->data_precision) - 1U) &&
#if RGB_RED == 0 && RGB_GREEN == 1 && RGB_BLUE == 2 && RGB_PIXELSIZE == 3
               (cinfo->in_color_space == JCS_EXT_RGB ||
                cinfo->in_color_space == JCS_RGB)) {
#else
               cinfo->in_color_space == JCS_EXT_RGB) {
#endif
      source->pub.get_pixel_rows = get_raw_row;
      use_raw_buffer = TRUE;
      need_rescale = FALSE;
    } else {
      if (IsExtRGB(cinfo->in_color_space))
        source->pub.get_pixel_rows = get_rgb_row;
      else if (cinfo->in_color_space == JCS_CMYK)
        source->pub.get_pixel_rows = get_rgb_cmyk_row;
      else
        ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    }
    break;
  }

  if (IsExtRGB(cinfo->in_color_space))
    cinfo->input_components = rgb_pixelsize[cinfo->in_color_space];
  else if (cinfo->in_color_space == JCS_GRAYSCALE)
    cinfo->input_components = 1;
  else if (cinfo->in_color_space == JCS_CMYK)
    cinfo->input_components = 4;

  /* Allocate space for I/O buffer: 1 or 3 bytes or words/pixel. */
  if (need_iobuffer) {
    if (c == '6')
      source->buffer_width = (size_t)w * 3 *
        ((maxval <= 255) ? sizeof(U_CHAR) : (2 * sizeof(U_CHAR)));
    else
      source->buffer_width = (size_t)w *
        ((maxval <= 255) ? sizeof(U_CHAR) : (2 * sizeof(U_CHAR)));
    source->iobuffer = (U_CHAR *)
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                  source->buffer_width);
  }

  /* Create compressor input buffer. */
  if (use_raw_buffer) {
    /* For unscaled raw-input case, we can just map it onto the I/O buffer. */
    /* Synthesize a _JSAMPARRAY pointer structure */
    source->pixrow = (_JSAMPROW)source->iobuffer;
    source->pub._buffer = &source->pixrow;
    source->pub.buffer_height = 1;
  } else {
    /* Need to translate anyway, so make a separate sample buffer. */
    source->pub._buffer = (_JSAMPARRAY)(*cinfo->mem->alloc_sarray)
      ((j_common_ptr)cinfo, JPOOL_IMAGE,
       (JDIMENSION)w * cinfo->input_components, (JDIMENSION)1);
    source->pub.buffer_height = 1;
  }

  /* Compute the rescaling array if required. */
  if (need_rescale) {
    long val, half_maxval;

    /* On 16-bit-int machines we have to be careful of maxval = 65535 */
    source->rescale = (_JSAMPLE *)
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                  (size_t)(((long)MAX(maxval, 255) + 1L) *
                                           sizeof(_JSAMPLE)));
    memset(source->rescale, 0, (size_t)(((long)MAX(maxval, 255) + 1L) *
                                        sizeof(_JSAMPLE)));
    half_maxval = maxval / 2;
    for (val = 0; val <= (long)maxval; val++) {
      /* The multiplication here must be done in 32 bits to avoid overflow */
      source->rescale[val] =
        (_JSAMPLE)((val * ((1 << cinfo->data_precision) - 1) + half_maxval) /
                   maxval);
    }
  }
}


/*
 * Finish up at the end of the file.
 */

METHODDEF(void)
finish_input_ppm(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  /* no work */
}


/*
 * The module selection routine for PPM format input.
 */

GLOBAL(cjpeg_source_ptr)
_jinit_read_ppm(j_compress_ptr cinfo)
{
  ppm_source_ptr source;

#if BITS_IN_JSAMPLE == 8
  if (cinfo->data_precision > BITS_IN_JSAMPLE || cinfo->data_precision < 2)
#else
  if (cinfo->data_precision > BITS_IN_JSAMPLE ||
      cinfo->data_precision < BITS_IN_JSAMPLE - 3)
#endif
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  /* Create module interface object */
  source = (ppm_source_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(ppm_source_struct));
  /* Fill in method ptrs, except get_pixel_rows which start_input sets */
  source->pub.start_input = start_input_ppm;
  source->pub.finish_input = finish_input_ppm;
  source->pub.max_pixels = 0;

  return (cjpeg_source_ptr)source;
}

#endif /* defined(PPM_SUPPORTED) &&
          (BITS_IN_JSAMPLE != 16 || defined(C_LOSSLESS_SUPPORTED)) */
