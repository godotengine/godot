/*
 * rdcolmap.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file implements djpeg's "-map file" switch.  It reads a source image
 * and constructs a colormap to be supplied to the JPEG decompressor.
 *
 * Currently, these file formats are supported for the map file:
 *   GIF: the contents of the GIF's global colormap are used.
 *   PPM (either text or raw flavor): the entire file is read and
 *      each unique pixel value is entered in the map.
 * Note that reading a large PPM file will be horrendously slow.
 * Typically, a PPM-format map file should contain just one pixel
 * of each desired color.  Such a file can be extracted from an
 * ordinary image PPM file with ppmtomap(1).
 *
 * Rescaling a PPM that has a maxval unequal to _MAXJSAMPLE is not
 * currently implemented.
 */

#include "cdjpeg.h"             /* Common decls for cjpeg/djpeg applications */
#include "jsamplecomp.h"

#ifdef QUANT_2PASS_SUPPORTED    /* otherwise can't quantize to supplied map */
#if BITS_IN_JSAMPLE != 16 || defined(D_LOSSLESS_SUPPORTED)

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


/*
 * Add a (potentially) new color to the color map.
 */

LOCAL(void)
add_map_entry(j_decompress_ptr cinfo, int R, int G, int B)
{
  _JSAMPROW colormap0 = ((_JSAMPARRAY)cinfo->colormap)[0];
  _JSAMPROW colormap1 = ((_JSAMPARRAY)cinfo->colormap)[1];
  _JSAMPROW colormap2 = ((_JSAMPARRAY)cinfo->colormap)[2];
  int ncolors = cinfo->actual_number_of_colors;
  int index;

  /* Check for duplicate color. */
  for (index = 0; index < ncolors; index++) {
    if (colormap0[index] == R && colormap1[index] == G &&
        colormap2[index] == B)
      return;                   /* color is already in map */
  }

  /* Check for map overflow. */
  if (ncolors >= (_MAXJSAMPLE + 1))
    ERREXIT1(cinfo, JERR_QUANT_MANY_COLORS, (_MAXJSAMPLE + 1));

  /* OK, add color to map. */
  colormap0[ncolors] = (_JSAMPLE)R;
  colormap1[ncolors] = (_JSAMPLE)G;
  colormap2[ncolors] = (_JSAMPLE)B;
  cinfo->actual_number_of_colors++;
}


/*
 * Extract color map from a GIF file.
 */

LOCAL(void)
read_gif_map(j_decompress_ptr cinfo, FILE *infile)
{
  int header[13];
  int i, colormaplen;
  int R, G, B;

  /* Initial 'G' has already been read by read_color_map */
  /* Read the rest of the GIF header and logical screen descriptor */
  for (i = 1; i < 13; i++) {
    if ((header[i] = getc(infile)) == EOF)
      ERREXIT(cinfo, JERR_BAD_CMAP_FILE);
  }

  /* Verify GIF Header */
  if (header[1] != 'I' || header[2] != 'F')
    ERREXIT(cinfo, JERR_BAD_CMAP_FILE);

  /* There must be a global color map. */
  if ((header[10] & 0x80) == 0)
    ERREXIT(cinfo, JERR_BAD_CMAP_FILE);

  /* OK, fetch it. */
  colormaplen = 2 << (header[10] & 0x07);

  for (i = 0; i < colormaplen; i++) {
    R = getc(infile);
    G = getc(infile);
    B = getc(infile);
    if (R == EOF || G == EOF || B == EOF)
      ERREXIT(cinfo, JERR_BAD_CMAP_FILE);
    add_map_entry(cinfo,
                  R << (BITS_IN_JSAMPLE - 8),
                  G << (BITS_IN_JSAMPLE - 8),
                  B << (BITS_IN_JSAMPLE - 8));
  }
}


/* Support routines for reading PPM */


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
read_pbm_integer(j_decompress_ptr cinfo, FILE *infile)
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
      ERREXIT(cinfo, JERR_BAD_CMAP_FILE);
  } while (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r');

  if (ch < '0' || ch > '9')
    ERREXIT(cinfo, JERR_BAD_CMAP_FILE);

  val = ch - '0';
  while ((ch = pbm_getc(infile)) >= '0' && ch <= '9') {
    val *= 10;
    val += ch - '0';
  }
  return val;
}


/*
 * Extract color map from a PPM file.
 */

LOCAL(void)
read_ppm_map(j_decompress_ptr cinfo, FILE *infile)
{
  int c;
  unsigned int w, h, maxval, row, col;
  int R, G, B;

  /* Initial 'P' has already been read by read_color_map */
  c = getc(infile);             /* save format discriminator for a sec */

  /* while we fetch the remaining header info */
  w = read_pbm_integer(cinfo, infile);
  h = read_pbm_integer(cinfo, infile);
  maxval = read_pbm_integer(cinfo, infile);

  if (w <= 0 || h <= 0 || maxval <= 0) /* error check */
    ERREXIT(cinfo, JERR_BAD_CMAP_FILE);

  /* For now, we don't support rescaling from an unusual maxval. */
  if (maxval != (unsigned int)_MAXJSAMPLE)
    ERREXIT(cinfo, JERR_BAD_CMAP_FILE);

  switch (c) {
  case '3':                     /* it's a text-format PPM file */
    for (row = 0; row < h; row++) {
      for (col = 0; col < w; col++) {
        R = read_pbm_integer(cinfo, infile);
        G = read_pbm_integer(cinfo, infile);
        B = read_pbm_integer(cinfo, infile);
        add_map_entry(cinfo, R, G, B);
      }
    }
    break;

  case '6':                     /* it's a raw-format PPM file */
    for (row = 0; row < h; row++) {
      for (col = 0; col < w; col++) {
        R = getc(infile);
        G = getc(infile);
        B = getc(infile);
        if (R == EOF || G == EOF || B == EOF)
          ERREXIT(cinfo, JERR_BAD_CMAP_FILE);
        add_map_entry(cinfo, R, G, B);
      }
    }
    break;

  default:
    ERREXIT(cinfo, JERR_BAD_CMAP_FILE);
    break;
  }
}


/*
 * Main entry point from djpeg.c.
 *  Input: opened input file (from file name argument on command line).
 *  Output: colormap and actual_number_of_colors fields are set in cinfo.
 */

GLOBAL(void)
_read_color_map(j_decompress_ptr cinfo, FILE *infile)
{
  if (cinfo->data_precision != BITS_IN_JSAMPLE)
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  /* Allocate space for a color map of maximum supported size. */
  cinfo->colormap = (*cinfo->mem->alloc_sarray)
    ((j_common_ptr)cinfo, JPOOL_IMAGE,
     (JDIMENSION)(_MAXJSAMPLE + 1), (JDIMENSION)3);
  cinfo->actual_number_of_colors = 0; /* initialize map to empty */

  /* Read first byte to determine file format */
  switch (getc(infile)) {
  case 'G':
    read_gif_map(cinfo, infile);
    break;
  case 'P':
    read_ppm_map(cinfo, infile);
    break;
  default:
    ERREXIT(cinfo, JERR_BAD_CMAP_FILE);
    break;
  }
}

#endif /* BITS_IN_JSAMPLE != 16 || defined(D_LOSSLESS_SUPPORTED) */
#endif /* QUANT_2PASS_SUPPORTED */
