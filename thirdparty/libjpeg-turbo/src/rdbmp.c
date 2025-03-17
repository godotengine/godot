/*
 * rdbmp.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * Modified 2009-2017 by Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Modified 2011 by Siarhei Siamashka.
 * Copyright (C) 2015, 2017-2018, 2021-2024, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains routines to read input images in Microsoft "BMP"
 * format (MS Windows 3.x, OS/2 1.x, and OS/2 2.x flavors).
 * Currently, only 8-, 24-, and 32-bit images are supported, not 1-bit or
 * 4-bit (feeding such low-depth images into JPEG would be silly anyway).
 * Also, we don't support RLE-compressed files.
 *
 * These routines may need modification for non-Unix environments or
 * specialized applications.  As they stand, they assume input from
 * an ordinary stdio stream.  They further assume that reading begins
 * at the start of the file; start_input may need work if the
 * user interface has already read some data (e.g., to determine that
 * the file is indeed BMP format).
 *
 * This code contributed by James Arthur Boucher.
 */

#include "cmyk.h"
#include "cdjpeg.h"             /* Common decls for cjpeg/djpeg applications */

#ifdef BMP_SUPPORTED


/* Macros to deal with unsigned chars as efficiently as compiler allows */

typedef unsigned char U_CHAR;
#define UCH(x)  ((int)(x))


#define ReadOK(file, buffer, len) \
  (fread(buffer, 1, len, file) == ((size_t)(len)))

static int alpha_index[JPEG_NUMCS] = {
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 3, 0, 0, -1
};


/* Private version of data source object */

typedef struct _bmp_source_struct *bmp_source_ptr;

typedef struct _bmp_source_struct {
  struct cjpeg_source_struct pub; /* public fields */

  j_compress_ptr cinfo;         /* back link saves passing separate parm */

  JSAMPARRAY colormap;          /* BMP colormap (converted to my format) */

  jvirt_sarray_ptr whole_image; /* Needed to reverse row order */
  JDIMENSION source_row;        /* Current source row number */
  JDIMENSION row_width;         /* Physical width of scanlines in file */

  int bits_per_pixel;           /* remembers 8-, 24-, or 32-bit format */
  int cmap_length;              /* colormap length */

  boolean use_inversion_array;  /* TRUE = preload the whole image, which is
                                   stored in bottom-up order, and feed it to
                                   the calling program in top-down order

                                   FALSE = the calling program will maintain
                                   its own image buffer and read the rows in
                                   bottom-up order */

  U_CHAR *iobuffer;             /* I/O buffer (used to buffer a single row from
                                   disk if use_inversion_array == FALSE) */
} bmp_source_struct;


LOCAL(int)
read_byte(bmp_source_ptr sinfo)
/* Read next byte from BMP file */
{
  register FILE *infile = sinfo->pub.input_file;
  register int c;

  if ((c = getc(infile)) == EOF)
    ERREXIT(sinfo->cinfo, JERR_INPUT_EOF);
  return c;
}


LOCAL(void)
read_colormap(bmp_source_ptr sinfo, int cmaplen, int mapentrysize)
/* Read the colormap from a BMP file */
{
  int i, gray = 1;

  switch (mapentrysize) {
  case 3:
    /* BGR format (occurs in OS/2 files) */
    for (i = 0; i < cmaplen; i++) {
      sinfo->colormap[2][i] = (JSAMPLE)read_byte(sinfo);
      sinfo->colormap[1][i] = (JSAMPLE)read_byte(sinfo);
      sinfo->colormap[0][i] = (JSAMPLE)read_byte(sinfo);
      if (sinfo->colormap[2][i] != sinfo->colormap[1][i] ||
          sinfo->colormap[1][i] != sinfo->colormap[0][i])
        gray = 0;
    }
    break;
  case 4:
    /* BGR0 format (occurs in MS Windows files) */
    for (i = 0; i < cmaplen; i++) {
      sinfo->colormap[2][i] = (JSAMPLE)read_byte(sinfo);
      sinfo->colormap[1][i] = (JSAMPLE)read_byte(sinfo);
      sinfo->colormap[0][i] = (JSAMPLE)read_byte(sinfo);
      (void)read_byte(sinfo);
      if (sinfo->colormap[2][i] != sinfo->colormap[1][i] ||
          sinfo->colormap[1][i] != sinfo->colormap[0][i])
        gray = 0;
    }
    break;
  default:
    ERREXIT(sinfo->cinfo, JERR_BMP_BADCMAP);
    break;
  }

  if ((sinfo->cinfo->in_color_space == JCS_UNKNOWN ||
       sinfo->cinfo->in_color_space == JCS_RGB) && gray)
    sinfo->cinfo->in_color_space = JCS_GRAYSCALE;

  if (sinfo->cinfo->in_color_space == JCS_GRAYSCALE && !gray)
    ERREXIT(sinfo->cinfo, JERR_BAD_IN_COLORSPACE);
}


/*
 * Read one row of pixels.
 * The image has been read into the whole_image array, but is otherwise
 * unprocessed.  We must read it out in top-to-bottom row order, and if
 * it is an 8-bit image, we must expand colormapped pixels to 24bit format.
 */

METHODDEF(JDIMENSION)
get_8bit_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading 8-bit colormap indexes */
{
  bmp_source_ptr source = (bmp_source_ptr)sinfo;
  register JSAMPARRAY colormap = source->colormap;
  int cmaplen = source->cmap_length;
  JSAMPARRAY image_ptr;
  register int t;
  register JSAMPROW inptr, outptr;
  register JDIMENSION col;

  if (source->use_inversion_array) {
    /* Fetch next row from virtual array */
    source->source_row--;
    image_ptr = (*cinfo->mem->access_virt_sarray)
      ((j_common_ptr)cinfo, source->whole_image,
       source->source_row, (JDIMENSION)1, FALSE);
    inptr = image_ptr[0];
  } else {
    if (!ReadOK(source->pub.input_file, source->iobuffer, source->row_width))
      ERREXIT(cinfo, JERR_INPUT_EOF);
    inptr = source->iobuffer;
  }

  /* Expand the colormap indexes to real data */
  outptr = source->pub.buffer[0];
  if (cinfo->in_color_space == JCS_GRAYSCALE) {
    for (col = cinfo->image_width; col > 0; col--) {
      t = *inptr++;
      if (t >= cmaplen)
        ERREXIT(cinfo, JERR_BMP_OUTOFRANGE);
      *outptr++ = colormap[0][t];
    }
  } else if (cinfo->in_color_space == JCS_CMYK) {
    for (col = cinfo->image_width; col > 0; col--) {
      t = *inptr++;
      if (t >= cmaplen)
        ERREXIT(cinfo, JERR_BMP_OUTOFRANGE);
      rgb_to_cmyk(255, colormap[0][t], colormap[1][t], colormap[2][t], outptr,
                  outptr + 1, outptr + 2, outptr + 3);
      outptr += 4;
    }
  } else {
    register int rindex = rgb_red[cinfo->in_color_space];
    register int gindex = rgb_green[cinfo->in_color_space];
    register int bindex = rgb_blue[cinfo->in_color_space];
    register int aindex = alpha_index[cinfo->in_color_space];
    register int ps = rgb_pixelsize[cinfo->in_color_space];

    if (aindex >= 0) {
      for (col = cinfo->image_width; col > 0; col--) {
        t = *inptr++;
        if (t >= cmaplen)
          ERREXIT(cinfo, JERR_BMP_OUTOFRANGE);
        outptr[rindex] = colormap[0][t];
        outptr[gindex] = colormap[1][t];
        outptr[bindex] = colormap[2][t];
        outptr[aindex] = 0xFF;
        outptr += ps;
      }
    } else {
      for (col = cinfo->image_width; col > 0; col--) {
        t = *inptr++;
        if (t >= cmaplen)
          ERREXIT(cinfo, JERR_BMP_OUTOFRANGE);
        outptr[rindex] = colormap[0][t];
        outptr[gindex] = colormap[1][t];
        outptr[bindex] = colormap[2][t];
        outptr += ps;
      }
    }
  }

  return 1;
}


METHODDEF(JDIMENSION)
get_24bit_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading 24-bit pixels */
{
  bmp_source_ptr source = (bmp_source_ptr)sinfo;
  JSAMPARRAY image_ptr;
  register JSAMPROW inptr, outptr;
  register JDIMENSION col;

  if (source->use_inversion_array) {
    /* Fetch next row from virtual array */
    source->source_row--;
    image_ptr = (*cinfo->mem->access_virt_sarray)
      ((j_common_ptr)cinfo, source->whole_image,
       source->source_row, (JDIMENSION)1, FALSE);
    inptr = image_ptr[0];
  } else {
    if (!ReadOK(source->pub.input_file, source->iobuffer, source->row_width))
      ERREXIT(cinfo, JERR_INPUT_EOF);
    inptr = source->iobuffer;
  }

  /* Transfer data.  Note source values are in BGR order
   * (even though Microsoft's own documents say the opposite).
   */
  outptr = source->pub.buffer[0];
  if (cinfo->in_color_space == JCS_EXT_BGR) {
    memcpy(outptr, inptr, source->row_width);
  } else if (cinfo->in_color_space == JCS_CMYK) {
    for (col = cinfo->image_width; col > 0; col--) {
      JSAMPLE b = *inptr++, g = *inptr++, r = *inptr++;
      rgb_to_cmyk(255, r, g, b, outptr, outptr + 1, outptr + 2, outptr + 3);
      outptr += 4;
    }
  } else {
    register int rindex = rgb_red[cinfo->in_color_space];
    register int gindex = rgb_green[cinfo->in_color_space];
    register int bindex = rgb_blue[cinfo->in_color_space];
    register int aindex = alpha_index[cinfo->in_color_space];
    register int ps = rgb_pixelsize[cinfo->in_color_space];

    if (aindex >= 0) {
      for (col = cinfo->image_width; col > 0; col--) {
        outptr[bindex] = *inptr++;
        outptr[gindex] = *inptr++;
        outptr[rindex] = *inptr++;
        outptr[aindex] = 0xFF;
        outptr += ps;
      }
    } else {
      for (col = cinfo->image_width; col > 0; col--) {
        outptr[bindex] = *inptr++;
        outptr[gindex] = *inptr++;
        outptr[rindex] = *inptr++;
        outptr += ps;
      }
    }
  }

  return 1;
}


METHODDEF(JDIMENSION)
get_32bit_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading 32-bit pixels */
{
  bmp_source_ptr source = (bmp_source_ptr)sinfo;
  JSAMPARRAY image_ptr;
  register JSAMPROW inptr, outptr;
  register JDIMENSION col;

  if (source->use_inversion_array) {
    /* Fetch next row from virtual array */
    source->source_row--;
    image_ptr = (*cinfo->mem->access_virt_sarray)
      ((j_common_ptr)cinfo, source->whole_image,
       source->source_row, (JDIMENSION)1, FALSE);
    inptr = image_ptr[0];
  } else {
    if (!ReadOK(source->pub.input_file, source->iobuffer, source->row_width))
      ERREXIT(cinfo, JERR_INPUT_EOF);
    inptr = source->iobuffer;
  }

  /* Transfer data.  Note source values are in BGR order
   * (even though Microsoft's own documents say the opposite).
   */
  outptr = source->pub.buffer[0];
  if (cinfo->in_color_space == JCS_EXT_BGRX ||
      cinfo->in_color_space == JCS_EXT_BGRA) {
    memcpy(outptr, inptr, source->row_width);
  } else if (cinfo->in_color_space == JCS_CMYK) {
    for (col = cinfo->image_width; col > 0; col--) {
      JSAMPLE b = *inptr++, g = *inptr++, r = *inptr++;
      rgb_to_cmyk(255, r, g, b, outptr, outptr + 1, outptr + 2, outptr + 3);
      inptr++;                          /* skip the 4th byte (Alpha channel) */
      outptr += 4;
    }
  } else {
    register int rindex = rgb_red[cinfo->in_color_space];
    register int gindex = rgb_green[cinfo->in_color_space];
    register int bindex = rgb_blue[cinfo->in_color_space];
    register int aindex = alpha_index[cinfo->in_color_space];
    register int ps = rgb_pixelsize[cinfo->in_color_space];

    if (aindex >= 0) {
      for (col = cinfo->image_width; col > 0; col--) {
        outptr[bindex] = *inptr++;
        outptr[gindex] = *inptr++;
        outptr[rindex] = *inptr++;
        outptr[aindex] = *inptr++;
        outptr += ps;
      }
    } else {
      for (col = cinfo->image_width; col > 0; col--) {
        outptr[bindex] = *inptr++;
        outptr[gindex] = *inptr++;
        outptr[rindex] = *inptr++;
        inptr++;                        /* skip the 4th byte (Alpha channel) */
        outptr += ps;
      }
    }
  }

  return 1;
}


/*
 * This method loads the image into whole_image during the first call on
 * get_pixel_rows.  The get_pixel_rows pointer is then adjusted to call
 * get_8bit_row, get_24bit_row, or get_32bit_row on subsequent calls.
 */

METHODDEF(JDIMENSION)
preload_image(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  bmp_source_ptr source = (bmp_source_ptr)sinfo;
  register FILE *infile = source->pub.input_file;
  register JSAMPROW out_ptr;
  JSAMPARRAY image_ptr;
  JDIMENSION row;
  cd_progress_ptr progress = (cd_progress_ptr)cinfo->progress;

  /* Read the data into a virtual array in input-file row order. */
  for (row = 0; row < cinfo->image_height; row++) {
    if (progress != NULL) {
      progress->pub.pass_counter = (long)row;
      progress->pub.pass_limit = (long)cinfo->image_height;
      (*progress->pub.progress_monitor) ((j_common_ptr)cinfo);
    }
    image_ptr = (*cinfo->mem->access_virt_sarray)
      ((j_common_ptr)cinfo, source->whole_image, row, (JDIMENSION)1, TRUE);
    out_ptr = image_ptr[0];
    if (fread(out_ptr, 1, source->row_width, infile) != source->row_width) {
      if (feof(infile))
        ERREXIT(cinfo, JERR_INPUT_EOF);
      else
        ERREXIT(cinfo, JERR_FILE_READ);
    }
  }
  if (progress != NULL)
    progress->completed_extra_passes++;

  /* Set up to read from the virtual array in top-to-bottom order */
  switch (source->bits_per_pixel) {
  case 8:
    source->pub.get_pixel_rows = get_8bit_row;
    break;
  case 24:
    source->pub.get_pixel_rows = get_24bit_row;
    break;
  case 32:
    source->pub.get_pixel_rows = get_32bit_row;
    break;
  default:
    ERREXIT(cinfo, JERR_BMP_BADDEPTH);
  }
  source->source_row = cinfo->image_height;

  /* And read the first row */
  return (*source->pub.get_pixel_rows) (cinfo, sinfo);
}


/*
 * Read the file header; return image size and component count.
 */

METHODDEF(void)
start_input_bmp(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  bmp_source_ptr source = (bmp_source_ptr)sinfo;
  U_CHAR bmpfileheader[14];
  U_CHAR bmpinfoheader[64];

#define GET_2B(array, offset) \
  ((unsigned short)UCH(array[offset]) + \
   (((unsigned short)UCH(array[offset + 1])) << 8))
#define GET_4B(array, offset) \
  ((unsigned int)UCH(array[offset]) + \
   (((unsigned int)UCH(array[offset + 1])) << 8) + \
   (((unsigned int)UCH(array[offset + 2])) << 16) + \
   (((unsigned int)UCH(array[offset + 3])) << 24))

  int bfOffBits;
  int headerSize;
  int biWidth;
  int biHeight;
  unsigned short biPlanes;
  unsigned int biCompression;
  int biXPelsPerMeter, biYPelsPerMeter;
  int biClrUsed = 0;
  int mapentrysize = 0;         /* 0 indicates no colormap */
  int bPad;
  JDIMENSION row_width = 0;

  /* Read and verify the bitmap file header */
  if (!ReadOK(source->pub.input_file, bmpfileheader, 14))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  if (GET_2B(bmpfileheader, 0) != 0x4D42) /* 'BM' */
    ERREXIT(cinfo, JERR_BMP_NOT);
  bfOffBits = GET_4B(bmpfileheader, 10);
  /* We ignore the remaining fileheader fields */

  /* The infoheader might be 12 bytes (OS/2 1.x), 40 bytes (Windows),
   * or 64 bytes (OS/2 2.x).  Check the first 4 bytes to find out which.
   */
  if (!ReadOK(source->pub.input_file, bmpinfoheader, 4))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  headerSize = GET_4B(bmpinfoheader, 0);
  if (headerSize < 12 || headerSize > 64 || (headerSize + 14) > bfOffBits)
    ERREXIT(cinfo, JERR_BMP_BADHEADER);
  if (!ReadOK(source->pub.input_file, bmpinfoheader + 4, headerSize - 4))
    ERREXIT(cinfo, JERR_INPUT_EOF);

  switch (headerSize) {
  case 12:
    /* Decode OS/2 1.x header (Microsoft calls this a BITMAPCOREHEADER) */
    biWidth = (int)GET_2B(bmpinfoheader, 4);
    biHeight = (int)GET_2B(bmpinfoheader, 6);
    biPlanes = GET_2B(bmpinfoheader, 8);
    source->bits_per_pixel = (int)GET_2B(bmpinfoheader, 10);

    switch (source->bits_per_pixel) {
    case 8:                     /* colormapped image */
      mapentrysize = 3;         /* OS/2 uses RGBTRIPLE colormap */
      TRACEMS2(cinfo, 1, JTRC_BMP_OS2_MAPPED, biWidth, biHeight);
      break;
    case 24:                    /* RGB image */
    case 32:                    /* RGB image + Alpha channel */
      TRACEMS3(cinfo, 1, JTRC_BMP_OS2, biWidth, biHeight,
               source->bits_per_pixel);
      break;
    default:
      ERREXIT(cinfo, JERR_BMP_BADDEPTH);
      break;
    }
    break;
  case 40:
  case 64:
    /* Decode Windows 3.x header (Microsoft calls this a BITMAPINFOHEADER) */
    /* or OS/2 2.x header, which has additional fields that we ignore */
    biWidth = (int)GET_4B(bmpinfoheader, 4);
    biHeight = (int)GET_4B(bmpinfoheader, 8);
    biPlanes = GET_2B(bmpinfoheader, 12);
    source->bits_per_pixel = (int)GET_2B(bmpinfoheader, 14);
    biCompression = GET_4B(bmpinfoheader, 16);
    biXPelsPerMeter = (int)GET_4B(bmpinfoheader, 24);
    biYPelsPerMeter = (int)GET_4B(bmpinfoheader, 28);
    biClrUsed = GET_4B(bmpinfoheader, 32);
    /* biSizeImage, biClrImportant fields are ignored */

    switch (source->bits_per_pixel) {
    case 8:                     /* colormapped image */
      mapentrysize = 4;         /* Windows uses RGBQUAD colormap */
      TRACEMS2(cinfo, 1, JTRC_BMP_MAPPED, biWidth, biHeight);
      break;
    case 24:                    /* RGB image */
    case 32:                    /* RGB image + Alpha channel */
      TRACEMS3(cinfo, 1, JTRC_BMP, biWidth, biHeight, source->bits_per_pixel);
      break;
    default:
      ERREXIT(cinfo, JERR_BMP_BADDEPTH);
      break;
    }
    if (biCompression != 0)
      ERREXIT(cinfo, JERR_BMP_COMPRESSED);

    if (biXPelsPerMeter > 0 && biYPelsPerMeter > 0) {
      /* Set JFIF density parameters from the BMP data */
      cinfo->X_density = (UINT16)(biXPelsPerMeter / 100); /* 100 cm per meter */
      cinfo->Y_density = (UINT16)(biYPelsPerMeter / 100);
      cinfo->density_unit = 2;  /* dots/cm */
    }
    break;
  default:
    ERREXIT(cinfo, JERR_BMP_BADHEADER);
    return;
  }

  if (biWidth <= 0 || biHeight <= 0)
    ERREXIT(cinfo, JERR_BMP_EMPTY);
  if (sinfo->max_pixels &&
      (unsigned long long)biWidth * biHeight > sinfo->max_pixels)
    ERREXIT1(cinfo, JERR_IMAGE_TOO_BIG, sinfo->max_pixels);
  if (biPlanes != 1)
    ERREXIT(cinfo, JERR_BMP_BADPLANES);

  /* Compute distance to bitmap data --- will adjust for colormap below */
  bPad = bfOffBits - (headerSize + 14);

  /* Read the colormap, if any */
  if (mapentrysize > 0) {
    if (biClrUsed <= 0)
      biClrUsed = 256;          /* assume it's 256 */
    else if (biClrUsed > 256)
      ERREXIT(cinfo, JERR_BMP_BADCMAP);
    /* Allocate space to store the colormap */
    source->colormap = (*cinfo->mem->alloc_sarray)
      ((j_common_ptr)cinfo, JPOOL_IMAGE, (JDIMENSION)biClrUsed, (JDIMENSION)3);
    source->cmap_length = (int)biClrUsed;
    /* and read it from the file */
    read_colormap(source, (int)biClrUsed, mapentrysize);
    /* account for size of colormap */
    bPad -= biClrUsed * mapentrysize;
  }

  /* Skip any remaining pad bytes */
  if (bPad < 0)                 /* incorrect bfOffBits value? */
    ERREXIT(cinfo, JERR_BMP_BADHEADER);
  while (--bPad >= 0) {
    (void)read_byte(source);
  }

  /* Compute row width in file, including padding to 4-byte boundary */
  switch (source->bits_per_pixel) {
  case 8:
    if (cinfo->in_color_space == JCS_UNKNOWN)
      cinfo->in_color_space = JCS_EXT_RGB;
    if (IsExtRGB(cinfo->in_color_space))
      cinfo->input_components = rgb_pixelsize[cinfo->in_color_space];
    else if (cinfo->in_color_space == JCS_GRAYSCALE)
      cinfo->input_components = 1;
    else if (cinfo->in_color_space == JCS_CMYK)
      cinfo->input_components = 4;
    else
      ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    row_width = (JDIMENSION)biWidth;
    break;
  case 24:
    if (cinfo->in_color_space == JCS_UNKNOWN)
      cinfo->in_color_space = JCS_EXT_BGR;
    if (IsExtRGB(cinfo->in_color_space))
      cinfo->input_components = rgb_pixelsize[cinfo->in_color_space];
    else if (cinfo->in_color_space == JCS_CMYK)
      cinfo->input_components = 4;
    else
      ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    if ((unsigned long long)biWidth * 3ULL > 0xFFFFFFFFULL)
      ERREXIT(cinfo, JERR_WIDTH_OVERFLOW);
    row_width = (JDIMENSION)biWidth * 3;
    break;
  case 32:
    if (cinfo->in_color_space == JCS_UNKNOWN)
      cinfo->in_color_space = JCS_EXT_BGRA;
    if (IsExtRGB(cinfo->in_color_space))
      cinfo->input_components = rgb_pixelsize[cinfo->in_color_space];
    else if (cinfo->in_color_space == JCS_CMYK)
      cinfo->input_components = 4;
    else
      ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    if ((unsigned long long)biWidth * 4ULL > 0xFFFFFFFFULL)
      ERREXIT(cinfo, JERR_WIDTH_OVERFLOW);
    row_width = (JDIMENSION)biWidth * 4;
    break;
  default:
    ERREXIT(cinfo, JERR_BMP_BADDEPTH);
  }
  while ((row_width & 3) != 0) row_width++;
  source->row_width = row_width;

  if (source->use_inversion_array) {
    /* Allocate space for inversion array, prepare for preload pass */
    source->whole_image = (*cinfo->mem->request_virt_sarray)
      ((j_common_ptr)cinfo, JPOOL_IMAGE, FALSE,
       row_width, (JDIMENSION)biHeight, (JDIMENSION)1);
    source->pub.get_pixel_rows = preload_image;
    if (cinfo->progress != NULL) {
      cd_progress_ptr progress = (cd_progress_ptr)cinfo->progress;
      progress->total_extra_passes++; /* count file input as separate pass */
    }
  } else {
    source->iobuffer = (U_CHAR *)
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE, row_width);
    switch (source->bits_per_pixel) {
    case 8:
      source->pub.get_pixel_rows = get_8bit_row;
      break;
    case 24:
      source->pub.get_pixel_rows = get_24bit_row;
      break;
    case 32:
      source->pub.get_pixel_rows = get_32bit_row;
      break;
    default:
      ERREXIT(cinfo, JERR_BMP_BADDEPTH);
    }
  }

  /* Ensure that biWidth * cinfo->input_components doesn't exceed the maximum
     value of the JDIMENSION type.  This is only a danger with BMP files, since
     their width and height fields are 32-bit integers. */
  if ((unsigned long long)biWidth *
      (unsigned long long)cinfo->input_components > 0xFFFFFFFFULL)
    ERREXIT(cinfo, JERR_WIDTH_OVERFLOW);
  /* Allocate one-row buffer for returned data */
  source->pub.buffer = (*cinfo->mem->alloc_sarray)
    ((j_common_ptr)cinfo, JPOOL_IMAGE,
     (JDIMENSION)biWidth * (JDIMENSION)cinfo->input_components, (JDIMENSION)1);
  source->pub.buffer_height = 1;

  cinfo->data_precision = 8;
  cinfo->image_width = (JDIMENSION)biWidth;
  cinfo->image_height = (JDIMENSION)biHeight;
}


/*
 * Finish up at the end of the file.
 */

METHODDEF(void)
finish_input_bmp(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  /* no work */
}


/*
 * The module selection routine for BMP format input.
 */

GLOBAL(cjpeg_source_ptr)
jinit_read_bmp(j_compress_ptr cinfo, boolean use_inversion_array)
{
  bmp_source_ptr source;

  if (cinfo->data_precision != 8)
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  /* Create module interface object */
  source = (bmp_source_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(bmp_source_struct));
  source->cinfo = cinfo;        /* make back link for subroutines */
  /* Fill in method ptrs, except get_pixel_rows which start_input sets */
  source->pub.start_input = start_input_bmp;
  source->pub.finish_input = finish_input_bmp;
  source->pub.max_pixels = 0;

  source->use_inversion_array = use_inversion_array;

  return (cjpeg_source_ptr)source;
}

#endif /* BMP_SUPPORTED */
