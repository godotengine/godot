/*
 * wrbmp.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2013, Linaro Limited.
 * Copyright (C) 2014-2015, 2017, 2019, 2022, 2024, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains routines to write output images in Microsoft "BMP"
 * format (MS Windows 3.x and OS/2 1.x flavors).
 * Either 8-bit colormapped or 24-bit full-color format can be written.
 * No compression is supported.
 *
 * These routines may need modification for non-Unix environments or
 * specialized applications.  As they stand, they assume output to
 * an ordinary stdio stream.
 *
 * This code contributed by James Arthur Boucher.
 */

#include "cmyk.h"
#include "cdjpeg.h"             /* Common decls for cjpeg/djpeg applications */
#include "jconfigint.h"

#ifdef BMP_SUPPORTED


/*
 * To support 12-bit JPEG data, we'd have to scale output down to 8 bits.
 * This is not yet implemented.
 */

#if BITS_IN_JSAMPLE != 8
  Sorry, this code only copes with 8-bit JSAMPLEs. /* deliberate syntax err */
#endif

/*
 * Since BMP stores scanlines bottom-to-top, we have to invert the image
 * from JPEG's top-to-bottom order.  To do this, we save the outgoing data
 * in a virtual array during put_pixel_row calls, then actually emit the
 * BMP file during finish_output.  The virtual array contains one JSAMPLE per
 * pixel if the output is grayscale or colormapped, three if it is full color.
 */

/* Private version of data destination object */

typedef struct {
  struct djpeg_dest_struct pub; /* public fields */

  boolean is_os2;               /* saves the OS2 format request flag */

  jvirt_sarray_ptr whole_image; /* needed to reverse row order */
  JDIMENSION data_width;        /* JSAMPLEs per row */
  JDIMENSION row_width;         /* physical width of one row in the BMP file */
  int pad_bytes;                /* number of padding bytes needed per row */
  JDIMENSION cur_output_row;    /* next row# to write to virtual array */

  boolean use_inversion_array;  /* TRUE = buffer the whole image, which is
                                   stored to disk in bottom-up order, and
                                   receive rows from the calling program in
                                   top-down order

                                   FALSE = the calling program will maintain
                                   its own image buffer and write the rows in
                                   bottom-up order */

  JSAMPLE *iobuffer;            /* I/O buffer (used to buffer a single row to
                                   disk if use_inversion_array == FALSE) */
} bmp_dest_struct;

typedef bmp_dest_struct *bmp_dest_ptr;


/* Forward declarations */
LOCAL(void) write_colormap(j_decompress_ptr cinfo, bmp_dest_ptr dest,
                           int map_colors, int map_entry_size);


static INLINE boolean is_big_endian(void)
{
  int test_value = 1;
  if (*(char *)&test_value != 1)
    return TRUE;
  return FALSE;
}


/*
 * Write some pixel data.
 * In this module rows_supplied will always be 1.
 */

METHODDEF(void)
put_pixel_rows(j_decompress_ptr cinfo, djpeg_dest_ptr dinfo,
               JDIMENSION rows_supplied)
/* This version is for writing 24-bit pixels */
{
  bmp_dest_ptr dest = (bmp_dest_ptr)dinfo;
  JSAMPARRAY image_ptr;
  register JSAMPROW inptr, outptr;
  register JDIMENSION col;
  int pad;

  if (dest->use_inversion_array) {
    /* Access next row in virtual array */
    image_ptr = (*cinfo->mem->access_virt_sarray)
      ((j_common_ptr)cinfo, dest->whole_image,
       dest->cur_output_row, (JDIMENSION)1, TRUE);
    dest->cur_output_row++;
    outptr = image_ptr[0];
  } else {
    outptr = dest->iobuffer;
  }

  /* Transfer data.  Note destination values must be in BGR order
   * (even though Microsoft's own documents say the opposite).
   */
  inptr = dest->pub.buffer[0];

  if (cinfo->out_color_space == JCS_EXT_BGR) {
    memcpy(outptr, inptr, dest->row_width);
    outptr += cinfo->output_width * 3;
  } else if (cinfo->out_color_space == JCS_RGB565) {
    boolean big_endian = is_big_endian();
    unsigned short *inptr2 = (unsigned short *)inptr;
    for (col = cinfo->output_width; col > 0; col--) {
      if (big_endian) {
        outptr[0] = (*inptr2 >> 5) & 0xF8;
        outptr[1] = ((*inptr2 << 5) & 0xE0) | ((*inptr2 >> 11) & 0x1C);
        outptr[2] = *inptr2 & 0xF8;
      } else {
        outptr[0] = (*inptr2 << 3) & 0xF8;
        outptr[1] = (*inptr2 >> 3) & 0xFC;
        outptr[2] = (*inptr2 >> 8) & 0xF8;
      }
      outptr += 3;
      inptr2++;
    }
  } else if (cinfo->out_color_space == JCS_CMYK) {
    for (col = cinfo->output_width; col > 0; col--) {
      JSAMPLE c = *inptr++, m = *inptr++, y = *inptr++, k = *inptr++;
      cmyk_to_rgb(255, c, m, y, k, outptr + 2, outptr + 1, outptr);
      outptr += 3;
    }
  } else {
    register int rindex = rgb_red[cinfo->out_color_space];
    register int gindex = rgb_green[cinfo->out_color_space];
    register int bindex = rgb_blue[cinfo->out_color_space];
    register int ps = rgb_pixelsize[cinfo->out_color_space];

    for (col = cinfo->output_width; col > 0; col--) {
      outptr[0] = inptr[bindex];
      outptr[1] = inptr[gindex];
      outptr[2] = inptr[rindex];
      outptr += 3;  inptr += ps;
    }
  }

  /* Zero out the pad bytes. */
  pad = dest->pad_bytes;
  while (--pad >= 0)
    *outptr++ = 0;

  if (!dest->use_inversion_array)
    fwrite(dest->iobuffer, 1, dest->row_width, dest->pub.output_file);
}

METHODDEF(void)
put_gray_rows(j_decompress_ptr cinfo, djpeg_dest_ptr dinfo,
              JDIMENSION rows_supplied)
/* This version is for grayscale OR quantized color output */
{
  bmp_dest_ptr dest = (bmp_dest_ptr)dinfo;
  JSAMPARRAY image_ptr;
  register JSAMPROW inptr, outptr;
  int pad;

  if (dest->use_inversion_array) {
    /* Access next row in virtual array */
    image_ptr = (*cinfo->mem->access_virt_sarray)
      ((j_common_ptr)cinfo, dest->whole_image,
       dest->cur_output_row, (JDIMENSION)1, TRUE);
    dest->cur_output_row++;
    outptr = image_ptr[0];
  } else {
    outptr = dest->iobuffer;
  }

  /* Transfer data. */
  inptr = dest->pub.buffer[0];
  memcpy(outptr, inptr, cinfo->output_width);
  outptr += cinfo->output_width;

  /* Zero out the pad bytes. */
  pad = dest->pad_bytes;
  while (--pad >= 0)
    *outptr++ = 0;

  if (!dest->use_inversion_array)
    fwrite(dest->iobuffer, 1, dest->row_width, dest->pub.output_file);
}


/*
 * Finish up at the end of the file.
 *
 * Here is where we really output the BMP file.
 *
 * First, routines to write the Windows and OS/2 variants of the file header.
 */

LOCAL(void)
write_bmp_header(j_decompress_ptr cinfo, bmp_dest_ptr dest)
/* Write a Windows-style BMP file header, including colormap if needed */
{
  char bmpfileheader[14];
  char bmpinfoheader[40];

#define PUT_2B(array, offset, value) \
  (array[offset] = (char)((value) & 0xFF), \
   array[offset + 1] = (char)(((value) >> 8) & 0xFF))
#define PUT_4B(array, offset, value) \
  (array[offset] = (char)((value) & 0xFF), \
   array[offset + 1] = (char)(((value) >> 8) & 0xFF), \
   array[offset + 2] = (char)(((value) >> 16) & 0xFF), \
   array[offset + 3] = (char)(((value) >> 24) & 0xFF))

  long headersize, bfSize;
  int bits_per_pixel, cmap_entries;

  /* Compute colormap size and total file size */
  if (IsExtRGB(cinfo->out_color_space)) {
    if (cinfo->quantize_colors) {
      /* Colormapped RGB */
      bits_per_pixel = 8;
      cmap_entries = 256;
    } else {
      /* Unquantized, full color RGB */
      bits_per_pixel = 24;
      cmap_entries = 0;
    }
  } else if (cinfo->out_color_space == JCS_RGB565 ||
             cinfo->out_color_space == JCS_CMYK) {
    bits_per_pixel = 24;
    cmap_entries   = 0;
  } else {
    /* Grayscale output.  We need to fake a 256-entry colormap. */
    bits_per_pixel = 8;
    cmap_entries = 256;
  }
  /* File size */
  headersize = 14 + 40 + cmap_entries * 4; /* Header and colormap */
  bfSize = headersize + (long)dest->row_width * (long)cinfo->output_height;

  /* Set unused fields of header to 0 */
  memset(bmpfileheader, 0, sizeof(bmpfileheader));
  memset(bmpinfoheader, 0, sizeof(bmpinfoheader));

  /* Fill the file header */
  bmpfileheader[0] = 0x42;      /* first 2 bytes are ASCII 'B', 'M' */
  bmpfileheader[1] = 0x4D;
  PUT_4B(bmpfileheader, 2, bfSize); /* bfSize */
  /* we leave bfReserved1 & bfReserved2 = 0 */
  PUT_4B(bmpfileheader, 10, headersize); /* bfOffBits */

  /* Fill the info header (Microsoft calls this a BITMAPINFOHEADER) */
  PUT_2B(bmpinfoheader, 0, 40); /* biSize */
  PUT_4B(bmpinfoheader, 4, cinfo->output_width); /* biWidth */
  PUT_4B(bmpinfoheader, 8, cinfo->output_height); /* biHeight */
  PUT_2B(bmpinfoheader, 12, 1); /* biPlanes - must be 1 */
  PUT_2B(bmpinfoheader, 14, bits_per_pixel); /* biBitCount */
  /* we leave biCompression = 0, for none */
  /* we leave biSizeImage = 0; this is correct for uncompressed data */
  if (cinfo->density_unit == 2) { /* if have density in dots/cm, then */
    PUT_4B(bmpinfoheader, 24, (long)(cinfo->X_density * 100)); /* XPels/M */
    PUT_4B(bmpinfoheader, 28, (long)(cinfo->Y_density * 100)); /* XPels/M */
  }
  PUT_2B(bmpinfoheader, 32, cmap_entries); /* biClrUsed */
  /* we leave biClrImportant = 0 */

  if (fwrite(bmpfileheader, 1, 14, dest->pub.output_file) != (size_t)14)
    ERREXIT(cinfo, JERR_FILE_WRITE);
  if (fwrite(bmpinfoheader, 1, 40, dest->pub.output_file) != (size_t)40)
    ERREXIT(cinfo, JERR_FILE_WRITE);

  if (cmap_entries > 0)
    write_colormap(cinfo, dest, cmap_entries, 4);
}


LOCAL(void)
write_os2_header(j_decompress_ptr cinfo, bmp_dest_ptr dest)
/* Write an OS2-style BMP file header, including colormap if needed */
{
  char bmpfileheader[14];
  char bmpcoreheader[12];
  long headersize, bfSize;
  int bits_per_pixel, cmap_entries;

  /* Compute colormap size and total file size */
  if (IsExtRGB(cinfo->out_color_space)) {
    if (cinfo->quantize_colors) {
      /* Colormapped RGB */
      bits_per_pixel = 8;
      cmap_entries = 256;
    } else {
      /* Unquantized, full color RGB */
      bits_per_pixel = 24;
      cmap_entries = 0;
    }
  } else if (cinfo->out_color_space == JCS_RGB565 ||
             cinfo->out_color_space == JCS_CMYK) {
    bits_per_pixel = 24;
    cmap_entries   = 0;
  } else {
    /* Grayscale output.  We need to fake a 256-entry colormap. */
    bits_per_pixel = 8;
    cmap_entries = 256;
  }
  /* File size */
  headersize = 14 + 12 + cmap_entries * 3; /* Header and colormap */
  bfSize = headersize + (long)dest->row_width * (long)cinfo->output_height;

  /* Set unused fields of header to 0 */
  memset(bmpfileheader, 0, sizeof(bmpfileheader));
  memset(bmpcoreheader, 0, sizeof(bmpcoreheader));

  /* Fill the file header */
  bmpfileheader[0] = 0x42;      /* first 2 bytes are ASCII 'B', 'M' */
  bmpfileheader[1] = 0x4D;
  PUT_4B(bmpfileheader, 2, bfSize); /* bfSize */
  /* we leave bfReserved1 & bfReserved2 = 0 */
  PUT_4B(bmpfileheader, 10, headersize); /* bfOffBits */

  /* Fill the info header (Microsoft calls this a BITMAPCOREHEADER) */
  PUT_2B(bmpcoreheader, 0, 12); /* bcSize */
  PUT_2B(bmpcoreheader, 4, cinfo->output_width); /* bcWidth */
  PUT_2B(bmpcoreheader, 6, cinfo->output_height); /* bcHeight */
  PUT_2B(bmpcoreheader, 8, 1);  /* bcPlanes - must be 1 */
  PUT_2B(bmpcoreheader, 10, bits_per_pixel); /* bcBitCount */

  if (fwrite(bmpfileheader, 1, 14, dest->pub.output_file) != (size_t)14)
    ERREXIT(cinfo, JERR_FILE_WRITE);
  if (fwrite(bmpcoreheader, 1, 12, dest->pub.output_file) != (size_t)12)
    ERREXIT(cinfo, JERR_FILE_WRITE);

  if (cmap_entries > 0)
    write_colormap(cinfo, dest, cmap_entries, 3);
}


/*
 * Write the colormap.
 * Windows uses BGR0 map entries; OS/2 uses BGR entries.
 */

LOCAL(void)
write_colormap(j_decompress_ptr cinfo, bmp_dest_ptr dest, int map_colors,
               int map_entry_size)
{
  JSAMPARRAY colormap = cinfo->colormap;
  int num_colors = cinfo->actual_number_of_colors;
  FILE *outfile = dest->pub.output_file;
  int i;

  if (colormap != NULL) {
    if (cinfo->out_color_components == 3) {
      /* Normal case with RGB colormap */
      for (i = 0; i < num_colors; i++) {
        putc(colormap[2][i], outfile);
        putc(colormap[1][i], outfile);
        putc(colormap[0][i], outfile);
        if (map_entry_size == 4)
          putc(0, outfile);
      }
    } else {
      /* Grayscale colormap (only happens with grayscale quantization) */
      for (i = 0; i < num_colors; i++) {
        putc(colormap[0][i], outfile);
        putc(colormap[0][i], outfile);
        putc(colormap[0][i], outfile);
        if (map_entry_size == 4)
          putc(0, outfile);
      }
    }
  } else {
    /* If no colormap, must be grayscale data.  Generate a linear "map". */
    for (i = 0; i < 256; i++) {
      putc(i, outfile);
      putc(i, outfile);
      putc(i, outfile);
      if (map_entry_size == 4)
        putc(0, outfile);
    }
  }
  /* Pad colormap with zeros to ensure specified number of colormap entries */
  if (i > map_colors)
    ERREXIT1(cinfo, JERR_TOO_MANY_COLORS, i);
  for (; i < map_colors; i++) {
    putc(0, outfile);
    putc(0, outfile);
    putc(0, outfile);
    if (map_entry_size == 4)
      putc(0, outfile);
  }
}


/*
 * Startup: write the file header unless the inversion array is being used.
 */

METHODDEF(void)
start_output_bmp(j_decompress_ptr cinfo, djpeg_dest_ptr dinfo)
{
  bmp_dest_ptr dest = (bmp_dest_ptr)dinfo;

  if (!dest->use_inversion_array) {
    /* Write the header and colormap */
    if (dest->is_os2)
      write_os2_header(cinfo, dest);
    else
      write_bmp_header(cinfo, dest);
  }
}


METHODDEF(void)
finish_output_bmp(j_decompress_ptr cinfo, djpeg_dest_ptr dinfo)
{
  bmp_dest_ptr dest = (bmp_dest_ptr)dinfo;
  register FILE *outfile = dest->pub.output_file;
  JSAMPARRAY image_ptr;
  register JSAMPROW data_ptr;
  JDIMENSION row;
  cd_progress_ptr progress = (cd_progress_ptr)cinfo->progress;

  if (dest->use_inversion_array) {
    /* Write the header and colormap */
    if (dest->is_os2)
      write_os2_header(cinfo, dest);
    else
      write_bmp_header(cinfo, dest);

    /* Write the file body from our virtual array */
    for (row = cinfo->output_height; row > 0; row--) {
      if (progress != NULL) {
        progress->pub.pass_counter = (long)(cinfo->output_height - row);
        progress->pub.pass_limit = (long)cinfo->output_height;
        (*progress->pub.progress_monitor) ((j_common_ptr)cinfo);
      }
      image_ptr = (*cinfo->mem->access_virt_sarray)
        ((j_common_ptr)cinfo, dest->whole_image, row - 1, (JDIMENSION)1,
         FALSE);
      data_ptr = image_ptr[0];
      fwrite(data_ptr, 1, dest->row_width, outfile);
    }
    if (progress != NULL)
      progress->completed_extra_passes++;
  }

  /* Make sure we wrote the output file OK */
  fflush(outfile);
  if (ferror(outfile))
    ERREXIT(cinfo, JERR_FILE_WRITE);
}


/*
 * The module selection routine for BMP format output.
 */

GLOBAL(djpeg_dest_ptr)
jinit_write_bmp(j_decompress_ptr cinfo, boolean is_os2,
                boolean use_inversion_array)
{
  bmp_dest_ptr dest;
  JDIMENSION row_width;

  if (cinfo->data_precision != 8)
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  /* Create module interface object, fill in method pointers */
  dest = (bmp_dest_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(bmp_dest_struct));
  dest->pub.start_output = start_output_bmp;
  dest->pub.finish_output = finish_output_bmp;
  dest->pub.calc_buffer_dimensions = NULL;
  dest->is_os2 = is_os2;

  if (cinfo->out_color_space == JCS_GRAYSCALE) {
    dest->pub.put_pixel_rows = put_gray_rows;
  } else if (IsExtRGB(cinfo->out_color_space)) {
    if (cinfo->quantize_colors)
      dest->pub.put_pixel_rows = put_gray_rows;
    else
      dest->pub.put_pixel_rows = put_pixel_rows;
  } else if (!cinfo->quantize_colors &&
             (cinfo->out_color_space == JCS_RGB565 ||
              cinfo->out_color_space == JCS_CMYK)) {
    dest->pub.put_pixel_rows = put_pixel_rows;
  } else {
    ERREXIT(cinfo, JERR_BMP_COLORSPACE);
  }

  /* Calculate output image dimensions so we can allocate space */
  jpeg_calc_output_dimensions(cinfo);

  /* Determine width of rows in the BMP file (padded to 4-byte boundary). */
  if (cinfo->out_color_space == JCS_RGB565) {
    row_width = cinfo->output_width * 2;
    dest->row_width = dest->data_width = cinfo->output_width * 3;
    while ((row_width & 3) != 0) row_width++;
  } else if (!cinfo->quantize_colors &&
             (IsExtRGB(cinfo->out_color_space) ||
              cinfo->out_color_space == JCS_CMYK)) {
    row_width = cinfo->output_width * cinfo->output_components;
    dest->row_width = dest->data_width = cinfo->output_width * 3;
  } else {
    row_width = cinfo->output_width * cinfo->output_components;
    dest->row_width = dest->data_width = row_width;
  }
  while ((dest->row_width & 3) != 0) dest->row_width++;
  dest->pad_bytes = (int)(dest->row_width - dest->data_width);


  if (use_inversion_array) {
    /* Allocate space for inversion array, prepare for write pass */
    dest->whole_image = (*cinfo->mem->request_virt_sarray)
      ((j_common_ptr)cinfo, JPOOL_IMAGE, FALSE,
       dest->row_width, cinfo->output_height, (JDIMENSION)1);
    dest->cur_output_row = 0;
    if (cinfo->progress != NULL) {
      cd_progress_ptr progress = (cd_progress_ptr)cinfo->progress;
      progress->total_extra_passes++; /* count file input as separate pass */
    }
  } else {
    dest->iobuffer = (JSAMPLE *)(*cinfo->mem->alloc_small)
      ((j_common_ptr)cinfo, JPOOL_IMAGE, dest->row_width);
  }
  dest->use_inversion_array = use_inversion_array;

  /* Create decompressor output buffer. */
  dest->pub.buffer = (*cinfo->mem->alloc_sarray)
    ((j_common_ptr)cinfo, JPOOL_IMAGE, row_width, (JDIMENSION)1);
  dest->pub.buffer_height = 1;

  return (djpeg_dest_ptr)dest;
}

#endif /* BMP_SUPPORTED */
