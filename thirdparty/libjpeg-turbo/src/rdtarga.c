/*
 * rdtarga.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1996, Thomas G. Lane.
 * Modified 2017 by Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2018, 2021-2023, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains routines to read input images in Targa format.
 *
 * These routines may need modification for non-Unix environments or
 * specialized applications.  As they stand, they assume input from
 * an ordinary stdio stream.  They further assume that reading begins
 * at the start of the file; start_input may need work if the
 * user interface has already read some data (e.g., to determine that
 * the file is indeed Targa format).
 *
 * Based on code contributed by Lee Daniel Crocker.
 */

#include "cdjpeg.h"             /* Common decls for cjpeg/djpeg applications */

#ifdef TARGA_SUPPORTED


/* Macros to deal with unsigned chars as efficiently as compiler allows */

typedef unsigned char U_CHAR;
#define UCH(x)  ((int)(x))


#define ReadOK(file, buffer, len) \
  (fread(buffer, 1, len, file) == ((size_t)(len)))


/* Private version of data source object */

typedef struct _tga_source_struct *tga_source_ptr;

typedef struct _tga_source_struct {
  struct cjpeg_source_struct pub; /* public fields */

  j_compress_ptr cinfo;         /* back link saves passing separate parm */

  JSAMPARRAY colormap;          /* Targa colormap (converted to my format) */

  jvirt_sarray_ptr whole_image; /* Needed if funny input row order */
  JDIMENSION current_row;       /* Current logical row number to read */

  /* Pointer to routine to extract next Targa pixel from input file */
  void (*read_pixel) (tga_source_ptr sinfo);

  /* Result of read_pixel is delivered here: */
  U_CHAR tga_pixel[4];

  int pixel_size;               /* Bytes per Targa pixel (1 to 4) */
  int cmap_length;              /* colormap length */

  /* State info for reading RLE-coded pixels; both counts must be init to 0 */
  int block_count;              /* # of pixels remaining in RLE block */
  int dup_pixel_count;          /* # of times to duplicate previous pixel */

  /* This saves the correct pixel-row-expansion method for preload_image */
  JDIMENSION (*get_pixel_rows) (j_compress_ptr cinfo, cjpeg_source_ptr sinfo);
} tga_source_struct;


/* For expanding 5-bit pixel values to 8-bit with best rounding */

static const UINT8 c5to8bits[32] = {
    0,   8,  16,  25,  33,  41,  49,  58,
   66,  74,  82,  90,  99, 107, 115, 123,
  132, 140, 148, 156, 165, 173, 181, 189,
  197, 206, 214, 222, 230, 239, 247, 255
};



LOCAL(int)
read_byte(tga_source_ptr sinfo)
/* Read next byte from Targa file */
{
  register FILE *infile = sinfo->pub.input_file;
  register int c;

  if ((c = getc(infile)) == EOF)
    ERREXIT(sinfo->cinfo, JERR_INPUT_EOF);
  return c;
}


LOCAL(void)
read_colormap(tga_source_ptr sinfo, int cmaplen, int mapentrysize)
/* Read the colormap from a Targa file */
{
  int i;

  /* Presently only handles 24-bit BGR format */
  if (mapentrysize != 24)
    ERREXIT(sinfo->cinfo, JERR_TGA_BADCMAP);

  for (i = 0; i < cmaplen; i++) {
    sinfo->colormap[2][i] = (JSAMPLE)read_byte(sinfo);
    sinfo->colormap[1][i] = (JSAMPLE)read_byte(sinfo);
    sinfo->colormap[0][i] = (JSAMPLE)read_byte(sinfo);
  }
}


/*
 * read_pixel methods: get a single pixel from Targa file into tga_pixel[]
 */

METHODDEF(void)
read_non_rle_pixel(tga_source_ptr sinfo)
/* Read one Targa pixel from the input file; no RLE expansion */
{
  register int i;

  for (i = 0; i < sinfo->pixel_size; i++) {
    sinfo->tga_pixel[i] = (U_CHAR)read_byte(sinfo);
  }
}


METHODDEF(void)
read_rle_pixel(tga_source_ptr sinfo)
/* Read one Targa pixel from the input file, expanding RLE data as needed */
{
  register int i;

  /* Duplicate previously read pixel? */
  if (sinfo->dup_pixel_count > 0) {
    sinfo->dup_pixel_count--;
    return;
  }

  /* Time to read RLE block header? */
  if (--sinfo->block_count < 0) { /* decrement pixels remaining in block */
    i = read_byte(sinfo);
    if (i & 0x80) {             /* Start of duplicate-pixel block? */
      sinfo->dup_pixel_count = i & 0x7F; /* number of dups after this one */
      sinfo->block_count = 0;   /* then read new block header */
    } else {
      sinfo->block_count = i & 0x7F; /* number of pixels after this one */
    }
  }

  /* Read next pixel */
  for (i = 0; i < sinfo->pixel_size; i++) {
    sinfo->tga_pixel[i] = (U_CHAR)read_byte(sinfo);
  }
}


/*
 * Read one row of pixels.
 *
 * We provide several different versions depending on input file format.
 */


METHODDEF(JDIMENSION)
get_8bit_gray_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading 8-bit grayscale pixels */
{
  tga_source_ptr source = (tga_source_ptr)sinfo;
  register JSAMPROW ptr;
  register JDIMENSION col;

  ptr = source->pub.buffer[0];
  for (col = cinfo->image_width; col > 0; col--) {
    (*source->read_pixel) (source); /* Load next pixel into tga_pixel */
    *ptr++ = (JSAMPLE)UCH(source->tga_pixel[0]);
  }
  return 1;
}

METHODDEF(JDIMENSION)
get_8bit_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading 8-bit colormap indexes */
{
  tga_source_ptr source = (tga_source_ptr)sinfo;
  register int t;
  register JSAMPROW ptr;
  register JDIMENSION col;
  register JSAMPARRAY colormap = source->colormap;
  int cmaplen = source->cmap_length;

  ptr = source->pub.buffer[0];
  for (col = cinfo->image_width; col > 0; col--) {
    (*source->read_pixel) (source); /* Load next pixel into tga_pixel */
    t = UCH(source->tga_pixel[0]);
    if (t >= cmaplen)
      ERREXIT(cinfo, JERR_TGA_BADPARMS);
    *ptr++ = colormap[0][t];
    *ptr++ = colormap[1][t];
    *ptr++ = colormap[2][t];
  }
  return 1;
}

METHODDEF(JDIMENSION)
get_16bit_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading 16-bit pixels */
{
  tga_source_ptr source = (tga_source_ptr)sinfo;
  register int t;
  register JSAMPROW ptr;
  register JDIMENSION col;

  ptr = source->pub.buffer[0];
  for (col = cinfo->image_width; col > 0; col--) {
    (*source->read_pixel) (source); /* Load next pixel into tga_pixel */
    t = UCH(source->tga_pixel[0]);
    t += UCH(source->tga_pixel[1]) << 8;
    /* We expand 5 bit data to 8 bit sample width.
     * The format of the 16-bit (LSB first) input word is
     *     xRRRRRGGGGGBBBBB
     */
    ptr[2] = (JSAMPLE)c5to8bits[t & 0x1F];
    t >>= 5;
    ptr[1] = (JSAMPLE)c5to8bits[t & 0x1F];
    t >>= 5;
    ptr[0] = (JSAMPLE)c5to8bits[t & 0x1F];
    ptr += 3;
  }
  return 1;
}

METHODDEF(JDIMENSION)
get_24bit_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
/* This version is for reading 24-bit pixels */
{
  tga_source_ptr source = (tga_source_ptr)sinfo;
  register JSAMPROW ptr;
  register JDIMENSION col;

  ptr = source->pub.buffer[0];
  for (col = cinfo->image_width; col > 0; col--) {
    (*source->read_pixel) (source); /* Load next pixel into tga_pixel */
    *ptr++ = (JSAMPLE)UCH(source->tga_pixel[2]); /* change BGR to RGB order */
    *ptr++ = (JSAMPLE)UCH(source->tga_pixel[1]);
    *ptr++ = (JSAMPLE)UCH(source->tga_pixel[0]);
  }
  return 1;
}

/*
 * Targa also defines a 32-bit pixel format with order B,G,R,A.
 * We presently ignore the attribute byte, so the code for reading
 * these pixels is identical to the 24-bit routine above.
 * This works because the actual pixel length is only known to read_pixel.
 */

#define get_32bit_row  get_24bit_row


/*
 * This method is for re-reading the input data in standard top-down
 * row order.  The entire image has already been read into whole_image
 * with proper conversion of pixel format, but it's in a funny row order.
 */

METHODDEF(JDIMENSION)
get_memory_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  tga_source_ptr source = (tga_source_ptr)sinfo;
  JDIMENSION source_row;

  /* Compute row of source that maps to current_row of normal order */
  /* For now, assume image is bottom-up and not interlaced. */
  /* NEEDS WORK to support interlaced images! */
  source_row = cinfo->image_height - source->current_row - 1;

  /* Fetch that row from virtual array */
  source->pub.buffer = (*cinfo->mem->access_virt_sarray)
    ((j_common_ptr)cinfo, source->whole_image,
     source_row, (JDIMENSION)1, FALSE);

  source->current_row++;
  return 1;
}


/*
 * This method loads the image into whole_image during the first call on
 * get_pixel_rows.  The get_pixel_rows pointer is then adjusted to call
 * get_memory_row on subsequent calls.
 */

METHODDEF(JDIMENSION)
preload_image(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  tga_source_ptr source = (tga_source_ptr)sinfo;
  JDIMENSION row;
  cd_progress_ptr progress = (cd_progress_ptr)cinfo->progress;

  /* Read the data into a virtual array in input-file row order. */
  for (row = 0; row < cinfo->image_height; row++) {
    if (progress != NULL) {
      progress->pub.pass_counter = (long)row;
      progress->pub.pass_limit = (long)cinfo->image_height;
      (*progress->pub.progress_monitor) ((j_common_ptr)cinfo);
    }
    source->pub.buffer = (*cinfo->mem->access_virt_sarray)
      ((j_common_ptr)cinfo, source->whole_image, row, (JDIMENSION)1, TRUE);
    (*source->get_pixel_rows) (cinfo, sinfo);
  }
  if (progress != NULL)
    progress->completed_extra_passes++;

  /* Set up to read from the virtual array in unscrambled order */
  source->pub.get_pixel_rows = get_memory_row;
  source->current_row = 0;
  /* And read the first row */
  return get_memory_row(cinfo, sinfo);
}


/*
 * Read the file header; return image size and component count.
 */

METHODDEF(void)
start_input_tga(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  tga_source_ptr source = (tga_source_ptr)sinfo;
  U_CHAR targaheader[18];
  int idlen, cmaptype, subtype, flags, interlace_type, components;
  unsigned int width, height, maplen;
  boolean is_bottom_up;

#define GET_2B(offset) \
  ((unsigned int)UCH(targaheader[offset]) + \
   (((unsigned int)UCH(targaheader[offset + 1])) << 8))

  if (!ReadOK(source->pub.input_file, targaheader, 18))
    ERREXIT(cinfo, JERR_INPUT_EOF);

  /* Pretend "15-bit" pixels are 16-bit --- we ignore attribute bit anyway */
  if (targaheader[16] == 15)
    targaheader[16] = 16;

  idlen = UCH(targaheader[0]);
  cmaptype = UCH(targaheader[1]);
  subtype = UCH(targaheader[2]);
  maplen = GET_2B(5);
  width = GET_2B(12);
  height = GET_2B(14);
  source->pixel_size = UCH(targaheader[16]) >> 3;
  flags = UCH(targaheader[17]); /* Image Descriptor byte */

  is_bottom_up = ((flags & 0x20) == 0); /* bit 5 set => top-down */
  interlace_type = flags >> 6;  /* bits 6/7 are interlace code */

  if (cmaptype > 1 ||           /* cmaptype must be 0 or 1 */
      source->pixel_size < 1 || source->pixel_size > 4 ||
      (UCH(targaheader[16]) & 7) != 0 || /* bits/pixel must be multiple of 8 */
      interlace_type != 0 ||      /* currently don't allow interlaced image */
      width == 0 || height == 0)  /* image width/height must be non-zero */
    ERREXIT(cinfo, JERR_TGA_BADPARMS);
  if (sinfo->max_pixels &&
      (unsigned long long)width * height > sinfo->max_pixels)
    ERREXIT1(cinfo, JERR_IMAGE_TOO_BIG, sinfo->max_pixels);

  if (subtype > 8) {
    /* It's an RLE-coded file */
    source->read_pixel = read_rle_pixel;
    source->block_count = source->dup_pixel_count = 0;
    subtype -= 8;
  } else {
    /* Non-RLE file */
    source->read_pixel = read_non_rle_pixel;
  }

  /* Now should have subtype 1, 2, or 3 */
  components = 3;               /* until proven different */
  cinfo->in_color_space = JCS_RGB;

  switch (subtype) {
  case 1:                       /* Colormapped image */
    if (source->pixel_size == 1 && cmaptype == 1)
      source->get_pixel_rows = get_8bit_row;
    else
      ERREXIT(cinfo, JERR_TGA_BADPARMS);
    TRACEMS2(cinfo, 1, JTRC_TGA_MAPPED, width, height);
    break;
  case 2:                       /* RGB image */
    switch (source->pixel_size) {
    case 2:
      source->get_pixel_rows = get_16bit_row;
      break;
    case 3:
      source->get_pixel_rows = get_24bit_row;
      break;
    case 4:
      source->get_pixel_rows = get_32bit_row;
      break;
    default:
      ERREXIT(cinfo, JERR_TGA_BADPARMS);
      break;
    }
    TRACEMS2(cinfo, 1, JTRC_TGA, width, height);
    break;
  case 3:                       /* Grayscale image */
    components = 1;
    cinfo->in_color_space = JCS_GRAYSCALE;
    if (source->pixel_size == 1)
      source->get_pixel_rows = get_8bit_gray_row;
    else
      ERREXIT(cinfo, JERR_TGA_BADPARMS);
    TRACEMS2(cinfo, 1, JTRC_TGA_GRAY, width, height);
    break;
  default:
    ERREXIT(cinfo, JERR_TGA_BADPARMS);
    break;
  }

  if (is_bottom_up) {
    /* Create a virtual array to buffer the upside-down image. */
    source->whole_image = (*cinfo->mem->request_virt_sarray)
      ((j_common_ptr)cinfo, JPOOL_IMAGE, FALSE,
       (JDIMENSION)width * components, (JDIMENSION)height, (JDIMENSION)1);
    if (cinfo->progress != NULL) {
      cd_progress_ptr progress = (cd_progress_ptr)cinfo->progress;
      progress->total_extra_passes++; /* count file input as separate pass */
    }
    /* source->pub.buffer will point to the virtual array. */
    source->pub.buffer_height = 1; /* in case anyone looks at it */
    source->pub.get_pixel_rows = preload_image;
  } else {
    /* Don't need a virtual array, but do need a one-row input buffer. */
    source->whole_image = NULL;
    source->pub.buffer = (*cinfo->mem->alloc_sarray)
      ((j_common_ptr)cinfo, JPOOL_IMAGE,
       (JDIMENSION)width * components, (JDIMENSION)1);
    source->pub.buffer_height = 1;
    source->pub.get_pixel_rows = source->get_pixel_rows;
  }

  while (idlen--)               /* Throw away ID field */
    (void)read_byte(source);

  if (maplen > 0) {
    if (maplen > 256 || GET_2B(3) != 0)
      ERREXIT(cinfo, JERR_TGA_BADCMAP);
    /* Allocate space to store the colormap */
    source->colormap = (*cinfo->mem->alloc_sarray)
      ((j_common_ptr)cinfo, JPOOL_IMAGE, (JDIMENSION)maplen, (JDIMENSION)3);
    source->cmap_length = (int)maplen;
    /* and read it from the file */
    read_colormap(source, (int)maplen, UCH(targaheader[7]));
  } else {
    if (cmaptype)               /* but you promised a cmap! */
      ERREXIT(cinfo, JERR_TGA_BADPARMS);
    source->colormap = NULL;
    source->cmap_length = 0;
  }

  cinfo->input_components = components;
  cinfo->data_precision = 8;
  cinfo->image_width = width;
  cinfo->image_height = height;
}


/*
 * Finish up at the end of the file.
 */

METHODDEF(void)
finish_input_tga(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  /* no work */
}


/*
 * The module selection routine for Targa format input.
 */

GLOBAL(cjpeg_source_ptr)
jinit_read_targa(j_compress_ptr cinfo)
{
  tga_source_ptr source;

  if (cinfo->data_precision != 8)
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  /* Create module interface object */
  source = (tga_source_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(tga_source_struct));
  source->cinfo = cinfo;        /* make back link for subroutines */
  /* Fill in method ptrs, except get_pixel_rows which start_input sets */
  source->pub.start_input = start_input_tga;
  source->pub.finish_input = finish_input_tga;
  source->pub.max_pixels = 0;

  return (cjpeg_source_ptr)source;
}

#endif /* TARGA_SUPPORTED */
