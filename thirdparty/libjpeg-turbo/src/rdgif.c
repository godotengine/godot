/*
 * rdgif.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Modified 2019 by Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2021-2023, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains routines to read input images in GIF format.
 *
 * These routines may need modification for non-Unix environments or
 * specialized applications.  As they stand, they assume input from
 * an ordinary stdio stream.  They further assume that reading begins
 * at the start of the file; start_input may need work if the
 * user interface has already read some data (e.g., to determine that
 * the file is indeed GIF format).
 */

/*
 * This code is loosely based on giftoppm from the PBMPLUS distribution
 * of Feb. 1991.  That file contains the following copyright notice:
 * +-------------------------------------------------------------------+
 * | Copyright 1990, David Koblas.                                     |
 * |   Permission to use, copy, modify, and distribute this software   |
 * |   and its documentation for any purpose and without fee is hereby |
 * |   granted, provided that the above copyright notice appear in all |
 * |   copies and that both that copyright notice and this permission  |
 * |   notice appear in supporting documentation.  This software is    |
 * |   provided "as is" without express or implied warranty.           |
 * +-------------------------------------------------------------------+
 */

#include "cdjpeg.h"             /* Common decls for cjpeg/djpeg applications */

#ifdef GIF_SUPPORTED


/* Macros to deal with unsigned chars as efficiently as compiler allows */

typedef unsigned char U_CHAR;
#define UCH(x)  ((int)(x))


#define ReadOK(file, buffer, len) \
  (fread(buffer, 1, len, file) == ((size_t)(len)))


#define MAXCOLORMAPSIZE  256    /* max # of colors in a GIF colormap */
#define NUMCOLORS        3      /* # of colors */
#define CM_RED           0      /* color component numbers */
#define CM_GREEN         1
#define CM_BLUE          2

#define MAX_LZW_BITS     12     /* maximum LZW code size */
#define LZW_TABLE_SIZE   (1 << MAX_LZW_BITS) /* # of possible LZW symbols */

/* Macros for extracting header data --- note we assume chars may be signed */

#define LM_to_uint(array, offset) \
  ((unsigned int)UCH(array[offset]) + \
   (((unsigned int)UCH(array[offset + 1])) << 8))

#define BitSet(byte, bit)       ((byte) & (bit))
#define INTERLACE       0x40    /* mask for bit signifying interlaced image */
#define COLORMAPFLAG    0x80    /* mask for bit signifying colormap presence */


/*
 * LZW decompression tables look like this:
 *   symbol_head[K] = prefix symbol of any LZW symbol K (0..LZW_TABLE_SIZE-1)
 *   symbol_tail[K] = suffix byte   of any LZW symbol K (0..LZW_TABLE_SIZE-1)
 * Note that entries 0..end_code of the above tables are not used,
 * since those symbols represent raw bytes or special codes.
 *
 * The stack represents the not-yet-used expansion of the last LZW symbol.
 * In the worst case, a symbol could expand to as many bytes as there are
 * LZW symbols, so we allocate LZW_TABLE_SIZE bytes for the stack.
 * (This is conservative since that number includes the raw-byte symbols.)
 */


/* Private version of data source object */

typedef struct {
  struct cjpeg_source_struct pub; /* public fields */

  j_compress_ptr cinfo;         /* back link saves passing separate parm */

  JSAMPARRAY colormap;          /* GIF colormap (converted to my format) */

  /* State for GetCode and LZWReadByte */
  U_CHAR code_buf[256 + 4];     /* current input data block */
  int last_byte;                /* # of bytes in code_buf */
  int last_bit;                 /* # of bits in code_buf */
  int cur_bit;                  /* next bit index to read */
  boolean first_time;           /* flags first call to GetCode */
  boolean out_of_blocks;        /* TRUE if hit terminator data block */

  int input_code_size;          /* codesize given in GIF file */
  int clear_code, end_code;     /* values for Clear and End codes */

  int code_size;                /* current actual code size */
  int limit_code;               /* 2^code_size */
  int max_code;                 /* first unused code value */

  /* Private state for LZWReadByte */
  int oldcode;                  /* previous LZW symbol */
  int firstcode;                /* first byte of oldcode's expansion */

  /* LZW symbol table and expansion stack */
  UINT16 *symbol_head;          /* => table of prefix symbols */
  UINT8  *symbol_tail;          /* => table of suffix bytes */
  UINT8  *symbol_stack;         /* => stack for symbol expansions */
  UINT8  *sp;                   /* stack pointer */

  /* State for interlaced image processing */
  boolean is_interlaced;        /* TRUE if have interlaced image */
  jvirt_sarray_ptr interlaced_image; /* full image in interlaced order */
  JDIMENSION cur_row_number;    /* need to know actual row number */
  JDIMENSION pass2_offset;      /* # of pixel rows in pass 1 */
  JDIMENSION pass3_offset;      /* # of pixel rows in passes 1&2 */
  JDIMENSION pass4_offset;      /* # of pixel rows in passes 1,2,3 */
} gif_source_struct;

typedef gif_source_struct *gif_source_ptr;


/* Forward declarations */
METHODDEF(JDIMENSION) get_pixel_rows(j_compress_ptr cinfo,
                                     cjpeg_source_ptr sinfo);
METHODDEF(JDIMENSION) load_interlaced_image(j_compress_ptr cinfo,
                                            cjpeg_source_ptr sinfo);
METHODDEF(JDIMENSION) get_interlaced_row(j_compress_ptr cinfo,
                                         cjpeg_source_ptr sinfo);


LOCAL(int)
ReadByte(gif_source_ptr sinfo)
/* Read next byte from GIF file */
{
  register FILE *infile = sinfo->pub.input_file;
  register int c;

  if ((c = getc(infile)) == EOF)
    ERREXIT(sinfo->cinfo, JERR_INPUT_EOF);
  return c;
}


LOCAL(int)
GetDataBlock(gif_source_ptr sinfo, U_CHAR *buf)
/* Read a GIF data block, which has a leading count byte */
/* A zero-length block marks the end of a data block sequence */
{
  int count;

  count = ReadByte(sinfo);
  if (count > 0) {
    if (!ReadOK(sinfo->pub.input_file, buf, count))
      ERREXIT(sinfo->cinfo, JERR_INPUT_EOF);
  }
  return count;
}


LOCAL(void)
SkipDataBlocks(gif_source_ptr sinfo)
/* Skip a series of data blocks, until a block terminator is found */
{
  U_CHAR buf[256];

  while (GetDataBlock(sinfo, buf) > 0)
    /* skip */;
}


LOCAL(void)
ReInitLZW(gif_source_ptr sinfo)
/* (Re)initialize LZW state; shared code for startup and Clear processing */
{
  sinfo->code_size = sinfo->input_code_size + 1;
  sinfo->limit_code = sinfo->clear_code << 1;   /* 2^code_size */
  sinfo->max_code = sinfo->clear_code + 2;      /* first unused code value */
  sinfo->sp = sinfo->symbol_stack;              /* init stack to empty */
}


LOCAL(void)
InitLZWCode(gif_source_ptr sinfo)
/* Initialize for a series of LZWReadByte (and hence GetCode) calls */
{
  /* GetCode initialization */
  sinfo->last_byte = 2;         /* make safe to "recopy last two bytes" */
  sinfo->code_buf[0] = 0;
  sinfo->code_buf[1] = 0;
  sinfo->last_bit = 0;          /* nothing in the buffer */
  sinfo->cur_bit = 0;           /* force buffer load on first call */
  sinfo->first_time = TRUE;
  sinfo->out_of_blocks = FALSE;

  /* LZWReadByte initialization: */
  /* compute special code values (note that these do not change later) */
  sinfo->clear_code = 1 << sinfo->input_code_size;
  sinfo->end_code = sinfo->clear_code + 1;
  ReInitLZW(sinfo);
}


LOCAL(int)
GetCode(gif_source_ptr sinfo)
/* Fetch the next code_size bits from the GIF data */
/* We assume code_size is less than 16 */
{
  register int accum;
  int offs, count;

  while (sinfo->cur_bit + sinfo->code_size > sinfo->last_bit) {
    /* Time to reload the buffer */
    /* First time, share code with Clear case */
    if (sinfo->first_time) {
      sinfo->first_time = FALSE;
      return sinfo->clear_code;
    }
    if (sinfo->out_of_blocks) {
      WARNMS(sinfo->cinfo, JWRN_GIF_NOMOREDATA);
      return sinfo->end_code;   /* fake something useful */
    }
    /* preserve last two bytes of what we have -- assume code_size <= 16 */
    sinfo->code_buf[0] = sinfo->code_buf[sinfo->last_byte-2];
    sinfo->code_buf[1] = sinfo->code_buf[sinfo->last_byte-1];
    /* Load more bytes; set flag if we reach the terminator block */
    if ((count = GetDataBlock(sinfo, &sinfo->code_buf[2])) == 0) {
      sinfo->out_of_blocks = TRUE;
      WARNMS(sinfo->cinfo, JWRN_GIF_NOMOREDATA);
      return sinfo->end_code;   /* fake something useful */
    }
    /* Reset counters */
    sinfo->cur_bit = (sinfo->cur_bit - sinfo->last_bit) + 16;
    sinfo->last_byte = 2 + count;
    sinfo->last_bit = sinfo->last_byte * 8;
  }

  /* Form up next 24 bits in accum */
  offs = sinfo->cur_bit >> 3;   /* byte containing cur_bit */
  accum = UCH(sinfo->code_buf[offs + 2]);
  accum <<= 8;
  accum |= UCH(sinfo->code_buf[offs + 1]);
  accum <<= 8;
  accum |= UCH(sinfo->code_buf[offs]);

  /* Right-align cur_bit in accum, then mask off desired number of bits */
  accum >>= (sinfo->cur_bit & 7);
  sinfo->cur_bit += sinfo->code_size;
  return accum & ((1 << sinfo->code_size) - 1);
}


LOCAL(int)
LZWReadByte(gif_source_ptr sinfo)
/* Read an LZW-compressed byte */
{
  register int code;            /* current working code */
  int incode;                   /* saves actual input code */

  /* If any codes are stacked from a previously read symbol, return them */
  if (sinfo->sp > sinfo->symbol_stack)
    return (int)(*(--sinfo->sp));

  /* Time to read a new symbol */
  code = GetCode(sinfo);

  if (code == sinfo->clear_code) {
    /* Reinit state, swallow any extra Clear codes, and */
    /* return next code, which is expected to be a raw byte. */
    ReInitLZW(sinfo);
    do {
      code = GetCode(sinfo);
    } while (code == sinfo->clear_code);
    if (code > sinfo->clear_code) { /* make sure it is a raw byte */
      WARNMS(sinfo->cinfo, JWRN_GIF_BADDATA);
      code = 0;                 /* use something valid */
    }
    /* make firstcode, oldcode valid! */
    sinfo->firstcode = sinfo->oldcode = code;
    return code;
  }

  if (code == sinfo->end_code) {
    /* Skip the rest of the image, unless GetCode already read terminator */
    if (!sinfo->out_of_blocks) {
      SkipDataBlocks(sinfo);
      sinfo->out_of_blocks = TRUE;
    }
    /* Complain that there's not enough data */
    WARNMS(sinfo->cinfo, JWRN_GIF_ENDCODE);
    /* Pad data with 0's */
    return 0;                   /* fake something usable */
  }

  /* Got normal raw byte or LZW symbol */
  incode = code;                /* save for a moment */

  if (code >= sinfo->max_code) { /* special case for not-yet-defined symbol */
    /* code == max_code is OK; anything bigger is bad data */
    if (code > sinfo->max_code) {
      WARNMS(sinfo->cinfo, JWRN_GIF_BADDATA);
      incode = 0;               /* prevent creation of loops in symbol table */
    }
    /* this symbol will be defined as oldcode/firstcode */
    *(sinfo->sp++) = (UINT8)sinfo->firstcode;
    code = sinfo->oldcode;
  }

  /* If it's a symbol, expand it into the stack */
  while (code >= sinfo->clear_code) {
    *(sinfo->sp++) = sinfo->symbol_tail[code]; /* tail is a byte value */
    code = sinfo->symbol_head[code]; /* head is another LZW symbol */
  }
  /* At this point code just represents a raw byte */
  sinfo->firstcode = code;      /* save for possible future use */

  /* If there's room in table... */
  if ((code = sinfo->max_code) < LZW_TABLE_SIZE) {
    /* Define a new symbol = prev sym + head of this sym's expansion */
    sinfo->symbol_head[code] = (UINT16)sinfo->oldcode;
    sinfo->symbol_tail[code] = (UINT8)sinfo->firstcode;
    sinfo->max_code++;
    /* Is it time to increase code_size? */
    if (sinfo->max_code >= sinfo->limit_code &&
        sinfo->code_size < MAX_LZW_BITS) {
      sinfo->code_size++;
      sinfo->limit_code <<= 1;  /* keep equal to 2^code_size */
    }
  }

  sinfo->oldcode = incode;      /* save last input symbol for future use */
  return sinfo->firstcode;      /* return first byte of symbol's expansion */
}


LOCAL(void)
ReadColorMap(gif_source_ptr sinfo, int cmaplen, JSAMPARRAY cmap)
/* Read a GIF colormap */
{
  int i, gray = 1;

  for (i = 0; i < cmaplen; i++) {
#if BITS_IN_JSAMPLE == 8
#define UPSCALE(x)  (x)
#else
#define UPSCALE(x)  ((x) << (BITS_IN_JSAMPLE - 8))
#endif
    cmap[CM_RED][i]   = (JSAMPLE)UPSCALE(ReadByte(sinfo));
    cmap[CM_GREEN][i] = (JSAMPLE)UPSCALE(ReadByte(sinfo));
    cmap[CM_BLUE][i]  = (JSAMPLE)UPSCALE(ReadByte(sinfo));
    if (cmap[CM_RED][i] != cmap[CM_GREEN][i] ||
        cmap[CM_GREEN][i] != cmap[CM_BLUE][i])
      gray = 0;
  }

  if (sinfo->cinfo->in_color_space == JCS_RGB && gray) {
    sinfo->cinfo->in_color_space = JCS_GRAYSCALE;
    sinfo->cinfo->input_components = 1;
  }
}


LOCAL(void)
DoExtension(gif_source_ptr sinfo)
/* Process an extension block */
/* Currently we ignore 'em all */
{
  int extlabel;

  /* Read extension label byte */
  extlabel = ReadByte(sinfo);
  TRACEMS1(sinfo->cinfo, 1, JTRC_GIF_EXTENSION, extlabel);
  /* Skip the data block(s) associated with the extension */
  SkipDataBlocks(sinfo);
}


/*
 * Read the file header; return image size and component count.
 */

METHODDEF(void)
start_input_gif(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  gif_source_ptr source = (gif_source_ptr)sinfo;
  U_CHAR hdrbuf[10];            /* workspace for reading control blocks */
  unsigned int width, height;   /* image dimensions */
  int colormaplen, aspectRatio;
  int c;

  /* Read and verify GIF Header */
  if (!ReadOK(source->pub.input_file, hdrbuf, 6))
    ERREXIT(cinfo, JERR_GIF_NOT);
  if (hdrbuf[0] != 'G' || hdrbuf[1] != 'I' || hdrbuf[2] != 'F')
    ERREXIT(cinfo, JERR_GIF_NOT);
  /* Check for expected version numbers.
   * If unknown version, give warning and try to process anyway;
   * this is per recommendation in GIF89a standard.
   */
  if ((hdrbuf[3] != '8' || hdrbuf[4] != '7' || hdrbuf[5] != 'a') &&
      (hdrbuf[3] != '8' || hdrbuf[4] != '9' || hdrbuf[5] != 'a'))
    TRACEMS3(cinfo, 1, JTRC_GIF_BADVERSION, hdrbuf[3], hdrbuf[4], hdrbuf[5]);

  /* Read and decipher Logical Screen Descriptor */
  if (!ReadOK(source->pub.input_file, hdrbuf, 7))
    ERREXIT(cinfo, JERR_INPUT_EOF);
  width = LM_to_uint(hdrbuf, 0);
  height = LM_to_uint(hdrbuf, 2);
  if (width == 0 || height == 0)
    ERREXIT(cinfo, JERR_GIF_EMPTY);
  if (sinfo->max_pixels &&
      (unsigned long long)width * height > sinfo->max_pixels)
    ERREXIT1(cinfo, JERR_IMAGE_TOO_BIG, sinfo->max_pixels);
  /* we ignore the color resolution, sort flag, and background color index */
  aspectRatio = UCH(hdrbuf[6]);
  if (aspectRatio != 0 && aspectRatio != 49)
    TRACEMS(cinfo, 1, JTRC_GIF_NONSQUARE);

  /* Allocate space to store the colormap */
  source->colormap = (*cinfo->mem->alloc_sarray)
    ((j_common_ptr)cinfo, JPOOL_IMAGE, (JDIMENSION)MAXCOLORMAPSIZE,
     (JDIMENSION)NUMCOLORS);
  colormaplen = 0;              /* indicate initialization */

  /* Read global colormap if header indicates it is present */
  if (BitSet(hdrbuf[4], COLORMAPFLAG)) {
    colormaplen = 2 << (hdrbuf[4] & 0x07);
    ReadColorMap(source, colormaplen, source->colormap);
  }

  /* Scan until we reach start of desired image.
   * We don't currently support skipping images, but could add it easily.
   */
  for (;;) {
    c = ReadByte(source);

    if (c == ';')               /* GIF terminator?? */
      ERREXIT(cinfo, JERR_GIF_IMAGENOTFOUND);

    if (c == '!') {             /* Extension */
      DoExtension(source);
      continue;
    }

    if (c != ',') {             /* Not an image separator? */
      WARNMS1(cinfo, JWRN_GIF_CHAR, c);
      continue;
    }

    /* Read and decipher Local Image Descriptor */
    if (!ReadOK(source->pub.input_file, hdrbuf, 9))
      ERREXIT(cinfo, JERR_INPUT_EOF);
    /* we ignore top/left position info, also sort flag */
    width = LM_to_uint(hdrbuf, 4);
    height = LM_to_uint(hdrbuf, 6);
    if (width == 0 || height == 0)
      ERREXIT(cinfo, JERR_GIF_EMPTY);
    if (sinfo->max_pixels &&
        (unsigned long long)width * height > sinfo->max_pixels)
      ERREXIT1(cinfo, JERR_IMAGE_TOO_BIG, sinfo->max_pixels);
    source->is_interlaced = (BitSet(hdrbuf[8], INTERLACE) != 0);

    /* Read local colormap if header indicates it is present */
    /* Note: if we wanted to support skipping images, */
    /* we'd need to skip rather than read colormap for ignored images */
    if (BitSet(hdrbuf[8], COLORMAPFLAG)) {
      colormaplen = 2 << (hdrbuf[8] & 0x07);
      ReadColorMap(source, colormaplen, source->colormap);
    }

    source->input_code_size = ReadByte(source); /* get min-code-size byte */
    if (source->input_code_size < 2 || source->input_code_size > 8)
      ERREXIT1(cinfo, JERR_GIF_CODESIZE, source->input_code_size);

    /* Reached desired image, so break out of loop */
    /* If we wanted to skip this image, */
    /* we'd call SkipDataBlocks and then continue the loop */
    break;
  }

  /* Prepare to read selected image: first initialize LZW decompressor */
  source->symbol_head = (UINT16 *)
    (*cinfo->mem->alloc_large) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                LZW_TABLE_SIZE * sizeof(UINT16));
  source->symbol_tail = (UINT8 *)
    (*cinfo->mem->alloc_large) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                LZW_TABLE_SIZE * sizeof(UINT8));
  source->symbol_stack = (UINT8 *)
    (*cinfo->mem->alloc_large) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                LZW_TABLE_SIZE * sizeof(UINT8));
  InitLZWCode(source);

  /*
   * If image is interlaced, we read it into a full-size sample array,
   * decompressing as we go; then get_interlaced_row selects rows from the
   * sample array in the proper order.
   */
  if (source->is_interlaced) {
    /* We request the virtual array now, but can't access it until virtual
     * arrays have been allocated.  Hence, the actual work of reading the
     * image is postponed until the first call to get_pixel_rows.
     */
    source->interlaced_image = (*cinfo->mem->request_virt_sarray)
      ((j_common_ptr)cinfo, JPOOL_IMAGE, FALSE,
       (JDIMENSION)width, (JDIMENSION)height, (JDIMENSION)1);
    if (cinfo->progress != NULL) {
      cd_progress_ptr progress = (cd_progress_ptr)cinfo->progress;
      progress->total_extra_passes++; /* count file input as separate pass */
    }
    source->pub.get_pixel_rows = load_interlaced_image;
  } else {
    source->pub.get_pixel_rows = get_pixel_rows;
  }

  if (cinfo->in_color_space != JCS_GRAYSCALE) {
    cinfo->in_color_space = JCS_RGB;
    cinfo->input_components = NUMCOLORS;
  }

  /* Create compressor input buffer. */
  source->pub.buffer = (*cinfo->mem->alloc_sarray)
    ((j_common_ptr)cinfo, JPOOL_IMAGE,
     (JDIMENSION)width * cinfo->input_components, (JDIMENSION)1);
  source->pub.buffer_height = 1;

  /* Pad colormap for safety. */
  for (c = colormaplen; c < source->clear_code; c++) {
    source->colormap[CM_RED][c]   =
    source->colormap[CM_GREEN][c] =
    source->colormap[CM_BLUE][c]  = CENTERJSAMPLE;
  }

  /* Return info about the image. */
  cinfo->data_precision = BITS_IN_JSAMPLE; /* we always rescale data to this */
  cinfo->image_width = width;
  cinfo->image_height = height;

  TRACEMS3(cinfo, 1, JTRC_GIF, width, height, colormaplen);
}


/*
 * Read one row of pixels.
 * This version is used for noninterlaced GIF images:
 * we read directly from the GIF file.
 */

METHODDEF(JDIMENSION)
get_pixel_rows(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  gif_source_ptr source = (gif_source_ptr)sinfo;
  register int c;
  register JSAMPROW ptr;
  register JDIMENSION col;
  register JSAMPARRAY colormap = source->colormap;

  ptr = source->pub.buffer[0];
  if (cinfo->in_color_space == JCS_GRAYSCALE) {
    for (col = cinfo->image_width; col > 0; col--) {
      c = LZWReadByte(source);
      *ptr++ = colormap[CM_RED][c];
    }
  } else {
    for (col = cinfo->image_width; col > 0; col--) {
      c = LZWReadByte(source);
      *ptr++ = colormap[CM_RED][c];
      *ptr++ = colormap[CM_GREEN][c];
      *ptr++ = colormap[CM_BLUE][c];
    }
  }
  return 1;
}


/*
 * Read one row of pixels.
 * This version is used for the first call on get_pixel_rows when
 * reading an interlaced GIF file: we read the whole image into memory.
 */

METHODDEF(JDIMENSION)
load_interlaced_image(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  gif_source_ptr source = (gif_source_ptr)sinfo;
  register JSAMPROW sptr;
  register JDIMENSION col;
  JDIMENSION row;
  cd_progress_ptr progress = (cd_progress_ptr)cinfo->progress;

  /* Read the interlaced image into the virtual array we've created. */
  for (row = 0; row < cinfo->image_height; row++) {
    if (progress != NULL) {
      progress->pub.pass_counter = (long)row;
      progress->pub.pass_limit = (long)cinfo->image_height;
      (*progress->pub.progress_monitor) ((j_common_ptr)cinfo);
    }
    sptr = *(*cinfo->mem->access_virt_sarray)
      ((j_common_ptr)cinfo, source->interlaced_image, row, (JDIMENSION)1,
       TRUE);
    for (col = cinfo->image_width; col > 0; col--) {
      *sptr++ = (JSAMPLE)LZWReadByte(source);
    }
  }
  if (progress != NULL)
    progress->completed_extra_passes++;

  /* Replace method pointer so subsequent calls don't come here. */
  source->pub.get_pixel_rows = get_interlaced_row;
  /* Initialize for get_interlaced_row, and perform first call on it. */
  source->cur_row_number = 0;
  source->pass2_offset = (cinfo->image_height + 7) / 8;
  source->pass3_offset = source->pass2_offset + (cinfo->image_height + 3) / 8;
  source->pass4_offset = source->pass3_offset + (cinfo->image_height + 1) / 4;

  return get_interlaced_row(cinfo, sinfo);
}


/*
 * Read one row of pixels.
 * This version is used for interlaced GIF images:
 * we read from the virtual array.
 */

METHODDEF(JDIMENSION)
get_interlaced_row(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  gif_source_ptr source = (gif_source_ptr)sinfo;
  register int c;
  register JSAMPROW sptr, ptr;
  register JDIMENSION col;
  register JSAMPARRAY colormap = source->colormap;
  JDIMENSION irow;

  /* Figure out which row of interlaced image is needed, and access it. */
  switch ((int)(source->cur_row_number & 7)) {
  case 0:                       /* first-pass row */
    irow = source->cur_row_number >> 3;
    break;
  case 4:                       /* second-pass row */
    irow = (source->cur_row_number >> 3) + source->pass2_offset;
    break;
  case 2:                       /* third-pass row */
  case 6:
    irow = (source->cur_row_number >> 2) + source->pass3_offset;
    break;
  default:                      /* fourth-pass row */
    irow = (source->cur_row_number >> 1) + source->pass4_offset;
  }
  sptr = *(*cinfo->mem->access_virt_sarray)
    ((j_common_ptr)cinfo, source->interlaced_image, irow, (JDIMENSION)1,
     FALSE);
  /* Scan the row, expand colormap, and output */
  ptr = source->pub.buffer[0];
  if (cinfo->in_color_space == JCS_GRAYSCALE) {
    for (col = cinfo->image_width; col > 0; col--) {
      c = *sptr++;
      *ptr++ = colormap[CM_RED][c];
    }
  } else {
    for (col = cinfo->image_width; col > 0; col--) {
      c = *sptr++;
      *ptr++ = colormap[CM_RED][c];
      *ptr++ = colormap[CM_GREEN][c];
      *ptr++ = colormap[CM_BLUE][c];
    }
  }
  source->cur_row_number++;     /* for next time */
  return 1;
}


/*
 * Finish up at the end of the file.
 */

METHODDEF(void)
finish_input_gif(j_compress_ptr cinfo, cjpeg_source_ptr sinfo)
{
  /* no work */
}


/*
 * The module selection routine for GIF format input.
 */

GLOBAL(cjpeg_source_ptr)
jinit_read_gif(j_compress_ptr cinfo)
{
  gif_source_ptr source;

  if (cinfo->data_precision != 8)
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  /* Create module interface object */
  source = (gif_source_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(gif_source_struct));
  source->cinfo = cinfo;        /* make back link for subroutines */
  /* Fill in method ptrs, except get_pixel_rows which start_input sets */
  source->pub.start_input = start_input_gif;
  source->pub.finish_input = finish_input_gif;
  source->pub.max_pixels = 0;

  return (cjpeg_source_ptr)source;
}

#endif /* GIF_SUPPORTED */
