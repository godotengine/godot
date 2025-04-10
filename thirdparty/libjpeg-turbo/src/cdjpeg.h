/*
 * cdjpeg.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1997, Thomas G. Lane.
 * Modified 2019 by Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2017, 2019, 2021-2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains common declarations for the sample applications
 * cjpeg and djpeg.  It is NOT used by the core JPEG library.
 */

#define JPEG_CJPEG_DJPEG        /* define proper options in jconfig.h */
#define JPEG_INTERNAL_OPTIONS   /* cjpeg.c,djpeg.c need to see xxx_SUPPORTED */
#include "jinclude.h"
#include "jpeglib.h"
#include "jerror.h"             /* get library error codes too */
#include "cderror.h"            /* get application-specific error codes */


/*
 * Object interface for cjpeg's source file decoding modules
 */

typedef struct cjpeg_source_struct *cjpeg_source_ptr;

struct cjpeg_source_struct {
  void (*start_input) (j_compress_ptr cinfo, cjpeg_source_ptr sinfo);
  JDIMENSION (*get_pixel_rows) (j_compress_ptr cinfo, cjpeg_source_ptr sinfo);
  void (*finish_input) (j_compress_ptr cinfo, cjpeg_source_ptr sinfo);

  FILE *input_file;

  JSAMPARRAY buffer;
  J12SAMPARRAY buffer12;
#ifdef C_LOSSLESS_SUPPORTED
  J16SAMPARRAY buffer16;
#endif
  JDIMENSION buffer_height;
  JDIMENSION max_pixels;
};


/*
 * Object interface for djpeg's output file encoding modules
 */

typedef struct djpeg_dest_struct *djpeg_dest_ptr;

struct djpeg_dest_struct {
  /* start_output is called after jpeg_start_decompress finishes.
   * The color map will be ready at this time, if one is needed.
   */
  void (*start_output) (j_decompress_ptr cinfo, djpeg_dest_ptr dinfo);
  /* Emit the specified number of pixel rows from the buffer. */
  void (*put_pixel_rows) (j_decompress_ptr cinfo, djpeg_dest_ptr dinfo,
                          JDIMENSION rows_supplied);
  /* Finish up at the end of the image. */
  void (*finish_output) (j_decompress_ptr cinfo, djpeg_dest_ptr dinfo);
  /* Re-calculate buffer dimensions based on output dimensions (for use with
     partial image decompression.)  If this is NULL, then the output format
     does not support partial image decompression (BMP, in particular, cannot
     support partial decompression because it uses an inversion buffer to write
     the image in bottom-up order.) */
  void (*calc_buffer_dimensions) (j_decompress_ptr cinfo,
                                  djpeg_dest_ptr dinfo);


  /* Target file spec; filled in by djpeg.c after object is created. */
  FILE *output_file;

  /* Output pixel-row buffer.  Created by module init or start_output.
   * Width is cinfo->output_width * cinfo->output_components;
   * height is buffer_height.
   */
  JSAMPARRAY buffer;
  J12SAMPARRAY buffer12;
#ifdef D_LOSSLESS_SUPPORTED
  J16SAMPARRAY buffer16;
#endif
  JDIMENSION buffer_height;
};


/*
 * cjpeg/djpeg may need to perform extra passes to convert to or from
 * the source/destination file format.  The JPEG library does not know
 * about these passes, but we'd like them to be counted by the progress
 * monitor.  We use an expanded progress monitor object to hold the
 * additional pass count.
 */

struct cdjpeg_progress_mgr {
  struct jpeg_progress_mgr pub; /* fields known to JPEG library */
  int completed_extra_passes;   /* extra passes completed */
  int total_extra_passes;       /* total extra */
  JDIMENSION max_scans;         /* abort if the number of scans exceeds this
                                   value and the value is non-zero */
  boolean report;               /* whether or not to report progress */
  /* last printed percentage stored here to avoid multiple printouts */
  int percent_done;
};

typedef struct cdjpeg_progress_mgr *cd_progress_ptr;


/* Module selection routines for I/O modules. */

EXTERN(cjpeg_source_ptr) jinit_read_bmp(j_compress_ptr cinfo,
                                        boolean use_inversion_array);
EXTERN(djpeg_dest_ptr) jinit_write_bmp(j_decompress_ptr cinfo, boolean is_os2,
                                       boolean use_inversion_array);
EXTERN(cjpeg_source_ptr) jinit_read_gif(j_compress_ptr cinfo);
EXTERN(djpeg_dest_ptr) jinit_write_gif(j_decompress_ptr cinfo, boolean is_lzw);
EXTERN(djpeg_dest_ptr) j12init_write_gif(j_decompress_ptr cinfo,
                                         boolean is_lzw);
EXTERN(cjpeg_source_ptr) jinit_read_ppm(j_compress_ptr cinfo);
EXTERN(cjpeg_source_ptr) j12init_read_ppm(j_compress_ptr cinfo);
#ifdef C_LOSSLESS_SUPPORTED
EXTERN(cjpeg_source_ptr) j16init_read_ppm(j_compress_ptr cinfo);
#endif
EXTERN(djpeg_dest_ptr) jinit_write_ppm(j_decompress_ptr cinfo);
EXTERN(djpeg_dest_ptr) j12init_write_ppm(j_decompress_ptr cinfo);
#ifdef D_LOSSLESS_SUPPORTED
EXTERN(djpeg_dest_ptr) j16init_write_ppm(j_decompress_ptr cinfo);
#endif
EXTERN(cjpeg_source_ptr) jinit_read_targa(j_compress_ptr cinfo);
EXTERN(djpeg_dest_ptr) jinit_write_targa(j_decompress_ptr cinfo);

/* cjpeg support routines (in rdswitch.c) */

EXTERN(boolean) read_quant_tables(j_compress_ptr cinfo, char *filename,
                                  boolean force_baseline);
EXTERN(boolean) read_scan_script(j_compress_ptr cinfo, char *filename);
EXTERN(boolean) set_quality_ratings(j_compress_ptr cinfo, char *arg,
                                    boolean force_baseline);
EXTERN(boolean) set_quant_slots(j_compress_ptr cinfo, char *arg);
EXTERN(boolean) set_sample_factors(j_compress_ptr cinfo, char *arg);

/* djpeg support routines (in rdcolmap.c) */

EXTERN(void) read_color_map(j_decompress_ptr cinfo, FILE *infile);
EXTERN(void) read_color_map_12(j_decompress_ptr cinfo, FILE *infile);

/* common support routines (in cdjpeg.c) */

EXTERN(void) start_progress_monitor(j_common_ptr cinfo,
                                    cd_progress_ptr progress);
EXTERN(void) end_progress_monitor(j_common_ptr cinfo);
EXTERN(boolean) keymatch(char *arg, const char *keyword, int minchars);
EXTERN(FILE *) read_stdin(void);
EXTERN(FILE *) write_stdout(void);

/* miscellaneous useful macros */

#ifdef DONT_USE_B_MODE          /* define mode parameters for fopen() */
#define READ_BINARY     "r"
#define WRITE_BINARY    "w"
#else
#define READ_BINARY     "rb"
#define WRITE_BINARY    "wb"
#endif

#ifndef EXIT_FAILURE            /* define exit() codes if not provided */
#define EXIT_FAILURE  1
#endif
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS  0
#endif
#ifndef EXIT_WARNING
#define EXIT_WARNING  2
#endif
