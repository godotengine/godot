/*
 * djpeg.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Modified 2013-2019 by Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2010-2011, 2013-2017, 2019-2020, 2022-2024, D. R. Commander.
 * Copyright (C) 2015, Google, Inc.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains a command-line user interface for the JPEG decompressor.
 * It should work on any system with Unix- or MS-DOS-style command lines.
 *
 * Two different command line styles are permitted, depending on the
 * compile-time switch TWO_FILE_COMMANDLINE:
 *      djpeg [options]  inputfile outputfile
 *      djpeg [options]  [inputfile]
 * In the second style, output is always to standard output, which you'd
 * normally redirect to a file or pipe to some other program.  Input is
 * either from a named file or from standard input (typically redirected).
 * The second style is convenient on Unix but is unhelpful on systems that
 * don't support pipes.  Also, you MUST use the first style if your system
 * doesn't do binary I/O to stdin/stdout.
 * To simplify script writing, the "-outfile" switch is provided.  The syntax
 *      djpeg [options]  -outfile outputfile  inputfile
 * works regardless of which command line style is used.
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include "cdjpeg.h"             /* Common decls for cjpeg/djpeg applications */
#include "jversion.h"           /* for version message */
#include "jconfigint.h"

#include <ctype.h>              /* to declare isprint() */


/* Create the add-on message string table. */

#define JMESSAGE(code, string)  string,

static const char * const cdjpeg_message_table[] = {
#include "cderror.h"
  NULL
};


/*
 * This list defines the known output image formats
 * (not all of which need be supported by a given version).
 * You can change the default output format by defining DEFAULT_FMT;
 * indeed, you had better do so if you undefine PPM_SUPPORTED.
 */

typedef enum {
  FMT_BMP,                      /* BMP format (Windows flavor) */
  FMT_GIF,                      /* GIF format (LZW-compressed) */
  FMT_GIF0,                     /* GIF format (uncompressed) */
  FMT_OS2,                      /* BMP format (OS/2 flavor) */
  FMT_PPM,                      /* PPM/PGM (PBMPLUS formats) */
  FMT_TARGA,                    /* Targa format */
  FMT_TIFF                      /* TIFF format */
} IMAGE_FORMATS;

#ifndef DEFAULT_FMT             /* so can override from CFLAGS in Makefile */
#define DEFAULT_FMT     FMT_PPM
#endif

static IMAGE_FORMATS requested_fmt;


/*
 * Argument-parsing code.
 * The switch parser is designed to be useful with DOS-style command line
 * syntax, ie, intermixed switches and file names, where only the switches
 * to the left of a given file name affect processing of that file.
 * The main program in this file doesn't actually use this capability...
 */


static const char *progname;    /* program name for error messages */
static char *icc_filename;      /* for -icc switch */
static JDIMENSION max_scans;    /* for -maxscans switch */
static char *outfilename;       /* for -outfile switch */
static boolean memsrc;          /* for -memsrc switch */
static boolean report;          /* for -report switch */
static boolean skip, crop;
static JDIMENSION skip_start, skip_end;
static JDIMENSION crop_x, crop_y, crop_width, crop_height;
static boolean strict;          /* for -strict switch */
#define INPUT_BUF_SIZE  4096


LOCAL(void)
usage(void)
/* complain about bad command line */
{
  fprintf(stderr, "usage: %s [switches] ", progname);
#ifdef TWO_FILE_COMMANDLINE
  fprintf(stderr, "inputfile outputfile\n");
#else
  fprintf(stderr, "[inputfile]\n");
#endif

  fprintf(stderr, "Switches (names may be abbreviated):\n");
  fprintf(stderr, "  -colors N      Reduce image to no more than N colors [legacy feature]\n");
  fprintf(stderr, "  -fast          Low-quality processing [legacy feature]\n");
  fprintf(stderr, "  -grayscale     Force grayscale output\n");
  fprintf(stderr, "  -rgb           Force RGB output\n");
  fprintf(stderr, "  -rgb565        Force RGB565 output\n");
#ifdef IDCT_SCALING_SUPPORTED
  fprintf(stderr, "  -scale M/N     Scale output image by fraction M/N, eg, 1/8\n");
#endif
#ifdef BMP_SUPPORTED
  fprintf(stderr, "  -bmp           Select BMP output format (Windows style)%s\n",
          (DEFAULT_FMT == FMT_BMP ? " (default)" : ""));
#endif
#ifdef GIF_SUPPORTED
  fprintf(stderr, "  -gif           Select GIF output format (LZW-compressed)%s [legacy feature]\n",
          (DEFAULT_FMT == FMT_GIF ? " (default)" : ""));
  fprintf(stderr, "  -gif0          Select GIF output format (uncompressed)%s [legacy feature]\n",
          (DEFAULT_FMT == FMT_GIF0 ? " (default)" : ""));
#endif
#ifdef BMP_SUPPORTED
  fprintf(stderr, "  -os2           Select BMP output format (OS/2 style)%s [legacy feature]\n",
          (DEFAULT_FMT == FMT_OS2 ? " (default)" : ""));
#endif
#ifdef PPM_SUPPORTED
  fprintf(stderr, "  -pnm           Select PBMPLUS (PPM/PGM) output format%s\n",
          (DEFAULT_FMT == FMT_PPM ? " (default)" : ""));
#endif
#ifdef TARGA_SUPPORTED
  fprintf(stderr, "  -targa         Select Targa output format%s [legacy feature]\n",
          (DEFAULT_FMT == FMT_TARGA ? " (default)" : ""));
#endif
  fprintf(stderr, "Switches for advanced users:\n");
#ifdef DCT_ISLOW_SUPPORTED
  fprintf(stderr, "  -dct int       Use accurate integer DCT method%s\n",
          (JDCT_DEFAULT == JDCT_ISLOW ? " (default)" : ""));
#endif
#ifdef DCT_IFAST_SUPPORTED
  fprintf(stderr, "  -dct fast      Use less accurate integer DCT method [legacy feature]%s\n",
          (JDCT_DEFAULT == JDCT_IFAST ? " (default)" : ""));
#endif
#ifdef DCT_FLOAT_SUPPORTED
  fprintf(stderr, "  -dct float     Use floating-point DCT method [legacy feature]%s\n",
          (JDCT_DEFAULT == JDCT_FLOAT ? " (default)" : ""));
#endif
  fprintf(stderr, "  -dither fs     Use Floyd-Steinberg dithering when quantizing colors (default)\n");
  fprintf(stderr, "                 [legacy feature]\n");
  fprintf(stderr, "  -dither none   Don't use dithering when quantizing colors [legacy feature]\n");
  fprintf(stderr, "  -dither ordered  Use ordered dithering when quantizing colors\n");
  fprintf(stderr, "                   [legacy feature]\n");
  fprintf(stderr, "  -icc FILE      Extract ICC profile to FILE\n");
#ifdef QUANT_2PASS_SUPPORTED
  fprintf(stderr, "  -map FILE      Quantize to colors used in named image file [legacy feature]\n");
#endif
  fprintf(stderr, "  -nosmooth      Use faster, lower-quality upsampling\n");
#ifdef QUANT_1PASS_SUPPORTED
  fprintf(stderr, "  -onepass       Use 1-pass color quantization (low quality) [legacy feature]\n");
#endif
  fprintf(stderr, "  -maxmemory N   Maximum memory to use (in kbytes)\n");
  fprintf(stderr, "  -maxscans N    Maximum number of scans to allow in input file\n");
  fprintf(stderr, "  -outfile name  Specify name for output file\n");
  fprintf(stderr, "  -memsrc        Load input file into memory before decompressing\n");
  fprintf(stderr, "  -report        Report decompression progress\n");
  fprintf(stderr, "  -skip Y0,Y1    Decompress all rows except those between Y0 and Y1 (inclusive)\n");
  fprintf(stderr, "  -crop WxH+X+Y  Decompress only a rectangular subregion of the image\n");
  fprintf(stderr, "                 [requires PBMPLUS (PPM/PGM), GIF, or Targa output format]\n");
  fprintf(stderr, "  -strict        Treat all warnings as fatal\n");
  fprintf(stderr, "  -verbose  or  -debug   Emit debug output\n");
  fprintf(stderr, "  -version       Print version information and exit\n");
  exit(EXIT_FAILURE);
}


LOCAL(int)
parse_switches(j_decompress_ptr cinfo, int argc, char **argv,
               int last_file_arg_seen, boolean for_real)
/* Parse optional switches.
 * Returns argv[] index of first file-name argument (== argc if none).
 * Any file names with indexes <= last_file_arg_seen are ignored;
 * they have presumably been processed in a previous iteration.
 * (Pass 0 for last_file_arg_seen on the first or only iteration.)
 * for_real is FALSE on the first (dummy) pass; we may skip any expensive
 * processing.
 */
{
  int argn;
  char *arg;

  /* Set up default JPEG parameters. */
  requested_fmt = DEFAULT_FMT;  /* set default output file format */
  icc_filename = NULL;
  max_scans = 0;
  outfilename = NULL;
  memsrc = FALSE;
  report = FALSE;
  skip = FALSE;
  crop = FALSE;
  strict = FALSE;
  cinfo->err->trace_level = 0;

  /* Scan command line options, adjust parameters */

  for (argn = 1; argn < argc; argn++) {
    arg = argv[argn];
    if (*arg != '-') {
      /* Not a switch, must be a file name argument */
      if (argn <= last_file_arg_seen) {
        outfilename = NULL;     /* -outfile applies to just one input file */
        continue;               /* ignore this name if previously processed */
      }
      break;                    /* else done parsing switches */
    }
    arg++;                      /* advance past switch marker character */

    if (keymatch(arg, "bmp", 1)) {
      /* BMP output format (Windows flavor). */
      requested_fmt = FMT_BMP;

    } else if (keymatch(arg, "colors", 1) || keymatch(arg, "colours", 1) ||
               keymatch(arg, "quantize", 1) || keymatch(arg, "quantise", 1)) {
      /* Do color quantization. */
      int val;

      if (++argn >= argc)       /* advance to next argument */
        usage();
      if (sscanf(argv[argn], "%d", &val) != 1)
        usage();
      cinfo->desired_number_of_colors = val;
      cinfo->quantize_colors = TRUE;

    } else if (keymatch(arg, "dct", 2)) {
      /* Select IDCT algorithm. */
      if (++argn >= argc)       /* advance to next argument */
        usage();
      if (keymatch(argv[argn], "int", 1)) {
        cinfo->dct_method = JDCT_ISLOW;
      } else if (keymatch(argv[argn], "fast", 2)) {
        cinfo->dct_method = JDCT_IFAST;
      } else if (keymatch(argv[argn], "float", 2)) {
        cinfo->dct_method = JDCT_FLOAT;
      } else
        usage();

    } else if (keymatch(arg, "dither", 2)) {
      /* Select dithering algorithm. */
      if (++argn >= argc)       /* advance to next argument */
        usage();
      if (keymatch(argv[argn], "fs", 2)) {
        cinfo->dither_mode = JDITHER_FS;
      } else if (keymatch(argv[argn], "none", 2)) {
        cinfo->dither_mode = JDITHER_NONE;
      } else if (keymatch(argv[argn], "ordered", 2)) {
        cinfo->dither_mode = JDITHER_ORDERED;
      } else
        usage();

    } else if (keymatch(arg, "debug", 1) || keymatch(arg, "verbose", 1)) {
      /* Enable debug printouts. */
      /* On first -d, print version identification */
      static boolean printed_version = FALSE;

      if (!printed_version) {
        fprintf(stderr, "%s version %s (build %s)\n",
                PACKAGE_NAME, VERSION, BUILD);
        fprintf(stderr, JCOPYRIGHT1);
        fprintf(stderr, JCOPYRIGHT2 "\n");
        fprintf(stderr, "Emulating The Independent JPEG Group's software, version %s\n\n",
                JVERSION);
        printed_version = TRUE;
      }
      cinfo->err->trace_level++;

    } else if (keymatch(arg, "version", 4)) {
      fprintf(stderr, "%s version %s (build %s)\n",
              PACKAGE_NAME, VERSION, BUILD);
      exit(EXIT_SUCCESS);

    } else if (keymatch(arg, "fast", 1)) {
      /* Select recommended processing options for quick-and-dirty output. */
      cinfo->two_pass_quantize = FALSE;
      cinfo->dither_mode = JDITHER_ORDERED;
      if (!cinfo->quantize_colors) /* don't override an earlier -colors */
        cinfo->desired_number_of_colors = 216;
      cinfo->dct_method = JDCT_FASTEST;
      cinfo->do_fancy_upsampling = FALSE;

    } else if (keymatch(arg, "gif", 1)) {
      /* GIF output format (LZW-compressed). */
      requested_fmt = FMT_GIF;

    } else if (keymatch(arg, "gif0", 4)) {
      /* GIF output format (uncompressed). */
      requested_fmt = FMT_GIF0;

    } else if (keymatch(arg, "grayscale", 2) ||
               keymatch(arg, "greyscale", 2)) {
      /* Force monochrome output. */
      cinfo->out_color_space = JCS_GRAYSCALE;

    } else if (keymatch(arg, "rgb", 2)) {
      /* Force RGB output. */
      cinfo->out_color_space = JCS_RGB;

    } else if (keymatch(arg, "rgb565", 2)) {
      /* Force RGB565 output. */
      cinfo->out_color_space = JCS_RGB565;

    } else if (keymatch(arg, "icc", 1)) {
      /* Set ICC filename. */
      if (++argn >= argc)       /* advance to next argument */
        usage();
      icc_filename = argv[argn];
#ifdef SAVE_MARKERS_SUPPORTED
      jpeg_save_markers(cinfo, JPEG_APP0 + 2, 0xFFFF);
#endif

    } else if (keymatch(arg, "map", 3)) {
      /* Quantize to a color map taken from an input file. */
      if (++argn >= argc)       /* advance to next argument */
        usage();
      if (for_real) {           /* too expensive to do twice! */
#ifdef QUANT_2PASS_SUPPORTED    /* otherwise can't quantize to supplied map */
        FILE *mapfile;

        if ((mapfile = fopen(argv[argn], READ_BINARY)) == NULL) {
          fprintf(stderr, "%s: can't open %s\n", progname, argv[argn]);
          exit(EXIT_FAILURE);
        }
        if (cinfo->data_precision == 12)
          read_color_map_12(cinfo, mapfile);
        else
          read_color_map(cinfo, mapfile);
        fclose(mapfile);
        cinfo->quantize_colors = TRUE;
#else
        ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
      }

    } else if (keymatch(arg, "maxmemory", 3)) {
      /* Maximum memory in Kb (or Mb with 'm'). */
      long lval;
      char ch = 'x';

      if (++argn >= argc)       /* advance to next argument */
        usage();
      if (sscanf(argv[argn], "%ld%c", &lval, &ch) < 1)
        usage();
      if (ch == 'm' || ch == 'M')
        lval *= 1000L;
      cinfo->mem->max_memory_to_use = lval * 1000L;

    } else if (keymatch(arg, "maxscans", 4)) {
      if (++argn >= argc)       /* advance to next argument */
        usage();
      if (sscanf(argv[argn], "%u", &max_scans) != 1)
        usage();

    } else if (keymatch(arg, "nosmooth", 3)) {
      /* Suppress fancy upsampling */
      cinfo->do_fancy_upsampling = FALSE;

    } else if (keymatch(arg, "onepass", 3)) {
      /* Use fast one-pass quantization. */
      cinfo->two_pass_quantize = FALSE;

    } else if (keymatch(arg, "os2", 3)) {
      /* BMP output format (OS/2 flavor). */
      requested_fmt = FMT_OS2;

    } else if (keymatch(arg, "outfile", 4)) {
      /* Set output file name. */
      if (++argn >= argc)       /* advance to next argument */
        usage();
      outfilename = argv[argn]; /* save it away for later use */

    } else if (keymatch(arg, "memsrc", 2)) {
      /* Use in-memory source manager */
      memsrc = TRUE;

    } else if (keymatch(arg, "pnm", 1) || keymatch(arg, "ppm", 1)) {
      /* PPM/PGM output format. */
      requested_fmt = FMT_PPM;

    } else if (keymatch(arg, "report", 2)) {
      report = TRUE;

    } else if (keymatch(arg, "scale", 1)) {
      /* Scale the output image by a fraction M/N. */
      if (++argn >= argc)       /* advance to next argument */
        usage();
      if (sscanf(argv[argn], "%u/%u",
                 &cinfo->scale_num, &cinfo->scale_denom) != 2)
        usage();

    } else if (keymatch(arg, "skip", 2)) {
      int temp_start = -1, temp_end = -1;
      if (++argn >= argc)
        usage();
      if (sscanf(argv[argn], "%d,%d", &temp_start, &temp_end) != 2 ||
          temp_start < 0 || temp_end < 0 || temp_start > temp_end)
        usage();
      skip = TRUE;
      skip_start = temp_start;
      skip_end = temp_end;

    } else if (keymatch(arg, "crop", 2)) {
      int temp_width = -1, temp_height = -1, temp_x = -1, temp_y = -1;
      char c;
      if (++argn >= argc)
        usage();
      if (sscanf(argv[argn], "%d%c%d+%d+%d", &temp_width, &c, &temp_height,
                 &temp_x, &temp_y) != 5 ||
          (c != 'X' && c != 'x') || temp_width < 1 || temp_height < 1 ||
          temp_x < 0 || temp_y < 0)
        usage();
      crop = TRUE;
      crop_width = temp_width;
      crop_height = temp_height;
      crop_x = temp_x;
      crop_y = temp_y;

    } else if (keymatch(arg, "strict", 2)) {
      strict = TRUE;

    } else if (keymatch(arg, "targa", 1)) {
      /* Targa output format. */
      requested_fmt = FMT_TARGA;

    } else {
      usage();                  /* bogus switch */
    }
  }

  return argn;                  /* return index of next arg (file name) */
}


/*
 * Marker processor for COM and interesting APPn markers.
 * This replaces the library's built-in processor, which just skips the marker.
 * We want to print out the marker as text, to the extent possible.
 * Note this code relies on a non-suspending data source.
 */

LOCAL(unsigned int)
jpeg_getc(j_decompress_ptr cinfo)
/* Read next byte */
{
  struct jpeg_source_mgr *datasrc = cinfo->src;

  if (datasrc->bytes_in_buffer == 0) {
    if (!(*datasrc->fill_input_buffer) (cinfo))
      ERREXIT(cinfo, JERR_CANT_SUSPEND);
  }
  datasrc->bytes_in_buffer--;
  return *datasrc->next_input_byte++;
}


METHODDEF(boolean)
print_text_marker(j_decompress_ptr cinfo)
{
  boolean traceit = (cinfo->err->trace_level >= 1);
  long length;
  unsigned int ch;
  unsigned int lastch = 0;

  length = jpeg_getc(cinfo) << 8;
  length += jpeg_getc(cinfo);
  length -= 2;                  /* discount the length word itself */

  if (traceit) {
    if (cinfo->unread_marker == JPEG_COM)
      fprintf(stderr, "Comment, length %ld:\n", (long)length);
    else                        /* assume it is an APPn otherwise */
      fprintf(stderr, "APP%d, length %ld:\n",
              cinfo->unread_marker - JPEG_APP0, (long)length);
  }

  while (--length >= 0) {
    ch = jpeg_getc(cinfo);
    if (traceit) {
      /* Emit the character in a readable form.
       * Nonprintables are converted to \nnn form,
       * while \ is converted to \\.
       * Newlines in CR, CR/LF, or LF form will be printed as one newline.
       */
      if (ch == '\r') {
        fprintf(stderr, "\n");
      } else if (ch == '\n') {
        if (lastch != '\r')
          fprintf(stderr, "\n");
      } else if (ch == '\\') {
        fprintf(stderr, "\\\\");
      } else if (isprint(ch)) {
        putc(ch, stderr);
      } else {
        fprintf(stderr, "\\%03o", ch);
      }
      lastch = ch;
    }
  }

  if (traceit)
    fprintf(stderr, "\n");

  return TRUE;
}


METHODDEF(void)
my_emit_message(j_common_ptr cinfo, int msg_level)
{
  if (msg_level < 0) {
    /* Treat warning as fatal */
    cinfo->err->error_exit(cinfo);
  } else {
    if (cinfo->err->trace_level >= msg_level)
      cinfo->err->output_message(cinfo);
  }
}


/*
 * The main program.
 */

int
main(int argc, char **argv)
{
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  struct cdjpeg_progress_mgr progress;
  int file_index;
  djpeg_dest_ptr dest_mgr = NULL;
  FILE *input_file;
  FILE *output_file;
  unsigned char *inbuffer = NULL;
  unsigned long insize = 0;
  JDIMENSION num_scanlines;

  progname = argv[0];
  if (progname == NULL || progname[0] == 0)
    progname = "djpeg";         /* in case C library doesn't provide it */

  /* Initialize the JPEG decompression object with default error handling. */
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  /* Add some application-specific error messages (from cderror.h) */
  jerr.addon_message_table = cdjpeg_message_table;
  jerr.first_addon_message = JMSG_FIRSTADDONCODE;
  jerr.last_addon_message = JMSG_LASTADDONCODE;

  /* Insert custom marker processor for COM and APP12.
   * APP12 is used by some digital camera makers for textual info,
   * so we provide the ability to display it as text.
   * If you like, additional APPn marker types can be selected for display,
   * but don't try to override APP0 or APP14 this way (see libjpeg.txt).
   */
  jpeg_set_marker_processor(&cinfo, JPEG_COM, print_text_marker);
  jpeg_set_marker_processor(&cinfo, JPEG_APP0 + 12, print_text_marker);

  /* Scan command line to find file names. */
  /* It is convenient to use just one switch-parsing routine, but the switch
   * values read here are ignored; we will rescan the switches after opening
   * the input file.
   * (Exception: tracing level set here controls verbosity for COM markers
   * found during jpeg_read_header...)
   */

  file_index = parse_switches(&cinfo, argc, argv, 0, FALSE);

  if (strict)
    jerr.emit_message = my_emit_message;

#ifdef TWO_FILE_COMMANDLINE
  /* Must have either -outfile switch or explicit output file name */
  if (outfilename == NULL) {
    if (file_index != argc - 2) {
      fprintf(stderr, "%s: must name one input and one output file\n",
              progname);
      usage();
    }
    outfilename = argv[file_index + 1];
  } else {
    if (file_index != argc - 1) {
      fprintf(stderr, "%s: must name one input and one output file\n",
              progname);
      usage();
    }
  }
#else
  /* Unix style: expect zero or one file name */
  if (file_index < argc - 1) {
    fprintf(stderr, "%s: only one input file\n", progname);
    usage();
  }
#endif /* TWO_FILE_COMMANDLINE */

  /* Open the input file. */
  if (file_index < argc) {
    if ((input_file = fopen(argv[file_index], READ_BINARY)) == NULL) {
      fprintf(stderr, "%s: can't open %s\n", progname, argv[file_index]);
      exit(EXIT_FAILURE);
    }
  } else {
    /* default input file is stdin */
    input_file = read_stdin();
  }

  /* Open the output file. */
  if (outfilename != NULL) {
    if ((output_file = fopen(outfilename, WRITE_BINARY)) == NULL) {
      fprintf(stderr, "%s: can't open %s\n", progname, outfilename);
      exit(EXIT_FAILURE);
    }
  } else {
    /* default output file is stdout */
    output_file = write_stdout();
  }

  if (report || max_scans != 0) {
    start_progress_monitor((j_common_ptr)&cinfo, &progress);
    progress.report = report;
    progress.max_scans = max_scans;
  }

  /* Specify data source for decompression */
  if (memsrc) {
    size_t nbytes;
    do {
      inbuffer = (unsigned char *)realloc(inbuffer, insize + INPUT_BUF_SIZE);
      if (inbuffer == NULL) {
        fprintf(stderr, "%s: memory allocation failure\n", progname);
        exit(EXIT_FAILURE);
      }
      nbytes = fread(&inbuffer[insize], 1, INPUT_BUF_SIZE, input_file);
      if (nbytes < INPUT_BUF_SIZE && ferror(input_file)) {
        if (file_index < argc)
          fprintf(stderr, "%s: can't read from %s\n", progname,
                  argv[file_index]);
        else
          fprintf(stderr, "%s: can't read from stdin\n", progname);
      }
      insize += (unsigned long)nbytes;
    } while (nbytes == INPUT_BUF_SIZE);
    fprintf(stderr, "Compressed size:  %lu bytes\n", insize);
    jpeg_mem_src(&cinfo, inbuffer, insize);
  } else
    jpeg_stdio_src(&cinfo, input_file);

  /* Read file header, set default decompression parameters */
  (void)jpeg_read_header(&cinfo, TRUE);

  /* Adjust default decompression parameters by re-parsing the options */
  file_index = parse_switches(&cinfo, argc, argv, 0, TRUE);

  /* Initialize the output module now to let it override any crucial
   * option settings (for instance, GIF wants to force color quantization).
   */
  switch (requested_fmt) {
#ifdef BMP_SUPPORTED
  case FMT_BMP:
    dest_mgr = jinit_write_bmp(&cinfo, FALSE, TRUE);
    break;
  case FMT_OS2:
    dest_mgr = jinit_write_bmp(&cinfo, TRUE, TRUE);
    break;
#endif
#ifdef GIF_SUPPORTED
  case FMT_GIF:
    if (cinfo.data_precision == 8)
      dest_mgr = jinit_write_gif(&cinfo, TRUE);
    else if (cinfo.data_precision == 12)
      dest_mgr = j12init_write_gif(&cinfo, TRUE);
    else
      ERREXIT1(&cinfo, JERR_BAD_PRECISION, cinfo.data_precision);
    break;
  case FMT_GIF0:
    dest_mgr = jinit_write_gif(&cinfo, FALSE);
    break;
#endif
#ifdef PPM_SUPPORTED
  case FMT_PPM:
    if (cinfo.data_precision <= 8)
      dest_mgr = jinit_write_ppm(&cinfo);
    else if (cinfo.data_precision <= 12)
      dest_mgr = j12init_write_ppm(&cinfo);
    else
#ifdef D_LOSSLESS_SUPPORTED
      dest_mgr = j16init_write_ppm(&cinfo);
#else
      ERREXIT1(&cinfo, JERR_BAD_PRECISION, cinfo.data_precision);
#endif
    break;
#endif
#ifdef TARGA_SUPPORTED
  case FMT_TARGA:
    dest_mgr = jinit_write_targa(&cinfo);
    break;
#endif
  default:
    ERREXIT(&cinfo, JERR_UNSUPPORTED_FORMAT);
    break;
  }
  dest_mgr->output_file = output_file;

  /* Start decompressor */
  (void)jpeg_start_decompress(&cinfo);

  /* Skip rows */
  if (skip) {
    JDIMENSION tmp;

    /* Check for valid skip_end.  We cannot check this value until after
     * jpeg_start_decompress() is called.  Note that we have already verified
     * that skip_start <= skip_end.
     */
    if (skip_end > cinfo.output_height - 1) {
      fprintf(stderr, "%s: skip region exceeds image height %u\n", progname,
              cinfo.output_height);
      exit(EXIT_FAILURE);
    }

    /* Write output file header.  This is a hack to ensure that the destination
     * manager creates an output image of the proper size.
     */
    tmp = cinfo.output_height;
    cinfo.output_height -= (skip_end - skip_start + 1);
    (*dest_mgr->start_output) (&cinfo, dest_mgr);
    cinfo.output_height = tmp;

    if (cinfo.data_precision == 8) {
      /* Process data */
      while (cinfo.output_scanline < skip_start) {
        num_scanlines = jpeg_read_scanlines(&cinfo, dest_mgr->buffer,
                                            dest_mgr->buffer_height);
        (*dest_mgr->put_pixel_rows) (&cinfo, dest_mgr, num_scanlines);
      }
      if ((tmp = jpeg_skip_scanlines(&cinfo, skip_end - skip_start + 1)) !=
          skip_end - skip_start + 1) {
        fprintf(stderr, "%s: jpeg_skip_scanlines() returned %u rather than %u\n",
                progname, tmp, skip_end - skip_start + 1);
        exit(EXIT_FAILURE);
      }
      while (cinfo.output_scanline < cinfo.output_height) {
        num_scanlines = jpeg_read_scanlines(&cinfo, dest_mgr->buffer,
                                            dest_mgr->buffer_height);
        (*dest_mgr->put_pixel_rows) (&cinfo, dest_mgr, num_scanlines);
      }
    } else if (cinfo.data_precision == 12) {
      /* Process data */
      while (cinfo.output_scanline < skip_start) {
        num_scanlines = jpeg12_read_scanlines(&cinfo, dest_mgr->buffer12,
                                              dest_mgr->buffer_height);
        (*dest_mgr->put_pixel_rows) (&cinfo, dest_mgr, num_scanlines);
      }
      if ((tmp = jpeg12_skip_scanlines(&cinfo, skip_end - skip_start + 1)) !=
          skip_end - skip_start + 1) {
        fprintf(stderr, "%s: jpeg12_skip_scanlines() returned %u rather than %u\n",
                progname, tmp, skip_end - skip_start + 1);
        exit(EXIT_FAILURE);
      }
      while (cinfo.output_scanline < cinfo.output_height) {
        num_scanlines = jpeg12_read_scanlines(&cinfo, dest_mgr->buffer12,
                                              dest_mgr->buffer_height);
        (*dest_mgr->put_pixel_rows) (&cinfo, dest_mgr, num_scanlines);
      }
    } else
      ERREXIT(&cinfo, JERR_NOTIMPL);

  /* Decompress a subregion */
  } else if (crop) {
    JDIMENSION tmp;

    /* Check for valid crop dimensions.  We cannot check these values until
     * after jpeg_start_decompress() is called.
     */
    if ((unsigned long long)crop_x + crop_width > cinfo.output_width ||
        (unsigned long long)crop_y + crop_height > cinfo.output_height) {
      fprintf(stderr, "%s: crop dimensions exceed image dimensions %u x %u\n",
              progname, cinfo.output_width, cinfo.output_height);
      exit(EXIT_FAILURE);
    }

    if (cinfo.data_precision == 8)
      jpeg_crop_scanline(&cinfo, &crop_x, &crop_width);
    else if (cinfo.data_precision == 12)
      jpeg12_crop_scanline(&cinfo, &crop_x, &crop_width);
    else
      ERREXIT(&cinfo, JERR_NOTIMPL);
    if (dest_mgr->calc_buffer_dimensions)
      (*dest_mgr->calc_buffer_dimensions) (&cinfo, dest_mgr);
    else
      ERREXIT(&cinfo, JERR_UNSUPPORTED_FORMAT);

    /* Write output file header.  This is a hack to ensure that the destination
     * manager creates an output image of the proper size.
     */
    tmp = cinfo.output_height;
    cinfo.output_height = crop_height;
    (*dest_mgr->start_output) (&cinfo, dest_mgr);
    cinfo.output_height = tmp;

    if (cinfo.data_precision == 8) {
      /* Process data */
      if ((tmp = jpeg_skip_scanlines(&cinfo, crop_y)) != crop_y) {
        fprintf(stderr, "%s: jpeg_skip_scanlines() returned %u rather than %u\n",
                progname, tmp, crop_y);
        exit(EXIT_FAILURE);
      }
      while (cinfo.output_scanline < crop_y + crop_height) {
        num_scanlines = jpeg_read_scanlines(&cinfo, dest_mgr->buffer,
                                            dest_mgr->buffer_height);
        (*dest_mgr->put_pixel_rows) (&cinfo, dest_mgr, num_scanlines);
      }
      if ((tmp =
           jpeg_skip_scanlines(&cinfo,
                               cinfo.output_height - crop_y - crop_height)) !=
          cinfo.output_height - crop_y - crop_height) {
        fprintf(stderr, "%s: jpeg_skip_scanlines() returned %u rather than %u\n",
                progname, tmp, cinfo.output_height - crop_y - crop_height);
        exit(EXIT_FAILURE);
      }
    } else if (cinfo.data_precision == 12) {
      /* Process data */
      if ((tmp = jpeg12_skip_scanlines(&cinfo, crop_y)) != crop_y) {
        fprintf(stderr, "%s: jpeg12_skip_scanlines() returned %u rather than %u\n",
                progname, tmp, crop_y);
        exit(EXIT_FAILURE);
      }
      while (cinfo.output_scanline < crop_y + crop_height) {
        num_scanlines = jpeg12_read_scanlines(&cinfo, dest_mgr->buffer12,
                                              dest_mgr->buffer_height);
        (*dest_mgr->put_pixel_rows) (&cinfo, dest_mgr, num_scanlines);
      }
      if ((tmp =
           jpeg12_skip_scanlines(&cinfo, cinfo.output_height - crop_y -
                                         crop_height)) !=
          cinfo.output_height - crop_y - crop_height) {
        fprintf(stderr, "%s: jpeg12_skip_scanlines() returned %u rather than %u\n",
                progname, tmp, cinfo.output_height - crop_y - crop_height);
        exit(EXIT_FAILURE);
      }
    } else
      ERREXIT(&cinfo, JERR_NOTIMPL);

  /* Normal full-image decompress */
  } else {
    /* Write output file header */
    (*dest_mgr->start_output) (&cinfo, dest_mgr);

    if (cinfo.data_precision <= 8) {
      /* Process data */
      while (cinfo.output_scanline < cinfo.output_height) {
        num_scanlines = jpeg_read_scanlines(&cinfo, dest_mgr->buffer,
                                            dest_mgr->buffer_height);
        (*dest_mgr->put_pixel_rows) (&cinfo, dest_mgr, num_scanlines);
      }
    } else if (cinfo.data_precision <= 12) {
      /* Process data */
      while (cinfo.output_scanline < cinfo.output_height) {
        num_scanlines = jpeg12_read_scanlines(&cinfo, dest_mgr->buffer12,
                                              dest_mgr->buffer_height);
        (*dest_mgr->put_pixel_rows) (&cinfo, dest_mgr, num_scanlines);
      }
    } else {
#ifdef D_LOSSLESS_SUPPORTED
      /* Process data */
      while (cinfo.output_scanline < cinfo.output_height) {
        num_scanlines = jpeg16_read_scanlines(&cinfo, dest_mgr->buffer16,
                                              dest_mgr->buffer_height);
        (*dest_mgr->put_pixel_rows) (&cinfo, dest_mgr, num_scanlines);
      }
#else
      ERREXIT1(&cinfo, JERR_BAD_PRECISION, cinfo.data_precision);
#endif
    }
  }

  /* Hack: count final pass as done in case finish_output does an extra pass.
   * The library won't have updated completed_passes.
   */
  if (report || max_scans != 0)
    progress.pub.completed_passes = progress.pub.total_passes;

  if (icc_filename != NULL) {
    FILE *icc_file;
    JOCTET *icc_profile;
    unsigned int icc_len;

    if ((icc_file = fopen(icc_filename, WRITE_BINARY)) == NULL) {
      fprintf(stderr, "%s: can't open %s\n", progname, icc_filename);
      exit(EXIT_FAILURE);
    }
    if (jpeg_read_icc_profile(&cinfo, &icc_profile, &icc_len)) {
      if (fwrite(icc_profile, icc_len, 1, icc_file) < 1) {
        fprintf(stderr, "%s: can't read ICC profile from %s\n", progname,
                icc_filename);
        free(icc_profile);
        fclose(icc_file);
        exit(EXIT_FAILURE);
      }
      free(icc_profile);
      fclose(icc_file);
    } else if (cinfo.err->msg_code != JWRN_BOGUS_ICC)
      fprintf(stderr, "%s: no ICC profile data in JPEG file\n", progname);
  }

  /* Finish decompression and release memory.
   * I must do it in this order because output module has allocated memory
   * of lifespan JPOOL_IMAGE; it needs to finish before releasing memory.
   */
  (*dest_mgr->finish_output) (&cinfo, dest_mgr);
  (void)jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  /* Close files, if we opened them */
  if (input_file != stdin)
    fclose(input_file);
  if (output_file != stdout)
    fclose(output_file);

  if (report || max_scans != 0)
    end_progress_monitor((j_common_ptr)&cinfo);

  if (memsrc)
    free(inbuffer);

  /* All done. */
  exit(jerr.num_warnings ? EXIT_WARNING : EXIT_SUCCESS);
  return 0;                     /* suppress no-return-value warnings */
}
