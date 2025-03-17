/*
 * cjpeg.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1998, Thomas G. Lane.
 * Modified 2003-2011 by Guido Vollbeding.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2010, 2013-2014, 2017, 2019-2022, 2024, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains a command-line user interface for the JPEG compressor.
 * It should work on any system with Unix- or MS-DOS-style command lines.
 *
 * Two different command line styles are permitted, depending on the
 * compile-time switch TWO_FILE_COMMANDLINE:
 *      cjpeg [options]  inputfile outputfile
 *      cjpeg [options]  [inputfile]
 * In the second style, output is always to standard output, which you'd
 * normally redirect to a file or pipe to some other program.  Input is
 * either from a named file or from standard input (typically redirected).
 * The second style is convenient on Unix but is unhelpful on systems that
 * don't support pipes.  Also, you MUST use the first style if your system
 * doesn't do binary I/O to stdin/stdout.
 * To simplify script writing, the "-outfile" switch is provided.  The syntax
 *      cjpeg [options]  -outfile outputfile  inputfile
 * works regardless of which command line style is used.
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#ifdef CJPEG_FUZZER
#define JPEG_INTERNALS
#endif
#include "cdjpeg.h"             /* Common decls for cjpeg/djpeg applications */
#include "jversion.h"           /* for version message */
#include "jconfigint.h"


/* Create the add-on message string table. */

#define JMESSAGE(code, string)  string,

static const char * const cdjpeg_message_table[] = {
#include "cderror.h"
  NULL
};


/*
 * This routine determines what format the input file is,
 * and selects the appropriate input-reading module.
 *
 * To determine which family of input formats the file belongs to,
 * we may look only at the first byte of the file, since C does not
 * guarantee that more than one character can be pushed back with ungetc.
 * Looking at additional bytes would require one of these approaches:
 *     1) assume we can fseek() the input file (fails for piped input);
 *     2) assume we can push back more than one character (works in
 *        some C implementations, but unportable);
 *     3) provide our own buffering (breaks input readers that want to use
 *        stdio directly);
 * or  4) don't put back the data, and modify the input_init methods to assume
 *        they start reading after the start of file.
 * #1 is attractive for MS-DOS but is untenable on Unix.
 *
 * The most portable solution for file types that can't be identified by their
 * first byte is to make the user tell us what they are.  This is also the
 * only approach for "raw" file types that contain only arbitrary values.
 * We presently apply this method for Targa files.  Most of the time Targa
 * files start with 0x00, so we recognize that case.  Potentially, however,
 * a Targa file could start with any byte value (byte 0 is the length of the
 * seldom-used ID field), so we provide a switch to force Targa input mode.
 */

static boolean is_targa;        /* records user -targa switch */


LOCAL(cjpeg_source_ptr)
select_file_type(j_compress_ptr cinfo, FILE *infile)
{
  int c;

  if (is_targa) {
#ifdef TARGA_SUPPORTED
    return jinit_read_targa(cinfo);
#else
    ERREXIT(cinfo, JERR_TGA_NOTCOMP);
#endif
  }

  if ((c = getc(infile)) == EOF)
    ERREXIT(cinfo, JERR_INPUT_EMPTY);
  if (ungetc(c, infile) == EOF)
    ERREXIT(cinfo, JERR_UNGETC_FAILED);

  switch (c) {
#ifdef BMP_SUPPORTED
  case 'B':
    return jinit_read_bmp(cinfo, TRUE);
#endif
#ifdef GIF_SUPPORTED
  case 'G':
    return jinit_read_gif(cinfo);
#endif
#ifdef PPM_SUPPORTED
  case 'P':
    if (cinfo->data_precision <= 8)
      return jinit_read_ppm(cinfo);
    else if (cinfo->data_precision <= 12)
      return j12init_read_ppm(cinfo);
    else {
#ifdef C_LOSSLESS_SUPPORTED
      return j16init_read_ppm(cinfo);
#else
      ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);
      break;
#endif
    }
#endif
#ifdef TARGA_SUPPORTED
  case 0x00:
    return jinit_read_targa(cinfo);
#endif
  default:
    ERREXIT(cinfo, JERR_UNKNOWN_FORMAT);
    break;
  }

  return NULL;                  /* suppress compiler warnings */
}


/*
 * Argument-parsing code.
 * The switch parser is designed to be useful with DOS-style command line
 * syntax, ie, intermixed switches and file names, where only the switches
 * to the left of a given file name affect processing of that file.
 * The main program in this file doesn't actually use this capability...
 */


static const char *progname;    /* program name for error messages */
static char *icc_filename;      /* for -icc switch */
static char *outfilename;       /* for -outfile switch */
static boolean memdst;          /* for -memdst switch */
static boolean report;          /* for -report switch */
static boolean strict;          /* for -strict switch */


#ifdef CJPEG_FUZZER

#include <setjmp.h>

struct my_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
};

void my_error_exit(j_common_ptr cinfo)
{
  struct my_error_mgr *myerr = (struct my_error_mgr *)cinfo->err;

  longjmp(myerr->setjmp_buffer, 1);
}

static void my_emit_message_fuzzer(j_common_ptr cinfo, int msg_level)
{
  if (msg_level < 0)
    cinfo->err->num_warnings++;
}

#define HANDLE_ERROR() { \
  if (cinfo.global_state > CSTATE_START) { \
    if (memdst && outbuffer) \
      (*cinfo.dest->term_destination) (&cinfo); \
    jpeg_abort_compress(&cinfo); \
  } \
  jpeg_destroy_compress(&cinfo); \
  if (input_file != stdin && input_file != NULL) \
    fclose(input_file); \
  if (memdst) \
    free(outbuffer); \
  return EXIT_FAILURE; \
}

#endif


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
  fprintf(stderr, "  -quality N[,...]   Compression quality (0..100; 5-95 is most useful range,\n");
  fprintf(stderr, "                     default is 75)\n");
  fprintf(stderr, "  -grayscale     Create monochrome JPEG file\n");
  fprintf(stderr, "  -rgb           Create RGB JPEG file\n");
#ifdef ENTROPY_OPT_SUPPORTED
  fprintf(stderr, "  -optimize      Optimize Huffman table (smaller file, but slow compression)\n");
#endif
#ifdef C_PROGRESSIVE_SUPPORTED
  fprintf(stderr, "  -progressive   Create progressive JPEG file\n");
#endif
#ifdef TARGA_SUPPORTED
  fprintf(stderr, "  -targa         Input file is Targa format (usually not needed)\n");
#endif
  fprintf(stderr, "Switches for advanced users:\n");
  fprintf(stderr, "  -precision N   Create JPEG file with N-bit data precision\n");
#ifdef C_LOSSLESS_SUPPORTED
  fprintf(stderr, "                 (N=2..16; default is 8; if N is not 8 or 12, then -lossless\n");
  fprintf(stderr, "                 must also be specified)\n");
#else
  fprintf(stderr, "                 (N is 8 or 12; default is 8)\n");
#endif
#ifdef C_LOSSLESS_SUPPORTED
  fprintf(stderr, "  -lossless psv[,Pt]  Create lossless JPEG file\n");
#endif
#ifdef C_ARITH_CODING_SUPPORTED
  fprintf(stderr, "  -arithmetic    Use arithmetic coding\n");
#endif
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
  fprintf(stderr, "  -icc FILE      Embed ICC profile contained in FILE\n");
  fprintf(stderr, "  -restart N     Set restart interval in rows, or in blocks with B\n");
#ifdef INPUT_SMOOTHING_SUPPORTED
  fprintf(stderr, "  -smooth N      Smooth dithered input (N=1..100 is strength)\n");
#endif
  fprintf(stderr, "  -maxmemory N   Maximum memory to use (in kbytes)\n");
  fprintf(stderr, "  -outfile name  Specify name for output file\n");
  fprintf(stderr, "  -memdst        Compress to memory instead of file (useful for benchmarking)\n");
  fprintf(stderr, "  -report        Report compression progress\n");
  fprintf(stderr, "  -strict        Treat all warnings as fatal\n");
  fprintf(stderr, "  -verbose  or  -debug   Emit debug output\n");
  fprintf(stderr, "  -version       Print version information and exit\n");
  fprintf(stderr, "Switches for wizards:\n");
  fprintf(stderr, "  -baseline      Force baseline quantization tables\n");
  fprintf(stderr, "  -qtables FILE  Use quantization tables given in FILE\n");
  fprintf(stderr, "  -qslots N[,...]    Set component quantization tables\n");
  fprintf(stderr, "  -sample HxV[,...]  Set component sampling factors\n");
#ifdef C_MULTISCAN_FILES_SUPPORTED
  fprintf(stderr, "  -scans FILE    Create multi-scan JPEG per script FILE\n");
#endif
  exit(EXIT_FAILURE);
}


LOCAL(int)
parse_switches(j_compress_ptr cinfo, int argc, char **argv,
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
#ifdef C_LOSSLESS_SUPPORTED
  int psv = 0, pt = 0;
#endif
  boolean force_baseline;
  boolean simple_progressive;
  char *qualityarg = NULL;      /* saves -quality parm if any */
  char *qtablefile = NULL;      /* saves -qtables filename if any */
  char *qslotsarg = NULL;       /* saves -qslots parm if any */
  char *samplearg = NULL;       /* saves -sample parm if any */
  char *scansarg = NULL;        /* saves -scans parm if any */

  /* Set up default JPEG parameters. */

  force_baseline = FALSE;       /* by default, allow 16-bit quantizers */
  simple_progressive = FALSE;
  is_targa = FALSE;
  icc_filename = NULL;
  outfilename = NULL;
  memdst = FALSE;
  report = FALSE;
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

    if (keymatch(arg, "arithmetic", 1)) {
      /* Use arithmetic coding. */
#ifdef C_ARITH_CODING_SUPPORTED
      cinfo->arith_code = TRUE;
#else
      fprintf(stderr, "%s: sorry, arithmetic coding not supported\n",
              progname);
      exit(EXIT_FAILURE);
#endif

    } else if (keymatch(arg, "baseline", 1)) {
      /* Force baseline-compatible output (8-bit quantizer values). */
      force_baseline = TRUE;

    } else if (keymatch(arg, "dct", 2)) {
      /* Select DCT algorithm. */
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

    } else if (keymatch(arg, "grayscale", 2) ||
               keymatch(arg, "greyscale", 2)) {
      /* Force a monochrome JPEG file to be generated. */
      jpeg_set_colorspace(cinfo, JCS_GRAYSCALE);

    } else if (keymatch(arg, "rgb", 3)) {
      /* Force an RGB JPEG file to be generated. */
      jpeg_set_colorspace(cinfo, JCS_RGB);

    } else if (keymatch(arg, "icc", 1)) {
      /* Set ICC filename. */
      if (++argn >= argc)       /* advance to next argument */
        usage();
      icc_filename = argv[argn];

    } else if (keymatch(arg, "lossless", 1)) {
      /* Enable lossless mode. */
#ifdef C_LOSSLESS_SUPPORTED
      char ch = ',', *ptr;

      if (++argn >= argc)       /* advance to next argument */
        usage();
      if (sscanf(argv[argn], "%d%c", &psv, &ch) < 1 || ch != ',')
        usage();
      ptr = argv[argn];
      while (*ptr && *ptr++ != ','); /* advance to next segment of arg
                                        string */
      if (*ptr)
        sscanf(ptr, "%d", &pt);

      /* We must postpone execution until data_precision is known. */
#else
      fprintf(stderr, "%s: sorry, lossless output was not compiled\n",
              progname);
      exit(EXIT_FAILURE);
#endif

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

    } else if (keymatch(arg, "optimize", 1) || keymatch(arg, "optimise", 1)) {
      /* Enable entropy parm optimization. */
#ifdef ENTROPY_OPT_SUPPORTED
      cinfo->optimize_coding = TRUE;
#else
      fprintf(stderr, "%s: sorry, entropy optimization was not compiled in\n",
              progname);
      exit(EXIT_FAILURE);
#endif

    } else if (keymatch(arg, "outfile", 4)) {
      /* Set output file name. */
      if (++argn >= argc)       /* advance to next argument */
        usage();
      outfilename = argv[argn]; /* save it away for later use */

    } else if (keymatch(arg, "precision", 3)) {
      /* Set data precision. */
      int val;

      if (++argn >= argc)       /* advance to next argument */
        usage();
      if (sscanf(argv[argn], "%d", &val) != 1)
        usage();
#ifdef C_LOSSLESS_SUPPORTED
      if (val < 2 || val > 16)
#else
      if (val != 8 && val != 12)
#endif
        usage();
      cinfo->data_precision = val;

    } else if (keymatch(arg, "progressive", 1)) {
      /* Select simple progressive mode. */
#ifdef C_PROGRESSIVE_SUPPORTED
      simple_progressive = TRUE;
      /* We must postpone execution until num_components is known. */
#else
      fprintf(stderr, "%s: sorry, progressive output was not compiled in\n",
              progname);
      exit(EXIT_FAILURE);
#endif

    } else if (keymatch(arg, "memdst", 2)) {
      /* Use in-memory destination manager */
      memdst = TRUE;

    } else if (keymatch(arg, "quality", 1)) {
      /* Quality ratings (quantization table scaling factors). */
      if (++argn >= argc)       /* advance to next argument */
        usage();
      qualityarg = argv[argn];

    } else if (keymatch(arg, "qslots", 2)) {
      /* Quantization table slot numbers. */
      if (++argn >= argc)       /* advance to next argument */
        usage();
      qslotsarg = argv[argn];
      /* Must delay setting qslots until after we have processed any
       * colorspace-determining switches, since jpeg_set_colorspace sets
       * default quant table numbers.
       */

    } else if (keymatch(arg, "qtables", 2)) {
      /* Quantization tables fetched from file. */
      if (++argn >= argc)       /* advance to next argument */
        usage();
      qtablefile = argv[argn];
      /* We postpone actually reading the file in case -quality comes later. */

    } else if (keymatch(arg, "report", 3)) {
      report = TRUE;

    } else if (keymatch(arg, "restart", 1)) {
      /* Restart interval in MCU rows (or in MCUs with 'b'). */
      long lval;
      char ch = 'x';

      if (++argn >= argc)       /* advance to next argument */
        usage();
      if (sscanf(argv[argn], "%ld%c", &lval, &ch) < 1)
        usage();
      if (lval < 0 || lval > 65535L)
        usage();
      if (ch == 'b' || ch == 'B') {
        cinfo->restart_interval = (unsigned int)lval;
        cinfo->restart_in_rows = 0; /* else prior '-restart n' overrides me */
      } else {
        cinfo->restart_in_rows = (int)lval;
        /* restart_interval will be computed during startup */
      }

    } else if (keymatch(arg, "sample", 2)) {
      /* Set sampling factors. */
      if (++argn >= argc)       /* advance to next argument */
        usage();
      samplearg = argv[argn];
      /* Must delay setting sample factors until after we have processed any
       * colorspace-determining switches, since jpeg_set_colorspace sets
       * default sampling factors.
       */

    } else if (keymatch(arg, "scans", 2)) {
      /* Set scan script. */
#ifdef C_MULTISCAN_FILES_SUPPORTED
      if (++argn >= argc)       /* advance to next argument */
        usage();
      scansarg = argv[argn];
      /* We must postpone reading the file in case -progressive appears. */
#else
      fprintf(stderr, "%s: sorry, multi-scan output was not compiled in\n",
              progname);
      exit(EXIT_FAILURE);
#endif

    } else if (keymatch(arg, "smooth", 2)) {
      /* Set input smoothing factor. */
      int val;

      if (++argn >= argc)       /* advance to next argument */
        usage();
      if (sscanf(argv[argn], "%d", &val) != 1)
        usage();
      if (val < 0 || val > 100)
        usage();
      cinfo->smoothing_factor = val;

    } else if (keymatch(arg, "strict", 2)) {
      strict = TRUE;

    } else if (keymatch(arg, "targa", 1)) {
      /* Input file is Targa format. */
      is_targa = TRUE;

    } else {
      usage();                  /* bogus switch */
    }
  }

  /* Post-switch-scanning cleanup */

  if (for_real) {

    /* Set quantization tables for selected quality. */
    /* Some or all may be overridden if -qtables is present. */
    if (qualityarg != NULL)     /* process -quality if it was present */
      if (!set_quality_ratings(cinfo, qualityarg, force_baseline))
        usage();

    if (qtablefile != NULL)     /* process -qtables if it was present */
      if (!read_quant_tables(cinfo, qtablefile, force_baseline))
        usage();

    if (qslotsarg != NULL)      /* process -qslots if it was present */
      if (!set_quant_slots(cinfo, qslotsarg))
        usage();

    if (samplearg != NULL)      /* process -sample if it was present */
      if (!set_sample_factors(cinfo, samplearg))
        usage();

#ifdef C_PROGRESSIVE_SUPPORTED
    if (simple_progressive)     /* process -progressive; -scans can override */
      jpeg_simple_progression(cinfo);
#endif

#ifdef C_LOSSLESS_SUPPORTED
    if (psv != 0)               /* process -lossless */
      jpeg_enable_lossless(cinfo, psv, pt);
#endif

#ifdef C_MULTISCAN_FILES_SUPPORTED
    if (scansarg != NULL)       /* process -scans if it was present */
      if (!read_scan_script(cinfo, scansarg))
        usage();
#endif
  }

  return argn;                  /* return index of next arg (file name) */
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
  struct jpeg_compress_struct cinfo;
#ifdef CJPEG_FUZZER
  struct my_error_mgr myerr;
  struct jpeg_error_mgr &jerr = myerr.pub;
#else
  struct jpeg_error_mgr jerr;
#endif
  struct cdjpeg_progress_mgr progress;
  int file_index;
  cjpeg_source_ptr src_mgr;
  FILE *input_file = NULL;
  FILE *icc_file;
  JOCTET *icc_profile = NULL;
  long icc_len = 0;
  FILE *output_file = NULL;
  unsigned char *outbuffer = NULL;
  unsigned long outsize = 0;
  JDIMENSION num_scanlines;

  progname = argv[0];
  if (progname == NULL || progname[0] == 0)
    progname = "cjpeg";         /* in case C library doesn't provide it */

  /* Initialize the JPEG compression object with default error handling. */
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  /* Add some application-specific error messages (from cderror.h) */
  jerr.addon_message_table = cdjpeg_message_table;
  jerr.first_addon_message = JMSG_FIRSTADDONCODE;
  jerr.last_addon_message = JMSG_LASTADDONCODE;

  /* Initialize JPEG parameters.
   * Much of this may be overridden later.
   * In particular, we don't yet know the input file's color space,
   * but we need to provide some value for jpeg_set_defaults() to work.
   */

  cinfo.in_color_space = JCS_RGB; /* arbitrary guess */
  jpeg_set_defaults(&cinfo);

  /* Scan command line to find file names.
   * It is convenient to use just one switch-parsing routine, but the switch
   * values read here are ignored; we will rescan the switches after opening
   * the input file.
   */

  file_index = parse_switches(&cinfo, argc, argv, 0, FALSE);

  if (strict)
    jerr.emit_message = my_emit_message;

#ifdef TWO_FILE_COMMANDLINE
  if (!memdst) {
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
  } else if (!memdst) {
    /* default output file is stdout */
    output_file = write_stdout();
  }

  if (icc_filename != NULL) {
    if ((icc_file = fopen(icc_filename, READ_BINARY)) == NULL) {
      fprintf(stderr, "%s: can't open %s\n", progname, icc_filename);
      exit(EXIT_FAILURE);
    }
    if (fseek(icc_file, 0, SEEK_END) < 0 ||
        (icc_len = ftell(icc_file)) < 1 ||
        fseek(icc_file, 0, SEEK_SET) < 0) {
      fprintf(stderr, "%s: can't determine size of %s\n", progname,
              icc_filename);
      exit(EXIT_FAILURE);
    }
    if ((icc_profile = (JOCTET *)malloc(icc_len)) == NULL) {
      fprintf(stderr, "%s: can't allocate memory for ICC profile\n", progname);
      fclose(icc_file);
      exit(EXIT_FAILURE);
    }
    if (fread(icc_profile, icc_len, 1, icc_file) < 1) {
      fprintf(stderr, "%s: can't read ICC profile from %s\n", progname,
              icc_filename);
      free(icc_profile);
      fclose(icc_file);
      exit(EXIT_FAILURE);
    }
    fclose(icc_file);
  }

#ifdef CJPEG_FUZZER
  jerr.error_exit = my_error_exit;
  jerr.emit_message = my_emit_message_fuzzer;
  if (setjmp(myerr.setjmp_buffer))
    HANDLE_ERROR()
#endif

  if (report) {
    start_progress_monitor((j_common_ptr)&cinfo, &progress);
    progress.report = report;
  }

  /* Figure out the input file format, and set up to read it. */
  src_mgr = select_file_type(&cinfo, input_file);
  src_mgr->input_file = input_file;
#ifdef CJPEG_FUZZER
  src_mgr->max_pixels = 1048576;
#endif

  /* Read the input file header to obtain file size & colorspace. */
  (*src_mgr->start_input) (&cinfo, src_mgr);

  /* Now that we know input colorspace, fix colorspace-dependent defaults */
  jpeg_default_colorspace(&cinfo);

  /* Adjust default compression parameters by re-parsing the options */
  file_index = parse_switches(&cinfo, argc, argv, 0, TRUE);

  /* Specify data destination for compression */
  if (memdst)
    jpeg_mem_dest(&cinfo, &outbuffer, &outsize);
  else
    jpeg_stdio_dest(&cinfo, output_file);

#ifdef CJPEG_FUZZER
  if (setjmp(myerr.setjmp_buffer))
    HANDLE_ERROR()
#endif

  /* Start compressor */
  jpeg_start_compress(&cinfo, TRUE);

  if (icc_profile != NULL)
    jpeg_write_icc_profile(&cinfo, icc_profile, (unsigned int)icc_len);

  /* Process data */
  if (cinfo.data_precision <= 8) {
    while (cinfo.next_scanline < cinfo.image_height) {
      num_scanlines = (*src_mgr->get_pixel_rows) (&cinfo, src_mgr);
      (void)jpeg_write_scanlines(&cinfo, src_mgr->buffer, num_scanlines);
    }
  } else if (cinfo.data_precision <= 12) {
    while (cinfo.next_scanline < cinfo.image_height) {
      num_scanlines = (*src_mgr->get_pixel_rows) (&cinfo, src_mgr);
      (void)jpeg12_write_scanlines(&cinfo, src_mgr->buffer12, num_scanlines);
    }
  } else {
#ifdef C_LOSSLESS_SUPPORTED
    while (cinfo.next_scanline < cinfo.image_height) {
      num_scanlines = (*src_mgr->get_pixel_rows) (&cinfo, src_mgr);
      (void)jpeg16_write_scanlines(&cinfo, src_mgr->buffer16, num_scanlines);
    }
#else
    ERREXIT1(&cinfo, JERR_BAD_PRECISION, cinfo.data_precision);
#endif
  }

  /* Finish compression and release memory */
  (*src_mgr->finish_input) (&cinfo, src_mgr);
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  /* Close files, if we opened them */
  if (input_file != stdin)
    fclose(input_file);
  if (output_file != stdout && output_file != NULL)
    fclose(output_file);

  if (report)
    end_progress_monitor((j_common_ptr)&cinfo);

  if (memdst) {
#ifndef CJPEG_FUZZER
    fprintf(stderr, "Compressed size:  %lu bytes\n", outsize);
#endif
    free(outbuffer);
  }

  free(icc_profile);

  /* All done. */
  return (jerr.num_warnings ? EXIT_WARNING : EXIT_SUCCESS);
}
