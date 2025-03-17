/*
 * wrjpgcom.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1997, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2014, 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains a very simple stand-alone application that inserts
 * user-supplied text as a COM (comment) marker in a JFIF file.
 * This may be useful as an example of the minimum logic needed to parse
 * JPEG markers.
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#define JPEG_CJPEG_DJPEG        /* to get the command-line config symbols */
#include "jinclude.h"           /* get auto-config symbols, <stdio.h> */

#include <ctype.h>              /* to declare isupper(), tolower() */
#ifdef USE_SETMODE
#include <fcntl.h>              /* to declare setmode()'s parameter macros */
/* If you have setmode() but not <io.h>, just delete this line: */
#include <io.h>                 /* to declare setmode() */
#endif

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

/* Reduce this value if your malloc() can't allocate blocks up to 64K.
 * On DOS, compiling in large model is usually a better solution.
 */

#ifndef MAX_COM_LENGTH
#define MAX_COM_LENGTH  65000L  /* must be <= 65533 in any case */
#endif


/*
 * These macros are used to read the input file and write the output file.
 * To reuse this code in another application, you might need to change these.
 */

static FILE *infile;            /* input JPEG file */

/* Return next input byte, or EOF if no more */
#define NEXTBYTE()  getc(infile)

static FILE *outfile;           /* output JPEG file */

/* Emit an output byte */
#define PUTBYTE(x)  putc((x), outfile)


/* Error exit handler */
#define ERREXIT(msg)  (fprintf(stderr, "%s\n", msg), exit(EXIT_FAILURE))


/* Read one byte, testing for EOF */
static int
read_1_byte(void)
{
  int c;

  c = NEXTBYTE();
  if (c == EOF)
    ERREXIT("Premature EOF in JPEG file");
  return c;
}

/* Read 2 bytes, convert to unsigned int */
/* All 2-byte quantities in JPEG markers are MSB first */
static unsigned int
read_2_bytes(void)
{
  int c1, c2;

  c1 = NEXTBYTE();
  if (c1 == EOF)
    ERREXIT("Premature EOF in JPEG file");
  c2 = NEXTBYTE();
  if (c2 == EOF)
    ERREXIT("Premature EOF in JPEG file");
  return (((unsigned int)c1) << 8) + ((unsigned int)c2);
}


/* Routines to write data to output file */

static void
write_1_byte(int c)
{
  PUTBYTE(c);
}

static void
write_2_bytes(unsigned int val)
{
  PUTBYTE((val >> 8) & 0xFF);
  PUTBYTE(val & 0xFF);
}

static void
write_marker(int marker)
{
  PUTBYTE(0xFF);
  PUTBYTE(marker);
}

static void
copy_rest_of_file(void)
{
  int c;

  while ((c = NEXTBYTE()) != EOF)
    PUTBYTE(c);
}


/*
 * JPEG markers consist of one or more 0xFF bytes, followed by a marker
 * code byte (which is not an FF).  Here are the marker codes of interest
 * in this program.  (See jdmarker.c for a more complete list.)
 */

#define M_SOF0   0xC0           /* Start Of Frame N */
#define M_SOF1   0xC1           /* N indicates which compression process */
#define M_SOF2   0xC2           /* Only SOF0-SOF2 are now in common use */
#define M_SOF3   0xC3
#define M_SOF5   0xC5           /* NB: codes C4 and CC are NOT SOF markers */
#define M_SOF6   0xC6
#define M_SOF7   0xC7
#define M_SOF9   0xC9
#define M_SOF10  0xCA
#define M_SOF11  0xCB
#define M_SOF13  0xCD
#define M_SOF14  0xCE
#define M_SOF15  0xCF
#define M_SOI    0xD8           /* Start Of Image (beginning of datastream) */
#define M_EOI    0xD9           /* End Of Image (end of datastream) */
#define M_SOS    0xDA           /* Start Of Scan (begins compressed data) */
#define M_COM    0xFE           /* COMment */


/*
 * Find the next JPEG marker and return its marker code.
 * We expect at least one FF byte, possibly more if the compressor used FFs
 * to pad the file.  (Padding FFs will NOT be replicated in the output file.)
 * There could also be non-FF garbage between markers.  The treatment of such
 * garbage is unspecified; we choose to skip over it but emit a warning msg.
 * NB: this routine must not be used after seeing SOS marker, since it will
 * not deal correctly with FF/00 sequences in the compressed image data...
 */

static int
next_marker(void)
{
  int c;
  int discarded_bytes = 0;

  /* Find 0xFF byte; count and skip any non-FFs. */
  c = read_1_byte();
  while (c != 0xFF) {
    discarded_bytes++;
    c = read_1_byte();
  }
  /* Get marker code byte, swallowing any duplicate FF bytes.  Extra FFs
   * are legal as pad bytes, so don't count them in discarded_bytes.
   */
  do {
    c = read_1_byte();
  } while (c == 0xFF);

  if (discarded_bytes != 0) {
    fprintf(stderr, "Warning: garbage data found in JPEG file\n");
  }

  return c;
}


/*
 * Read the initial marker, which should be SOI.
 * For a JFIF file, the first two bytes of the file should be literally
 * 0xFF M_SOI.  To be more general, we could use next_marker, but if the
 * input file weren't actually JPEG at all, next_marker might read the whole
 * file and then return a misleading error message...
 */

static int
first_marker(void)
{
  int c1, c2;

  c1 = NEXTBYTE();
  c2 = NEXTBYTE();
  if (c1 != 0xFF || c2 != M_SOI)
    ERREXIT("Not a JPEG file");
  return c2;
}


/*
 * Most types of marker are followed by a variable-length parameter segment.
 * This routine skips over the parameters for any marker we don't otherwise
 * want to process.
 * Note that we MUST skip the parameter segment explicitly in order not to
 * be fooled by 0xFF bytes that might appear within the parameter segment;
 * such bytes do NOT introduce new markers.
 */

static void
copy_variable(void)
/* Copy an unknown or uninteresting variable-length marker */
{
  unsigned int length;

  /* Get the marker parameter length count */
  length = read_2_bytes();
  write_2_bytes(length);
  /* Length includes itself, so must be at least 2 */
  if (length < 2)
    ERREXIT("Erroneous JPEG marker length");
  length -= 2;
  /* Copy the remaining bytes */
  while (length > 0) {
    write_1_byte(read_1_byte());
    length--;
  }
}

static void
skip_variable(void)
/* Skip over an unknown or uninteresting variable-length marker */
{
  unsigned int length;

  /* Get the marker parameter length count */
  length = read_2_bytes();
  /* Length includes itself, so must be at least 2 */
  if (length < 2)
    ERREXIT("Erroneous JPEG marker length");
  length -= 2;
  /* Skip over the remaining bytes */
  while (length > 0) {
    (void)read_1_byte();
    length--;
  }
}


/*
 * Parse the marker stream until SOFn or EOI is seen;
 * copy data to output, but discard COM markers unless keep_COM is true.
 */

static int
scan_JPEG_header(int keep_COM)
{
  int marker;

  /* Expect SOI at start of file */
  if (first_marker() != M_SOI)
    ERREXIT("Expected SOI marker first");
  write_marker(M_SOI);

  /* Scan miscellaneous markers until we reach SOFn. */
  for (;;) {
    marker = next_marker();
    switch (marker) {
      /* Note that marker codes 0xC4, 0xC8, 0xCC are not, and must not be,
       * treated as SOFn.  C4 in particular is actually DHT.
       */
    case M_SOF0:                /* Baseline */
    case M_SOF1:                /* Extended sequential, Huffman */
    case M_SOF2:                /* Progressive, Huffman */
    case M_SOF3:                /* Lossless, Huffman */
    case M_SOF5:                /* Differential sequential, Huffman */
    case M_SOF6:                /* Differential progressive, Huffman */
    case M_SOF7:                /* Differential lossless, Huffman */
    case M_SOF9:                /* Extended sequential, arithmetic */
    case M_SOF10:               /* Progressive, arithmetic */
    case M_SOF11:               /* Lossless, arithmetic */
    case M_SOF13:               /* Differential sequential, arithmetic */
    case M_SOF14:               /* Differential progressive, arithmetic */
    case M_SOF15:               /* Differential lossless, arithmetic */
      return marker;

    case M_SOS:                 /* should not see compressed data before SOF */
      ERREXIT("SOS without prior SOFn");
      break;

    case M_EOI:                 /* in case it's a tables-only JPEG stream */
      return marker;

    case M_COM:                 /* Existing COM: conditionally discard */
      if (keep_COM) {
        write_marker(marker);
        copy_variable();
      } else {
        skip_variable();
      }
      break;

    default:                    /* Anything else just gets copied */
      write_marker(marker);
      copy_variable();          /* we assume it has a parameter count... */
      break;
    }
  } /* end loop */
}


/* Command line parsing code */

static const char *progname;    /* program name for error messages */


static void
usage(void)
/* complain about bad command line */
{
  fprintf(stderr, "wrjpgcom inserts a textual comment in a JPEG file.\n");
  fprintf(stderr, "You can add to or replace any existing comment(s).\n");

  fprintf(stderr, "Usage: %s [switches] ", progname);
#ifdef TWO_FILE_COMMANDLINE
  fprintf(stderr, "inputfile outputfile\n");
#else
  fprintf(stderr, "[inputfile]\n");
#endif

  fprintf(stderr, "Switches (names may be abbreviated):\n");
  fprintf(stderr, "  -replace         Delete any existing comments\n");
  fprintf(stderr, "  -comment \"text\"  Insert comment with given text\n");
  fprintf(stderr, "  -cfile name      Read comment from named file\n");
  fprintf(stderr, "Notice that you must put quotes around the comment text\n");
  fprintf(stderr, "when you use -comment.\n");
  fprintf(stderr, "If you do not give either -comment or -cfile on the command line,\n");
  fprintf(stderr, "then the comment text is read from standard input.\n");
  fprintf(stderr, "It can be multiple lines, up to %u characters total.\n",
          (unsigned int)MAX_COM_LENGTH);
#ifndef TWO_FILE_COMMANDLINE
  fprintf(stderr, "You must specify an input JPEG file name when supplying\n");
  fprintf(stderr, "comment text from standard input.\n");
#endif

  exit(EXIT_FAILURE);
}


static int
keymatch(char *arg, const char *keyword, int minchars)
/* Case-insensitive matching of (possibly abbreviated) keyword switches. */
/* keyword is the constant keyword (must be lower case already), */
/* minchars is length of minimum legal abbreviation. */
{
  register int ca, ck;
  register int nmatched = 0;

  while ((ca = *arg++) != '\0') {
    if ((ck = *keyword++) == '\0')
      return 0;                 /* arg longer than keyword, no good */
    if (isupper(ca))            /* force arg to lcase (assume ck is already) */
      ca = tolower(ca);
    if (ca != ck)
      return 0;                 /* no good */
    nmatched++;                 /* count matched characters */
  }
  /* reached end of argument; fail if it's too short for unique abbrev */
  if (nmatched < minchars)
    return 0;
  return 1;                     /* A-OK */
}


/*
 * The main program.
 */

int
main(int argc, char **argv)
{
  int argn;
  char *arg;
  int keep_COM = 1;
  char *comment_arg = NULL;
  FILE *comment_file = NULL;
  unsigned int comment_length = 0;
  int marker;

  progname = argv[0];
  if (progname == NULL || progname[0] == 0)
    progname = "wrjpgcom";      /* in case C library doesn't provide it */

  /* Parse switches, if any */
  for (argn = 1; argn < argc; argn++) {
    arg = argv[argn];
    if (arg[0] != '-')
      break;                    /* not switch, must be file name */
    arg++;                      /* advance over '-' */
    if (keymatch(arg, "replace", 1)) {
      keep_COM = 0;
    } else if (keymatch(arg, "cfile", 2)) {
      if (++argn >= argc) usage();
      if ((comment_file = fopen(argv[argn], "r")) == NULL) {
        fprintf(stderr, "%s: can't open %s\n", progname, argv[argn]);
        exit(EXIT_FAILURE);
      }
    } else if (keymatch(arg, "comment", 1)) {
      if (++argn >= argc) usage();
      comment_arg = argv[argn];
      /* If the comment text starts with '"', then we are probably running
       * under MS-DOG and must parse out the quoted string ourselves.  Sigh.
       */
      if (comment_arg[0] == '"') {
        comment_arg = (char *)malloc((size_t)MAX_COM_LENGTH);
        if (comment_arg == NULL)
          ERREXIT("Insufficient memory");
        if (strlen(argv[argn]) + 2 >= (size_t)MAX_COM_LENGTH) {
          fprintf(stderr, "Comment text may not exceed %u bytes\n",
                  (unsigned int)MAX_COM_LENGTH);
          exit(EXIT_FAILURE);
        }
        strcpy(comment_arg, argv[argn] + 1);
        for (;;) {
          comment_length = (unsigned int)strlen(comment_arg);
          if (comment_length > 0 && comment_arg[comment_length - 1] == '"') {
            comment_arg[comment_length - 1] = '\0'; /* zap terminating quote */
            break;
          }
          if (++argn >= argc)
            ERREXIT("Missing ending quote mark");
          if (strlen(comment_arg) + strlen(argv[argn]) + 2 >=
              (size_t)MAX_COM_LENGTH) {
            fprintf(stderr, "Comment text may not exceed %u bytes\n",
                    (unsigned int)MAX_COM_LENGTH);
            exit(EXIT_FAILURE);
          }
          strcat(comment_arg, " ");
          strcat(comment_arg, argv[argn]);
        }
      } else if (strlen(argv[argn]) >= (size_t)MAX_COM_LENGTH) {
        fprintf(stderr, "Comment text may not exceed %u bytes\n",
                (unsigned int)MAX_COM_LENGTH);
        exit(EXIT_FAILURE);
      }
      comment_length = (unsigned int)strlen(comment_arg);
    } else
      usage();
  }

  /* Cannot use both -comment and -cfile. */
  if (comment_arg != NULL && comment_file != NULL)
    usage();
  /* If there is neither -comment nor -cfile, we will read the comment text
   * from stdin; in this case there MUST be an input JPEG file name.
   */
  if (comment_arg == NULL && comment_file == NULL && argn >= argc)
    usage();

  /* Open the input file. */
  if (argn < argc) {
    if ((infile = fopen(argv[argn], READ_BINARY)) == NULL) {
      fprintf(stderr, "%s: can't open %s\n", progname, argv[argn]);
      exit(EXIT_FAILURE);
    }
  } else {
    /* default input file is stdin */
#ifdef USE_SETMODE              /* need to hack file mode? */
    setmode(fileno(stdin), O_BINARY);
#endif
#ifdef USE_FDOPEN               /* need to re-open in binary mode? */
    if ((infile = fdopen(fileno(stdin), READ_BINARY)) == NULL) {
      fprintf(stderr, "%s: can't open stdin\n", progname);
      exit(EXIT_FAILURE);
    }
#else
    infile = stdin;
#endif
  }

  /* Open the output file. */
#ifdef TWO_FILE_COMMANDLINE
  /* Must have explicit output file name */
  if (argn != argc - 2) {
    fprintf(stderr, "%s: must name one input and one output file\n", progname);
    usage();
  }
  if ((outfile = fopen(argv[argn + 1], WRITE_BINARY)) == NULL) {
    fprintf(stderr, "%s: can't open %s\n", progname, argv[argn + 1]);
    exit(EXIT_FAILURE);
  }
#else
  /* Unix style: expect zero or one file name */
  if (argn < argc - 1) {
    fprintf(stderr, "%s: only one input file\n", progname);
    usage();
  }
  /* default output file is stdout */
#ifdef USE_SETMODE              /* need to hack file mode? */
  setmode(fileno(stdout), O_BINARY);
#endif
#ifdef USE_FDOPEN               /* need to re-open in binary mode? */
  if ((outfile = fdopen(fileno(stdout), WRITE_BINARY)) == NULL) {
    fprintf(stderr, "%s: can't open stdout\n", progname);
    exit(EXIT_FAILURE);
  }
#else
  outfile = stdout;
#endif
#endif /* TWO_FILE_COMMANDLINE */

  /* Collect comment text from comment_file or stdin, if necessary */
  if (comment_arg == NULL) {
    FILE *src_file;
    int c;

    comment_arg = (char *)malloc((size_t)MAX_COM_LENGTH);
    if (comment_arg == NULL)
      ERREXIT("Insufficient memory");
    comment_length = 0;
    src_file = (comment_file != NULL ? comment_file : stdin);
    while ((c = getc(src_file)) != EOF) {
      if (comment_length >= (unsigned int)MAX_COM_LENGTH) {
        fprintf(stderr, "Comment text may not exceed %u bytes\n",
                (unsigned int)MAX_COM_LENGTH);
        exit(EXIT_FAILURE);
      }
      comment_arg[comment_length++] = (char)c;
    }
    if (comment_file != NULL)
      fclose(comment_file);
  }

  /* Copy JPEG headers until SOFn marker;
   * we will insert the new comment marker just before SOFn.
   * This (a) causes the new comment to appear after, rather than before,
   * existing comments; and (b) ensures that comments come after any JFIF
   * or JFXX markers, as required by the JFIF specification.
   */
  marker = scan_JPEG_header(keep_COM);
  /* Insert the new COM marker, but only if nonempty text has been supplied */
  if (comment_length > 0) {
    write_marker(M_COM);
    write_2_bytes(comment_length + 2);
    while (comment_length > 0) {
      write_1_byte(*comment_arg++);
      comment_length--;
    }
  }
  /* Duplicate the remainder of the source file.
   * Note that any COM markers occurring after SOF will not be touched.
   */
  write_marker(marker);
  copy_rest_of_file();

  /* All done. */
  exit(EXIT_SUCCESS);
  return 0;                     /* suppress no-return-value warnings */
}
