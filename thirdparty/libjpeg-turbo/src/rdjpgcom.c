/*
 * rdjpgcom.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1997, Thomas G. Lane.
 * Modified 2009 by Bill Allombert, Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains a very simple stand-alone application that displays
 * the text in COM (comment) markers in a JFIF file.
 * This may be useful as an example of the minimum logic needed to parse
 * JPEG markers.
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#define JPEG_CJPEG_DJPEG        /* to get the command-line config symbols */
#include "jinclude.h"           /* get auto-config symbols, <stdio.h> */

#include <locale.h>             /* Bill Allombert: use locale for isprint */
#include <ctype.h>              /* to declare isupper(), tolower() */
#ifdef USE_SETMODE
#include <fcntl.h>              /* to declare setmode()'s parameter macros */
/* If you have setmode() but not <io.h>, just delete this line: */
#include <io.h>                 /* to declare setmode() */
#endif

#ifdef DONT_USE_B_MODE          /* define mode parameters for fopen() */
#define READ_BINARY     "r"
#else
#define READ_BINARY     "rb"
#endif

#ifndef EXIT_FAILURE            /* define exit() codes if not provided */
#define EXIT_FAILURE  1
#endif
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS  0
#endif


/*
 * These macros are used to read the input file.
 * To reuse this code in another application, you might need to change these.
 */

static FILE *infile;            /* input JPEG file */

/* Return next input byte, or EOF if no more */
#define NEXTBYTE()  getc(infile)


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
#define M_APP12  0xEC           /* (we don't bother to list all 16 APPn's) */
#define M_COM    0xFE           /* COMment */


/*
 * Find the next JPEG marker and return its marker code.
 * We expect at least one FF byte, possibly more if the compressor used FFs
 * to pad the file.
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
 * Process a COM marker.
 * We want to print out the marker contents as legible text;
 * we must guard against non-text junk and varying newline representations.
 */

static void
process_COM(int raw)
{
  unsigned int length;
  int ch;
  int lastch = 0;

  /* Bill Allombert: set locale properly for isprint */
  setlocale(LC_CTYPE, "");

  /* Get the marker parameter length count */
  length = read_2_bytes();
  /* Length includes itself, so must be at least 2 */
  if (length < 2)
    ERREXIT("Erroneous JPEG marker length");
  length -= 2;

  while (length > 0) {
    ch = read_1_byte();
    if (raw) {
      putc(ch, stdout);
    /* Emit the character in a readable form.
     * Nonprintables are converted to \nnn form,
     * while \ is converted to \\.
     * Newlines in CR, CR/LF, or LF form will be printed as one newline.
     */
    } else if (ch == '\r') {
      printf("\n");
    } else if (ch == '\n') {
      if (lastch != '\r')
        printf("\n");
    } else if (ch == '\\') {
      printf("\\\\");
    } else if (isprint(ch)) {
      putc(ch, stdout);
    } else {
      printf("\\%03o", (unsigned int)ch);
    }
    lastch = ch;
    length--;
  }
  printf("\n");

  /* Bill Allombert: revert to C locale */
  setlocale(LC_CTYPE, "C");
}


/*
 * Process a SOFn marker.
 * This code is only needed if you want to know the image dimensions...
 */

static void
process_SOFn(int marker)
{
  unsigned int length;
  unsigned int image_height, image_width;
  int data_precision, num_components;
  const char *process;
  int ci;

  length = read_2_bytes();      /* usual parameter length count */

  data_precision = read_1_byte();
  image_height = read_2_bytes();
  image_width = read_2_bytes();
  num_components = read_1_byte();

  switch (marker) {
  case M_SOF0:  process = "Baseline";  break;
  case M_SOF1:  process = "Extended sequential";  break;
  case M_SOF2:  process = "Progressive";  break;
  case M_SOF3:  process = "Lossless";  break;
  case M_SOF5:  process = "Differential sequential";  break;
  case M_SOF6:  process = "Differential progressive";  break;
  case M_SOF7:  process = "Differential lossless";  break;
  case M_SOF9:  process = "Extended sequential, arithmetic coding";  break;
  case M_SOF10: process = "Progressive, arithmetic coding";  break;
  case M_SOF11: process = "Lossless, arithmetic coding";  break;
  case M_SOF13: process = "Differential sequential, arithmetic coding";  break;
  case M_SOF14:
    process = "Differential progressive, arithmetic coding";  break;
  case M_SOF15: process = "Differential lossless, arithmetic coding";  break;
  default:      process = "Unknown";  break;
  }

  printf("JPEG image is %uw * %uh, %d color components, %d bits per sample\n",
         image_width, image_height, num_components, data_precision);
  printf("JPEG process: %s\n", process);

  if (length != (unsigned int)(8 + num_components * 3))
    ERREXIT("Bogus SOF marker length");

  for (ci = 0; ci < num_components; ci++) {
    (void)read_1_byte();        /* Component ID code */
    (void)read_1_byte();        /* H, V sampling factors */
    (void)read_1_byte();        /* Quantization table number */
  }
}


/*
 * Parse the marker stream until SOS or EOI is seen;
 * display any COM markers.
 * While the companion program wrjpgcom will always insert COM markers before
 * SOFn, other implementations might not, so we scan to SOS before stopping.
 * If we were only interested in the image dimensions, we would stop at SOFn.
 * (Conversely, if we only cared about COM markers, there would be no need
 * for special code to handle SOFn; we could treat it like other markers.)
 */

static int
scan_JPEG_header(int verbose, int raw)
{
  int marker;

  /* Expect SOI at start of file */
  if (first_marker() != M_SOI)
    ERREXIT("Expected SOI marker first");

  /* Scan miscellaneous markers until we reach SOS. */
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
      if (verbose)
        process_SOFn(marker);
      else
        skip_variable();
      break;

    case M_SOS:                 /* stop before hitting compressed data */
      return marker;

    case M_EOI:                 /* in case it's a tables-only JPEG stream */
      return marker;

    case M_COM:
      process_COM(raw);
      break;

    case M_APP12:
      /* Some digital camera makers put useful textual information into
       * APP12 markers, so we print those out too when in -verbose mode.
       */
      if (verbose) {
        printf("APP12 contains:\n");
        process_COM(raw);
      } else
        skip_variable();
      break;

    default:                    /* Anything else just gets skipped */
      skip_variable();          /* we assume it has a parameter count... */
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
  fprintf(stderr, "rdjpgcom displays any textual comments in a JPEG file.\n");

  fprintf(stderr, "Usage: %s [switches] [inputfile]\n", progname);

  fprintf(stderr, "Switches (names may be abbreviated):\n");
  fprintf(stderr, "  -raw        Display non-printable characters in comments (unsafe)\n");
  fprintf(stderr, "  -verbose    Also display dimensions of JPEG image\n");

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
  int verbose = 0, raw = 0;

  progname = argv[0];
  if (progname == NULL || progname[0] == 0)
    progname = "rdjpgcom";      /* in case C library doesn't provide it */

  /* Parse switches, if any */
  for (argn = 1; argn < argc; argn++) {
    arg = argv[argn];
    if (arg[0] != '-')
      break;                    /* not switch, must be file name */
    arg++;                      /* advance over '-' */
    if (keymatch(arg, "verbose", 1)) {
      verbose++;
    } else if (keymatch(arg, "raw", 1)) {
      raw = 1;
    } else
      usage();
  }

  /* Open the input file. */
  /* Unix style: expect zero or one file name */
  if (argn < argc - 1) {
    fprintf(stderr, "%s: only one input file\n", progname);
    usage();
  }
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

  /* Scan the JPEG headers. */
  (void)scan_JPEG_header(verbose, raw);

  /* All done. */
  exit(EXIT_SUCCESS);
  return 0;                     /* suppress no-return-value warnings */
}
