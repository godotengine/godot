/*
 * cderror.h
 *
 * Copyright (C) 1994-1997, Thomas G. Lane.
 * Modified 2009-2020 by Guido Vollbeding.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file defines the error and message codes for the cjpeg/djpeg
 * applications.  These strings are not needed as part of the JPEG library
 * proper.
 * Edit this file to add new codes, or to translate the message strings to
 * some other language.
 */

/*
 * To define the enum list of message codes, include this file without
 * defining macro JMESSAGE.  To create a message string table, include it
 * again with a suitable JMESSAGE definition (see jerror.c for an example).
 */
#ifndef JMESSAGE
#ifndef CDERROR_H
#define CDERROR_H
/* First time through, define the enum list */
#define JMAKE_ENUM_LIST
#else
/* Repeated inclusions of this file are no-ops unless JMESSAGE is defined */
#define JMESSAGE(code,string)
#endif /* CDERROR_H */
#endif /* JMESSAGE */

#ifdef JMAKE_ENUM_LIST

typedef enum {

#define JMESSAGE(code,string)	code ,

#endif /* JMAKE_ENUM_LIST */

JMESSAGE(JMSG_FIRSTADDONCODE=1000, NULL) /* Must be first entry! */

#ifdef BMP_SUPPORTED
JMESSAGE(JERR_BMP_BADCMAP, "Unsupported BMP colormap format")
JMESSAGE(JERR_BMP_BADDEPTH, "Only 8-, 24-, and 32-bit BMP files are supported")
JMESSAGE(JERR_BMP_BADHEADER, "Invalid BMP file: bad header length")
JMESSAGE(JERR_BMP_BADPLANES, "Invalid BMP file: biPlanes not equal to 1")
JMESSAGE(JERR_BMP_COLORSPACE, "BMP output must be grayscale or RGB")
JMESSAGE(JERR_BMP_COMPRESSED, "Sorry, compressed BMPs not yet supported")
JMESSAGE(JERR_BMP_NOT, "Not a BMP file - does not start with BM")
JMESSAGE(JERR_BMP_OUTOFRANGE, "Numeric value out of range in BMP file")
JMESSAGE(JTRC_BMP, "%ux%u %d-bit BMP image")
JMESSAGE(JTRC_BMP_MAPPED, "%ux%u 8-bit colormapped BMP image")
JMESSAGE(JTRC_BMP_OS2, "%ux%u %d-bit OS2 BMP image")
JMESSAGE(JTRC_BMP_OS2_MAPPED, "%ux%u 8-bit colormapped OS2 BMP image")
#endif /* BMP_SUPPORTED */

#ifdef GIF_SUPPORTED
JMESSAGE(JERR_GIF_BUG, "GIF output got confused")
JMESSAGE(JERR_GIF_CODESIZE, "Bogus GIF codesize %d")
JMESSAGE(JERR_GIF_COLORSPACE, "GIF output must be grayscale or RGB")
JMESSAGE(JERR_GIF_IMAGENOTFOUND, "Too few images in GIF file")
JMESSAGE(JERR_GIF_NOT, "Not a GIF file")
JMESSAGE(JERR_GIF_OUTOFRANGE, "Numeric value out of range in GIF file")
JMESSAGE(JTRC_GIF, "%ux%ux%d GIF image")
JMESSAGE(JTRC_GIF_BADVERSION,
	 "Warning: unexpected GIF version number '%c%c%c'")
JMESSAGE(JTRC_GIF_EXTENSION, "Ignoring GIF extension block of type 0x%02x")
JMESSAGE(JTRC_GIF_NONSQUARE, "Caution: nonsquare pixels in input")
JMESSAGE(JWRN_GIF_BADDATA, "Corrupt data in GIF file")
JMESSAGE(JWRN_GIF_CHAR, "Bogus char 0x%02x in GIF file, ignoring")
JMESSAGE(JWRN_GIF_ENDCODE, "Premature end of GIF image")
JMESSAGE(JWRN_GIF_NOMOREDATA, "Ran out of GIF bits")
#endif /* GIF_SUPPORTED */

#ifdef PPM_SUPPORTED
JMESSAGE(JERR_PPM_COLORSPACE, "PPM output must be grayscale or RGB")
JMESSAGE(JERR_PPM_NONNUMERIC, "Nonnumeric data in PPM file")
JMESSAGE(JERR_PPM_NOT, "Not a PPM/PGM file")
JMESSAGE(JERR_PPM_OUTOFRANGE, "Numeric value out of range in PPM file")
JMESSAGE(JTRC_PGM, "%ux%u PGM image")
JMESSAGE(JTRC_PGM_TEXT, "%ux%u text PGM image")
JMESSAGE(JTRC_PPM, "%ux%u PPM image")
JMESSAGE(JTRC_PPM_TEXT, "%ux%u text PPM image")
#endif /* PPM_SUPPORTED */

#ifdef RLE_SUPPORTED
JMESSAGE(JERR_RLE_BADERROR, "Bogus error code from RLE library")
JMESSAGE(JERR_RLE_COLORSPACE, "RLE output must be grayscale or RGB")
JMESSAGE(JERR_RLE_DIMENSIONS, "Image dimensions (%ux%u) too large for RLE")
JMESSAGE(JERR_RLE_EMPTY, "Empty RLE file")
JMESSAGE(JERR_RLE_EOF, "Premature EOF in RLE header")
JMESSAGE(JERR_RLE_MEM, "Insufficient memory for RLE header")
JMESSAGE(JERR_RLE_NOT, "Not an RLE file")
JMESSAGE(JERR_RLE_TOOMANYCHANNELS, "Cannot handle %d output channels for RLE")
JMESSAGE(JERR_RLE_UNSUPPORTED, "Cannot handle this RLE setup")
JMESSAGE(JTRC_RLE, "%ux%u full-color RLE file")
JMESSAGE(JTRC_RLE_FULLMAP, "%ux%u full-color RLE file with map of length %d")
JMESSAGE(JTRC_RLE_GRAY, "%ux%u grayscale RLE file")
JMESSAGE(JTRC_RLE_MAPGRAY, "%ux%u grayscale RLE file with map of length %d")
JMESSAGE(JTRC_RLE_MAPPED, "%ux%u colormapped RLE file with map of length %d")
#endif /* RLE_SUPPORTED */

#ifdef TARGA_SUPPORTED
JMESSAGE(JERR_TGA_BADCMAP, "Unsupported Targa colormap format")
JMESSAGE(JERR_TGA_BADPARMS, "Invalid or unsupported Targa file")
JMESSAGE(JERR_TGA_COLORSPACE, "Targa output must be grayscale or RGB")
JMESSAGE(JTRC_TGA, "%ux%u RGB Targa image")
JMESSAGE(JTRC_TGA_GRAY, "%ux%u grayscale Targa image")
JMESSAGE(JTRC_TGA_MAPPED, "%ux%u colormapped Targa image")
#else
JMESSAGE(JERR_TGA_NOTCOMP, "Targa support was not compiled")
#endif /* TARGA_SUPPORTED */

JMESSAGE(JERR_BAD_CMAP_FILE,
	 "Color map file is invalid or of unsupported format")
JMESSAGE(JERR_TOO_MANY_COLORS,
	 "Output file format cannot handle %d colormap entries")
JMESSAGE(JERR_UNGETC_FAILED, "ungetc failed")
#ifdef TARGA_SUPPORTED
JMESSAGE(JERR_UNKNOWN_FORMAT,
	 "Unrecognized input file format --- perhaps you need -targa")
#else
JMESSAGE(JERR_UNKNOWN_FORMAT, "Unrecognized input file format")
#endif
JMESSAGE(JERR_UNSUPPORTED_FORMAT, "Unsupported output file format")

#ifdef JMAKE_ENUM_LIST

  JMSG_LASTADDONCODE
} ADDON_MESSAGE_CODE;

#undef JMAKE_ENUM_LIST
#endif /* JMAKE_ENUM_LIST */

/* Zap JMESSAGE macro so that future re-inclusions do nothing by default */
#undef JMESSAGE
