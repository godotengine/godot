/* include/X11/Xcursor/Xcursor.h.  Generated from Xcursor.h.in by configure.  */
/*
 * Copyright © 2002 Keith Packard
 *
 * Permission to use, copy, modify, distribute, and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies and that both that
 * copyright notice and this permission notice appear in supporting
 * documentation, and that the name of Keith Packard not be used in
 * advertising or publicity pertaining to distribution of the software without
 * specific, written prior permission.  Keith Packard makes no
 * representations about the suitability of this software for any purpose.  It
 * is provided "as is" without express or implied warranty.
 *
 * KEITH PACKARD DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
 * INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO
 * EVENT SHALL KEITH PACKARD BE LIABLE FOR ANY SPECIAL, INDIRECT OR
 * CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
 * DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
 * TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef _XCURSOR_H_
#define _XCURSOR_H_
#include <stdio.h>
#include <X11/Xfuncproto.h>
#include <X11/Xlib.h>

typedef int		XcursorBool;
typedef unsigned int	XcursorUInt;

typedef XcursorUInt	XcursorDim;
typedef XcursorUInt	XcursorPixel;

#define XcursorTrue	1
#define XcursorFalse	0

/*
 * Cursor files start with a header.  The header
 * contains a magic number, a version number and a
 * table of contents which has type and offset information
 * for the remaining tables in the file.
 *
 * File minor versions increment for compatible changes
 * File major versions increment for incompatible changes (never, we hope)
 *
 * Chunks of the same type are always upward compatible.  Incompatible
 * changes are made with new chunk types; the old data can remain under
 * the old type.  Upward compatible changes can add header data as the
 * header lengths are specified in the file.
 *
 *  File:
 *	FileHeader
 *	LISTofChunk
 *
 *  FileHeader:
 *	CARD32		magic	    magic number
 *	CARD32		header	    bytes in file header
 *	CARD32		version	    file version
 *	CARD32		ntoc	    number of toc entries
 *	LISTofFileToc   toc	    table of contents
 *
 *  FileToc:
 *	CARD32		type	    entry type
 *	CARD32		subtype	    entry subtype (size for images)
 *	CARD32		position    absolute file position
 */

#define XCURSOR_MAGIC	0x72756358  /* "Xcur" LSBFirst */

/*
 * Current Xcursor version number.  Will be substituted by configure
 * from the version in the libXcursor configure.ac file.
 */

#define XCURSOR_LIB_MAJOR 1
#define XCURSOR_LIB_MINOR 2
#define XCURSOR_LIB_REVISION 0
#define XCURSOR_LIB_VERSION	((XCURSOR_LIB_MAJOR * 10000) + \
				 (XCURSOR_LIB_MINOR * 100) + \
				 (XCURSOR_LIB_REVISION))

/*
 * This version number is stored in cursor files; changes to the
 * file format require updating this version number
 */
#define XCURSOR_FILE_MAJOR	1
#define XCURSOR_FILE_MINOR	0
#define XCURSOR_FILE_VERSION	((XCURSOR_FILE_MAJOR << 16) | (XCURSOR_FILE_MINOR))
#define XCURSOR_FILE_HEADER_LEN	(4 * 4)
#define XCURSOR_FILE_TOC_LEN	(3 * 4)

typedef struct _XcursorFileToc {
    XcursorUInt	    type;	/* chunk type */
    XcursorUInt	    subtype;	/* subtype (size for images) */
    XcursorUInt	    position;	/* absolute position in file */
} XcursorFileToc;

typedef struct _XcursorFileHeader {
    XcursorUInt	    magic;	/* magic number */
    XcursorUInt	    header;	/* byte length of header */
    XcursorUInt	    version;	/* file version number */
    XcursorUInt	    ntoc;	/* number of toc entries */
    XcursorFileToc  *tocs;	/* table of contents */
} XcursorFileHeader;

/*
 * The rest of the file is a list of chunks, each tagged by type
 * and version.
 *
 *  Chunk:
 *	ChunkHeader
 *	<extra type-specific header fields>
 *	<type-specific data>
 *
 *  ChunkHeader:
 *	CARD32	    header	bytes in chunk header + type header
 *	CARD32	    type	chunk type
 *	CARD32	    subtype	chunk subtype
 *	CARD32	    version	chunk type version
 */

#define XCURSOR_CHUNK_HEADER_LEN    (4 * 4)

typedef struct _XcursorChunkHeader {
    XcursorUInt	    header;	/* bytes in chunk header */
    XcursorUInt	    type;	/* chunk type */
    XcursorUInt	    subtype;	/* chunk subtype (size for images) */
    XcursorUInt	    version;	/* version of this type */
} XcursorChunkHeader;

/*
 * Here's a list of the known chunk types
 */

/*
 * Comments consist of a 4-byte length field followed by
 * UTF-8 encoded text
 *
 *  Comment:
 *	ChunkHeader header	chunk header
 *	CARD32	    length	bytes in text
 *	LISTofCARD8 text	UTF-8 encoded text
 */

#define XCURSOR_COMMENT_TYPE	    0xfffe0001
#define XCURSOR_COMMENT_VERSION	    1
#define XCURSOR_COMMENT_HEADER_LEN  (XCURSOR_CHUNK_HEADER_LEN + (1 *4))
#define XCURSOR_COMMENT_COPYRIGHT   1
#define XCURSOR_COMMENT_LICENSE	    2
#define XCURSOR_COMMENT_OTHER	    3
#define XCURSOR_COMMENT_MAX_LEN	    0x100000

typedef struct _XcursorComment {
    XcursorUInt	    version;
    XcursorUInt	    comment_type;
    char	    *comment;
} XcursorComment;

/*
 * Each cursor image occupies a separate image chunk.
 * The length of the image header follows the chunk header
 * so that future versions can extend the header without
 * breaking older applications
 *
 *  Image:
 *	ChunkHeader	header	chunk header
 *	CARD32		width	actual width
 *	CARD32		height	actual height
 *	CARD32		xhot	hot spot x
 *	CARD32		yhot	hot spot y
 *	CARD32		delay	animation delay
 *	LISTofCARD32	pixels	ARGB pixels
 */

#define XCURSOR_IMAGE_TYPE    	    0xfffd0002
#define XCURSOR_IMAGE_VERSION	    1
#define XCURSOR_IMAGE_HEADER_LEN    (XCURSOR_CHUNK_HEADER_LEN + (5*4))
#define XCURSOR_IMAGE_MAX_SIZE	    0x7fff	/* 32767x32767 max cursor size */

typedef struct _XcursorImage {
    XcursorUInt	    version;	/* version of the image data */
    XcursorDim	    size;	/* nominal size for matching */
    XcursorDim	    width;	/* actual width */
    XcursorDim	    height;	/* actual height */
    XcursorDim	    xhot;	/* hot spot x (must be inside image) */
    XcursorDim	    yhot;	/* hot spot y (must be inside image) */
    XcursorUInt	    delay;	/* animation delay to next frame (ms) */
    XcursorPixel    *pixels;	/* pointer to pixels */
} XcursorImage;

/*
 * Other data structures exposed by the library API
 */
typedef struct _XcursorImages {
    int		    nimage;	/* number of images */
    XcursorImage    **images;	/* array of XcursorImage pointers */
    char	    *name;	/* name used to load images */
} XcursorImages;

typedef struct _XcursorCursors {
    Display	    *dpy;	/* Display holding cursors */
    int		    ref;	/* reference count */
    int		    ncursor;	/* number of cursors */
    Cursor	    *cursors;	/* array of cursors */
} XcursorCursors;

typedef struct _XcursorAnimate {
    XcursorCursors   *cursors;	/* list of cursors to use */
    int		    sequence;	/* which cursor is next */
} XcursorAnimate;

typedef struct _XcursorFile XcursorFile;

struct _XcursorFile {
    void    *closure;
    int	    (*read)  (XcursorFile *file, unsigned char *buf, int len);
    int	    (*write) (XcursorFile *file, unsigned char *buf, int len);
    int	    (*seek)  (XcursorFile *file, long offset, int whence);
};

typedef struct _XcursorComments {
    int		    ncomment;	/* number of comments */
    XcursorComment  **comments;	/* array of XcursorComment pointers */
} XcursorComments;

#define XCURSOR_CORE_THEME  "core"

_XFUNCPROTOBEGIN

/*
 * Manage Image objects
 */
XcursorImage *
XcursorImageCreate (int width, int height);

void
XcursorImageDestroy (XcursorImage *image);

/*
 * Manage Images objects
 */
XcursorImages *
XcursorImagesCreate (int size);

void
XcursorImagesDestroy (XcursorImages *images);

void
XcursorImagesSetName (XcursorImages *images, const char *name);

/*
 * Manage Cursor objects
 */
XcursorCursors *
XcursorCursorsCreate (Display *dpy, int size);

void
XcursorCursorsDestroy (XcursorCursors *cursors);

/*
 * Manage Animate objects
 */
XcursorAnimate *
XcursorAnimateCreate (XcursorCursors *cursors);

void
XcursorAnimateDestroy (XcursorAnimate *animate);

Cursor
XcursorAnimateNext (XcursorAnimate *animate);

/*
 * Manage Comment objects
 */
XcursorComment *
XcursorCommentCreate (XcursorUInt comment_type, int length);

void
XcursorCommentDestroy (XcursorComment *comment);

XcursorComments *
XcursorCommentsCreate (int size);

void
XcursorCommentsDestroy (XcursorComments *comments);

/*
 * XcursorFile/Image APIs
 */
XcursorImage *
XcursorXcFileLoadImage (XcursorFile *file, int size);

XcursorImages *
XcursorXcFileLoadImages (XcursorFile *file, int size);

XcursorImages *
XcursorXcFileLoadAllImages (XcursorFile *file);

XcursorBool
XcursorXcFileLoad (XcursorFile	    *file,
		   XcursorComments  **commentsp,
		   XcursorImages    **imagesp);

XcursorBool
XcursorXcFileSave (XcursorFile		    *file,
		   const XcursorComments    *comments,
		   const XcursorImages	    *images);

/*
 * FILE/Image APIs
 */
XcursorImage *
XcursorFileLoadImage (FILE *file, int size);

XcursorImages *
XcursorFileLoadImages (FILE *file, int size);

XcursorImages *
XcursorFileLoadAllImages (FILE *file);

XcursorBool
XcursorFileLoad (FILE		    *file,
		 XcursorComments    **commentsp,
		 XcursorImages	    **imagesp);

XcursorBool
XcursorFileSaveImages (FILE *file, const XcursorImages *images);

XcursorBool
XcursorFileSave (FILE *			file,
		 const XcursorComments	*comments,
		 const XcursorImages	*images);

/*
 * Filename/Image APIs
 */
XcursorImage *
XcursorFilenameLoadImage (const char *filename, int size);

XcursorImages *
XcursorFilenameLoadImages (const char *filename, int size);

XcursorImages *
XcursorFilenameLoadAllImages (const char *filename);

XcursorBool
XcursorFilenameLoad (const char		*file,
		     XcursorComments	**commentsp,
		     XcursorImages	**imagesp);

XcursorBool
XcursorFilenameSaveImages (const char *filename, const XcursorImages *images);

XcursorBool
XcursorFilenameSave (const char		    *file,
		     const XcursorComments  *comments,
		     const XcursorImages    *images);

/*
 * Library/Image APIs
 */
XcursorImage *
XcursorLibraryLoadImage (const char *library, const char *theme, int size);

XcursorImages *
XcursorLibraryLoadImages (const char *library, const char *theme, int size);

/*
 * Library/shape API
 */

const char *
XcursorLibraryPath (void);

int
XcursorLibraryShape (const char *library);

/*
 * Image/Cursor APIs
 */

Cursor
XcursorImageLoadCursor (Display *dpy, const XcursorImage *image);

XcursorCursors *
XcursorImagesLoadCursors (Display *dpy, const XcursorImages *images);

Cursor
XcursorImagesLoadCursor (Display *dpy, const XcursorImages *images);

/*
 * Filename/Cursor APIs
 */
Cursor
XcursorFilenameLoadCursor (Display *dpy, const char *file);

XcursorCursors *
XcursorFilenameLoadCursors (Display *dpy, const char *file);

/*
 * Library/Cursor APIs
 */
Cursor
XcursorLibraryLoadCursor (Display *dpy, const char *file);

XcursorCursors *
XcursorLibraryLoadCursors (Display *dpy, const char *file);

/*
 * Shape/Image APIs
 */

XcursorImage *
XcursorShapeLoadImage (unsigned int shape, const char *theme, int size);

XcursorImages *
XcursorShapeLoadImages (unsigned int shape, const char *theme, int size);

/*
 * Shape/Cursor APIs
 */
Cursor
XcursorShapeLoadCursor (Display *dpy, unsigned int shape);

XcursorCursors *
XcursorShapeLoadCursors (Display *dpy, unsigned int shape);

/*
 * This is the function called by Xlib when attempting to
 * load cursors from XCreateGlyphCursor.  The interface must
 * not change as Xlib loads 'libXcursor.so' instead of
 * a specific major version
 */
Cursor
XcursorTryShapeCursor (Display	    *dpy,
		       Font	    source_font,
		       Font	    mask_font,
		       unsigned int source_char,
		       unsigned int mask_char,
		       XColor _Xconst *foreground,
		       XColor _Xconst *background);

void
XcursorNoticeCreateBitmap (Display	*dpy,
			   Pixmap	pid,
			   unsigned int width,
			   unsigned int height);

void
XcursorNoticePutBitmap (Display	    *dpy,
			Drawable    draw,
			XImage	    *image);

Cursor
XcursorTryShapeBitmapCursor (Display		*dpy,
			     Pixmap		source,
			     Pixmap		mask,
			     XColor		*foreground,
			     XColor		*background,
			     unsigned int	x,
			     unsigned int	y);

#define XCURSOR_BITMAP_HASH_SIZE    16

void
XcursorImageHash (XImage	*image,
		  unsigned char	hash[XCURSOR_BITMAP_HASH_SIZE]);

/*
 * Display information APIs
 */
XcursorBool
XcursorSupportsARGB (Display *dpy);

XcursorBool
XcursorSupportsAnim (Display *dpy);

XcursorBool
XcursorSetDefaultSize (Display *dpy, int size);

int
XcursorGetDefaultSize (Display *dpy);

XcursorBool
XcursorSetTheme (Display *dpy, const char *theme);

char *
XcursorGetTheme (Display *dpy);

XcursorBool
XcursorGetThemeCore (Display *dpy);

XcursorBool
XcursorSetThemeCore (Display *dpy, XcursorBool theme_core);

_XFUNCPROTOEND

#endif
