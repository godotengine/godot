/*
 *
 * Copyright © 2000 SuSE, Inc.
 *
 * Permission to use, copy, modify, distribute, and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies and that both that
 * copyright notice and this permission notice appear in supporting
 * documentation, and that the name of SuSE not be used in advertising or
 * publicity pertaining to distribution of the software without specific,
 * written prior permission.  SuSE makes no representations about the
 * suitability of this software for any purpose.  It is provided "as is"
 * without express or implied warranty.
 *
 * SuSE DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL SuSE
 * BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
 * OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * Author:  Keith Packard, SuSE, Inc.
 */

#ifndef _XRENDER_H_
#define _XRENDER_H_

#include <X11/Xlib.h>
#include <X11/Xfuncproto.h>
#include <X11/Xosdefs.h>
#include <X11/Xutil.h>

#include <X11/extensions/render.h>

typedef struct {
    short   red;
    short   redMask;
    short   green;
    short   greenMask;
    short   blue;
    short   blueMask;
    short   alpha;
    short   alphaMask;
} XRenderDirectFormat;

typedef struct {
    PictFormat		id;
    int			type;
    int			depth;
    XRenderDirectFormat	direct;
    Colormap		colormap;
} XRenderPictFormat;

#define PictFormatID	    (1 << 0)
#define PictFormatType	    (1 << 1)
#define PictFormatDepth	    (1 << 2)
#define PictFormatRed	    (1 << 3)
#define PictFormatRedMask   (1 << 4)
#define PictFormatGreen	    (1 << 5)
#define PictFormatGreenMask (1 << 6)
#define PictFormatBlue	    (1 << 7)
#define PictFormatBlueMask  (1 << 8)
#define PictFormatAlpha	    (1 << 9)
#define PictFormatAlphaMask (1 << 10)
#define PictFormatColormap  (1 << 11)

typedef struct _XRenderPictureAttributes {
    int 		repeat;
    Picture		alpha_map;
    int			alpha_x_origin;
    int			alpha_y_origin;
    int			clip_x_origin;
    int			clip_y_origin;
    Pixmap		clip_mask;
    Bool		graphics_exposures;
    int			subwindow_mode;
    int			poly_edge;
    int			poly_mode;
    Atom		dither;
    Bool		component_alpha;
} XRenderPictureAttributes;

typedef struct {
    unsigned short   red;
    unsigned short   green;
    unsigned short   blue;
    unsigned short   alpha;
} XRenderColor;

typedef struct _XGlyphInfo {
    unsigned short  width;
    unsigned short  height;
    short	    x;
    short	    y;
    short	    xOff;
    short	    yOff;
} XGlyphInfo;

typedef struct _XGlyphElt8 {
    GlyphSet		    glyphset;
    _Xconst char	    *chars;
    int			    nchars;
    int			    xOff;
    int			    yOff;
} XGlyphElt8;

typedef struct _XGlyphElt16 {
    GlyphSet		    glyphset;
    _Xconst unsigned short  *chars;
    int			    nchars;
    int			    xOff;
    int			    yOff;
} XGlyphElt16;

typedef struct _XGlyphElt32 {
    GlyphSet		    glyphset;
    _Xconst unsigned int    *chars;
    int			    nchars;
    int			    xOff;
    int			    yOff;
} XGlyphElt32;

typedef double	XDouble;

typedef struct _XPointDouble {
    XDouble  x, y;
} XPointDouble;

#define XDoubleToFixed(f)    ((XFixed) ((f) * 65536))
#define XFixedToDouble(f)    (((XDouble) (f)) / 65536)

typedef int XFixed;

typedef struct _XPointFixed {
    XFixed  x, y;
} XPointFixed;

typedef struct _XLineFixed {
    XPointFixed	p1, p2;
} XLineFixed;

typedef struct _XTriangle {
    XPointFixed	p1, p2, p3;
} XTriangle;

typedef struct _XCircle {
    XFixed x;
    XFixed y;
    XFixed radius;
} XCircle;

typedef struct _XTrapezoid {
    XFixed  top, bottom;
    XLineFixed	left, right;
} XTrapezoid;

typedef struct _XTransform {
    XFixed  matrix[3][3];
} XTransform;

typedef struct _XFilters {
    int	    nfilter;
    char    **filter;
    int	    nalias;
    short   *alias;
} XFilters;

typedef struct _XIndexValue {
    unsigned long    pixel;
    unsigned short   red, green, blue, alpha;
} XIndexValue;

typedef struct _XAnimCursor {
    Cursor	    cursor;
    unsigned long   delay;
} XAnimCursor;

typedef struct _XSpanFix {
    XFixed	    left, right, y;
} XSpanFix;

typedef struct _XTrap {
    XSpanFix	    top, bottom;
} XTrap;

typedef struct _XLinearGradient {
    XPointFixed p1;
    XPointFixed p2;
} XLinearGradient;

typedef struct _XRadialGradient {
    XCircle inner;
    XCircle outer;
} XRadialGradient;

typedef struct _XConicalGradient {
    XPointFixed center;
    XFixed angle; /* in degrees */
} XConicalGradient;

_XFUNCPROTOBEGIN

Bool XRenderQueryExtension (Display *dpy, int *event_basep, int *error_basep);

Status XRenderQueryVersion (Display *dpy,
			    int     *major_versionp,
			    int     *minor_versionp);

Status XRenderQueryFormats (Display *dpy);

int XRenderQuerySubpixelOrder (Display *dpy, int screen);

Bool XRenderSetSubpixelOrder (Display *dpy, int screen, int subpixel);

XRenderPictFormat *
XRenderFindVisualFormat (Display *dpy, _Xconst Visual *visual);

XRenderPictFormat *
XRenderFindFormat (Display			*dpy,
		   unsigned long		mask,
		   _Xconst XRenderPictFormat	*templ,
		   int				count);

#define PictStandardARGB32  0
#define PictStandardRGB24   1
#define PictStandardA8	    2
#define PictStandardA4	    3
#define PictStandardA1	    4
#define PictStandardNUM	    5

XRenderPictFormat *
XRenderFindStandardFormat (Display		*dpy,
			   int			format);

XIndexValue *
XRenderQueryPictIndexValues(Display			*dpy,
			    _Xconst XRenderPictFormat	*format,
			    int				*num);

Picture
XRenderCreatePicture (Display				*dpy,
		      Drawable				drawable,
		      _Xconst XRenderPictFormat		*format,
		      unsigned long			valuemask,
		      _Xconst XRenderPictureAttributes	*attributes);

void
XRenderChangePicture (Display				*dpy,
		      Picture				picture,
		      unsigned long			valuemask,
		      _Xconst XRenderPictureAttributes  *attributes);

void
XRenderSetPictureClipRectangles (Display	    *dpy,
				 Picture	    picture,
				 int		    xOrigin,
				 int		    yOrigin,
				 _Xconst XRectangle *rects,
				 int		    n);

void
XRenderSetPictureClipRegion (Display	    *dpy,
			     Picture	    picture,
			     Region	    r);

void
XRenderSetPictureTransform (Display	    *dpy,
			    Picture	    picture,
			    XTransform	    *transform);

void
XRenderFreePicture (Display                   *dpy,
		    Picture                   picture);

void
XRenderComposite (Display   *dpy,
		  int	    op,
		  Picture   src,
		  Picture   mask,
		  Picture   dst,
		  int	    src_x,
		  int	    src_y,
		  int	    mask_x,
		  int	    mask_y,
		  int	    dst_x,
		  int	    dst_y,
		  unsigned int	width,
		  unsigned int	height);

GlyphSet
XRenderCreateGlyphSet (Display *dpy, _Xconst XRenderPictFormat *format);

GlyphSet
XRenderReferenceGlyphSet (Display *dpy, GlyphSet existing);

void
XRenderFreeGlyphSet (Display *dpy, GlyphSet glyphset);

void
XRenderAddGlyphs (Display		*dpy,
		  GlyphSet		glyphset,
		  _Xconst Glyph		*gids,
		  _Xconst XGlyphInfo	*glyphs,
		  int			nglyphs,
		  _Xconst char		*images,
		  int			nbyte_images);

void
XRenderFreeGlyphs (Display	    *dpy,
		   GlyphSet	    glyphset,
		   _Xconst Glyph    *gids,
		   int		    nglyphs);

void
XRenderCompositeString8 (Display		    *dpy,
			 int			    op,
			 Picture		    src,
			 Picture		    dst,
			 _Xconst XRenderPictFormat  *maskFormat,
			 GlyphSet		    glyphset,
			 int			    xSrc,
			 int			    ySrc,
			 int			    xDst,
			 int			    yDst,
			 _Xconst char		    *string,
			 int			    nchar);

void
XRenderCompositeString16 (Display		    *dpy,
			  int			    op,
			  Picture		    src,
			  Picture		    dst,
			  _Xconst XRenderPictFormat *maskFormat,
			  GlyphSet		    glyphset,
			  int			    xSrc,
			  int			    ySrc,
			  int			    xDst,
			  int			    yDst,
			  _Xconst unsigned short    *string,
			  int			    nchar);

void
XRenderCompositeString32 (Display		    *dpy,
			  int			    op,
			  Picture		    src,
			  Picture		    dst,
			  _Xconst XRenderPictFormat *maskFormat,
			  GlyphSet		    glyphset,
			  int			    xSrc,
			  int			    ySrc,
			  int			    xDst,
			  int			    yDst,
			  _Xconst unsigned int	    *string,
			  int			    nchar);

void
XRenderCompositeText8 (Display			    *dpy,
		       int			    op,
		       Picture			    src,
		       Picture			    dst,
		       _Xconst XRenderPictFormat    *maskFormat,
		       int			    xSrc,
		       int			    ySrc,
		       int			    xDst,
		       int			    yDst,
		       _Xconst XGlyphElt8	    *elts,
		       int			    nelt);

void
XRenderCompositeText16 (Display			    *dpy,
			int			    op,
			Picture			    src,
			Picture			    dst,
			_Xconst XRenderPictFormat   *maskFormat,
			int			    xSrc,
			int			    ySrc,
			int			    xDst,
			int			    yDst,
			_Xconst XGlyphElt16	    *elts,
			int			    nelt);

void
XRenderCompositeText32 (Display			    *dpy,
			int			    op,
			Picture			    src,
			Picture			    dst,
			_Xconst XRenderPictFormat   *maskFormat,
			int			    xSrc,
			int			    ySrc,
			int			    xDst,
			int			    yDst,
			_Xconst XGlyphElt32	    *elts,
			int			    nelt);

void
XRenderFillRectangle (Display		    *dpy,
		      int		    op,
		      Picture		    dst,
		      _Xconst XRenderColor  *color,
		      int		    x,
		      int		    y,
		      unsigned int	    width,
		      unsigned int	    height);

void
XRenderFillRectangles (Display		    *dpy,
		       int		    op,
		       Picture		    dst,
		       _Xconst XRenderColor *color,
		       _Xconst XRectangle   *rectangles,
		       int		    n_rects);

void
XRenderCompositeTrapezoids (Display		*dpy,
			    int			op,
			    Picture		src,
			    Picture		dst,
			    _Xconst XRenderPictFormat	*maskFormat,
			    int			xSrc,
			    int			ySrc,
			    _Xconst XTrapezoid	*traps,
			    int			ntrap);

void
XRenderCompositeTriangles (Display		*dpy,
			   int			op,
			   Picture		src,
			   Picture		dst,
			    _Xconst XRenderPictFormat	*maskFormat,
			   int			xSrc,
			   int			ySrc,
			   _Xconst XTriangle	*triangles,
			   int			ntriangle);

void
XRenderCompositeTriStrip (Display		*dpy,
			  int			op,
			  Picture		src,
			  Picture		dst,
			    _Xconst XRenderPictFormat	*maskFormat,
			  int			xSrc,
			  int			ySrc,
			  _Xconst XPointFixed	*points,
			  int			npoint);

void
XRenderCompositeTriFan (Display			*dpy,
			int			op,
			Picture			src,
			Picture			dst,
			_Xconst XRenderPictFormat	*maskFormat,
			int			xSrc,
			int			ySrc,
			_Xconst XPointFixed	*points,
			int			npoint);

void
XRenderCompositeDoublePoly (Display		    *dpy,
			    int			    op,
			    Picture		    src,
			    Picture		    dst,
			    _Xconst XRenderPictFormat	*maskFormat,
			    int			    xSrc,
			    int			    ySrc,
			    int			    xDst,
			    int			    yDst,
			    _Xconst XPointDouble    *fpoints,
			    int			    npoints,
			    int			    winding);
Status
XRenderParseColor(Display	*dpy,
		  char		*spec,
		  XRenderColor	*def);

Cursor
XRenderCreateCursor (Display	    *dpy,
		     Picture	    source,
		     unsigned int   x,
		     unsigned int   y);

XFilters *
XRenderQueryFilters (Display *dpy, Drawable drawable);

void
XRenderSetPictureFilter (Display    *dpy,
			 Picture    picture,
			 const char *filter,
			 XFixed	    *params,
			 int	    nparams);

Cursor
XRenderCreateAnimCursor (Display	*dpy,
			 int		ncursor,
			 XAnimCursor	*cursors);


void
XRenderAddTraps (Display	    *dpy,
		 Picture	    picture,
		 int		    xOff,
		 int		    yOff,
		 _Xconst XTrap	    *traps,
		 int		    ntrap);

Picture XRenderCreateSolidFill (Display *dpy,
                                const XRenderColor *color);

Picture XRenderCreateLinearGradient (Display *dpy,
                                     const XLinearGradient *gradient,
                                     const XFixed *stops,
                                     const XRenderColor *colors,
                                     int nstops);

Picture XRenderCreateRadialGradient (Display *dpy,
                                     const XRadialGradient *gradient,
                                     const XFixed *stops,
                                     const XRenderColor *colors,
                                     int nstops);

Picture XRenderCreateConicalGradient (Display *dpy,
                                      const XConicalGradient *gradient,
                                      const XFixed *stops,
                                      const XRenderColor *colors,
                                      int nstops);

_XFUNCPROTOEND

#endif /* _XRENDER_H_ */
