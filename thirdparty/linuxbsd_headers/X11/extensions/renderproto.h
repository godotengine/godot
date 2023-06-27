/*
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

#ifndef _XRENDERP_H_
#define _XRENDERP_H_

#include <X11/Xmd.h>
#include <X11/extensions/render.h>

#define Window CARD32
#define Drawable CARD32
#define Font CARD32
#define Pixmap CARD32
#define Cursor CARD32
#define Colormap CARD32
#define GContext CARD32
#define Atom CARD32
#define VisualID CARD32
#define Time CARD32
#define KeyCode CARD8
#define KeySym CARD32

#define Picture	    CARD32
#define PictFormat  CARD32
#define Fixed	    INT32
#define Glyphset    CARD32

/*
 * data structures
 */

typedef struct {
    CARD16  red;
    CARD16  redMask;
    CARD16  green;
    CARD16  greenMask;
    CARD16  blue;
    CARD16  blueMask;
    CARD16  alpha;
    CARD16  alphaMask;
} xDirectFormat;

#define sz_xDirectFormat    16

typedef struct {
    PictFormat	id;
    CARD8	type;
    CARD8	depth;
    CARD16	pad1;
    xDirectFormat   direct;
    Colormap	colormap;
} xPictFormInfo;

#define sz_xPictFormInfo    28

typedef struct {
    VisualID	visual;
    PictFormat	format;
} xPictVisual;

#define sz_xPictVisual	    8

typedef struct {
    CARD8	depth;
    CARD8	pad1;
    CARD16	nPictVisuals;
    CARD32	pad2;
} xPictDepth;

#define sz_xPictDepth	8

typedef struct {
    CARD32	nDepth;
    PictFormat	fallback;
} xPictScreen;

#define sz_xPictScreen	8

typedef struct {
    CARD32	pixel;
    CARD16	red;
    CARD16	green;
    CARD16	blue;
    CARD16	alpha;
} xIndexValue;

#define sz_xIndexValue	12

typedef struct {
    CARD16	red;
    CARD16	green;
    CARD16	blue;
    CARD16	alpha;
} xRenderColor;

#define sz_xRenderColor	8

typedef struct {
    Fixed	x;
    Fixed	y;
} xPointFixed;

#define sz_xPointFixed	8

typedef struct {
    xPointFixed	p1;
    xPointFixed p2;
} xLineFixed;

#define sz_xLineFixed	16

typedef struct {
    xPointFixed	p1, p2, p3;
} xTriangle;

#define sz_xTriangle	24

typedef struct {
    Fixed	top;
    Fixed	bottom;
    xLineFixed	left;
    xLineFixed	right;
} xTrapezoid;

#define sz_xTrapezoid	40

typedef struct {
    CARD16  width;
    CARD16  height;
    INT16   x;
    INT16   y;
    INT16   xOff;
    INT16   yOff;
} xGlyphInfo;

#define sz_xGlyphInfo	12

typedef struct {
    CARD8   len;
    CARD8   pad1;
    CARD16  pad2;
    INT16   deltax;
    INT16   deltay;
} xGlyphElt;

#define sz_xGlyphElt	8

typedef struct {
    Fixed   l, r, y;
} xSpanFix;

#define sz_xSpanFix	12

typedef struct {
    xSpanFix	top, bot;
} xTrap;

#define sz_xTrap	24

/*
 * requests and replies
 */
typedef struct {
    CARD8   reqType;
    CARD8   renderReqType;
    CARD16  length;
    CARD32  majorVersion;
    CARD32  minorVersion;
} xRenderQueryVersionReq;

#define sz_xRenderQueryVersionReq   12

typedef struct {
    BYTE    type;   /* X_Reply */
    BYTE    pad1;
    CARD16  sequenceNumber;
    CARD32  length;
    CARD32  majorVersion;
    CARD32  minorVersion;
    CARD32  pad2;
    CARD32  pad3;
    CARD32  pad4;
    CARD32  pad5;
} xRenderQueryVersionReply;

#define sz_xRenderQueryVersionReply	32

typedef struct {
    CARD8   reqType;
    CARD8   renderReqType;
    CARD16  length;
} xRenderQueryPictFormatsReq;

#define sz_xRenderQueryPictFormatsReq	4

typedef struct {
    BYTE    type;   /* X_Reply */
    BYTE    pad1;
    CARD16  sequenceNumber;
    CARD32  length;
    CARD32  numFormats;
    CARD32  numScreens;
    CARD32  numDepths;
    CARD32  numVisuals;
    CARD32  numSubpixel;	    /* Version 0.6 */
    CARD32  pad5;
} xRenderQueryPictFormatsReply;

#define sz_xRenderQueryPictFormatsReply	32

typedef struct {
    CARD8   reqType;
    CARD8   renderReqType;
    CARD16  length;
    PictFormat	format;
} xRenderQueryPictIndexValuesReq;

#define sz_xRenderQueryPictIndexValuesReq   8

typedef struct {
    BYTE    type;   /* X_Reply */
    BYTE    pad1;
    CARD16  sequenceNumber;
    CARD32  length;
    CARD32  numIndexValues;
    CARD32  pad2;
    CARD32  pad3;
    CARD32  pad4;
    CARD32  pad5;
    CARD32  pad6;
} xRenderQueryPictIndexValuesReply;

#define sz_xRenderQueryPictIndexValuesReply 32

typedef struct {
    CARD8	reqType;
    CARD8	renderReqType;
    CARD16	length;
    Picture	pid;
    Drawable	drawable;
    PictFormat	format;
    CARD32	mask;
} xRenderCreatePictureReq;

#define sz_xRenderCreatePictureReq	    20

typedef struct {
    CARD8	reqType;
    CARD8	renderReqType;
    CARD16	length;
    Picture	picture;
    CARD32	mask;
} xRenderChangePictureReq;

#define sz_xRenderChangePictureReq	    12

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    Picture     picture;
    INT16       xOrigin;
    INT16       yOrigin;
} xRenderSetPictureClipRectanglesReq;

#define sz_xRenderSetPictureClipRectanglesReq	    12

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    Picture     picture;
} xRenderFreePictureReq;

#define sz_xRenderFreePictureReq	    8

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    CARD8	op;
    CARD8	pad1;
    CARD16	pad2;
    Picture	src;
    Picture	mask;
    Picture	dst;
    INT16	xSrc;
    INT16	ySrc;
    INT16	xMask;
    INT16	yMask;
    INT16	xDst;
    INT16	yDst;
    CARD16	width;
    CARD16	height;
} xRenderCompositeReq;

#define sz_xRenderCompositeReq		    36

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    Picture	src;
    Picture	dst;
    CARD32	colorScale;
    CARD32	alphaScale;
    INT16	xSrc;
    INT16	ySrc;
    INT16	xDst;
    INT16	yDst;
    CARD16	width;
    CARD16	height;
} xRenderScaleReq;

#define sz_xRenderScaleReq			    32

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    CARD8	op;
    CARD8	pad1;
    CARD16	pad2;
    Picture	src;
    Picture	dst;
    PictFormat	maskFormat;
    INT16	xSrc;
    INT16	ySrc;
} xRenderTrapezoidsReq;

#define sz_xRenderTrapezoidsReq			    24

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    CARD8	op;
    CARD8	pad1;
    CARD16	pad2;
    Picture	src;
    Picture	dst;
    PictFormat	maskFormat;
    INT16	xSrc;
    INT16	ySrc;
} xRenderTrianglesReq;

#define sz_xRenderTrianglesReq			    24

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    CARD8	op;
    CARD8	pad1;
    CARD16	pad2;
    Picture	src;
    Picture	dst;
    PictFormat	maskFormat;
    INT16	xSrc;
    INT16	ySrc;
} xRenderTriStripReq;

#define sz_xRenderTriStripReq			    24

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    CARD8	op;
    CARD8	pad1;
    CARD16	pad2;
    Picture	src;
    Picture	dst;
    PictFormat	maskFormat;
    INT16	xSrc;
    INT16	ySrc;
} xRenderTriFanReq;

#define sz_xRenderTriFanReq			    24

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    Glyphset    gsid;
    PictFormat  format;
} xRenderCreateGlyphSetReq;

#define sz_xRenderCreateGlyphSetReq		    12

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    Glyphset    gsid;
    Glyphset    existing;
} xRenderReferenceGlyphSetReq;

#define sz_xRenderReferenceGlyphSetReq		    24

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    Glyphset    glyphset;
} xRenderFreeGlyphSetReq;

#define sz_xRenderFreeGlyphSetReq		    8

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    Glyphset    glyphset;
    CARD32	nglyphs;
} xRenderAddGlyphsReq;

#define sz_xRenderAddGlyphsReq			    12

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    Glyphset    glyphset;
} xRenderFreeGlyphsReq;

#define sz_xRenderFreeGlyphsReq			    8

typedef struct {
    CARD8       reqType;
    CARD8       renderReqType;
    CARD16      length;
    CARD8	op;
    CARD8	pad1;
    CARD16	pad2;
    Picture	src;
    Picture	dst;
    PictFormat	maskFormat;
    Glyphset	glyphset;
    INT16	xSrc;
    INT16	ySrc;
} xRenderCompositeGlyphsReq, xRenderCompositeGlyphs8Req,
xRenderCompositeGlyphs16Req, xRenderCompositeGlyphs32Req;

#define sz_xRenderCompositeGlyphs8Req		    28
#define sz_xRenderCompositeGlyphs16Req		    28
#define sz_xRenderCompositeGlyphs32Req		    28

/* 0.1 and higher */

typedef struct {
    CARD8	reqType;
    CARD8       renderReqType;
    CARD16      length;
    CARD8	op;
    CARD8	pad1;
    CARD16	pad2;
    Picture	dst;
    xRenderColor    color;
} xRenderFillRectanglesReq;

#define sz_xRenderFillRectanglesReq		    20

/* 0.5 and higher */

typedef struct {
    CARD8	reqType;
    CARD8	renderReqType;
    CARD16	length;
    Cursor	cid;
    Picture	src;
    CARD16	x;
    CARD16	y;
} xRenderCreateCursorReq;

#define sz_xRenderCreateCursorReq		    16

/* 0.6 and higher */

/*
 * This can't use an array because 32-bit values may be in bitfields
 */
typedef struct {
    Fixed	matrix11;
    Fixed	matrix12;
    Fixed	matrix13;
    Fixed	matrix21;
    Fixed	matrix22;
    Fixed	matrix23;
    Fixed	matrix31;
    Fixed	matrix32;
    Fixed	matrix33;
} xRenderTransform;

#define sz_xRenderTransform 36

typedef struct {
    CARD8		reqType;
    CARD8		renderReqType;
    CARD16		length;
    Picture		picture;
    xRenderTransform	transform;
} xRenderSetPictureTransformReq;

#define sz_xRenderSetPictureTransformReq	    44

typedef struct {
    CARD8		reqType;
    CARD8		renderReqType;
    CARD16		length;
    Drawable		drawable;
} xRenderQueryFiltersReq;

#define sz_xRenderQueryFiltersReq		    8

typedef struct {
    BYTE    type;   /* X_Reply */
    BYTE    pad1;
    CARD16  sequenceNumber;
    CARD32  length;
    CARD32  numAliases;	/* LISTofCARD16 */
    CARD32  numFilters;	/* LISTofSTRING8 */
    CARD32  pad2;
    CARD32  pad3;
    CARD32  pad4;
    CARD32  pad5;
} xRenderQueryFiltersReply;

#define sz_xRenderQueryFiltersReply		    32

typedef struct {
    CARD8		reqType;
    CARD8		renderReqType;
    CARD16		length;
    Picture		picture;
    CARD16		nbytes; /* number of bytes in name */
    CARD16		pad;
} xRenderSetPictureFilterReq;

#define sz_xRenderSetPictureFilterReq		    12

/* 0.8 and higher */

typedef struct {
    Cursor		cursor;
    CARD32		delay;
} xAnimCursorElt;

#define sz_xAnimCursorElt			    8

typedef struct {
    CARD8		reqType;
    CARD8		renderReqType;
    CARD16		length;
    Cursor		cid;
} xRenderCreateAnimCursorReq;

#define sz_xRenderCreateAnimCursorReq		    8

/* 0.9 and higher */

typedef struct {
    CARD8		reqType;
    CARD8		renderReqType;
    CARD16		length;
    Picture		picture;
    INT16		xOff;
    INT16		yOff;
} xRenderAddTrapsReq;

#define sz_xRenderAddTrapsReq			    12

/* 0.10 and higher */

typedef struct {
    CARD8	reqType;
    CARD8	renderReqType;
    CARD16	length;
    Picture	pid;
    xRenderColor color;
} xRenderCreateSolidFillReq;

#define sz_xRenderCreateSolidFillReq                 16

typedef struct {
    CARD8	reqType;
    CARD8	renderReqType;
    CARD16	length;
    Picture	pid;
    xPointFixed p1;
    xPointFixed p2;
    CARD32      nStops;
} xRenderCreateLinearGradientReq;

#define sz_xRenderCreateLinearGradientReq                 28

typedef struct {
    CARD8	reqType;
    CARD8	renderReqType;
    CARD16	length;
    Picture	pid;
    xPointFixed inner;
    xPointFixed outer;
    Fixed       inner_radius;
    Fixed       outer_radius;
    CARD32      nStops;
} xRenderCreateRadialGradientReq;

#define sz_xRenderCreateRadialGradientReq                 36

typedef struct {
    CARD8	reqType;
    CARD8	renderReqType;
    CARD16	length;
    Picture	pid;
    xPointFixed center;
    Fixed       angle; /* in degrees */
    CARD32      nStops;
} xRenderCreateConicalGradientReq;

#define sz_xRenderCreateConicalGradientReq                 24

#undef Window
#undef Drawable
#undef Font
#undef Pixmap
#undef Cursor
#undef Colormap
#undef GContext
#undef Atom
#undef VisualID
#undef Time
#undef KeyCode
#undef KeySym

#undef Picture
#undef PictFormat
#undef Fixed
#undef Glyphset

#endif /* _XRENDERP_H_ */
