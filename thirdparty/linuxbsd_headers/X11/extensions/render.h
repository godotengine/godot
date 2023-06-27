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

#ifndef _RENDER_H_
#define _RENDER_H_

#include <X11/Xdefs.h>

typedef XID		Glyph;
typedef XID		GlyphSet;
typedef XID		Picture;
typedef XID		PictFormat;

#define RENDER_NAME	"RENDER"
#define RENDER_MAJOR	0
#define RENDER_MINOR	11

#define X_RenderQueryVersion		    0
#define X_RenderQueryPictFormats	    1
#define X_RenderQueryPictIndexValues	    2	/* 0.7 */
#define X_RenderQueryDithers		    3
#define X_RenderCreatePicture		    4
#define X_RenderChangePicture		    5
#define X_RenderSetPictureClipRectangles    6
#define X_RenderFreePicture		    7
#define X_RenderComposite		    8
#define X_RenderScale			    9
#define X_RenderTrapezoids		    10
#define X_RenderTriangles		    11
#define X_RenderTriStrip		    12
#define X_RenderTriFan			    13
#define X_RenderColorTrapezoids		    14
#define X_RenderColorTriangles		    15
/* #define X_RenderTransform		    16 */
#define X_RenderCreateGlyphSet		    17
#define X_RenderReferenceGlyphSet	    18
#define X_RenderFreeGlyphSet		    19
#define X_RenderAddGlyphs		    20
#define X_RenderAddGlyphsFromPicture	    21
#define X_RenderFreeGlyphs		    22
#define X_RenderCompositeGlyphs8	    23
#define X_RenderCompositeGlyphs16	    24
#define X_RenderCompositeGlyphs32	    25
#define X_RenderFillRectangles		    26
/* 0.5 */
#define X_RenderCreateCursor		    27
/* 0.6 */
#define X_RenderSetPictureTransform	    28
#define X_RenderQueryFilters		    29
#define X_RenderSetPictureFilter	    30
/* 0.8 */
#define X_RenderCreateAnimCursor	    31
/* 0.9 */
#define X_RenderAddTraps		    32
/* 0.10 */
#define X_RenderCreateSolidFill             33
#define X_RenderCreateLinearGradient        34
#define X_RenderCreateRadialGradient        35
#define X_RenderCreateConicalGradient       36
#define RenderNumberRequests		    (X_RenderCreateConicalGradient+1)

#define BadPictFormat			    0
#define BadPicture			    1
#define BadPictOp			    2
#define BadGlyphSet			    3
#define BadGlyph			    4
#define RenderNumberErrors		    (BadGlyph+1)

#define PictTypeIndexed			    0
#define PictTypeDirect			    1

#define PictOpMinimum			    0
#define PictOpClear			    0
#define PictOpSrc			    1
#define PictOpDst			    2
#define PictOpOver			    3
#define PictOpOverReverse		    4
#define PictOpIn			    5
#define PictOpInReverse			    6
#define PictOpOut			    7
#define PictOpOutReverse		    8
#define PictOpAtop			    9
#define PictOpAtopReverse		    10
#define PictOpXor			    11
#define PictOpAdd			    12
#define PictOpSaturate			    13
#define PictOpMaximum			    13

/*
 * Operators only available in version 0.2
 */
#define PictOpDisjointMinimum			    0x10
#define PictOpDisjointClear			    0x10
#define PictOpDisjointSrc			    0x11
#define PictOpDisjointDst			    0x12
#define PictOpDisjointOver			    0x13
#define PictOpDisjointOverReverse		    0x14
#define PictOpDisjointIn			    0x15
#define PictOpDisjointInReverse			    0x16
#define PictOpDisjointOut			    0x17
#define PictOpDisjointOutReverse		    0x18
#define PictOpDisjointAtop			    0x19
#define PictOpDisjointAtopReverse		    0x1a
#define PictOpDisjointXor			    0x1b
#define PictOpDisjointMaximum			    0x1b

#define PictOpConjointMinimum			    0x20
#define PictOpConjointClear			    0x20
#define PictOpConjointSrc			    0x21
#define PictOpConjointDst			    0x22
#define PictOpConjointOver			    0x23
#define PictOpConjointOverReverse		    0x24
#define PictOpConjointIn			    0x25
#define PictOpConjointInReverse			    0x26
#define PictOpConjointOut			    0x27
#define PictOpConjointOutReverse		    0x28
#define PictOpConjointAtop			    0x29
#define PictOpConjointAtopReverse		    0x2a
#define PictOpConjointXor			    0x2b
#define PictOpConjointMaximum			    0x2b

/*
 * Operators only available in version 0.11
 */
#define PictOpBlendMinimum			    0x30
#define PictOpMultiply				    0x30
#define PictOpScreen				    0x31
#define PictOpOverlay				    0x32
#define PictOpDarken				    0x33
#define PictOpLighten				    0x34
#define PictOpColorDodge			    0x35
#define PictOpColorBurn				    0x36
#define PictOpHardLight				    0x37
#define PictOpSoftLight				    0x38
#define PictOpDifference			    0x39
#define PictOpExclusion				    0x3a
#define PictOpHSLHue				    0x3b
#define PictOpHSLSaturation			    0x3c
#define PictOpHSLColor				    0x3d
#define PictOpHSLLuminosity			    0x3e
#define PictOpBlendMaximum			    0x3e

#define PolyEdgeSharp			    0
#define PolyEdgeSmooth			    1

#define PolyModePrecise			    0
#define PolyModeImprecise		    1

#define CPRepeat			    (1 << 0)
#define CPAlphaMap			    (1 << 1)
#define CPAlphaXOrigin			    (1 << 2)
#define CPAlphaYOrigin			    (1 << 3)
#define CPClipXOrigin			    (1 << 4)
#define CPClipYOrigin			    (1 << 5)
#define CPClipMask			    (1 << 6)
#define CPGraphicsExposure		    (1 << 7)
#define CPSubwindowMode			    (1 << 8)
#define CPPolyEdge			    (1 << 9)
#define CPPolyMode			    (1 << 10)
#define CPDither			    (1 << 11)
#define CPComponentAlpha		    (1 << 12)
#define CPLastBit			    12

/* Filters included in 0.6 */
#define FilterNearest			    "nearest"
#define FilterBilinear			    "bilinear"
/* Filters included in 0.10 */
#define FilterConvolution		    "convolution"

#define FilterFast			    "fast"
#define FilterGood			    "good"
#define FilterBest			    "best"

#define FilterAliasNone			    -1

/* Subpixel orders included in 0.6 */
#define SubPixelUnknown			    0
#define SubPixelHorizontalRGB		    1
#define SubPixelHorizontalBGR		    2
#define SubPixelVerticalRGB		    3
#define SubPixelVerticalBGR		    4
#define SubPixelNone			    5

/* Extended repeat attributes included in 0.10 */
#define RepeatNone                          0
#define RepeatNormal                        1
#define RepeatPad                           2
#define RepeatReflect                       3

#endif	/* _RENDER_H_ */
