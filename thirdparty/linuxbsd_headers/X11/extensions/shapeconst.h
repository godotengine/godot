/************************************************************

Copyright 1989, 1998  The Open Group

Permission to use, copy, modify, distribute, and sell this software and its
documentation for any purpose is hereby granted without fee, provided that
the above copyright notice appear in all copies and that both that
copyright notice and this permission notice appear in supporting
documentation.

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
OPEN GROUP BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of The Open Group shall not be
used in advertising or otherwise to promote the sale, use or other dealings
in this Software without prior written authorization from The Open Group.

********************************************************/

#ifndef _SHAPECONST_H_
#define _SHAPECONST_H_

/*
 * Protocol requests constants and alignment values
 * These would really be in SHAPE's X.h and Xproto.h equivalents
 */

#define SHAPENAME "SHAPE"

#define SHAPE_MAJOR_VERSION	1	/* current version numbers */
#define SHAPE_MINOR_VERSION	1

#define ShapeSet			0
#define ShapeUnion			1
#define ShapeIntersect			2
#define ShapeSubtract			3
#define ShapeInvert			4

#define ShapeBounding			0
#define ShapeClip			1
#define ShapeInput			2

#define ShapeNotifyMask			(1L << 0)
#define ShapeNotify			0

#define ShapeNumberEvents		(ShapeNotify + 1)

#endif /* _SHAPECONST_H_ */
