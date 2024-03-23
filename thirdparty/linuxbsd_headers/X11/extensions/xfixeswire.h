/*
 * Copyright (c) 2006, Oracle and/or its affiliates. All rights reserved.
 * Copyright 2010 Red Hat, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
/*
 * Copyright © 2002 Keith Packard, member of The XFree86 Project, Inc.
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


#ifndef _XFIXESWIRE_H_
#define _XFIXESWIRE_H_

#define XFIXES_NAME	"XFIXES"
#define XFIXES_MAJOR	6
#define XFIXES_MINOR	0

/*************** Version 1 ******************/
#define X_XFixesQueryVersion		    0
#define X_XFixesChangeSaveSet		    1
#define X_XFixesSelectSelectionInput	    2
#define X_XFixesSelectCursorInput	    3
#define X_XFixesGetCursorImage		    4
/*************** Version 2 ******************/
#define X_XFixesCreateRegion		    5
#define X_XFixesCreateRegionFromBitmap	    6
#define X_XFixesCreateRegionFromWindow	    7
#define X_XFixesCreateRegionFromGC	    8
#define X_XFixesCreateRegionFromPicture	    9
#define X_XFixesDestroyRegion		    10
#define X_XFixesSetRegion		    11
#define X_XFixesCopyRegion		    12
#define X_XFixesUnionRegion		    13
#define X_XFixesIntersectRegion		    14
#define X_XFixesSubtractRegion		    15
#define X_XFixesInvertRegion		    16
#define X_XFixesTranslateRegion		    17
#define X_XFixesRegionExtents		    18
#define X_XFixesFetchRegion		    19
#define X_XFixesSetGCClipRegion		    20
#define X_XFixesSetWindowShapeRegion	    21
#define X_XFixesSetPictureClipRegion	    22
#define X_XFixesSetCursorName		    23
#define X_XFixesGetCursorName		    24
#define X_XFixesGetCursorImageAndName	    25
#define X_XFixesChangeCursor		    26
#define X_XFixesChangeCursorByName	    27
/*************** Version 3 ******************/
#define X_XFixesExpandRegion		    28
/*************** Version 4 ******************/
#define X_XFixesHideCursor		    29
#define X_XFixesShowCursor		    30
/*************** Version 5 ******************/
#define X_XFixesCreatePointerBarrier	    31
#define X_XFixesDestroyPointerBarrier	    32
/*************** Version 6 ******************/
#define X_XFixesSetClientDisconnectMode	    33
#define X_XFixesGetClientDisconnectMode	    34

#define XFixesNumberRequests		    (X_XFixesGetClientDisconnectMode+1)

/* Selection events share one event number */
#define XFixesSelectionNotify		    0

/* Within the selection, the 'subtype' field distinguishes */
#define XFixesSetSelectionOwnerNotify	    0
#define XFixesSelectionWindowDestroyNotify  1
#define XFixesSelectionClientCloseNotify    2

#define XFixesSetSelectionOwnerNotifyMask	(1L << 0)
#define XFixesSelectionWindowDestroyNotifyMask	(1L << 1)
#define XFixesSelectionClientCloseNotifyMask	(1L << 2)

/* There's only one cursor event so far */
#define XFixesCursorNotify		    1

#define XFixesDisplayCursorNotify	    0

#define XFixesDisplayCursorNotifyMask	    (1L << 0)

#define XFixesNumberEvents		    (2)

/* errors */
#define BadRegion			    0
#define BadBarrier			    1
#define XFixesNumberErrors		    (BadBarrier+1)

#define SaveSetNearest			    0
#define SaveSetRoot			    1

#define SaveSetMap			    0
#define SaveSetUnmap			    1

/*************** Version 2 ******************/

#define WindowRegionBounding		    0
#define WindowRegionClip		    1

/*************** Version 5 ******************/

#define BarrierPositiveX		    (1L << 0)
#define BarrierPositiveY		    (1L << 1)
#define BarrierNegativeX		    (1L << 2)
#define BarrierNegativeY		    (1L << 3)

/*************** Version 6 ******************/

/* The default server behaviour */
#define XFixesClientDisconnectFlagDefault   0
/* The server may disconnect this client to shut down */
#define XFixesClientDisconnectFlagTerminate  (1L << 0)

#endif	/* _XFIXESWIRE_H_ */
