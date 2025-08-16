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

#ifndef _SHAPE_H_
#define _SHAPE_H_

#include <X11/Xfuncproto.h>
#include <X11/extensions/shapeconst.h>

#ifndef _SHAPE_SERVER_
#include <X11/Xutil.h>

typedef struct {
    int	type;		    /* of event */
    unsigned long serial;   /* # of last request processed by server */
    Bool send_event;	    /* true if this came frome a SendEvent request */
    Display *display;	    /* Display the event was read from */
    Window window;	    /* window of event */
    int kind;		    /* ShapeBounding or ShapeClip */
    int x, y;		    /* extents of new region */
    unsigned width, height;
    Time time;		    /* server timestamp when region changed */
    Bool shaped;	    /* true if the region exists */
} XShapeEvent;

_XFUNCPROTOBEGIN

extern Bool XShapeQueryExtension (
    Display*	/* display */,
    int*	/* event_base */,
    int*	/* error_base */
);

extern Status XShapeQueryVersion (
    Display*	/* display */,
    int*	/* major_version */,
    int*	/* minor_version */
);

extern void XShapeCombineRegion (
    Display*	/* display */,
    Window	/* dest */,
    int		/* dest_kind */,
    int		/* x_off */,
    int		/* y_off */,
    Region	/* region */,
    int		/* op */
);

extern void XShapeCombineRectangles (
    Display*	/* display */,
    Window	/* dest */,
    int		/* dest_kind */,
    int		/* x_off */,
    int		/* y_off */,
    XRectangle*	/* rectangles */,
    int		/* n_rects */,
    int		/* op */,
    int		/* ordering */
);

extern void XShapeCombineMask (
    Display*	/* display */,
    Window	/* dest */,
    int		/* dest_kind */,
    int		/* x_off */,
    int		/* y_off */,
    Pixmap	/* src */,
    int		/* op */
);

extern void XShapeCombineShape (
    Display*	/* display */,
    Window	/* dest */,
    int		/* dest_kind */,
    int		/* x_off */,
    int		/* y_off */,
    Window	/* src */,
    int		/* src_kind */,
    int		/* op */
);

extern void XShapeOffsetShape (
    Display*	/* display */,
    Window	/* dest */,
    int		/* dest_kind */,
    int		/* x_off */,
    int		/* y_off */
);

extern Status XShapeQueryExtents (
    Display*		/* display */,
    Window		/* window */,
    Bool*		/* bounding_shaped */,
    int*		/* x_bounding */,
    int*		/* y_bounding */,
    unsigned int*	/* w_bounding */,
    unsigned int*	/* h_bounding */,
    Bool*		/* clip_shaped */,
    int*		/* x_clip */,
    int*		/* y_clip */,
    unsigned int*	/* w_clip */,
    unsigned int*	/* h_clip */
);

extern void XShapeSelectInput (
    Display*		/* display */,
    Window		/* window */,
    unsigned long	/* mask */
);

extern unsigned long XShapeInputSelected (
    Display*	/* display */,
    Window	/* window */
);

extern XRectangle *XShapeGetRectangles (
    Display*	/* display */,
    Window	/* window */,
    int		/* kind */,
    int*	/* count */,
    int*	/* ordering */
);

_XFUNCPROTOEND

#endif /* !_SHAPE_SERVER_ */

#endif /* _SHAPE_H_ */
