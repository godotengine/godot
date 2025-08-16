
/***********************************************************

Copyright 1987, 1998  The Open Group

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


Copyright 1987 by Digital Equipment Corporation, Maynard, Massachusetts.

                        All Rights Reserved

Permission to use, copy, modify, and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appear in all copies and that
both that copyright notice and this permission notice appear in
supporting documentation, and that the name of Digital not be
used in advertising or publicity pertaining to distribution of the
software without specific, written prior permission.

DIGITAL DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
DIGITAL BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
SOFTWARE.

******************************************************************/

#ifndef _X11_XUTIL_H_
#define _X11_XUTIL_H_

/* You must include <X11/Xlib.h> before including this file */
#include <X11/Xlib.h>
#include <X11/keysym.h>

/* The Xlib structs are full of implicit padding to properly align members.
   We can't clean that up without breaking ABI, so tell clang not to bother
   complaining about it. */
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

/*
 * Bitmask returned by XParseGeometry().  Each bit tells if the corresponding
 * value (x, y, width, height) was found in the parsed string.
 */
#define NoValue		0x0000
#define XValue  	0x0001
#define YValue		0x0002
#define WidthValue  	0x0004
#define HeightValue  	0x0008
#define AllValues 	0x000F
#define XNegative 	0x0010
#define YNegative 	0x0020

/*
 * new version containing base_width, base_height, and win_gravity fields;
 * used with WM_NORMAL_HINTS.
 */
typedef struct {
    	long flags;	/* marks which fields in this structure are defined */
	int x, y;		/* obsolete for new window mgrs, but clients */
	int width, height;	/* should set so old wm's don't mess up */
	int min_width, min_height;
	int max_width, max_height;
    	int width_inc, height_inc;
	struct {
		int x;	/* numerator */
		int y;	/* denominator */
	} min_aspect, max_aspect;
	int base_width, base_height;		/* added by ICCCM version 1 */
	int win_gravity;			/* added by ICCCM version 1 */
} XSizeHints;

/*
 * The next block of definitions are for window manager properties that
 * clients and applications use for communication.
 */

/* flags argument in size hints */
#define USPosition	(1L << 0) /* user specified x, y */
#define USSize		(1L << 1) /* user specified width, height */

#define PPosition	(1L << 2) /* program specified position */
#define PSize		(1L << 3) /* program specified size */
#define PMinSize	(1L << 4) /* program specified minimum size */
#define PMaxSize	(1L << 5) /* program specified maximum size */
#define PResizeInc	(1L << 6) /* program specified resize increments */
#define PAspect		(1L << 7) /* program specified min and max aspect ratios */
#define PBaseSize	(1L << 8) /* program specified base for incrementing */
#define PWinGravity	(1L << 9) /* program specified window gravity */

/* obsolete */
#define PAllHints (PPosition|PSize|PMinSize|PMaxSize|PResizeInc|PAspect)



typedef struct {
	long flags;	/* marks which fields in this structure are defined */
	Bool input;	/* does this application rely on the window manager to
			get keyboard input? */
	int initial_state;	/* see below */
	Pixmap icon_pixmap;	/* pixmap to be used as icon */
	Window icon_window; 	/* window to be used as icon */
	int icon_x, icon_y; 	/* initial position of icon */
	Pixmap icon_mask;	/* icon mask bitmap */
	XID window_group;	/* id of related window group */
	/* this structure may be extended in the future */
} XWMHints;

/* definition for flags of XWMHints */

#define InputHint 		(1L << 0)
#define StateHint 		(1L << 1)
#define IconPixmapHint		(1L << 2)
#define IconWindowHint		(1L << 3)
#define IconPositionHint 	(1L << 4)
#define IconMaskHint		(1L << 5)
#define WindowGroupHint		(1L << 6)
#define AllHints (InputHint|StateHint|IconPixmapHint|IconWindowHint| \
IconPositionHint|IconMaskHint|WindowGroupHint)
#define XUrgencyHint		(1L << 8)

/* definitions for initial window state */
#define WithdrawnState 0	/* for windows that are not mapped */
#define NormalState 1	/* most applications want to start this way */
#define IconicState 3	/* application wants to start as an icon */

/*
 * Obsolete states no longer defined by ICCCM
 */
#define DontCareState 0	/* don't know or care */
#define ZoomState 2	/* application wants to start zoomed */
#define InactiveState 4	/* application believes it is seldom used; */
			/* some wm's may put it on inactive menu */


/*
 * new structure for manipulating TEXT properties; used with WM_NAME,
 * WM_ICON_NAME, WM_CLIENT_MACHINE, and WM_COMMAND.
 */
typedef struct {
    unsigned char *value;		/* same as Property routines */
    Atom encoding;			/* prop type */
    int format;				/* prop data format: 8, 16, or 32 */
    unsigned long nitems;		/* number of data items in value */
} XTextProperty;

#define XNoMemory -1
#define XLocaleNotSupported -2
#define XConverterNotFound -3

typedef enum {
    XStringStyle,		/* STRING */
    XCompoundTextStyle,		/* COMPOUND_TEXT */
    XTextStyle,			/* text in owner's encoding (current locale)*/
    XStdICCTextStyle,		/* STRING, else COMPOUND_TEXT */
    /* The following is an XFree86 extension, introduced in November 2000 */
    XUTF8StringStyle		/* UTF8_STRING */
} XICCEncodingStyle;

typedef struct {
	int min_width, min_height;
	int max_width, max_height;
	int width_inc, height_inc;
} XIconSize;

typedef struct {
	char *res_name;
	char *res_class;
} XClassHint;

#ifdef XUTIL_DEFINE_FUNCTIONS
extern int XDestroyImage(
        XImage *ximage);
extern unsigned long XGetPixel(
        XImage *ximage,
        int x, int y);
extern int XPutPixel(
        XImage *ximage,
        int x, int y,
        unsigned long pixel);
extern XImage *XSubImage(
        XImage *ximage,
        int x, int y,
        unsigned int width, unsigned int height);
extern int XAddPixel(
        XImage *ximage,
        long value);
#else
/*
 * These macros are used to give some sugar to the image routines so that
 * naive people are more comfortable with them.
 */
#define XDestroyImage(ximage) \
	((*((ximage)->f.destroy_image))((ximage)))
#define XGetPixel(ximage, x, y) \
	((*((ximage)->f.get_pixel))((ximage), (x), (y)))
#define XPutPixel(ximage, x, y, pixel) \
	((*((ximage)->f.put_pixel))((ximage), (x), (y), (pixel)))
#define XSubImage(ximage, x, y, width, height)  \
	((*((ximage)->f.sub_image))((ximage), (x), (y), (width), (height)))
#define XAddPixel(ximage, value) \
	((*((ximage)->f.add_pixel))((ximage), (value)))
#endif

/*
 * Compose sequence status structure, used in calling XLookupString.
 */
typedef struct _XComposeStatus {
    XPointer compose_ptr;	/* state table pointer */
    int chars_matched;		/* match state */
} XComposeStatus;

/*
 * Keysym macros, used on Keysyms to test for classes of symbols
 */
#define IsKeypadKey(keysym) \
  (((KeySym)(keysym) >= XK_KP_Space) && ((KeySym)(keysym) <= XK_KP_Equal))

#define IsPrivateKeypadKey(keysym) \
  (((KeySym)(keysym) >= 0x11000000) && ((KeySym)(keysym) <= 0x1100FFFF))

#define IsCursorKey(keysym) \
  (((KeySym)(keysym) >= XK_Home)     && ((KeySym)(keysym) <  XK_Select))

#define IsPFKey(keysym) \
  (((KeySym)(keysym) >= XK_KP_F1)     && ((KeySym)(keysym) <= XK_KP_F4))

#define IsFunctionKey(keysym) \
  (((KeySym)(keysym) >= XK_F1)       && ((KeySym)(keysym) <= XK_F35))

#define IsMiscFunctionKey(keysym) \
  (((KeySym)(keysym) >= XK_Select)   && ((KeySym)(keysym) <= XK_Break))

#ifdef XK_XKB_KEYS
#define IsModifierKey(keysym) \
  ((((KeySym)(keysym) >= XK_Shift_L) && ((KeySym)(keysym) <= XK_Hyper_R)) \
   || (((KeySym)(keysym) >= XK_ISO_Lock) && \
       ((KeySym)(keysym) <= XK_ISO_Level5_Lock)) \
   || ((KeySym)(keysym) == XK_Mode_switch) \
   || ((KeySym)(keysym) == XK_Num_Lock))
#else
#define IsModifierKey(keysym) \
  ((((KeySym)(keysym) >= XK_Shift_L) && ((KeySym)(keysym) <= XK_Hyper_R)) \
   || ((KeySym)(keysym) == XK_Mode_switch) \
   || ((KeySym)(keysym) == XK_Num_Lock))
#endif
/*
 * opaque reference to Region data type
 */
typedef struct _XRegion *Region;

/* Return values from XRectInRegion() */

#define RectangleOut 0
#define RectangleIn  1
#define RectanglePart 2


/*
 * Information used by the visual utility routines to find desired visual
 * type from the many visuals a display may support.
 */

typedef struct {
  Visual *visual;
  VisualID visualid;
  int screen;
  int depth;
#if defined(__cplusplus) || defined(c_plusplus)
  int c_class;					/* C++ */
#else
  int class;
#endif
  unsigned long red_mask;
  unsigned long green_mask;
  unsigned long blue_mask;
  int colormap_size;
  int bits_per_rgb;
} XVisualInfo;

#define VisualNoMask		0x0
#define VisualIDMask 		0x1
#define VisualScreenMask	0x2
#define VisualDepthMask		0x4
#define VisualClassMask		0x8
#define VisualRedMaskMask	0x10
#define VisualGreenMaskMask	0x20
#define VisualBlueMaskMask	0x40
#define VisualColormapSizeMask	0x80
#define VisualBitsPerRGBMask	0x100
#define VisualAllMask		0x1FF

/*
 * This defines a window manager property that clients may use to
 * share standard color maps of type RGB_COLOR_MAP:
 */
typedef struct {
	Colormap colormap;
	unsigned long red_max;
	unsigned long red_mult;
	unsigned long green_max;
	unsigned long green_mult;
	unsigned long blue_max;
	unsigned long blue_mult;
	unsigned long base_pixel;
	VisualID visualid;		/* added by ICCCM version 1 */
	XID killid;			/* added by ICCCM version 1 */
} XStandardColormap;

#define ReleaseByFreeingColormap ((XID) 1L)  /* for killid field above */


/*
 * return codes for XReadBitmapFile and XWriteBitmapFile
 */
#define BitmapSuccess		0
#define BitmapOpenFailed 	1
#define BitmapFileInvalid 	2
#define BitmapNoMemory		3

/****************************************************************
 *
 * Context Management
 *
 ****************************************************************/


/* Associative lookup table return codes */

#define XCSUCCESS 0	/* No error. */
#define XCNOMEM   1    /* Out of memory */
#define XCNOENT   2    /* No entry in table */

typedef int XContext;

#define XUniqueContext()       ((XContext) XrmUniqueQuark())
#define XStringToContext(string)   ((XContext) XrmStringToQuark(string))

_XFUNCPROTOBEGIN

/* The following declarations are alphabetized. */

extern XClassHint *XAllocClassHint (
    void
);

extern XIconSize *XAllocIconSize (
    void
);

extern XSizeHints *XAllocSizeHints (
    void
);

extern XStandardColormap *XAllocStandardColormap (
    void
);

extern XWMHints *XAllocWMHints (
    void
);

extern int XClipBox(
    Region		/* r */,
    XRectangle*		/* rect_return */
);

extern Region XCreateRegion(
    void
);

extern const char *XDefaultString (void);

extern int XDeleteContext(
    Display*		/* display */,
    XID			/* rid */,
    XContext		/* context */
);

extern int XDestroyRegion(
    Region		/* r */
);

extern int XEmptyRegion(
    Region		/* r */
);

extern int XEqualRegion(
    Region		/* r1 */,
    Region		/* r2 */
);

extern int XFindContext(
    Display*		/* display */,
    XID			/* rid */,
    XContext		/* context */,
    XPointer*		/* data_return */
);

extern Status XGetClassHint(
    Display*		/* display */,
    Window		/* w */,
    XClassHint*		/* class_hints_return */
);

extern Status XGetIconSizes(
    Display*		/* display */,
    Window		/* w */,
    XIconSize**		/* size_list_return */,
    int*		/* count_return */
);

extern Status XGetNormalHints(
    Display*		/* display */,
    Window		/* w */,
    XSizeHints*		/* hints_return */
);

extern Status XGetRGBColormaps(
    Display*		/* display */,
    Window		/* w */,
    XStandardColormap** /* stdcmap_return */,
    int*		/* count_return */,
    Atom		/* property */
);

extern Status XGetSizeHints(
    Display*		/* display */,
    Window		/* w */,
    XSizeHints*		/* hints_return */,
    Atom		/* property */
);

extern Status XGetStandardColormap(
    Display*		/* display */,
    Window		/* w */,
    XStandardColormap*	/* colormap_return */,
    Atom		/* property */
);

extern Status XGetTextProperty(
    Display*		/* display */,
    Window		/* window */,
    XTextProperty*	/* text_prop_return */,
    Atom		/* property */
);

extern XVisualInfo *XGetVisualInfo(
    Display*		/* display */,
    long		/* vinfo_mask */,
    XVisualInfo*	/* vinfo_template */,
    int*		/* nitems_return */
);

extern Status XGetWMClientMachine(
    Display*		/* display */,
    Window		/* w */,
    XTextProperty*	/* text_prop_return */
);

extern XWMHints *XGetWMHints(
    Display*		/* display */,
    Window		/* w */
);

extern Status XGetWMIconName(
    Display*		/* display */,
    Window		/* w */,
    XTextProperty*	/* text_prop_return */
);

extern Status XGetWMName(
    Display*		/* display */,
    Window		/* w */,
    XTextProperty*	/* text_prop_return */
);

extern Status XGetWMNormalHints(
    Display*		/* display */,
    Window		/* w */,
    XSizeHints*		/* hints_return */,
    long*		/* supplied_return */
);

extern Status XGetWMSizeHints(
    Display*		/* display */,
    Window		/* w */,
    XSizeHints*		/* hints_return */,
    long*		/* supplied_return */,
    Atom		/* property */
);

extern Status XGetZoomHints(
    Display*		/* display */,
    Window		/* w */,
    XSizeHints*		/* zhints_return */
);

extern int XIntersectRegion(
    Region		/* sra */,
    Region		/* srb */,
    Region		/* dr_return */
);

extern void XConvertCase(
    KeySym		/* sym */,
    KeySym*		/* lower */,
    KeySym*		/* upper */
);

extern int XLookupString(
    XKeyEvent*		/* event_struct */,
    char*		/* buffer_return */,
    int			/* bytes_buffer */,
    KeySym*		/* keysym_return */,
    XComposeStatus*	/* status_in_out */
);

extern Status XMatchVisualInfo(
    Display*		/* display */,
    int			/* screen */,
    int			/* depth */,
    int			/* class */,
    XVisualInfo*	/* vinfo_return */
);

extern int XOffsetRegion(
    Region		/* r */,
    int			/* dx */,
    int			/* dy */
);

extern Bool XPointInRegion(
    Region		/* r */,
    int			/* x */,
    int			/* y */
);

extern Region XPolygonRegion(
    XPoint*		/* points */,
    int			/* n */,
    int			/* fill_rule */
);

extern int XRectInRegion(
    Region		/* r */,
    int			/* x */,
    int			/* y */,
    unsigned int	/* width */,
    unsigned int	/* height */
);

extern int XSaveContext(
    Display*		/* display */,
    XID			/* rid */,
    XContext		/* context */,
    _Xconst char*	/* data */
);

extern int XSetClassHint(
    Display*		/* display */,
    Window		/* w */,
    XClassHint*		/* class_hints */
);

extern int XSetIconSizes(
    Display*		/* display */,
    Window		/* w */,
    XIconSize*		/* size_list */,
    int			/* count */
);

extern int XSetNormalHints(
    Display*		/* display */,
    Window		/* w */,
    XSizeHints*		/* hints */
);

extern void XSetRGBColormaps(
    Display*		/* display */,
    Window		/* w */,
    XStandardColormap*	/* stdcmaps */,
    int			/* count */,
    Atom		/* property */
);

extern int XSetSizeHints(
    Display*		/* display */,
    Window		/* w */,
    XSizeHints*		/* hints */,
    Atom		/* property */
);

extern int XSetStandardProperties(
    Display*		/* display */,
    Window		/* w */,
    _Xconst char*	/* window_name */,
    _Xconst char*	/* icon_name */,
    Pixmap		/* icon_pixmap */,
    char**		/* argv */,
    int			/* argc */,
    XSizeHints*		/* hints */
);

extern void XSetTextProperty(
    Display*		/* display */,
    Window		/* w */,
    XTextProperty*	/* text_prop */,
    Atom		/* property */
);

extern void XSetWMClientMachine(
    Display*		/* display */,
    Window		/* w */,
    XTextProperty*	/* text_prop */
);

extern int XSetWMHints(
    Display*		/* display */,
    Window		/* w */,
    XWMHints*		/* wm_hints */
);

extern void XSetWMIconName(
    Display*		/* display */,
    Window		/* w */,
    XTextProperty*	/* text_prop */
);

extern void XSetWMName(
    Display*		/* display */,
    Window		/* w */,
    XTextProperty*	/* text_prop */
);

extern void XSetWMNormalHints(
    Display*		/* display */,
    Window		/* w */,
    XSizeHints*		/* hints */
);

extern void XSetWMProperties(
    Display*		/* display */,
    Window		/* w */,
    XTextProperty*	/* window_name */,
    XTextProperty*	/* icon_name */,
    char**		/* argv */,
    int			/* argc */,
    XSizeHints*		/* normal_hints */,
    XWMHints*		/* wm_hints */,
    XClassHint*		/* class_hints */
);

extern void XmbSetWMProperties(
    Display*		/* display */,
    Window		/* w */,
    _Xconst char*	/* window_name */,
    _Xconst char*	/* icon_name */,
    char**		/* argv */,
    int			/* argc */,
    XSizeHints*		/* normal_hints */,
    XWMHints*		/* wm_hints */,
    XClassHint*		/* class_hints */
);

extern void Xutf8SetWMProperties(
    Display*		/* display */,
    Window		/* w */,
    _Xconst char*	/* window_name */,
    _Xconst char*	/* icon_name */,
    char**		/* argv */,
    int			/* argc */,
    XSizeHints*		/* normal_hints */,
    XWMHints*		/* wm_hints */,
    XClassHint*		/* class_hints */
);

extern void XSetWMSizeHints(
    Display*		/* display */,
    Window		/* w */,
    XSizeHints*		/* hints */,
    Atom		/* property */
);

extern int XSetRegion(
    Display*		/* display */,
    GC			/* gc */,
    Region		/* r */
);

extern void XSetStandardColormap(
    Display*		/* display */,
    Window		/* w */,
    XStandardColormap*	/* colormap */,
    Atom		/* property */
);

extern int XSetZoomHints(
    Display*		/* display */,
    Window		/* w */,
    XSizeHints*		/* zhints */
);

extern int XShrinkRegion(
    Region		/* r */,
    int			/* dx */,
    int			/* dy */
);

extern Status XStringListToTextProperty(
    char**		/* list */,
    int			/* count */,
    XTextProperty*	/* text_prop_return */
);

extern int XSubtractRegion(
    Region		/* sra */,
    Region		/* srb */,
    Region		/* dr_return */
);

extern int XmbTextListToTextProperty(
    Display*		display,
    char**		list,
    int			count,
    XICCEncodingStyle	style,
    XTextProperty*	text_prop_return
);

extern int XwcTextListToTextProperty(
    Display*		display,
    wchar_t**		list,
    int			count,
    XICCEncodingStyle	style,
    XTextProperty*	text_prop_return
);

extern int Xutf8TextListToTextProperty(
    Display*		display,
    char**		list,
    int			count,
    XICCEncodingStyle	style,
    XTextProperty*	text_prop_return
);

extern void XwcFreeStringList(
    wchar_t**		list
);

extern Status XTextPropertyToStringList(
    XTextProperty*	/* text_prop */,
    char***		/* list_return */,
    int*		/* count_return */
);

extern int XmbTextPropertyToTextList(
    Display*		display,
    const XTextProperty* text_prop,
    char***		list_return,
    int*		count_return
);

extern int XwcTextPropertyToTextList(
    Display*		display,
    const XTextProperty* text_prop,
    wchar_t***		list_return,
    int*		count_return
);

extern int Xutf8TextPropertyToTextList(
    Display*		display,
    const XTextProperty* text_prop,
    char***		list_return,
    int*		count_return
);

extern int XUnionRectWithRegion(
    XRectangle*		/* rectangle */,
    Region		/* src_region */,
    Region		/* dest_region_return */
);

extern int XUnionRegion(
    Region		/* sra */,
    Region		/* srb */,
    Region		/* dr_return */
);

extern int XWMGeometry(
    Display*		/* display */,
    int			/* screen_number */,
    _Xconst char*	/* user_geometry */,
    _Xconst char*	/* default_geometry */,
    unsigned int	/* border_width */,
    XSizeHints*		/* hints */,
    int*		/* x_return */,
    int*		/* y_return */,
    int*		/* width_return */,
    int*		/* height_return */,
    int*		/* gravity_return */
);

extern int XXorRegion(
    Region		/* sra */,
    Region		/* srb */,
    Region		/* dr_return */
);

#ifdef __clang__
#pragma clang diagnostic pop
#endif

_XFUNCPROTOEND

#endif /* _X11_XUTIL_H_ */
