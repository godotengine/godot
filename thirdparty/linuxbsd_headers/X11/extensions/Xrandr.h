/*
 * Copyright © 2000 Compaq Computer Corporation, Inc.
 * Copyright © 2002 Hewlett-Packard Company, Inc.
 * Copyright © 2006 Intel Corporation
 * Copyright © 2008 Red Hat, Inc.
 *
 * Permission to use, copy, modify, distribute, and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies and that both that copyright
 * notice and this permission notice appear in supporting documentation, and
 * that the name of the copyright holders not be used in advertising or
 * publicity pertaining to distribution of the software without specific,
 * written prior permission.  The copyright holders make no representations
 * about the suitability of this software for any purpose.  It is provided "as
 * is" without express or implied warranty.
 *
 * THE COPYRIGHT HOLDERS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
 * INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY SPECIAL, INDIRECT OR
 * CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
 * DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
 * TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THIS SOFTWARE.
 *
 * Author:  Jim Gettys, HP Labs, Hewlett-Packard, Inc.
 *	    Keith Packard, Intel Corporation
 */

#ifndef _XRANDR_H_
#define _XRANDR_H_

#include <X11/extensions/randr.h>
#include <X11/extensions/Xrender.h>

#include <X11/Xfuncproto.h>

_XFUNCPROTOBEGIN

typedef XID RROutput;
typedef XID RRCrtc;
typedef XID RRMode;
typedef XID RRProvider;

typedef struct {
    int	width, height;
    int	mwidth, mheight;
} XRRScreenSize;

/*
 *  Events.
 */

typedef struct {
    int type;			/* event base */
    unsigned long serial;	/* # of last request processed by server */
    Bool send_event;		/* true if this came from a SendEvent request */
    Display *display;		/* Display the event was read from */
    Window window;		/* window which selected for this event */
    Window root;		/* Root window for changed screen */
    Time timestamp;		/* when the screen change occurred */
    Time config_timestamp;	/* when the last configuration change */
    SizeID size_index;
    SubpixelOrder subpixel_order;
    Rotation rotation;
    int width;
    int height;
    int mwidth;
    int mheight;
} XRRScreenChangeNotifyEvent;

typedef struct {
    int type;			/* event base */
    unsigned long serial;	/* # of last request processed by server */
    Bool send_event;		/* true if this came from a SendEvent request */
    Display *display;		/* Display the event was read from */
    Window window;		/* window which selected for this event */
    int subtype;		/* RRNotify_ subtype */
} XRRNotifyEvent;

typedef struct {
    int type;			/* event base */
    unsigned long serial;	/* # of last request processed by server */
    Bool send_event;		/* true if this came from a SendEvent request */
    Display *display;		/* Display the event was read from */
    Window window;		/* window which selected for this event */
    int subtype;		/* RRNotify_OutputChange */
    RROutput output;		/* affected output */
    RRCrtc crtc;	    	/* current crtc (or None) */
    RRMode mode;	    	/* current mode (or None) */
    Rotation rotation;		/* current rotation of associated crtc */
    Connection connection;	/* current connection status */
    SubpixelOrder subpixel_order;
} XRROutputChangeNotifyEvent;

typedef struct {
    int type;			/* event base */
    unsigned long serial;	/* # of last request processed by server */
    Bool send_event;		/* true if this came from a SendEvent request */
    Display *display;		/* Display the event was read from */
    Window window;		/* window which selected for this event */
    int subtype;		/* RRNotify_CrtcChange */
    RRCrtc crtc;    		/* current crtc (or None) */
    RRMode mode;	    	/* current mode (or None) */
    Rotation rotation;		/* current rotation of associated crtc */
    int x, y;			/* position */
    unsigned int width, height;	/* size */
} XRRCrtcChangeNotifyEvent;

typedef struct {
    int type;			/* event base */
    unsigned long serial;	/* # of last request processed by server */
    Bool send_event;		/* true if this came from a SendEvent request */
    Display *display;		/* Display the event was read from */
    Window window;		/* window which selected for this event */
    int subtype;		/* RRNotify_OutputProperty */
    RROutput output;		/* related output */
    Atom property;		/* changed property */
    Time timestamp;		/* time of change */
    int state;			/* NewValue, Deleted */
} XRROutputPropertyNotifyEvent;

typedef struct {
    int type;			/* event base */
    unsigned long serial;	/* # of last request processed by server */
    Bool send_event;		/* true if this came from a SendEvent request */
    Display *display;		/* Display the event was read from */
    Window window;		/* window which selected for this event */
    int subtype;		/* RRNotify_ProviderChange */
    RRProvider provider; 	/* current provider (or None) */
    Time timestamp;		/* time of change */
    unsigned int current_role;
} XRRProviderChangeNotifyEvent;

typedef struct {
    int type;			/* event base */
    unsigned long serial;	/* # of last request processed by server */
    Bool send_event;		/* true if this came from a SendEvent request */
    Display *display;		/* Display the event was read from */
    Window window;		/* window which selected for this event */
    int subtype;		/* RRNotify_ProviderProperty */
    RRProvider provider;		/* related provider */
    Atom property;		/* changed property */
    Time timestamp;		/* time of change */
    int state;			/* NewValue, Deleted */
} XRRProviderPropertyNotifyEvent;

typedef struct {
    int type;			/* event base */
    unsigned long serial;	/* # of last request processed by server */
    Bool send_event;		/* true if this came from a SendEvent request */
    Display *display;		/* Display the event was read from */
    Window window;		/* window which selected for this event */
    int subtype;		/* RRNotify_ResourceChange */
    Time timestamp;		/* time of change */
} XRRResourceChangeNotifyEvent;

/* internal representation is private to the library */
typedef struct _XRRScreenConfiguration XRRScreenConfiguration;

Bool XRRQueryExtension (Display *dpy,
			int *event_base_return,
			int *error_base_return);
Status XRRQueryVersion (Display *dpy,
			    int     *major_version_return,
			    int     *minor_version_return);

XRRScreenConfiguration *XRRGetScreenInfo (Display *dpy,
					  Window window);

void XRRFreeScreenConfigInfo (XRRScreenConfiguration *config);

/*
 * Note that screen configuration changes are only permitted if the client can
 * prove it has up to date configuration information.  We are trying to
 * insist that it become possible for screens to change dynamically, so
 * we want to ensure the client knows what it is talking about when requesting
 * changes.
 */
Status XRRSetScreenConfig (Display *dpy,
			   XRRScreenConfiguration *config,
			   Drawable draw,
			   int size_index,
			   Rotation rotation,
			   Time timestamp);

/* added in v1.1, sorry for the lame name */
Status XRRSetScreenConfigAndRate (Display *dpy,
				  XRRScreenConfiguration *config,
				  Drawable draw,
				  int size_index,
				  Rotation rotation,
				  short rate,
				  Time timestamp);


Rotation XRRConfigRotations(XRRScreenConfiguration *config, Rotation *current_rotation);

Time XRRConfigTimes (XRRScreenConfiguration *config, Time *config_timestamp);

XRRScreenSize *XRRConfigSizes(XRRScreenConfiguration *config, int *nsizes);

short *XRRConfigRates (XRRScreenConfiguration *config, int sizeID, int *nrates);

SizeID XRRConfigCurrentConfiguration (XRRScreenConfiguration *config,
			      Rotation *rotation);

short XRRConfigCurrentRate (XRRScreenConfiguration *config);

int XRRRootToScreen(Display *dpy, Window root);

/*
 * returns the screen configuration for the specified screen; does a lazy
 * evalution to delay getting the information, and caches the result.
 * These routines should be used in preference to XRRGetScreenInfo
 * to avoid unneeded round trips to the X server.  These are new
 * in protocol version 0.1.
 */


void XRRSelectInput(Display *dpy, Window window, int mask);

/*
 * the following are always safe to call, even if RandR is not implemented
 * on a screen
 */


Rotation XRRRotations(Display *dpy, int screen, Rotation *current_rotation);
XRRScreenSize *XRRSizes(Display *dpy, int screen, int *nsizes);
short *XRRRates (Display *dpy, int screen, int sizeID, int *nrates);
Time XRRTimes (Display *dpy, int screen, Time *config_timestamp);


/* Version 1.2 additions */

/* despite returning a Status, this returns 1 for success */
Status
XRRGetScreenSizeRange (Display *dpy, Window window,
		       int *minWidth, int *minHeight,
		       int *maxWidth, int *maxHeight);

void
XRRSetScreenSize (Display *dpy, Window window,
		  int width, int height,
		  int mmWidth, int mmHeight);

typedef unsigned long XRRModeFlags;

typedef struct _XRRModeInfo {
    RRMode		id;
    unsigned int	width;
    unsigned int	height;
    unsigned long	dotClock;
    unsigned int	hSyncStart;
    unsigned int	hSyncEnd;
    unsigned int	hTotal;
    unsigned int	hSkew;
    unsigned int	vSyncStart;
    unsigned int	vSyncEnd;
    unsigned int	vTotal;
    char		*name;
    unsigned int	nameLength;
    XRRModeFlags	modeFlags;
} XRRModeInfo;

typedef struct _XRRScreenResources {
    Time	timestamp;
    Time	configTimestamp;
    int		ncrtc;
    RRCrtc	*crtcs;
    int		noutput;
    RROutput	*outputs;
    int		nmode;
    XRRModeInfo	*modes;
} XRRScreenResources;

XRRScreenResources *
XRRGetScreenResources (Display *dpy, Window window);

void
XRRFreeScreenResources (XRRScreenResources *resources);

typedef struct _XRROutputInfo {
    Time	    timestamp;
    RRCrtc	    crtc;
    char	    *name;
    int		    nameLen;
    unsigned long   mm_width;
    unsigned long   mm_height;
    Connection	    connection;
    SubpixelOrder   subpixel_order;
    int		    ncrtc;
    RRCrtc	    *crtcs;
    int		    nclone;
    RROutput	    *clones;
    int		    nmode;
    int		    npreferred;
    RRMode	    *modes;
} XRROutputInfo;

XRROutputInfo *
XRRGetOutputInfo (Display *dpy, XRRScreenResources *resources, RROutput output);

void
XRRFreeOutputInfo (XRROutputInfo *outputInfo);

Atom *
XRRListOutputProperties (Display *dpy, RROutput output, int *nprop);

typedef struct {
    Bool    pending;
    Bool    range;
    Bool    immutable;
    int	    num_values;
    long    *values;
} XRRPropertyInfo;

XRRPropertyInfo *
XRRQueryOutputProperty (Display *dpy, RROutput output, Atom property);

void
XRRConfigureOutputProperty (Display *dpy, RROutput output, Atom property,
			    Bool pending, Bool range, int num_values,
			    long *values);

void
XRRChangeOutputProperty (Display *dpy, RROutput output,
			 Atom property, Atom type,
			 int format, int mode,
			 _Xconst unsigned char *data, int nelements);

void
XRRDeleteOutputProperty (Display *dpy, RROutput output, Atom property);

int
XRRGetOutputProperty (Display *dpy, RROutput output,
		      Atom property, long offset, long length,
		      Bool _delete, Bool pending, Atom req_type,
		      Atom *actual_type, int *actual_format,
		      unsigned long *nitems, unsigned long *bytes_after,
		      unsigned char **prop);

XRRModeInfo *
XRRAllocModeInfo (_Xconst char *name, int nameLength);

RRMode
XRRCreateMode (Display *dpy, Window window, XRRModeInfo *modeInfo);

void
XRRDestroyMode (Display *dpy, RRMode mode);

void
XRRAddOutputMode (Display *dpy, RROutput output, RRMode mode);

void
XRRDeleteOutputMode (Display *dpy, RROutput output, RRMode mode);

void
XRRFreeModeInfo (XRRModeInfo *modeInfo);

typedef struct _XRRCrtcInfo {
    Time	    timestamp;
    int		    x, y;
    unsigned int    width, height;
    RRMode	    mode;
    Rotation	    rotation;
    int		    noutput;
    RROutput	    *outputs;
    Rotation	    rotations;
    int		    npossible;
    RROutput	    *possible;
} XRRCrtcInfo;

XRRCrtcInfo *
XRRGetCrtcInfo (Display *dpy, XRRScreenResources *resources, RRCrtc crtc);

void
XRRFreeCrtcInfo (XRRCrtcInfo *crtcInfo);

Status
XRRSetCrtcConfig (Display *dpy,
		  XRRScreenResources *resources,
		  RRCrtc crtc,
		  Time timestamp,
		  int x, int y,
		  RRMode mode,
		  Rotation rotation,
		  RROutput *outputs,
		  int noutputs);

int
XRRGetCrtcGammaSize (Display *dpy, RRCrtc crtc);

typedef struct _XRRCrtcGamma {
    int		    size;
    unsigned short  *red;
    unsigned short  *green;
    unsigned short  *blue;
} XRRCrtcGamma;

XRRCrtcGamma *
XRRGetCrtcGamma (Display *dpy, RRCrtc crtc);

XRRCrtcGamma *
XRRAllocGamma (int size);

void
XRRSetCrtcGamma (Display *dpy, RRCrtc crtc, XRRCrtcGamma *gamma);

void
XRRFreeGamma (XRRCrtcGamma *gamma);

/* Version 1.3 additions */

XRRScreenResources *
XRRGetScreenResourcesCurrent (Display *dpy, Window window);

void
XRRSetCrtcTransform (Display	*dpy,
		     RRCrtc	crtc,
		     XTransform	*transform,
		     _Xconst char *filter,
		     XFixed	*params,
		     int	nparams);

typedef struct _XRRCrtcTransformAttributes {
    XTransform	pendingTransform;
    char	*pendingFilter;
    int		pendingNparams;
    XFixed	*pendingParams;
    XTransform	currentTransform;
    char	*currentFilter;
    int		currentNparams;
    XFixed	*currentParams;
} XRRCrtcTransformAttributes;

/*
 * Get current crtc transforms and filters.
 * Pass *attributes to XFree to free
 */
Status
XRRGetCrtcTransform (Display	*dpy,
		     RRCrtc	crtc,
		     XRRCrtcTransformAttributes **attributes);

/*
 * intended to take RRScreenChangeNotify,  or
 * ConfigureNotify (on the root window)
 * returns 1 if it is an event type it understands, 0 if not
 */
int XRRUpdateConfiguration(XEvent *event);

typedef struct _XRRPanning {
    Time            timestamp;
    unsigned int left;
    unsigned int top;
    unsigned int width;
    unsigned int height;
    unsigned int track_left;
    unsigned int track_top;
    unsigned int track_width;
    unsigned int track_height;
    int          border_left;
    int          border_top;
    int          border_right;
    int          border_bottom;
} XRRPanning;

XRRPanning *
XRRGetPanning (Display *dpy, XRRScreenResources *resources, RRCrtc crtc);

void
XRRFreePanning (XRRPanning *panning);

Status
XRRSetPanning (Display *dpy,
	       XRRScreenResources *resources,
	       RRCrtc crtc,
	       XRRPanning *panning);

void
XRRSetOutputPrimary(Display *dpy,
		    Window window,
		    RROutput output);

RROutput
XRRGetOutputPrimary(Display *dpy,
		    Window window);

typedef struct _XRRProviderResources {
    Time timestamp;
    int nproviders;
    RRProvider *providers;
} XRRProviderResources;

XRRProviderResources *
XRRGetProviderResources(Display *dpy, Window window);

void
XRRFreeProviderResources(XRRProviderResources *resources);

typedef struct _XRRProviderInfo {
    unsigned int capabilities;
    int ncrtcs;
    RRCrtc	*crtcs;
    int noutputs;
    RROutput    *outputs;
    char	    *name;
    int nassociatedproviders;
    RRProvider *associated_providers;
    unsigned int *associated_capability;
    int		    nameLen;
} XRRProviderInfo;
  
XRRProviderInfo *
XRRGetProviderInfo(Display *dpy, XRRScreenResources *resources, RRProvider provider);

void
XRRFreeProviderInfo(XRRProviderInfo *provider);

int
XRRSetProviderOutputSource(Display *dpy, XID provider, XID source_provider);

int
XRRSetProviderOffloadSink(Display *dpy, XID provider, XID sink_provider);

Atom *
XRRListProviderProperties (Display *dpy, RRProvider provider, int *nprop);

XRRPropertyInfo *
XRRQueryProviderProperty (Display *dpy, RRProvider provider, Atom property);

void
XRRConfigureProviderProperty (Display *dpy, RRProvider provider, Atom property,
			    Bool pending, Bool range, int num_values,
			    long *values);
			
void
XRRChangeProviderProperty (Display *dpy, RRProvider provider,
			 Atom property, Atom type,
			 int format, int mode,
			 _Xconst unsigned char *data, int nelements);

void
XRRDeleteProviderProperty (Display *dpy, RRProvider provider, Atom property);

int
XRRGetProviderProperty (Display *dpy, RRProvider provider,
			Atom property, long offset, long length,
			Bool _delete, Bool pending, Atom req_type,
			Atom *actual_type, int *actual_format,
			unsigned long *nitems, unsigned long *bytes_after,
			unsigned char **prop);


typedef struct _XRRMonitorInfo {
    Atom name;
    Bool primary;
    Bool automatic;
    int noutput;
    int x;
    int y;
    int width;
    int height;
    int mwidth;
    int mheight;
    RROutput *outputs;
} XRRMonitorInfo;

XRRMonitorInfo *
XRRAllocateMonitor(Display *dpy, int noutput);

XRRMonitorInfo *
XRRGetMonitors(Display *dpy, Window window, Bool get_active, int *nmonitors);

void
XRRSetMonitor(Display *dpy, Window window, XRRMonitorInfo *monitor);

void
XRRDeleteMonitor(Display *dpy, Window window, Atom name);

void
XRRFreeMonitors(XRRMonitorInfo *monitors);

_XFUNCPROTOEND

#endif /* _XRANDR_H_ */
