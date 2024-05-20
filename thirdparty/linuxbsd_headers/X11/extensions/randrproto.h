/*
 * Copyright © 2000 Compaq Computer Corporation
 * Copyright © 2002 Hewlett-Packard Company
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
 * Author:  Jim Gettys, Hewlett-Packard Company, Inc.
 *	    Keith Packard, Intel Corporation
 */

/* note that RANDR 1.0 is incompatible with version 0.0, or 0.1 */
/* V1.0 removes depth switching from the protocol */
#ifndef _XRANDRP_H_
#define _XRANDRP_H_

#include <X11/extensions/randr.h>
#include <X11/extensions/renderproto.h>

#define Window CARD32
#define Drawable CARD32
#define Font CARD32
#define Pixmap CARD32
#define Cursor CARD32
#define Colormap CARD32
#define GContext CARD32
#define Atom CARD32
#define Time CARD32
#define KeyCode CARD8
#define KeySym CARD32
#define RROutput CARD32
#define RRMode CARD32
#define RRCrtc CARD32
#define RRProvider CARD32
#define RRModeFlags CARD32
#define RRLease CARD32

#define Rotation CARD16
#define SizeID CARD16
#define SubpixelOrder CARD16

/*
 * data structures
 */

typedef struct {
    CARD16 widthInPixels;
    CARD16 heightInPixels;
    CARD16 widthInMillimeters;
    CARD16 heightInMillimeters;
} xScreenSizes;
#define sz_xScreenSizes 8

/*
 * requests and replies
 */

typedef struct {
    CARD8   reqType;
    CARD8   randrReqType;
    CARD16  length;
    CARD32  majorVersion;
    CARD32  minorVersion;
} xRRQueryVersionReq;
#define sz_xRRQueryVersionReq   12

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
} xRRQueryVersionReply;
#define sz_xRRQueryVersionReply	32

typedef struct {
    CARD8   reqType;
    CARD8   randrReqType;
    CARD16  length;
    Window  window;
} xRRGetScreenInfoReq;
#define sz_xRRGetScreenInfoReq   8

/*
 * the xRRScreenInfoReply structure is followed by:
 *
 * the size information
 */


typedef struct {
    BYTE    type;   /* X_Reply */
    BYTE    setOfRotations;
    CARD16  sequenceNumber;
    CARD32  length;
    Window  root;
    Time    timestamp;
    Time    configTimestamp;
    CARD16  nSizes;
    SizeID  sizeID;
    Rotation  rotation;
    CARD16  rate;
    CARD16  nrateEnts;
    CARD16  pad;
} xRRGetScreenInfoReply;
#define sz_xRRGetScreenInfoReply	32

typedef struct {
    CARD8    reqType;
    CARD8    randrReqType;
    CARD16   length;
    Drawable drawable;
    Time     timestamp;
    Time     configTimestamp;
    SizeID   sizeID;
    Rotation rotation;
} xRR1_0SetScreenConfigReq;
#define sz_xRR1_0SetScreenConfigReq   20

typedef struct {
    CARD8    reqType;
    CARD8    randrReqType;
    CARD16   length;
    Drawable drawable;
    Time     timestamp;
    Time     configTimestamp;
    SizeID   sizeID;
    Rotation rotation;
    CARD16   rate;
    CARD16   pad;
} xRRSetScreenConfigReq;
#define sz_xRRSetScreenConfigReq   24

typedef struct {
    BYTE    type;   /* X_Reply */
    CARD8   status;
    CARD16  sequenceNumber;
    CARD32  length;
    Time    newTimestamp;
    Time    newConfigTimestamp;
    Window  root;
    CARD16  subpixelOrder;
    CARD16  pad4;
    CARD32  pad5;
    CARD32  pad6;
} xRRSetScreenConfigReply;
#define sz_xRRSetScreenConfigReply 32

typedef struct {
    CARD8   reqType;
    CARD8   randrReqType;
    CARD16  length;
    Window  window;
    CARD16  enable;
    CARD16  pad2;
} xRRSelectInputReq;
#define sz_xRRSelectInputReq   12

/*
 * Additions for version 1.2
 */

typedef struct _xRRModeInfo {
    RRMode		id;
    CARD16		width;
    CARD16		height;
    CARD32		dotClock;
    CARD16		hSyncStart;
    CARD16		hSyncEnd;
    CARD16		hTotal;
    CARD16		hSkew;
    CARD16		vSyncStart;
    CARD16		vSyncEnd;
    CARD16		vTotal;
    CARD16		nameLength;
    RRModeFlags		modeFlags;
} xRRModeInfo;
#define sz_xRRModeInfo		    32

typedef struct {
    CARD8   reqType;
    CARD8   randrReqType;
    CARD16  length;
    Window  window;
} xRRGetScreenSizeRangeReq;
#define sz_xRRGetScreenSizeRangeReq 8

typedef struct {
    BYTE    type;   /* X_Reply */
    CARD8   pad;
    CARD16  sequenceNumber;
    CARD32  length;
    CARD16  minWidth;
    CARD16  minHeight;
    CARD16  maxWidth;
    CARD16  maxHeight;
    CARD32  pad0;
    CARD32  pad1;
    CARD32  pad2;
    CARD32  pad3;
} xRRGetScreenSizeRangeReply;
#define sz_xRRGetScreenSizeRangeReply 32

typedef struct {
    CARD8   reqType;
    CARD8   randrReqType;
    CARD16  length;
    Window  window;
    CARD16  width;
    CARD16  height;
    CARD32  widthInMillimeters;
    CARD32  heightInMillimeters;
} xRRSetScreenSizeReq;
#define sz_xRRSetScreenSizeReq	    20

typedef struct {
    CARD8   reqType;
    CARD8   randrReqType;
    CARD16  length;
    Window  window;
} xRRGetScreenResourcesReq;
#define sz_xRRGetScreenResourcesReq 8

typedef struct {
    BYTE	type;
    CARD8	pad;
    CARD16	sequenceNumber;
    CARD32	length;
    Time	timestamp;
    Time	configTimestamp;
    CARD16	nCrtcs;
    CARD16	nOutputs;
    CARD16	nModes;
    CARD16	nbytesNames;
    CARD32	pad1;
    CARD32	pad2;
} xRRGetScreenResourcesReply;
#define sz_xRRGetScreenResourcesReply	32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RROutput	output;
    Time	configTimestamp;
} xRRGetOutputInfoReq;
#define sz_xRRGetOutputInfoReq		12

typedef struct {
    BYTE	type;
    CARD8	status;
    CARD16	sequenceNumber;
    CARD32	length;
    Time	timestamp;
    RRCrtc	crtc;
    CARD32	mmWidth;
    CARD32	mmHeight;
    CARD8	connection;
    CARD8	subpixelOrder;
    CARD16	nCrtcs;
    CARD16	nModes;
    CARD16	nPreferred;
    CARD16	nClones;
    CARD16	nameLength;
} xRRGetOutputInfoReply;
#define sz_xRRGetOutputInfoReply	36

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RROutput	output;
} xRRListOutputPropertiesReq;
#define sz_xRRListOutputPropertiesReq	8

typedef struct {
    BYTE	type;
    CARD8	pad0;
    CARD16	sequenceNumber;
    CARD32	length;
    CARD16	nAtoms;
    CARD16	pad1;
    CARD32	pad2;
    CARD32	pad3;
    CARD32	pad4;
    CARD32	pad5;
    CARD32	pad6;
} xRRListOutputPropertiesReply;
#define sz_xRRListOutputPropertiesReply	32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RROutput	output;
    Atom	property;
} xRRQueryOutputPropertyReq;
#define sz_xRRQueryOutputPropertyReq	12

typedef struct {
    BYTE	type;
    BYTE	pad0;
    CARD16	sequenceNumber;
    CARD32	length;
    BOOL	pending;
    BOOL	range;
    BOOL	immutable;
    BYTE	pad1;
    CARD32	pad2;
    CARD32	pad3;
    CARD32	pad4;
    CARD32	pad5;
    CARD32	pad6;
} xRRQueryOutputPropertyReply;
#define sz_xRRQueryOutputPropertyReply	32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RROutput	output;
    Atom	property;
    BOOL	pending;
    BOOL	range;
    CARD16	pad;
} xRRConfigureOutputPropertyReq;
#define sz_xRRConfigureOutputPropertyReq	16

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RROutput	output;
    Atom	property;
    Atom	type;
    CARD8	format;
    CARD8	mode;
    CARD16	pad;
    CARD32	nUnits;
} xRRChangeOutputPropertyReq;
#define sz_xRRChangeOutputPropertyReq	24

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RROutput	output;
    Atom	property;
} xRRDeleteOutputPropertyReq;
#define sz_xRRDeleteOutputPropertyReq	12

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RROutput	output;
    Atom	property;
    Atom	type;
    CARD32	longOffset;
    CARD32	longLength;
#ifdef __cplusplus
    BOOL	_delete;
#else
    BOOL	delete;
#endif
    BOOL	pending;
    CARD16	pad1;
} xRRGetOutputPropertyReq;
#define sz_xRRGetOutputPropertyReq	28

typedef struct {
    BYTE	type;
    CARD8	format;
    CARD16	sequenceNumber;
    CARD32	length;
    Atom	propertyType;
    CARD32	bytesAfter;
    CARD32	nItems;
    CARD32	pad1;
    CARD32	pad2;
    CARD32	pad3;
} xRRGetOutputPropertyReply;
#define sz_xRRGetOutputPropertyReply	32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    Window	window;
    xRRModeInfo	modeInfo;
} xRRCreateModeReq;
#define sz_xRRCreateModeReq		40

typedef struct {
    BYTE	type;
    CARD8	pad0;
    CARD16	sequenceNumber;
    CARD32	length;
    RRMode	mode;
    CARD32	pad1;
    CARD32	pad2;
    CARD32	pad3;
    CARD32	pad4;
    CARD32	pad5;
} xRRCreateModeReply;
#define sz_xRRCreateModeReply		32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRMode	mode;
} xRRDestroyModeReq;
#define sz_xRRDestroyModeReq		8

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RROutput	output;
    RRMode	mode;
} xRRAddOutputModeReq;
#define sz_xRRAddOutputModeReq		12

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RROutput	output;
    RRMode	mode;
} xRRDeleteOutputModeReq;
#define sz_xRRDeleteOutputModeReq	12

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRCrtc	crtc;
    Time	configTimestamp;
} xRRGetCrtcInfoReq;
#define sz_xRRGetCrtcInfoReq		12

typedef struct {
    BYTE	type;
    CARD8	status;
    CARD16	sequenceNumber;
    CARD32	length;
    Time	timestamp;
    INT16	x;
    INT16	y;
    CARD16	width;
    CARD16	height;
    RRMode	mode;
    Rotation	rotation;
    Rotation	rotations;
    CARD16	nOutput;
    CARD16	nPossibleOutput;
} xRRGetCrtcInfoReply;
#define sz_xRRGetCrtcInfoReply		32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRCrtc	crtc;
    Time	timestamp;
    Time    	configTimestamp;
    INT16	x;
    INT16	y;
    RRMode	mode;
    Rotation	rotation;
    CARD16	pad;
} xRRSetCrtcConfigReq;
#define sz_xRRSetCrtcConfigReq		28

typedef struct {
    BYTE	type;
    CARD8	status;
    CARD16	sequenceNumber;
    CARD32	length;
    Time	newTimestamp;
    CARD32	pad1;
    CARD32	pad2;
    CARD32	pad3;
    CARD32	pad4;
    CARD32	pad5;
} xRRSetCrtcConfigReply;
#define sz_xRRSetCrtcConfigReply	32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRCrtc	crtc;
} xRRGetCrtcGammaSizeReq;
#define sz_xRRGetCrtcGammaSizeReq	8

typedef struct {
    BYTE	type;
    CARD8	status;
    CARD16	sequenceNumber;
    CARD32	length;
    CARD16	size;
    CARD16	pad1;
    CARD32	pad2;
    CARD32	pad3;
    CARD32	pad4;
    CARD32	pad5;
    CARD32	pad6;
} xRRGetCrtcGammaSizeReply;
#define sz_xRRGetCrtcGammaSizeReply	32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRCrtc	crtc;
} xRRGetCrtcGammaReq;
#define sz_xRRGetCrtcGammaReq		8

typedef struct {
    BYTE	type;
    CARD8	status;
    CARD16	sequenceNumber;
    CARD32	length;
    CARD16	size;
    CARD16	pad1;
    CARD32	pad2;
    CARD32	pad3;
    CARD32	pad4;
    CARD32	pad5;
    CARD32	pad6;
} xRRGetCrtcGammaReply;
#define sz_xRRGetCrtcGammaReply		32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRCrtc	crtc;
    CARD16	size;
    CARD16	pad1;
} xRRSetCrtcGammaReq;
#define sz_xRRSetCrtcGammaReq		12

/*
 * Additions for V1.3
 */

typedef xRRGetScreenResourcesReq xRRGetScreenResourcesCurrentReq;

#define sz_xRRGetScreenResourcesCurrentReq sz_xRRGetScreenResourcesReq

typedef xRRGetScreenResourcesReply xRRGetScreenResourcesCurrentReply;
#define sz_xRRGetScreenResourcesCurrentReply	sz_xRRGetScreenResourcesReply

typedef struct {
    CARD8		reqType;
    CARD8		randrReqType;
    CARD16		length;
    RRCrtc		crtc;
    xRenderTransform	transform;
    CARD16		nbytesFilter;	/* number of bytes in filter name */
    CARD16		pad;
} xRRSetCrtcTransformReq;

#define sz_xRRSetCrtcTransformReq	48

typedef struct {
    CARD8		reqType;
    CARD8		randrReqType;
    CARD16		length;
    RRCrtc		crtc;
} xRRGetCrtcTransformReq;

#define sz_xRRGetCrtcTransformReq	8

typedef struct {
    BYTE		type;
    CARD8		status;
    CARD16		sequenceNumber;
    CARD32		length;
    xRenderTransform	pendingTransform;
    BYTE		hasTransforms;
    CARD8		pad0;
    CARD16		pad1;
    xRenderTransform	currentTransform;
    CARD32		pad2;
    CARD16		pendingNbytesFilter;    /* number of bytes in filter name */
    CARD16		pendingNparamsFilter;   /* number of filter params */
    CARD16		currentNbytesFilter;    /* number of bytes in filter name */
    CARD16		currentNparamsFilter;   /* number of filter params */
} xRRGetCrtcTransformReply;

#define sz_xRRGetCrtcTransformReply	96

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    Window	window;
    RROutput	output;
} xRRSetOutputPrimaryReq;
#define sz_xRRSetOutputPrimaryReq	12

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    Window	window;
} xRRGetOutputPrimaryReq;
#define sz_xRRGetOutputPrimaryReq	8

typedef struct {
    BYTE	type;
    CARD8	pad;
    CARD16	sequenceNumber;
    CARD32	length;
    RROutput	output;
    CARD32	pad1;
    CARD32	pad2;
    CARD32	pad3;
    CARD32	pad4;
    CARD32	pad5;
} xRRGetOutputPrimaryReply;
#define sz_xRRGetOutputPrimaryReply	32

/*
 * Additions for V1.4
 */

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    Window	window;
} xRRGetProvidersReq;
#define sz_xRRGetProvidersReq 8

typedef struct {
    BYTE	type;
    CARD8	pad;
    CARD16	sequenceNumber;
    CARD32	length;
    Time	timestamp;
    CARD16	nProviders;
    CARD16	pad1;
    CARD32	pad2;
    CARD32	pad3;
    CARD32	pad4;
    CARD32	pad5;
} xRRGetProvidersReply;
#define sz_xRRGetProvidersReply 32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRProvider	provider;
    Time	configTimestamp;
} xRRGetProviderInfoReq;
#define sz_xRRGetProviderInfoReq 12

typedef struct {
    BYTE	type;
    CARD8	status;
    CARD16	sequenceNumber;
    CARD32	length;
    Time	timestamp;
    CARD32	capabilities;
    CARD16	nCrtcs;
    CARD16	nOutputs;
    CARD16	nAssociatedProviders;
    CARD16	nameLength;
    CARD32	pad1;
    CARD32	pad2;
} xRRGetProviderInfoReply;
#define sz_xRRGetProviderInfoReply 32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRProvider  provider;
    RRProvider  source_provider;
    Time	configTimestamp;
} xRRSetProviderOutputSourceReq;
#define sz_xRRSetProviderOutputSourceReq 16

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRProvider  provider;
    RRProvider  sink_provider;
    Time	configTimestamp;
} xRRSetProviderOffloadSinkReq;
#define sz_xRRSetProviderOffloadSinkReq 16

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRProvider	provider;
} xRRListProviderPropertiesReq;
#define sz_xRRListProviderPropertiesReq	8

typedef struct {
    BYTE	type;
    CARD8	pad0;
    CARD16	sequenceNumber;
    CARD32	length;
    CARD16	nAtoms;
    CARD16	pad1;
    CARD32	pad2;
    CARD32	pad3;
    CARD32	pad4;
    CARD32	pad5;
    CARD32	pad6;
} xRRListProviderPropertiesReply;
#define sz_xRRListProviderPropertiesReply	32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRProvider	provider;
    Atom	property;
} xRRQueryProviderPropertyReq;
#define sz_xRRQueryProviderPropertyReq	12

typedef struct {
    BYTE	type;
    BYTE	pad0;
    CARD16	sequenceNumber;
    CARD32	length;
    BOOL	pending;
    BOOL	range;
    BOOL	immutable;
    BYTE	pad1;
    CARD32	pad2;
    CARD32	pad3;
    CARD32	pad4;
    CARD32	pad5;
    CARD32	pad6;
} xRRQueryProviderPropertyReply;
#define sz_xRRQueryProviderPropertyReply	32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRProvider	provider;
    Atom	property;
    BOOL	pending;
    BOOL	range;
    CARD16	pad;
} xRRConfigureProviderPropertyReq;
#define sz_xRRConfigureProviderPropertyReq	16

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRProvider	provider;
    Atom	property;
    Atom	type;
    CARD8	format;
    CARD8	mode;
    CARD16	pad;
    CARD32	nUnits;
} xRRChangeProviderPropertyReq;
#define sz_xRRChangeProviderPropertyReq	24

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRProvider	provider;
    Atom	property;
} xRRDeleteProviderPropertyReq;
#define sz_xRRDeleteProviderPropertyReq	12

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRProvider	provider;
    Atom	property;
    Atom	type;
    CARD32	longOffset;
    CARD32	longLength;
#ifdef __cplusplus
    BOOL	_delete;
#else
    BOOL	delete;
#endif
    BOOL	pending;
    CARD16	pad1;
} xRRGetProviderPropertyReq;
#define sz_xRRGetProviderPropertyReq	28

typedef struct {
    BYTE	type;
    CARD8	format;
    CARD16	sequenceNumber;
    CARD32	length;
    Atom	propertyType;
    CARD32	bytesAfter;
    CARD32	nItems;
    CARD32	pad1;
    CARD32	pad2;
    CARD32	pad3;
} xRRGetProviderPropertyReply;
#define sz_xRRGetProviderPropertyReply	32

/*
 * Additions for V1.6
 */

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    Window	window;
    RRLease	lid;
    CARD16	nCrtcs;
    CARD16	nOutputs;
} xRRCreateLeaseReq;
#define sz_xRRCreateLeaseReq	16

typedef struct {
    BYTE	type;
    CARD8	nfd;
    CARD16	sequenceNumber;
    CARD32	length;
    CARD32	pad2;
    CARD32	pad3;
    CARD32	pad4;
    CARD32	pad5;
    CARD32	pad6;
    CARD32	pad7;
} xRRCreateLeaseReply;
#define sz_xRRCreateLeaseReply		32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRLease	lid;
    BYTE	terminate;
    CARD8	pad1;
    CARD16	pad2;
} xRRFreeLeaseReq;
#define sz_xRRFreeLeaseReq		12

/*
 * event
 */
typedef struct {
    CARD8 type;				/* always evBase + ScreenChangeNotify */
    CARD8 rotation;			/* new rotation */
    CARD16 sequenceNumber;
    Time timestamp;			/* time screen was changed */
    Time configTimestamp;		/* time config data was changed */
    Window root;			/* root window */
    Window window;			/* window requesting notification */
    SizeID sizeID;			/* new size ID */
    CARD16 subpixelOrder;		/* subpixel order */
    CARD16 widthInPixels;		/* new size */
    CARD16 heightInPixels;
    CARD16 widthInMillimeters;
    CARD16 heightInMillimeters;
} xRRScreenChangeNotifyEvent;
#define sz_xRRScreenChangeNotifyEvent	32

typedef struct {
    CARD8 type;				/* always evBase + RRNotify */
    CARD8 subCode;			/* RRNotify_CrtcChange */
    CARD16 sequenceNumber;
    Time timestamp;			/* time crtc was changed */
    Window window;			/* window requesting notification */
    RRCrtc crtc;			/* affected CRTC */
    RRMode mode;			/* current mode */
    CARD16 rotation;			/* rotation and reflection */
    CARD16 pad1;			/* unused */
    INT16 x;				/* new location */
    INT16 y;
    CARD16 width;			/* new size */
    CARD16 height;
} xRRCrtcChangeNotifyEvent;
#define sz_xRRCrtcChangeNotifyEvent	32

typedef struct {
    CARD8 type;				/* always evBase + RRNotify */
    CARD8 subCode;			/* RRNotify_OutputChange */
    CARD16 sequenceNumber;
    Time timestamp;			/* time output was changed */
    Time configTimestamp;		/* time config was changed */
    Window window;			/* window requesting notification */
    RROutput output;			/* affected output */
    RRCrtc crtc;			/* current crtc */
    RRMode mode;			/* current mode */
    CARD16 rotation;			/* rotation and reflection */
    CARD8 connection;			/* connection status */
    CARD8 subpixelOrder;		/* subpixel order */
} xRROutputChangeNotifyEvent;
#define sz_xRROutputChangeNotifyEvent	32

typedef struct {
    CARD8 type;				/* always evBase + RRNotify */
    CARD8 subCode;			/* RRNotify_OutputProperty */
    CARD16 sequenceNumber;
    Window window;			/* window requesting notification */
    RROutput output;			/* affected output */
    Atom atom;				/* property name */
    Time timestamp;			/* time crtc was changed */
    CARD8 state;			/* NewValue or Deleted */
    CARD8 pad1;
    CARD16 pad2;
    CARD32 pad3;
    CARD32 pad4;
} xRROutputPropertyNotifyEvent;
#define sz_xRROutputPropertyNotifyEvent	32

typedef struct {
    CARD8 type;				/* always evBase + RRNotify */
    CARD8 subCode;			/* RRNotify_ProviderChange */
    CARD16 sequenceNumber;
    Time timestamp;			/* time provider was changed */
    Window window;			/* window requesting notification */
    RRProvider provider;		/* affected provider */
    CARD32 pad1;
    CARD32 pad2;
    CARD32 pad3;
    CARD32 pad4;
} xRRProviderChangeNotifyEvent;
#define sz_xRRProviderChangeNotifyEvent	32

typedef struct {
    CARD8 type;				/* always evBase + RRNotify */
    CARD8 subCode;			/* RRNotify_ProviderProperty */
    CARD16 sequenceNumber;
    Window window;			/* window requesting notification */
    RRProvider provider;		/* affected provider */
    Atom atom;				/* property name */
    Time timestamp;			/* time provider was changed */
    CARD8 state;			/* NewValue or Deleted */
    CARD8 pad1;
    CARD16 pad2;
    CARD32 pad3;
    CARD32 pad4;
} xRRProviderPropertyNotifyEvent;
#define sz_xRRProviderPropertyNotifyEvent	32

typedef struct {
    CARD8 type;				/* always evBase + RRNotify */
    CARD8 subCode;			/* RRNotify_ResourceChange */
    CARD16 sequenceNumber;
    Time timestamp;			/* time resource was changed */
    Window window;			/* window requesting notification */
    CARD32 pad1;
    CARD32 pad2;
    CARD32 pad3;
    CARD32 pad4;
    CARD32 pad5;
} xRRResourceChangeNotifyEvent;
#define sz_xRRResourceChangeNotifyEvent	32

typedef struct {
    CARD8 type;				/* always evBase + RRNotify */
    CARD8 subCode;			/* RRNotify_Lease */
    CARD16 sequenceNumber;
    Time timestamp;			/* time resource was changed */
    Window window;			/* window requesting notification */
    RRLease lease;
    CARD8 created;			/* created/deleted */
    CARD8 pad0;
    CARD16 pad1;
    CARD32 pad2;
    CARD32 pad3;
    CARD32 pad4;
} xRRLeaseNotifyEvent;
#define sz_xRRLeaseNotifyEvent		32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRCrtc	crtc;
} xRRGetPanningReq;
#define sz_xRRGetPanningReq		8

typedef struct {
    BYTE	type;
    CARD8	status;
    CARD16	sequenceNumber;
    CARD32	length;
    Time	timestamp;
    CARD16	left;
    CARD16	top;
    CARD16	width;
    CARD16	height;
    CARD16	track_left;
    CARD16	track_top;
    CARD16	track_width;
    CARD16	track_height;
    INT16	border_left;
    INT16	border_top;
    INT16	border_right;
    INT16	border_bottom;
} xRRGetPanningReply;
#define sz_xRRGetPanningReply		36

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    RRCrtc	crtc;
    Time	timestamp;
    CARD16	left;
    CARD16	top;
    CARD16	width;
    CARD16	height;
    CARD16	track_left;
    CARD16	track_top;
    CARD16	track_width;
    CARD16	track_height;
    INT16	border_left;
    INT16	border_top;
    INT16	border_right;
    INT16	border_bottom;
} xRRSetPanningReq;
#define sz_xRRSetPanningReq		36

typedef struct {
    BYTE	type;
    CARD8	status;
    CARD16	sequenceNumber;
    CARD32	length;
    Time	newTimestamp;
    CARD32	pad1;
    CARD32	pad2;
    CARD32	pad3;
    CARD32	pad4;
    CARD32	pad5;
} xRRSetPanningReply;
#define sz_xRRSetPanningReply	32

typedef struct {
    Atom	name;
    BOOL	primary;
    BOOL	automatic;
    CARD16	noutput;
    INT16	x;
    INT16	y;
    CARD16	width;
    CARD16	height;
    CARD32	widthInMillimeters;
    CARD32	heightInMillimeters;
} xRRMonitorInfo;
#define sz_xRRMonitorInfo	24

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    Window	window;
    BOOL	get_active;
    CARD8	pad;
    CARD16	pad2;
} xRRGetMonitorsReq;
#define sz_xRRGetMonitorsReq	12

typedef struct {
    BYTE	type;
    CARD8	status;
    CARD16	sequenceNumber;
    CARD32	length;
    Time	timestamp;
    CARD32	nmonitors;
    CARD32	noutputs;
    CARD32	pad1;
    CARD32	pad2;
    CARD32	pad3;
} xRRGetMonitorsReply;
#define sz_xRRGetMonitorsReply	32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    Window	window;
    xRRMonitorInfo	monitor;
} xRRSetMonitorReq;
#define sz_xRRSetMonitorReq	32

typedef struct {
    CARD8	reqType;
    CARD8	randrReqType;
    CARD16	length;
    Window	window;
    Atom	name;
} xRRDeleteMonitorReq;
#define sz_xRRDeleteMonitorReq	12

#undef RRLease
#undef RRModeFlags
#undef RRCrtc
#undef RRMode
#undef RROutput
#undef RRMode
#undef RRCrtc
#undef RRProvider
#undef Drawable
#undef Window
#undef Font
#undef Pixmap
#undef Cursor
#undef Colormap
#undef GContext
#undef Atom
#undef Time
#undef KeyCode
#undef KeySym
#undef Rotation
#undef SizeID
#undef SubpixelOrder

#endif /* _XRANDRP_H_ */
