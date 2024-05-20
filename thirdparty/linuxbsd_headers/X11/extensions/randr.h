/*
 * Copyright © 2000 Compaq Computer Corporation
 * Copyright © 2002 Hewlett Packard Company
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

#ifndef _RANDR_H_
#define _RANDR_H_

typedef unsigned short	Rotation;
typedef unsigned short	SizeID;
typedef unsigned short	SubpixelOrder;
typedef unsigned short	Connection;
typedef unsigned short	XRandrRotation;
typedef unsigned short	XRandrSizeID;
typedef unsigned short	XRandrSubpixelOrder;
typedef unsigned long	XRandrModeFlags;

#define RANDR_NAME		"RANDR"
#define RANDR_MAJOR		1
#define RANDR_MINOR		6

#define RRNumberErrors		5
#define RRNumberEvents		2
#define RRNumberRequests	47

#define X_RRQueryVersion	0
/* we skip 1 to make old clients fail pretty immediately */
#define X_RROldGetScreenInfo	1
#define X_RR1_0SetScreenConfig	2
/* V1.0 apps share the same set screen config request id */
#define X_RRSetScreenConfig	2
#define X_RROldScreenChangeSelectInput	3
/* 3 used to be ScreenChangeSelectInput; deprecated */
#define X_RRSelectInput		4
#define X_RRGetScreenInfo	5

/* V1.2 additions */
#define X_RRGetScreenSizeRange	    6
#define X_RRSetScreenSize	    7
#define X_RRGetScreenResources	    8
#define X_RRGetOutputInfo	    9
#define X_RRListOutputProperties    10
#define X_RRQueryOutputProperty	    11
#define X_RRConfigureOutputProperty 12
#define X_RRChangeOutputProperty    13
#define X_RRDeleteOutputProperty    14
#define X_RRGetOutputProperty	    15
#define X_RRCreateMode		    16
#define X_RRDestroyMode		    17
#define X_RRAddOutputMode	    18
#define X_RRDeleteOutputMode	    19
#define X_RRGetCrtcInfo		    20
#define X_RRSetCrtcConfig	    21
#define X_RRGetCrtcGammaSize	    22
#define X_RRGetCrtcGamma	    23
#define X_RRSetCrtcGamma	    24

/* V1.3 additions */
#define X_RRGetScreenResourcesCurrent	25
#define X_RRSetCrtcTransform	    26
#define X_RRGetCrtcTransform	    27
#define X_RRGetPanning		    28
#define X_RRSetPanning		    29
#define X_RRSetOutputPrimary	    30
#define X_RRGetOutputPrimary	    31

#define RRTransformUnit		    (1L << 0)
#define RRTransformScaleUp	    (1L << 1)
#define RRTransformScaleDown	    (1L << 2)
#define RRTransformProjective	    (1L << 3)

/* v1.4 */
#define X_RRGetProviders	      32
#define X_RRGetProviderInfo	      33
#define X_RRSetProviderOffloadSink    34
#define X_RRSetProviderOutputSource   35
#define X_RRListProviderProperties    36
#define X_RRQueryProviderProperty     37
#define X_RRConfigureProviderProperty 38
#define X_RRChangeProviderProperty    39
#define X_RRDeleteProviderProperty    40
#define X_RRGetProviderProperty	      41

/* v1.5 */
#define X_RRGetMonitors		      42
#define X_RRSetMonitor		      43
#define X_RRDeleteMonitor	      44

/* v1.6 */
#define X_RRCreateLease		      45
#define X_RRFreeLease		      46

/* Event selection bits */
#define RRScreenChangeNotifyMask  (1L << 0)
/* V1.2 additions */
#define RRCrtcChangeNotifyMask	    (1L << 1)
#define RROutputChangeNotifyMask    (1L << 2)
#define RROutputPropertyNotifyMask  (1L << 3)
/* V1.4 additions */
#define RRProviderChangeNotifyMask   (1L << 4)
#define RRProviderPropertyNotifyMask (1L << 5)
#define RRResourceChangeNotifyMask   (1L << 6)
/* V1.6 additions */
#define RRLeaseNotifyMask            (1L << 7)

/* Event codes */
#define RRScreenChangeNotify	0
/* V1.2 additions */
#define RRNotify		    1
/* RRNotify Subcodes */
#define  RRNotify_CrtcChange	    0
#define  RRNotify_OutputChange	    1
#define  RRNotify_OutputProperty    2
#define  RRNotify_ProviderChange    3
#define  RRNotify_ProviderProperty  4
#define  RRNotify_ResourceChange    5
/* V1.6 additions */
#define  RRNotify_Lease             6
/* used in the rotation field; rotation and reflection in 0.1 proto. */
#define RR_Rotate_0		1
#define RR_Rotate_90		2
#define RR_Rotate_180		4
#define RR_Rotate_270		8

/* new in 1.0 protocol, to allow reflection of screen */

#define RR_Reflect_X		16
#define RR_Reflect_Y		32

#define RRSetConfigSuccess		0
#define RRSetConfigInvalidConfigTime	1
#define RRSetConfigInvalidTime		2
#define RRSetConfigFailed		3

/* new in 1.2 protocol */

#define RR_HSyncPositive	0x00000001
#define RR_HSyncNegative	0x00000002
#define RR_VSyncPositive	0x00000004
#define RR_VSyncNegative	0x00000008
#define RR_Interlace		0x00000010
#define RR_DoubleScan		0x00000020
#define RR_CSync		0x00000040
#define RR_CSyncPositive	0x00000080
#define RR_CSyncNegative	0x00000100
#define RR_HSkewPresent		0x00000200
#define RR_BCast		0x00000400
#define RR_PixelMultiplex	0x00000800
#define RR_DoubleClock		0x00001000
#define RR_ClockDivideBy2	0x00002000

#define RR_Connected		0
#define RR_Disconnected		1
#define RR_UnknownConnection	2

#define BadRROutput		0
#define BadRRCrtc		1
#define BadRRMode		2
#define BadRRProvider		3
#define BadRRLease		4

/* Conventional RandR output properties */

#define RR_PROPERTY_BACKLIGHT		"Backlight"
#define RR_PROPERTY_RANDR_EDID		"EDID"
#define RR_PROPERTY_SIGNAL_FORMAT	"SignalFormat"
#define RR_PROPERTY_SIGNAL_PROPERTIES	"SignalProperties"
#define RR_PROPERTY_CONNECTOR_TYPE	"ConnectorType"
#define RR_PROPERTY_CONNECTOR_NUMBER	"ConnectorNumber"
#define RR_PROPERTY_COMPATIBILITY_LIST	"CompatibilityList"
#define RR_PROPERTY_CLONE_LIST		"CloneList"
#define RR_PROPERTY_BORDER		"Border"
#define RR_PROPERTY_BORDER_DIMENSIONS	"BorderDimensions"
#define RR_PROPERTY_GUID		"GUID"
#define RR_PROPERTY_RANDR_TILE		"TILE"
#define RR_PROPERTY_NON_DESKTOP		"non-desktop"

/* roles this device can carry out */
#define RR_Capability_None 0
#define RR_Capability_SourceOutput 1
#define RR_Capability_SinkOutput 2
#define RR_Capability_SourceOffload 4
#define RR_Capability_SinkOffload 8

#endif	/* _RANDR_H_ */
