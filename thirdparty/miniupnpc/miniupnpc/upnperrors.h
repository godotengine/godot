/* $Id: upnperrors.h,v 1.2 2008/07/02 23:31:15 nanard Exp $ */
/* (c) 2007-2015 Thomas Bernard
 * All rights reserved.
 * MiniUPnP Project.
 * http://miniupnp.free.fr/ or http://miniupnp.tuxfamily.org/
 * This software is subjet to the conditions detailed in the
 * provided LICENCE file. */
#ifndef UPNPERRORS_H_INCLUDED
#define UPNPERRORS_H_INCLUDED

#include "miniupnpc_declspec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* strupnperror()
 * Return a string description of the UPnP error code
 * or NULL for undefinded errors */
MINIUPNP_LIBSPEC const char * strupnperror(int err);

#ifdef __cplusplus
}
#endif

#endif
