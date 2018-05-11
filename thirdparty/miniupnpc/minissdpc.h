/* $Id: minissdpc.h,v 1.6 2015/09/18 12:45:16 nanard Exp $ */
/* Project: miniupnp
 * http://miniupnp.free.fr/ or http://miniupnp.tuxfamily.org/
 * Author: Thomas Bernard
 * Copyright (c) 2005-2015 Thomas Bernard
 * This software is subjects to the conditions detailed
 * in the LICENCE file provided within this distribution */
#ifndef MINISSDPC_H_INCLUDED
#define MINISSDPC_H_INCLUDED

#include "miniupnpc_declspec.h"
#include "upnpdev.h"

/* error codes : */
#define MINISSDPC_SUCCESS (0)
#define MINISSDPC_UNKNOWN_ERROR (-1)
#define MINISSDPC_SOCKET_ERROR (-101)
#define MINISSDPC_MEMORY_ERROR (-102)
#define MINISSDPC_INVALID_INPUT (-103)
#define MINISSDPC_INVALID_SERVER_REPLY (-104)

#ifdef __cplusplus
extern "C" {
#endif

#if !(defined(_WIN32) || defined(__amigaos__) || defined(__amigaos4__))

MINIUPNP_LIBSPEC struct UPNPDev *
getDevicesFromMiniSSDPD(const char * devtype, const char * socketpath, int * error);

MINIUPNP_LIBSPEC int
connectToMiniSSDPD(const char * socketpath);

MINIUPNP_LIBSPEC int
disconnectFromMiniSSDPD(int fd);

MINIUPNP_LIBSPEC int
requestDevicesFromMiniSSDPD(int fd, const char * devtype);

MINIUPNP_LIBSPEC struct UPNPDev *
receiveDevicesFromMiniSSDPD(int fd, int * error);

#endif /* !(defined(_WIN32) || defined(__amigaos__) || defined(__amigaos4__)) */

MINIUPNP_LIBSPEC struct UPNPDev *
ssdpDiscoverDevices(const char * const deviceTypes[],
                    int delay, const char * multicastif,
                    int localport,
                    int ipv6, unsigned char ttl,
                    int * error,
                    int searchalltypes);

#ifdef __cplusplus
}
#endif

#endif

