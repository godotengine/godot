/* $Id: upnpdev.h,v 1.6 2025/02/08 23:15:17 nanard Exp $ */
/* Project : miniupnp
 * Web : http://miniupnp.free.fr/ or https://miniupnp.tuxfamily.org/
 * Author : Thomas BERNARD
 * copyright (c) 2005-2025 Thomas Bernard
 * This software is subjet to the conditions detailed in the
 * provided LICENSE file. */
#ifndef UPNPDEV_H_INCLUDED
#define UPNPDEV_H_INCLUDED

/*! \file upnpdev.h
 * \brief UPNPDev device linked-list structure
 * \todo could be merged into miniupnpc.h
 */
#include "miniupnpc_declspec.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief UPnP device linked-list
 */
struct UPNPDev {
	/*! \brief pointer to the next element */
	struct UPNPDev * pNext;
	/*! \brief root description URL */
	char * descURL;
	/*! \brief ST: as advertised */
	char * st;
	/*! \brief USN: as advertised */
	char * usn;
	/*! \brief IPv6 scope id of the network interface */
	unsigned int scope_id;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
	/* C99 flexible array member */
	/*! \brief buffer for descURL, st and usn */
	char buffer[];
#elif defined(__GNUC__)
	char buffer[0];
#else
	/* Fallback to a hack */
	char buffer[1];
#endif
};

/*! \brief free list returned by upnpDiscover()
 * \param[in] devlist linked list to free
 */
MINIUPNP_LIBSPEC void freeUPNPDevlist(struct UPNPDev * devlist);


#ifdef __cplusplus
}
#endif


#endif /* UPNPDEV_H_INCLUDED */
