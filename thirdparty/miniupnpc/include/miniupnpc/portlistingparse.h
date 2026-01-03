/* $Id: portlistingparse.h,v 1.12 2025/02/08 23:15:17 nanard Exp $ */
/* MiniUPnP project
 * http://miniupnp.free.fr/ or http://miniupnp.tuxfamily.org/
 * (c) 2011-2025 Thomas Bernard
 * This software is subject to the conditions detailed
 * in the LICENCE file provided within the distribution */
#ifndef PORTLISTINGPARSE_H_INCLUDED
#define PORTLISTINGPARSE_H_INCLUDED

/*! \file portlistingparse.h
 * \brief Parsing of the list of port mappings
 *
 * As returned by GetListOfPortMappings.
 * Sample of PortMappingEntry :
 *
 *     <p:PortMappingEntry>
 *       <p:NewRemoteHost>202.233.2.1</p:NewRemoteHost>
 *       <p:NewExternalPort>2345</p:NewExternalPort>
 *       <p:NewProtocol>TCP</p:NewProtocol>
 *       <p:NewInternalPort>2345</p:NewInternalPort>
 *       <p:NewInternalClient>192.168.1.137</p:NewInternalClient>
 *       <p:NewEnabled>1</p:NewEnabled>
 *       <p:NewDescription>dooom</p:NewDescription>
 *       <p:NewLeaseTime>345</p:NewLeaseTime>
 *     </p:PortMappingEntry>
 */
#include "miniupnpc_declspec.h"
/* for the definition of UNSIGNED_INTEGER */
#include "miniupnpctypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief enum of all XML elements
 */
typedef enum { PortMappingEltNone,
       PortMappingEntry, NewRemoteHost,
       NewExternalPort, NewProtocol,
       NewInternalPort, NewInternalClient,
       NewEnabled, NewDescription,
       NewLeaseTime } portMappingElt;

/*!
 * \brief linked list of port mappings
 */
struct PortMapping {
	struct PortMapping * l_next;	/*!< \brief next list element */
	UNSIGNED_INTEGER leaseTime;	/*!< \brief in seconds */
	unsigned short externalPort;	/*!< \brief external port */
	unsigned short internalPort;	/*!< \brief internal port */
	char remoteHost[64];	/*!< \brief empty for wildcard */
	char internalClient[64];	/*!< \brief internal IP address */
	char description[64];	/*!< \brief description */
	char protocol[4];		/*!< \brief `TCP` or `UDP` */
	unsigned char enabled;	/*!< \brief 0 (false) or 1 (true) */
};

/*!
 * \brief structure for ParsePortListing()
 */
struct PortMappingParserData {
	struct PortMapping * l_head;	/*!< \brief list head */
	portMappingElt curelt;			/*!< \brief currently parsed element */
};

/*!
 * \brief parse the NewPortListing part of GetListOfPortMappings response
 *
 * \param[in] buffer XML data
 * \param[in] bufsize length of XML data
 * \param[out] pdata Parsed data
 */
MINIUPNP_LIBSPEC void
ParsePortListing(const char * buffer, int bufsize,
                 struct PortMappingParserData * pdata);

/*!
 * \brief free parsed data structure
 *
 * \param[in] pdata Parsed data to free
 */
MINIUPNP_LIBSPEC void
FreePortListing(struct PortMappingParserData * pdata);

#ifdef __cplusplus
}
#endif

#endif
