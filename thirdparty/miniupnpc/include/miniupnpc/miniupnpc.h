/* $Id: miniupnpc.h,v 1.80 2025/05/26 22:56:40 nanard Exp $ */
/* vim: tabstop=4 shiftwidth=4 noexpandtab
 * Project: miniupnp
 * http://miniupnp.free.fr/ or https://miniupnp.tuxfamily.org/
 * Author: Thomas Bernard
 * Copyright (c) 2005-2025 Thomas Bernard
 * This software is subjects to the conditions detailed
 * in the LICENCE file provided within this distribution */
#ifndef MINIUPNPC_H_INCLUDED
#define MINIUPNPC_H_INCLUDED

/*! \file miniupnpc.h
 * \brief Main C API for MiniUPnPc
 *
 * Contains functions to discover devices and check for device validity
 * or connectivity.
 *
 * \mainpage MiniUPnPc API documentation
 * MiniUPnPc (MiniUPnP client) is a library implementing a UPnP
 * Internet Gateway Device (IGD) control point.
 *
 * It should be used by applications that need to listen to incoming
 * traffic from the internet which are running on a LAN where a
 * UPnP IGD is running on the router (or gateway).
 *
 * See more documentation on the website http://miniupnp.free.fr
 * or https://miniupnp.tuxfamily.org/ or GitHub :
 * https://github.com/miniupnp/miniupnp/tree/master/miniupnpc
 */
#include "miniupnpc_declspec.h"
#include "igd_desc_parse.h"
#include "upnpdev.h"

/* error codes : */
/*! \brief value for success */
#define UPNPDISCOVER_SUCCESS (0)
/*! \brief value for unknown error */
#define UPNPDISCOVER_UNKNOWN_ERROR (-1)
/*! \brief value for a socket error */
#define UPNPDISCOVER_SOCKET_ERROR (-101)
/*! \brief value for a memory allocation error */
#define UPNPDISCOVER_MEMORY_ERROR (-102)

/*! \brief software version */
#define MINIUPNPC_VERSION	"2.3.3"
/*! \brief C API version */
#define MINIUPNPC_API_VERSION	21

/*! \brief any (ie system chosen) port */
#define UPNP_LOCAL_PORT_ANY     0
/*! \brief Use as an alias for 1900 for backwards compatibility */
#define UPNP_LOCAL_PORT_SAME    1

#ifdef __cplusplus
extern "C" {
#endif

/* Structures definitions : */

/*!
 * \brief UPnP method argument
 */
struct UPNParg {
	const char * elt;	/*!< \brief UPnP argument name */
	const char * val;	/*!< \brief UPnP argument value */
};

/*!
 * \brief execute a UPnP method (SOAP action)
 *
 * \todo error reporting should be improved
 *
 * \param[in] url Control URL for the service
 * \param[in] service service to use
 * \param[in] action action to call
 * \param[in] args action arguments
 * \param[out] bufsize the size of the returned buffer
 * \return NULL in case of error or the raw XML response
 */
char *
simpleUPnPcommand(const char * url, const char * service,
                  const char * action, const struct UPNParg * args,
                  int * bufsize);

/*!
 * \brief Discover UPnP IGD on the network.
 *
 * The discovered devices are returned as a chained list.
 * It is up to the caller to free the list with freeUPNPDevlist().
 * If available, device list will be obtained from MiniSSDPd.
 *
 * \param[in] delay (in millisecond) maximum time for waiting any device
 *            response
 * \param[in] multicastif If not NULL, used instead of the default
 *            multicast interface for sending SSDP discover packets
 * \param[in] minissdpdsock Path to minissdpd socket, default is used if
 *            NULL
 * \param[in] localport Source port to send SSDP packets.
 *            #UPNP_LOCAL_PORT_SAME for 1900 (same as destination port)
 *            #UPNP_LOCAL_PORT_ANY to let system assign a source port
 * \param[in] ipv6 0 for IPv4, 1 of IPv6
 * \param[in] ttl should default to 2 as advised by UDA 1.1
 * \param[out] error error code when NULL is returned
 * \return NULL or a linked list
 */
MINIUPNP_LIBSPEC struct UPNPDev *
upnpDiscover(int delay, const char * multicastif,
             const char * minissdpdsock, int localport,
             int ipv6, unsigned char ttl,
             int * error);

/*!
 * \brief Discover all UPnP devices on the network
 *
 * search for "ssdp:all"
 * \param[in] delay (in millisecond) maximum time for waiting any device
 *            response
 * \param[in] multicastif If not NULL, used instead of the default
 *            multicast interface for sending SSDP discover packets
 * \param[in] minissdpdsock Path to minissdpd socket, default is used if
 *            NULL
 * \param[in] localport Source port to send SSDP packets.
 *            #UPNP_LOCAL_PORT_SAME for 1900 (same as destination port)
 *            #UPNP_LOCAL_PORT_ANY to let system assign a source port
 * \param[in] ipv6 0 for IPv4, 1 of IPv6
 * \param[in] ttl should default to 2 as advised by UDA 1.1
 * \param[out] error error code when NULL is returned
 * \return NULL or a linked list
 */
MINIUPNP_LIBSPEC struct UPNPDev *
upnpDiscoverAll(int delay, const char * multicastif,
                const char * minissdpdsock, int localport,
                int ipv6, unsigned char ttl,
                int * error);

/*!
 * \brief Discover one type of UPnP devices
 *
 * \param[in] device device type to search
 * \param[in] delay (in millisecond) maximum time for waiting any device
 *            response
 * \param[in] multicastif If not NULL, used instead of the default
 *            multicast interface for sending SSDP discover packets
 * \param[in] minissdpdsock Path to minissdpd socket, default is used if
 *            NULL
 * \param[in] localport Source port to send SSDP packets.
 *            #UPNP_LOCAL_PORT_SAME for 1900 (same as destination port)
 *            #UPNP_LOCAL_PORT_ANY to let system assign a source port
 * \param[in] ipv6 0 for IPv4, 1 of IPv6
 * \param[in] ttl should default to 2 as advised by UDA 1.1
 * \param[out] error error code when NULL is returned
 * \return NULL or a linked list
 */
MINIUPNP_LIBSPEC struct UPNPDev *
upnpDiscoverDevice(const char * device, int delay, const char * multicastif,
                const char * minissdpdsock, int localport,
                int ipv6, unsigned char ttl,
                int * error);

/*!
 * \brief Discover one or several type of UPnP devices
 *
 * \param[in] deviceTypes array of device types to search (ending with NULL)
 * \param[in] delay (in millisecond) maximum time for waiting any device
 *            response
 * \param[in] multicastif If not NULL, used instead of the default
 *            multicast interface for sending SSDP discover packets
 * \param[in] minissdpdsock Path to minissdpd socket, default is used if
 *            NULL
 * \param[in] localport Source port to send SSDP packets.
 *            #UPNP_LOCAL_PORT_SAME for 1900 (same as destination port)
 *            #UPNP_LOCAL_PORT_ANY to let system assign a source port
 * \param[in] ipv6 0 for IPv4, 1 of IPv6
 * \param[in] ttl should default to 2 as advised by UDA 1.1
 * \param[out] error error code when NULL is returned
 * \param[in] searchalltypes 0 to stop with the first type returning results
 * \return NULL or a linked list
 */
MINIUPNP_LIBSPEC struct UPNPDev *
upnpDiscoverDevices(const char * const deviceTypes[],
                    int delay, const char * multicastif,
                    const char * minissdpdsock, int localport,
                    int ipv6, unsigned char ttl,
                    int * error,
                    int searchalltypes);

/*!
 * \brief parse root XML description of a UPnP device
 *
 * fill the IGDdatas structure.
 * \param[in] buffer character buffer containing the XML description
 * \param[in] bufsize size in bytes of the buffer
 * \param[out] data IGDdatas structure to fill
 */
MINIUPNP_LIBSPEC void parserootdesc(const char * buffer, int bufsize, struct IGDdatas * data);

/*!
 * \brief structure used to get fast access to urls
 */
struct UPNPUrls {
	/*! \brief controlURL of the WANIPConnection */
	char * controlURL;
	/*! \brief url of the description of the WANIPConnection */
	char * ipcondescURL;
	/*! \brief controlURL of the WANCommonInterfaceConfig */
	char * controlURL_CIF;
	/*! \brief controlURL of the WANIPv6FirewallControl */
	char * controlURL_6FC;
	/*! \brief url of the root description */
	char * rootdescURL;
};

/*! \brief NO IGD found */
#define UPNP_NO_IGD (0)
/*! \brief valid and connected IGD */
#define UPNP_CONNECTED_IGD (1)
/*! \brief valid and connected IGD but with a reserved address
 * (non routable) */
#define UPNP_PRIVATEIP_IGD (2)
/*! \brief valid but not connected IGD */
#define UPNP_DISCONNECTED_IGD (3)
/*! \brief UPnP device not recognized as an IGD */
#define UPNP_UNKNOWN_DEVICE (4)

/*!
 * \brief look for a valid and possibly connected IGD in the list
 *
 * In any non zero return case, the urls and data structures
 * passed as parameters are set. Donc forget to call FreeUPNPUrls(urls) to
 * free allocated memory.
 * \param[in] devlist A device list obtained with upnpDiscover() /
 *            upnpDiscoverAll() / upnpDiscoverDevice() / upnpDiscoverDevices()
 * \param[out] urls Urls for the IGD found
 * \param[out] data datas for the IGD found
 * \param[out] lanaddr buffer to copy the local address of the host to reach the IGD
 * \param[in] lanaddrlen size of the lanaddr buffer
 * \param[out] wanaddr buffer to copy the public address of the IGD
 * \param[in] wanaddrlen size of the wanaddr buffer
 * \return #UPNP_NO_IGD / #UPNP_CONNECTED_IGD / #UPNP_PRIVATEIP_IGD /
 *         #UPNP_DISCONNECTED_IGD / #UPNP_UNKNOWN_DEVICE
 */
MINIUPNP_LIBSPEC int
UPNP_GetValidIGD(struct UPNPDev * devlist,
                 struct UPNPUrls * urls,
                 struct IGDdatas * data,
                 char * lanaddr, int lanaddrlen,
                 char * wanaddr, int wanaddrlen);

/*!
 * \brief Get IGD URLs and data for URL
 *
 * Used when skipping the discovery process.
 * \param[in] rootdescurl Root description URL of the device
 * \param[out] urls Urls for the IGD found
 * \param[out] data datas for the IGD found
 * \param[out] lanaddr buffer to copy the local address of the host to reach the IGD
 * \param[in] lanaddrlen size of the lanaddr buffer
 * \return 0 Not ok / 1 OK
 */
MINIUPNP_LIBSPEC int
UPNP_GetIGDFromUrl(const char * rootdescurl,
                   struct UPNPUrls * urls,
                   struct IGDdatas * data,
                   char * lanaddr, int lanaddrlen);

/*!
 * \brief Prepare the URLs for usage
 *
 * build absolute URLs from the root description
 * \param[out] urls URL structure to initialize
 * \param[in] data datas for the IGD
 * \param[in] descURL root description URL for the IGD
 * \param[in] scope_id if not 0, add the scope to the linklocal IPv6
 *            addresses in URLs
 */
MINIUPNP_LIBSPEC void
GetUPNPUrls(struct UPNPUrls * urls, struct IGDdatas * data,
            const char * descURL, unsigned int scope_id);

/*!
 * \brief free the members of a UPNPUrls struct
 *
 * All URLs buffers are freed and zeroed
 * \param[out] urls URL structure to free
 */
MINIUPNP_LIBSPEC void
FreeUPNPUrls(struct UPNPUrls * urls);

/*!
 * \brief check the current connection status of an IGD
 *
 * it uses UPNP_GetStatusInfo()
 * \param[in] urls IGD URLs
 * \param[in] data IGD data
 * \return 1 Connected / 0 Disconnected
 */
MINIUPNP_LIBSPEC int UPNPIGD_IsConnected(struct UPNPUrls * urls, struct IGDdatas * data);


#ifdef __cplusplus
}
#endif

#endif

