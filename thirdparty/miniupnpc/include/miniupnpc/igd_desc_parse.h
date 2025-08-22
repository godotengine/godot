/* $Id: igd_desc_parse.h,v 1.14 2025/02/08 23:15:16 nanard Exp $ */
/* Project : miniupnp
 * http://miniupnp.free.fr/ or https://miniupnp.tuxfamily.org/
 * Author : Thomas Bernard
 * Copyright (c) 2005-2025 Thomas Bernard
 * This software is subject to the conditions detailed in the
 * LICENCE file provided in this distribution.
 * */
#ifndef IGD_DESC_PARSE_H_INCLUDED
#define IGD_DESC_PARSE_H_INCLUDED

/*! \file igd_desc_parse.h
 * \brief API to parse UPNP device description XML
 * \todo should not be exposed in the public API
 */

/*! \brief maximum lenght of URLs */
#define MINIUPNPC_URL_MAXSIZE (128)

/*! \brief Structure to store the result of the parsing of UPnP
 * descriptions of Internet Gateway Devices services */
struct IGDdatas_service {
	/*! \brief controlURL for the service */
	char controlurl[MINIUPNPC_URL_MAXSIZE];
	/*! \brief eventSubURL for the service */
	char eventsuburl[MINIUPNPC_URL_MAXSIZE];
	/*! \brief SCPDURL for the service */
	char scpdurl[MINIUPNPC_URL_MAXSIZE];
	/*! \brief serviceType */
	char servicetype[MINIUPNPC_URL_MAXSIZE];
	/*char devicetype[MINIUPNPC_URL_MAXSIZE];*/
};

/*! \brief Structure to store the result of the parsing of UPnP
 * descriptions of Internet Gateway Devices */
struct IGDdatas {
	/*! \brief current element name */
	char cureltname[MINIUPNPC_URL_MAXSIZE];
	/*! \brief URLBase */
	char urlbase[MINIUPNPC_URL_MAXSIZE];
	/*! \brief presentationURL */
	char presentationurl[MINIUPNPC_URL_MAXSIZE];
	/*! \brief depth into the XML tree */
	int level;
	/*! \brief "urn:schemas-upnp-org:service:WANCommonInterfaceConfig:1" */
	struct IGDdatas_service CIF;
	/*! \brief first of "urn:schemas-upnp-org:service:WANIPConnection:1"
	 * or "urn:schemas-upnp-org:service:WANPPPConnection:1" */
	struct IGDdatas_service first;
	/*! \brief second of "urn:schemas-upnp-org:service:WANIPConnection:1"
	 * or "urn:schemas-upnp-org:service:WANPPPConnection:1" */
	struct IGDdatas_service second;
	/*! \brief "urn:schemas-upnp-org:service:WANIPv6FirewallControl:1" */
	struct IGDdatas_service IPv6FC;
	/*! \brief currently parsed service */
	struct IGDdatas_service tmp;
};

/*!
 * \brief XML start element handler
 */
void IGDstartelt(void *, const char *, int);
/*!
 * \brief XML end element handler
 */
void IGDendelt(void *, const char *, int);
/*!
 * \brief XML characted data handler
 */
void IGDdata(void *, const char *, int);
#ifdef DEBUG
void printIGD(struct IGDdatas *);
#endif /* DEBUG */

#endif /* IGD_DESC_PARSE_H_INCLUDED */
