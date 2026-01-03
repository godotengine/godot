/* $Id: upnpcommands.c,v 1.56 2025/03/29 18:08:59 nanard Exp $ */
/* vim: tabstop=4 shiftwidth=4 noexpandtab
 * Project : miniupnp
 * Author : Thomas Bernard
 * Copyright (c) 2005-2025 Thomas Bernard
 * This software is subject to the conditions detailed in the
 * LICENCE file provided in this distribution.
 * */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "upnpcommands.h"
#include "miniupnpc.h"
#include "portlistingparse.h"
#include "upnpreplyparse.h"

/*! \file upnpcommands.c
 * \brief Internet Gateway Device methods implementations
 * \def STRTOUI
 * \brief strtoull() if available, strtol() if not
 */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#define STRTOUI	strtoull
#else
#define STRTOUI	strtoul
#endif

static UNSIGNED_INTEGER
my_atoui(const char * s)
{
	return s ? ((UNSIGNED_INTEGER)STRTOUI(s, NULL, 0)) : 0;
}

/*
 * */
MINIUPNP_LIBSPEC UNSIGNED_INTEGER
UPNP_GetTotalBytesSent(const char * controlURL,
					const char * servicetype)
{
	struct NameValueParserData pdata;
	char * buffer;
	int bufsize;
	unsigned int r = 0;
	char * p;
	if(!(buffer = simpleUPnPcommand(controlURL, servicetype,
	                                "GetTotalBytesSent", 0, &bufsize))) {
		return (UNSIGNED_INTEGER)UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	/*DisplayNameValueList(buffer, bufsize);*/
	free(buffer);
	p = GetValueFromNameValueList(&pdata, "NewTotalBytesSent");
	r = my_atoui(p);
	ClearNameValueList(&pdata);
	return r;
}

/*
 * */
MINIUPNP_LIBSPEC UNSIGNED_INTEGER
UPNP_GetTotalBytesReceived(const char * controlURL,
						const char * servicetype)
{
	struct NameValueParserData pdata;
	char * buffer;
	int bufsize;
	unsigned int r = 0;
	char * p;
	if(!(buffer = simpleUPnPcommand(controlURL, servicetype,
	                                "GetTotalBytesReceived", 0, &bufsize))) {
		return (UNSIGNED_INTEGER)UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	/*DisplayNameValueList(buffer, bufsize);*/
	free(buffer);
	p = GetValueFromNameValueList(&pdata, "NewTotalBytesReceived");
	r = my_atoui(p);
	ClearNameValueList(&pdata);
	return r;
}

/*
 * */
MINIUPNP_LIBSPEC UNSIGNED_INTEGER
UPNP_GetTotalPacketsSent(const char * controlURL,
						const char * servicetype)
{
	struct NameValueParserData pdata;
	char * buffer;
	int bufsize;
	unsigned int r = 0;
	char * p;
	if(!(buffer = simpleUPnPcommand(controlURL, servicetype,
	                                "GetTotalPacketsSent", 0, &bufsize))) {
		return (UNSIGNED_INTEGER)UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	/*DisplayNameValueList(buffer, bufsize);*/
	free(buffer);
	p = GetValueFromNameValueList(&pdata, "NewTotalPacketsSent");
	r = my_atoui(p);
	ClearNameValueList(&pdata);
	return r;
}

/*
 * */
MINIUPNP_LIBSPEC UNSIGNED_INTEGER
UPNP_GetTotalPacketsReceived(const char * controlURL,
						const char * servicetype)
{
	struct NameValueParserData pdata;
	char * buffer;
	int bufsize;
	unsigned int r = 0;
	char * p;
	if(!(buffer = simpleUPnPcommand(controlURL, servicetype,
	                                "GetTotalPacketsReceived", 0, &bufsize))) {
		return (UNSIGNED_INTEGER)UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	/*DisplayNameValueList(buffer, bufsize);*/
	free(buffer);
	p = GetValueFromNameValueList(&pdata, "NewTotalPacketsReceived");
	r = my_atoui(p);
	ClearNameValueList(&pdata);
	return r;
}

/* UPNP_GetStatusInfo() call the corresponding UPNP method
 * returns the current status and uptime */
MINIUPNP_LIBSPEC int
UPNP_GetStatusInfo(const char * controlURL,
				const char * servicetype,
				char * status,
				unsigned int * uptime,
				char * lastconnerror)
{
	struct NameValueParserData pdata;
	char * buffer;
	int bufsize;
	char * p;
	char * up;
	char * err;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;

	if(!status && !uptime)
		return UPNPCOMMAND_INVALID_ARGS;

	if(!(buffer = simpleUPnPcommand(controlURL, servicetype,
	                                "GetStatusInfo", 0, &bufsize))) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	/*DisplayNameValueList(buffer, bufsize);*/
	free(buffer);
	up = GetValueFromNameValueList(&pdata, "NewUptime");
	p = GetValueFromNameValueList(&pdata, "NewConnectionStatus");
	err = GetValueFromNameValueList(&pdata, "NewLastConnectionError");
	if(p && up)
	  ret = UPNPCOMMAND_SUCCESS;

	if(status) {
		if(p){
			strncpy(status, p, 64 );
			status[63] = '\0';
		}else
			status[0]= '\0';
	}

	if(uptime) {
		if(!up || sscanf(up,"%u",uptime) != 1)
			*uptime = 0;
	}

	if(lastconnerror) {
		if(err) {
			strncpy(lastconnerror, err, 64 );
			lastconnerror[63] = '\0';
		} else
			lastconnerror[0] = '\0';
	}

	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p) {
		if(sscanf(p, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}
	ClearNameValueList(&pdata);
	return ret;
}

/* UPNP_GetConnectionTypeInfo() call the corresponding UPNP method
 * returns the connection type */
MINIUPNP_LIBSPEC int
UPNP_GetConnectionTypeInfo(const char * controlURL,
                           const char * servicetype,
                           char * connectionType)
{
	struct NameValueParserData pdata;
	char * buffer;
	int bufsize;
	char * p;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;

	if(!connectionType)
		return UPNPCOMMAND_INVALID_ARGS;

	if(!(buffer = simpleUPnPcommand(controlURL, servicetype,
	                                "GetConnectionTypeInfo", 0, &bufsize))) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);
	p = GetValueFromNameValueList(&pdata, "NewConnectionType");
	/*p = GetValueFromNameValueList(&pdata, "NewPossibleConnectionTypes");*/
	/* PossibleConnectionTypes will have several values.... */
	if(p) {
		strncpy(connectionType, p, 64 );
		connectionType[63] = '\0';
		ret = UPNPCOMMAND_SUCCESS;
	} else
		connectionType[0] = '\0';
	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p) {
		if(sscanf(p, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}
	ClearNameValueList(&pdata);
	return ret;
}

/* UPNP_GetLinkLayerMaxBitRate() call the corresponding UPNP method.
 * Returns 2 values: Downloadlink bandwidth and Uplink bandwidth.
 * One of the values can be null
 * Note : GetLinkLayerMaxBitRates belongs to WANPPPConnection:1 only
 * We can use the GetCommonLinkProperties from WANCommonInterfaceConfig:1 */
MINIUPNP_LIBSPEC int
UPNP_GetLinkLayerMaxBitRates(const char * controlURL,
                             const char * servicetype,
                             unsigned int * bitrateDown,
                             unsigned int * bitrateUp)
{
	struct NameValueParserData pdata;
	char * buffer;
	int bufsize;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;
	char * down;
	char * up;
	char * p;

	if(!bitrateDown && !bitrateUp)
		return UPNPCOMMAND_INVALID_ARGS;

	/* shouldn't we use GetCommonLinkProperties ? */
	if(!(buffer = simpleUPnPcommand(controlURL, servicetype,
	                                "GetCommonLinkProperties", 0, &bufsize))) {
	                              /*"GetLinkLayerMaxBitRates", 0, &bufsize);*/
		return UPNPCOMMAND_HTTP_ERROR;
	}
	/*DisplayNameValueList(buffer, bufsize);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);
	/*down = GetValueFromNameValueList(&pdata, "NewDownstreamMaxBitRate");*/
	/*up = GetValueFromNameValueList(&pdata, "NewUpstreamMaxBitRate");*/
	down = GetValueFromNameValueList(&pdata, "NewLayer1DownstreamMaxBitRate");
	up = GetValueFromNameValueList(&pdata, "NewLayer1UpstreamMaxBitRate");
	/*GetValueFromNameValueList(&pdata, "NewWANAccessType");*/
	/*GetValueFromNameValueList(&pdata, "NewPhysicalLinkStatus");*/
	if(down && up)
		ret = UPNPCOMMAND_SUCCESS;

	if(bitrateDown) {
		if(!down || sscanf(down,"%u",bitrateDown) != 1)
			*bitrateDown = 0;
	}

	if(bitrateUp) {
		if(!up || sscanf(up,"%u",bitrateUp) != 1)
			*bitrateUp = 0;
	}
	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p) {
		if(sscanf(p, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}
	ClearNameValueList(&pdata);
	return ret;
}


/* UPNP_GetExternalIPAddress() call the corresponding UPNP method.
 * if the third arg is not null the value is copied to it.
 * at least 16 bytes must be available
 *
 * Return values :
 * 0 : SUCCESS
 * NON ZERO : ERROR Either an UPnP error code or an unknown error.
 *
 * 402 Invalid Args - See UPnP Device Architecture section on Control.
 * 501 Action Failed - See UPnP Device Architecture section on Control.
 */
MINIUPNP_LIBSPEC int
UPNP_GetExternalIPAddress(const char * controlURL,
                          const char * servicetype,
                          char * extIpAdd)
{
	struct NameValueParserData pdata;
	char * buffer;
	int bufsize;
	char * p;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;

	if(!extIpAdd || !controlURL || !servicetype)
		return UPNPCOMMAND_INVALID_ARGS;

	if(!(buffer = simpleUPnPcommand(controlURL, servicetype,
	                                "GetExternalIPAddress", 0, &bufsize))) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	/*DisplayNameValueList(buffer, bufsize);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);
	/*printf("external ip = %s\n", GetValueFromNameValueList(&pdata, "NewExternalIPAddress") );*/
	p = GetValueFromNameValueList(&pdata, "NewExternalIPAddress");
	if(p) {
		strncpy(extIpAdd, p, 16 );
		extIpAdd[15] = '\0';
		ret = UPNPCOMMAND_SUCCESS;
	} else
		extIpAdd[0] = '\0';

	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p) {
		if(sscanf(p, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}

	ClearNameValueList(&pdata);
	return ret;
}

MINIUPNP_LIBSPEC int
UPNP_AddPortMapping(const char * controlURL, const char * servicetype,
		    const char * extPort,
		    const char * inPort,
		    const char * inClient,
		    const char * desc,
		    const char * proto,
		    const char * remoteHost,
		    const char * leaseDuration)
{
	struct UPNParg AddPortMappingArgs[] = {
		{"NewRemoteHost", remoteHost},
		{"NewExternalPort", extPort},
		{"NewProtocol", proto},
		{"NewInternalPort", inPort},
		{"NewInternalClient", inClient},
		{"NewEnabled", "1"},
		{"NewPortMappingDescription", desc?desc:"libminiupnpc"},
		{"NewLeaseDuration", leaseDuration?leaseDuration:"0"},
		{NULL, NULL}
	};
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	int ret;

	if(!inPort || !inClient || !proto || !extPort)
		return UPNPCOMMAND_INVALID_ARGS;

	buffer = simpleUPnPcommand(controlURL, servicetype,
	                           "AddPortMapping", AddPortMappingArgs,
	                           &bufsize);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	/*DisplayNameValueList(buffer, bufsize);*/
	/*buffer[bufsize] = '\0';*/
	/*puts(buffer);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal) {
		/*printf("AddPortMapping errorCode = '%s'\n", resVal); */
		if(sscanf(resVal, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	} else {
		ret = UPNPCOMMAND_SUCCESS;
	}
	ClearNameValueList(&pdata);
	return ret;
}

MINIUPNP_LIBSPEC int
UPNP_AddAnyPortMapping(const char * controlURL, const char * servicetype,
		       const char * extPort,
		       const char * inPort,
		       const char * inClient,
		       const char * desc,
		       const char * proto,
		       const char * remoteHost,
		       const char * leaseDuration,
		       char * reservedPort)
{
	struct UPNParg AddAnyPortMappingArgs[] = {
		{"NewRemoteHost", remoteHost},
		{"NewExternalPort", extPort},
		{"NewProtocol", proto},
		{"NewInternalPort", inPort},
		{"NewInternalClient", inClient},
		{"NewEnabled", "1"},
		{"NewPortMappingDescription", desc?desc:"libminiupnpc"},
		{"NewLeaseDuration", leaseDuration?leaseDuration:"0"},
		{NULL, NULL}
	};
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	int ret;

	if(!inPort || !inClient || !proto || !extPort || !reservedPort)
		return UPNPCOMMAND_INVALID_ARGS;
	buffer = simpleUPnPcommand(controlURL, servicetype,
	                           "AddAnyPortMapping", AddAnyPortMappingArgs,
	                           &bufsize);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal) {
		if(sscanf(resVal, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	} else {
		char *p;

		p = GetValueFromNameValueList(&pdata, "NewReservedPort");
		if(p) {
			strncpy(reservedPort, p, 6);
			reservedPort[5] = '\0';
			ret = UPNPCOMMAND_SUCCESS;
		} else {
			ret = UPNPCOMMAND_INVALID_RESPONSE;
		}
	}
	ClearNameValueList(&pdata);
	return ret;
}

MINIUPNP_LIBSPEC int
UPNP_DeletePortMapping(const char * controlURL, const char * servicetype,
                       const char * extPort, const char * proto,
                       const char * remoteHost)
{
	/*struct NameValueParserData pdata;*/
	struct UPNParg DeletePortMappingArgs[] = {
		{"NewRemoteHost", remoteHost},
		{"NewExternalPort", extPort},
		{"NewProtocol", proto},
		{NULL, NULL}
	};
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	int ret;

	if(!extPort || !proto)
		return UPNPCOMMAND_INVALID_ARGS;

	buffer = simpleUPnPcommand(controlURL, servicetype,
	                          "DeletePortMapping",
	                          DeletePortMappingArgs, &bufsize);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	/*DisplayNameValueList(buffer, bufsize);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal) {
		if(sscanf(resVal, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	} else {
		ret = UPNPCOMMAND_SUCCESS;
	}
	ClearNameValueList(&pdata);
	return ret;
}

MINIUPNP_LIBSPEC int
UPNP_DeletePortMappingRange(const char * controlURL, const char * servicetype,
                            const char * extPortStart, const char * extPortEnd,
                            const char * proto,
                            const char * manage)
{
	struct UPNParg DeletePortMappingRangeArgs[] = {
		{"NewStartPort", extPortStart},
		{"NewEndPort", extPortEnd},
		{"NewProtocol", proto},
		{"NewManage", manage},
		{NULL, NULL}
	};
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	int ret;

	if(!extPortStart || !extPortEnd || !proto || !manage)
		return UPNPCOMMAND_INVALID_ARGS;


	buffer = simpleUPnPcommand(controlURL, servicetype,
	                           "DeletePortMappingRange",
	                           DeletePortMappingRangeArgs, &bufsize);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal) {
		if(sscanf(resVal, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	} else {
		ret = UPNPCOMMAND_SUCCESS;
	}
	ClearNameValueList(&pdata);
	return ret;
}

MINIUPNP_LIBSPEC int
UPNP_GetGenericPortMappingEntry(const char * controlURL,
                                const char * servicetype,
                                const char * index,
                                char * extPort,
                                char * intClient,
                                char * intPort,
                                char * protocol,
                                char * desc,
                                char * enabled,
                                char * rHost,
                                char * duration)
{
	return UPNP_GetGenericPortMappingEntryExt(controlURL, servicetype, index,
	                                          extPort, intClient, intPort,
	                                          protocol, desc, 80, enabled,
	                                          rHost, 64, duration);
}

MINIUPNP_LIBSPEC int
UPNP_GetGenericPortMappingEntryExt(const char * controlURL,
                                   const char * servicetype,
                                   const char * index,
                                   char * extPort,
                                   char * intClient,
                                   char * intPort,
                                   char * protocol,
                                   char * desc,
                                   size_t desclen,
                                   char * enabled,
                                   char * rHost,
                                   size_t rHostlen,
                                   char * duration)
{
	struct NameValueParserData pdata;
	struct UPNParg GetPortMappingArgs[] = {
		{"NewPortMappingIndex", index},
		{NULL, NULL}
	};
	char * buffer;
	int bufsize;
	char * p;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;
	if(!index || !extPort || !intClient || !intPort || !protocol)
		return UPNPCOMMAND_INVALID_ARGS;
	buffer = simpleUPnPcommand(controlURL, servicetype,
	                           "GetGenericPortMappingEntry",
	                           GetPortMappingArgs, &bufsize);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);

	p = GetValueFromNameValueList(&pdata, "NewRemoteHost");
	if(p && rHost)
	{
		strncpy(rHost, p, rHostlen);
		rHost[rHostlen-1] = '\0';
	}
	p = GetValueFromNameValueList(&pdata, "NewExternalPort");
	if(p)
	{
		strncpy(extPort, p, 6);
		extPort[5] = '\0';
		ret = UPNPCOMMAND_SUCCESS;
	}
	else
	{
		extPort[0] = '\0';
	}
	p = GetValueFromNameValueList(&pdata, "NewProtocol");
	if(p)
	{
		strncpy(protocol, p, 4);
		protocol[3] = '\0';
	}
	else
	{
		protocol[0] = '\0';
	}
	p = GetValueFromNameValueList(&pdata, "NewInternalClient");
	if(p)
	{
		strncpy(intClient, p, 16);
		intClient[15] = '\0';
		ret = 0;
	}
	else
	{
		intClient[0] = '\0';
	}
	p = GetValueFromNameValueList(&pdata, "NewInternalPort");
	if(p)
	{
		strncpy(intPort, p, 6);
		intPort[5] = '\0';
	}
	else
	{
		intPort[0] = '\0';
	}
	p = GetValueFromNameValueList(&pdata, "NewEnabled");
	if(p && enabled)
	{
		strncpy(enabled, p, 4);
		enabled[3] = '\0';
	}
	p = GetValueFromNameValueList(&pdata, "NewPortMappingDescription");
	if(p && desc)
	{
		strncpy(desc, p, desclen);
		desc[desclen-1] = '\0';
	}
	p = GetValueFromNameValueList(&pdata, "NewLeaseDuration");
	if(p && duration)
	{
		strncpy(duration, p, 16);
		duration[15] = '\0';
	}
	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p) {
		if(sscanf(p, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}
	ClearNameValueList(&pdata);
	return ret;
}

MINIUPNP_LIBSPEC int
UPNP_GetPortMappingNumberOfEntries(const char * controlURL,
                                   const char * servicetype,
                                   unsigned int * numEntries)
{
	struct NameValueParserData pdata;
	char * buffer;
	int bufsize;
	char* p;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;
	if(!(buffer = simpleUPnPcommand(controlURL, servicetype,
	                                "GetPortMappingNumberOfEntries", 0,
	                                &bufsize))) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
#ifdef DEBUG
	DisplayNameValueList(buffer, bufsize);
#endif
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);

	p = GetValueFromNameValueList(&pdata, "NewPortMappingNumberOfEntries");
	if(numEntries && p) {
		*numEntries = 0;
		sscanf(p, "%u", numEntries);
		ret = UPNPCOMMAND_SUCCESS;
	}

	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p) {
		if(sscanf(p, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}

	ClearNameValueList(&pdata);
	return ret;
}

/* UPNP_GetSpecificPortMappingEntry retrieves an existing port mapping
 * the result is returned in the intClient and intPort strings
 * please provide 16 and 6 bytes of data */
MINIUPNP_LIBSPEC int
UPNP_GetSpecificPortMappingEntry(const char * controlURL,
                                 const char * servicetype,
                                 const char * extPort,
                                 const char * proto,
                                 const char * remoteHost,
                                 char * intClient,
                                 char * intPort,
                                 char * desc,
                                 char * enabled,
                                 char * leaseDuration)
{
	return UPNP_GetSpecificPortMappingEntryExt(controlURL, servicetype,
	                                           extPort, proto, remoteHost,
	                                           intClient, intPort,
	                                           desc, 80, enabled,
	                                           leaseDuration);
}

MINIUPNP_LIBSPEC int
UPNP_GetSpecificPortMappingEntryExt(const char * controlURL,
                                    const char * servicetype,
                                    const char * extPort,
                                    const char * proto,
                                    const char * remoteHost,
                                    char * intClient,
                                    char * intPort,
                                    char * desc,
                                    size_t desclen,
                                    char * enabled,
                                    char * leaseDuration)
{
	struct NameValueParserData pdata;
	struct UPNParg GetPortMappingArgs[] = {
		{"NewRemoteHost", remoteHost},
		{"NewExternalPort", extPort},
		{"NewProtocol", proto},
		{NULL, NULL}
	};
	char * buffer;
	int bufsize;
	char * p;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;

	if(!intPort || !intClient || !extPort || !proto)
		return UPNPCOMMAND_INVALID_ARGS;

	buffer = simpleUPnPcommand(controlURL, servicetype,
	                           "GetSpecificPortMappingEntry",
	                           GetPortMappingArgs, &bufsize);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	/*DisplayNameValueList(buffer, bufsize);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);

	p = GetValueFromNameValueList(&pdata, "NewInternalClient");
	if(p) {
		strncpy(intClient, p, 16);
		intClient[15] = '\0';
		ret = UPNPCOMMAND_SUCCESS;
	} else
		intClient[0] = '\0';

	p = GetValueFromNameValueList(&pdata, "NewInternalPort");
	if(p) {
		strncpy(intPort, p, 6);
		intPort[5] = '\0';
	} else
		intPort[0] = '\0';

	p = GetValueFromNameValueList(&pdata, "NewEnabled");
	if(p && enabled) {
		strncpy(enabled, p, 4);
		enabled[3] = '\0';
	}

	p = GetValueFromNameValueList(&pdata, "NewPortMappingDescription");
	if(p && desc) {
		strncpy(desc, p, desclen);
		desc[desclen-1] = '\0';
	}

	p = GetValueFromNameValueList(&pdata, "NewLeaseDuration");
	if(p && leaseDuration)
	{
		strncpy(leaseDuration, p, 16);
		leaseDuration[15] = '\0';
	}

	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p) {
		if(sscanf(p, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}

	ClearNameValueList(&pdata);
	return ret;
}

/* UPNP_GetListOfPortMappings()
 *
 * Possible UPNP Error codes :
 * 606 Action not Authorized
 * 730 PortMappingNotFound - no port mapping is found in the specified range.
 * 733 InconsistantParameters - NewStartPort and NewEndPort values are not
 *                              consistent.
 */
MINIUPNP_LIBSPEC int
UPNP_GetListOfPortMappings(const char * controlURL,
                           const char * servicetype,
                           const char * startPort,
                           const char * endPort,
                           const char * protocol,
                           const char * numberOfPorts,
                           struct PortMappingParserData * data)
{
	struct NameValueParserData pdata;
	struct UPNParg GetListOfPortMappingsArgs[] = {
		{"NewStartPort", startPort},
		{"NewEndPort", endPort},
		{"NewProtocol", protocol},
		{"NewManage", "1"},
		{"NewNumberOfPorts", numberOfPorts?numberOfPorts:"1000"},
		{NULL, NULL}
	};
	const char * p;
	char * buffer;
	int bufsize;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;

	if(!startPort || !endPort || !protocol)
		return UPNPCOMMAND_INVALID_ARGS;

	buffer = simpleUPnPcommand(controlURL, servicetype,
	                           "GetListOfPortMappings",
	                           GetListOfPortMappingsArgs, &bufsize);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}

	/*DisplayNameValueList(buffer, bufsize);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);

	/*p = GetValueFromNameValueList(&pdata, "NewPortListing");*/
	/*if(p) {
		printf("NewPortListing : %s\n", p);
	}*/
	/*printf("NewPortListing(%d chars) : %s\n",
	       pdata.portListingLength, pdata.portListing);*/
	if(pdata.portListing)
	{
		/*struct PortMapping * pm;
		int i = 0;*/
		ParsePortListing(pdata.portListing, pdata.portListingLength,
		                 data);
		ret = UPNPCOMMAND_SUCCESS;
		/*
		for(pm = data->head.lh_first; pm != NULL; pm = pm->entries.le_next)
		{
			printf("%2d %s %5hu->%s:%-5hu '%s' '%s'\n",
			       i, pm->protocol, pm->externalPort, pm->internalClient,
			       pm->internalPort,
			       pm->description, pm->remoteHost);
			i++;
		}
		*/
		/*FreePortListing(&data);*/
	}

	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p) {
		if(sscanf(p, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}
	ClearNameValueList(&pdata);

	/*printf("%.*s", bufsize, buffer);*/

	return ret;
}

/* IGD:2, functions for service WANIPv6FirewallControl:1 */
MINIUPNP_LIBSPEC int
UPNP_GetFirewallStatus(const char * controlURL,
				const char * servicetype,
				int * firewallEnabled,
				int * inboundPinholeAllowed)
{
	struct NameValueParserData pdata;
	char * buffer;
	int bufsize;
	char * fe, *ipa, *p;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;

	if(!firewallEnabled || !inboundPinholeAllowed)
		return UPNPCOMMAND_INVALID_ARGS;

	buffer = simpleUPnPcommand(controlURL, servicetype,
	                           "GetFirewallStatus", 0, &bufsize);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);
	fe = GetValueFromNameValueList(&pdata, "FirewallEnabled");
	ipa = GetValueFromNameValueList(&pdata, "InboundPinholeAllowed");
	if(ipa && fe)
		ret = UPNPCOMMAND_SUCCESS;
	if(fe)
		*firewallEnabled = my_atoui(fe);
	/*else
		*firewallEnabled = 0;*/
	if(ipa)
		*inboundPinholeAllowed = my_atoui(ipa);
	/*else
		*inboundPinholeAllowed = 0;*/
	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p)
	{
		if(sscanf(p, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}
	ClearNameValueList(&pdata);
	return ret;
}

MINIUPNP_LIBSPEC int
UPNP_GetOutboundPinholeTimeout(const char * controlURL, const char * servicetype,
                    const char * remoteHost,
                    const char * remotePort,
                    const char * intClient,
                    const char * intPort,
                    const char * proto,
                    int * opTimeout)
{
	struct UPNParg GetOutboundPinholeTimeoutArgs[] = {
		{"RemoteHost", remoteHost},
		{"RemotePort", remotePort},
		{"Protocol", proto},
		{"InternalPort", intPort},
		{"InternalClient", intClient},
		{NULL, NULL}
	};
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	int ret;

	if(!intPort || !intClient || !proto || !remotePort || !remoteHost)
		return UPNPCOMMAND_INVALID_ARGS;

	buffer = simpleUPnPcommand(controlURL, servicetype,
	                           "GetOutboundPinholeTimeout", GetOutboundPinholeTimeoutArgs, &bufsize);
	if(!buffer)
		return UPNPCOMMAND_HTTP_ERROR;
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal)
	{
		if(sscanf(resVal, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}
	else
	{
		const char * p = GetValueFromNameValueList(&pdata, "OutboundPinholeTimeout");
		if(p)
			*opTimeout = my_atoui(p);
		ret = UPNPCOMMAND_SUCCESS;
	}
	ClearNameValueList(&pdata);
	return ret;
}

MINIUPNP_LIBSPEC int
UPNP_AddPinhole(const char * controlURL, const char * servicetype,
                    const char * remoteHost,
                    const char * remotePort,
                    const char * intClient,
                    const char * intPort,
                    const char * proto,
                    const char * leaseTime,
                    char * uniqueID)
{
	struct UPNParg AddPinholeArgs[] = {
		{"RemoteHost", ""},
		{"RemotePort", remotePort},
		{"Protocol", proto},
		{"InternalPort", intPort},
		{"InternalClient", ""},
		{"LeaseTime", leaseTime},
		{NULL, NULL}
	};
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	char * p;
	int ret;

	if(!intPort || !intClient || !proto || !remoteHost || !remotePort || !leaseTime)
		return UPNPCOMMAND_INVALID_ARGS;

	/* RemoteHost and InternalClient can be wilcarded
	 * accept both the empty string and "empty" as wildcard */
	if(strncmp(remoteHost, "empty", 5) != 0)
		AddPinholeArgs[0].val = remoteHost;
	if(strncmp(intClient, "empty", 5) != 0)
		AddPinholeArgs[4].val = intClient;
	buffer = simpleUPnPcommand(controlURL, servicetype,
	                           "AddPinhole", AddPinholeArgs, &bufsize);
	if(!buffer)
		return UPNPCOMMAND_HTTP_ERROR;
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);
	p = GetValueFromNameValueList(&pdata, "UniqueID");
	if(p)
	{
		strncpy(uniqueID, p, 8);
		uniqueID[7] = '\0';
	}
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal)
	{
		/*printf("AddPortMapping errorCode = '%s'\n", resVal);*/
		if(sscanf(resVal, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}
	else
	{
		ret = UPNPCOMMAND_SUCCESS;
	}
	ClearNameValueList(&pdata);
	return ret;
}

MINIUPNP_LIBSPEC int
UPNP_UpdatePinhole(const char * controlURL, const char * servicetype,
                    const char * uniqueID,
                    const char * leaseTime)
{
	struct UPNParg UpdatePinholeArgs[] = {
		{"UniqueID", uniqueID},
		{"NewLeaseTime", leaseTime},
		{NULL, NULL}
	};
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	int ret;

	if(!uniqueID || !leaseTime)
		return UPNPCOMMAND_INVALID_ARGS;

	buffer = simpleUPnPcommand(controlURL, servicetype,
	                           "UpdatePinhole", UpdatePinholeArgs, &bufsize);
	if(!buffer)
		return UPNPCOMMAND_HTTP_ERROR;
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal)
	{
		/*printf("AddPortMapping errorCode = '%s'\n", resVal); */
		if(sscanf(resVal, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}
	else
	{
		ret = UPNPCOMMAND_SUCCESS;
	}
	ClearNameValueList(&pdata);
	return ret;
}

MINIUPNP_LIBSPEC int
UPNP_DeletePinhole(const char * controlURL, const char * servicetype, const char * uniqueID)
{
	/*struct NameValueParserData pdata;*/
	struct UPNParg DeletePinholeArgs[] = {
		{"UniqueID", uniqueID},
		{NULL, NULL}
	};
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	int ret;

	if(!uniqueID)
		return UPNPCOMMAND_INVALID_ARGS;

	buffer = simpleUPnPcommand(controlURL, servicetype,
	                           "DeletePinhole", DeletePinholeArgs, &bufsize);
	if(!buffer)
		return UPNPCOMMAND_HTTP_ERROR;
	/*DisplayNameValueList(buffer, bufsize);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal)
	{
		if(sscanf(resVal, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}
	else
	{
		ret = UPNPCOMMAND_SUCCESS;
	}
	ClearNameValueList(&pdata);
	return ret;
}

MINIUPNP_LIBSPEC int
UPNP_CheckPinholeWorking(const char * controlURL, const char * servicetype,
                                 const char * uniqueID, int * isWorking)
{
	struct NameValueParserData pdata;
	struct UPNParg CheckPinholeWorkingArgs[] = {
		{"UniqueID", uniqueID},
		{NULL, NULL}
	};
	char * buffer;
	int bufsize;
	char * p;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;

	if(!uniqueID)
		return UPNPCOMMAND_INVALID_ARGS;

	buffer = simpleUPnPcommand(controlURL, servicetype,
	                           "CheckPinholeWorking", CheckPinholeWorkingArgs, &bufsize);
	if(!buffer)
	{
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);

	p = GetValueFromNameValueList(&pdata, "IsWorking");
	if(p)
	{
		*isWorking=my_atoui(p);
		ret = UPNPCOMMAND_SUCCESS;
	}
	else
		*isWorking = 0;

	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p)
	{
		if(sscanf(p, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}

	ClearNameValueList(&pdata);
	return ret;
}

MINIUPNP_LIBSPEC int
UPNP_GetPinholePackets(const char * controlURL, const char * servicetype,
                                 const char * uniqueID, int * packets)
{
	struct NameValueParserData pdata;
	struct UPNParg GetPinholePacketsArgs[] = {
		{"UniqueID", uniqueID},
		{NULL, NULL}
	};
	char * buffer;
	int bufsize;
	char * p;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;

	if(!uniqueID)
		return UPNPCOMMAND_INVALID_ARGS;

	buffer = simpleUPnPcommand(controlURL, servicetype,
	                           "GetPinholePackets", GetPinholePacketsArgs, &bufsize);
	if(!buffer)
		return UPNPCOMMAND_HTTP_ERROR;
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer);

	p = GetValueFromNameValueList(&pdata, "PinholePackets");
	if(p)
	{
		*packets=my_atoui(p);
		ret = UPNPCOMMAND_SUCCESS;
	}

	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p)
	{
		if(sscanf(p, "%d", &ret) != 1)
			ret = UPNPCOMMAND_UNKNOWN_ERROR;
	}

	ClearNameValueList(&pdata);
	return ret;
}
