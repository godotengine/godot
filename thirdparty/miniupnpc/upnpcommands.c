/* $Id: upnpcommands.c,v 1.49 2018/03/13 23:34:47 nanard Exp $ */
/* vim: tabstop=4 shiftwidth=4 noexpandtab
 * Project : miniupnp
 * Author : Thomas Bernard
 * Copyright (c) 2005-2018 Thomas Bernard
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
	if(!(buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                                "GetTotalBytesSent", 0, &bufsize))) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	/*DisplayNameValueList(buffer, bufsize);*/
	free(buffer); buffer = NULL;
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
	if(!(buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                                "GetTotalBytesReceived", 0, &bufsize))) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	/*DisplayNameValueList(buffer, bufsize);*/
	free(buffer); buffer = NULL;
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
	if(!(buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                                "GetTotalPacketsSent", 0, &bufsize))) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	/*DisplayNameValueList(buffer, bufsize);*/
	free(buffer); buffer = NULL;
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
	if(!(buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                                "GetTotalPacketsReceived", 0, &bufsize))) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	/*DisplayNameValueList(buffer, bufsize);*/
	free(buffer); buffer = NULL;
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

	if(!(buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                                "GetStatusInfo", 0, &bufsize))) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	/*DisplayNameValueList(buffer, bufsize);*/
	free(buffer); buffer = NULL;
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
		if(up)
			sscanf(up,"%u",uptime);
		else
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
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(p, "%d", &ret);
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

	if(!(buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                                "GetConnectionTypeInfo", 0, &bufsize))) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;
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
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(p, "%d", &ret);
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
	if(!(buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                                "GetCommonLinkProperties", 0, &bufsize))) {
	                              /*"GetLinkLayerMaxBitRates", 0, &bufsize);*/
		return UPNPCOMMAND_HTTP_ERROR;
	}
	/*DisplayNameValueList(buffer, bufsize);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;
	/*down = GetValueFromNameValueList(&pdata, "NewDownstreamMaxBitRate");*/
	/*up = GetValueFromNameValueList(&pdata, "NewUpstreamMaxBitRate");*/
	down = GetValueFromNameValueList(&pdata, "NewLayer1DownstreamMaxBitRate");
	up = GetValueFromNameValueList(&pdata, "NewLayer1UpstreamMaxBitRate");
	/*GetValueFromNameValueList(&pdata, "NewWANAccessType");*/
	/*GetValueFromNameValueList(&pdata, "NewPhysicalLinkStatus");*/
	if(down && up)
		ret = UPNPCOMMAND_SUCCESS;

	if(bitrateDown) {
		if(down)
			sscanf(down,"%u",bitrateDown);
		else
			*bitrateDown = 0;
	}

	if(bitrateUp) {
		if(up)
			sscanf(up,"%u",bitrateUp);
		else
			*bitrateUp = 0;
	}
	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p) {
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(p, "%d", &ret);
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

	if(!(buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                                "GetExternalIPAddress", 0, &bufsize))) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	/*DisplayNameValueList(buffer, bufsize);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;
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
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(p, "%d", &ret);
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
	struct UPNParg * AddPortMappingArgs;
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	int ret;

	if(!inPort || !inClient || !proto || !extPort)
		return UPNPCOMMAND_INVALID_ARGS;

	AddPortMappingArgs = calloc(9, sizeof(struct UPNParg));
	if(AddPortMappingArgs == NULL)
		return UPNPCOMMAND_MEM_ALLOC_ERROR;
	AddPortMappingArgs[0].elt = "NewRemoteHost";
	AddPortMappingArgs[0].val = remoteHost;
	AddPortMappingArgs[1].elt = "NewExternalPort";
	AddPortMappingArgs[1].val = extPort;
	AddPortMappingArgs[2].elt = "NewProtocol";
	AddPortMappingArgs[2].val = proto;
	AddPortMappingArgs[3].elt = "NewInternalPort";
	AddPortMappingArgs[3].val = inPort;
	AddPortMappingArgs[4].elt = "NewInternalClient";
	AddPortMappingArgs[4].val = inClient;
	AddPortMappingArgs[5].elt = "NewEnabled";
	AddPortMappingArgs[5].val = "1";
	AddPortMappingArgs[6].elt = "NewPortMappingDescription";
	AddPortMappingArgs[6].val = desc?desc:"libminiupnpc";
	AddPortMappingArgs[7].elt = "NewLeaseDuration";
	AddPortMappingArgs[7].val = leaseDuration?leaseDuration:"0";
	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                           "AddPortMapping", AddPortMappingArgs,
	                           &bufsize);
	free(AddPortMappingArgs);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	/*DisplayNameValueList(buffer, bufsize);*/
	/*buffer[bufsize] = '\0';*/
	/*puts(buffer);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal) {
		/*printf("AddPortMapping errorCode = '%s'\n", resVal); */
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(resVal, "%d", &ret);
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
	struct UPNParg * AddPortMappingArgs;
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	int ret;

	if(!inPort || !inClient || !proto || !extPort)
		return UPNPCOMMAND_INVALID_ARGS;

	AddPortMappingArgs = calloc(9, sizeof(struct UPNParg));
	if(AddPortMappingArgs == NULL)
		return UPNPCOMMAND_MEM_ALLOC_ERROR;
	AddPortMappingArgs[0].elt = "NewRemoteHost";
	AddPortMappingArgs[0].val = remoteHost;
	AddPortMappingArgs[1].elt = "NewExternalPort";
	AddPortMappingArgs[1].val = extPort;
	AddPortMappingArgs[2].elt = "NewProtocol";
	AddPortMappingArgs[2].val = proto;
	AddPortMappingArgs[3].elt = "NewInternalPort";
	AddPortMappingArgs[3].val = inPort;
	AddPortMappingArgs[4].elt = "NewInternalClient";
	AddPortMappingArgs[4].val = inClient;
	AddPortMappingArgs[5].elt = "NewEnabled";
	AddPortMappingArgs[5].val = "1";
	AddPortMappingArgs[6].elt = "NewPortMappingDescription";
	AddPortMappingArgs[6].val = desc?desc:"libminiupnpc";
	AddPortMappingArgs[7].elt = "NewLeaseDuration";
	AddPortMappingArgs[7].val = leaseDuration?leaseDuration:"0";
	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                           "AddAnyPortMapping", AddPortMappingArgs,
	                           &bufsize);
	free(AddPortMappingArgs);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal) {
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(resVal, "%d", &ret);
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
	struct UPNParg * DeletePortMappingArgs;
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	int ret;

	if(!extPort || !proto)
		return UPNPCOMMAND_INVALID_ARGS;

	DeletePortMappingArgs = calloc(4, sizeof(struct UPNParg));
	if(DeletePortMappingArgs == NULL)
		return UPNPCOMMAND_MEM_ALLOC_ERROR;
	DeletePortMappingArgs[0].elt = "NewRemoteHost";
	DeletePortMappingArgs[0].val = remoteHost;
	DeletePortMappingArgs[1].elt = "NewExternalPort";
	DeletePortMappingArgs[1].val = extPort;
	DeletePortMappingArgs[2].elt = "NewProtocol";
	DeletePortMappingArgs[2].val = proto;
	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                          "DeletePortMapping",
	                          DeletePortMappingArgs, &bufsize);
	free(DeletePortMappingArgs);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	/*DisplayNameValueList(buffer, bufsize);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal) {
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(resVal, "%d", &ret);
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
	struct UPNParg * DeletePortMappingArgs;
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	int ret;

	if(!extPortStart || !extPortEnd || !proto || !manage)
		return UPNPCOMMAND_INVALID_ARGS;

	DeletePortMappingArgs = calloc(5, sizeof(struct UPNParg));
	if(DeletePortMappingArgs == NULL)
		return UPNPCOMMAND_MEM_ALLOC_ERROR;
	DeletePortMappingArgs[0].elt = "NewStartPort";
	DeletePortMappingArgs[0].val = extPortStart;
	DeletePortMappingArgs[1].elt = "NewEndPort";
	DeletePortMappingArgs[1].val = extPortEnd;
	DeletePortMappingArgs[2].elt = "NewProtocol";
	DeletePortMappingArgs[2].val = proto;
	DeletePortMappingArgs[3].elt = "NewManage";
	DeletePortMappingArgs[3].val = manage;

	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                           "DeletePortMappingRange",
	                           DeletePortMappingArgs, &bufsize);
	free(DeletePortMappingArgs);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal) {
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(resVal, "%d", &ret);
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
	struct NameValueParserData pdata;
	struct UPNParg * GetPortMappingArgs;
	char * buffer;
	int bufsize;
	char * p;
	int r = UPNPCOMMAND_UNKNOWN_ERROR;
	if(!index)
		return UPNPCOMMAND_INVALID_ARGS;
	intClient[0] = '\0';
	intPort[0] = '\0';
	GetPortMappingArgs = calloc(2, sizeof(struct UPNParg));
	if(GetPortMappingArgs == NULL)
		return UPNPCOMMAND_MEM_ALLOC_ERROR;
	GetPortMappingArgs[0].elt = "NewPortMappingIndex";
	GetPortMappingArgs[0].val = index;
	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                           "GetGenericPortMappingEntry",
	                           GetPortMappingArgs, &bufsize);
	free(GetPortMappingArgs);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;

	p = GetValueFromNameValueList(&pdata, "NewRemoteHost");
	if(p && rHost)
	{
		strncpy(rHost, p, 64);
		rHost[63] = '\0';
	}
	p = GetValueFromNameValueList(&pdata, "NewExternalPort");
	if(p && extPort)
	{
		strncpy(extPort, p, 6);
		extPort[5] = '\0';
		r = UPNPCOMMAND_SUCCESS;
	}
	p = GetValueFromNameValueList(&pdata, "NewProtocol");
	if(p && protocol)
	{
		strncpy(protocol, p, 4);
		protocol[3] = '\0';
	}
	p = GetValueFromNameValueList(&pdata, "NewInternalClient");
	if(p)
	{
		strncpy(intClient, p, 16);
		intClient[15] = '\0';
		r = 0;
	}
	p = GetValueFromNameValueList(&pdata, "NewInternalPort");
	if(p)
	{
		strncpy(intPort, p, 6);
		intPort[5] = '\0';
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
		strncpy(desc, p, 80);
		desc[79] = '\0';
	}
	p = GetValueFromNameValueList(&pdata, "NewLeaseDuration");
	if(p && duration)
	{
		strncpy(duration, p, 16);
		duration[15] = '\0';
	}
	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p) {
		r = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(p, "%d", &r);
	}
	ClearNameValueList(&pdata);
	return r;
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
 	if(!(buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                                "GetPortMappingNumberOfEntries", 0,
	                                &bufsize))) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
#ifdef DEBUG
	DisplayNameValueList(buffer, bufsize);
#endif
 	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;

 	p = GetValueFromNameValueList(&pdata, "NewPortMappingNumberOfEntries");
 	if(numEntries && p) {
		*numEntries = 0;
 		sscanf(p, "%u", numEntries);
		ret = UPNPCOMMAND_SUCCESS;
 	}

	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p) {
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(p, "%d", &ret);
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
	struct NameValueParserData pdata;
	struct UPNParg * GetPortMappingArgs;
	char * buffer;
	int bufsize;
	char * p;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;

	if(!intPort || !intClient || !extPort || !proto)
		return UPNPCOMMAND_INVALID_ARGS;

	GetPortMappingArgs = calloc(4, sizeof(struct UPNParg));
	if(GetPortMappingArgs == NULL)
		return UPNPCOMMAND_MEM_ALLOC_ERROR;
	GetPortMappingArgs[0].elt = "NewRemoteHost";
	GetPortMappingArgs[0].val = remoteHost;
	GetPortMappingArgs[1].elt = "NewExternalPort";
	GetPortMappingArgs[1].val = extPort;
	GetPortMappingArgs[2].elt = "NewProtocol";
	GetPortMappingArgs[2].val = proto;
	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                           "GetSpecificPortMappingEntry",
	                           GetPortMappingArgs, &bufsize);
	free(GetPortMappingArgs);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	/*DisplayNameValueList(buffer, bufsize);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;

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
		strncpy(desc, p, 80);
		desc[79] = '\0';
	}

	p = GetValueFromNameValueList(&pdata, "NewLeaseDuration");
	if(p && leaseDuration)
	{
		strncpy(leaseDuration, p, 16);
		leaseDuration[15] = '\0';
	}

	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p) {
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(p, "%d", &ret);
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
	struct UPNParg * GetListOfPortMappingsArgs;
	const char * p;
	char * buffer;
	int bufsize;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;

	if(!startPort || !endPort || !protocol)
		return UPNPCOMMAND_INVALID_ARGS;

	GetListOfPortMappingsArgs = calloc(6, sizeof(struct UPNParg));
	if(GetListOfPortMappingsArgs == NULL)
		return UPNPCOMMAND_MEM_ALLOC_ERROR;
	GetListOfPortMappingsArgs[0].elt = "NewStartPort";
	GetListOfPortMappingsArgs[0].val = startPort;
	GetListOfPortMappingsArgs[1].elt = "NewEndPort";
	GetListOfPortMappingsArgs[1].val = endPort;
	GetListOfPortMappingsArgs[2].elt = "NewProtocol";
	GetListOfPortMappingsArgs[2].val = protocol;
	GetListOfPortMappingsArgs[3].elt = "NewManage";
	GetListOfPortMappingsArgs[3].val = "1";
	GetListOfPortMappingsArgs[4].elt = "NewNumberOfPorts";
	GetListOfPortMappingsArgs[4].val = numberOfPorts?numberOfPorts:"1000";

	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                           "GetListOfPortMappings",
	                           GetListOfPortMappingsArgs, &bufsize);
	free(GetListOfPortMappingsArgs);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}

	/*DisplayNameValueList(buffer, bufsize);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;

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
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(p, "%d", &ret);
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

	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                           "GetFirewallStatus", 0, &bufsize);
	if(!buffer) {
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;
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
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(p, "%d", &ret);
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
	struct UPNParg * GetOutboundPinholeTimeoutArgs;
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	char * p;
	int ret;

	if(!intPort || !intClient || !proto || !remotePort || !remoteHost)
		return UPNPCOMMAND_INVALID_ARGS;

	GetOutboundPinholeTimeoutArgs = calloc(6, sizeof(struct UPNParg));
	if(GetOutboundPinholeTimeoutArgs == NULL)
		return UPNPCOMMAND_MEM_ALLOC_ERROR;
	GetOutboundPinholeTimeoutArgs[0].elt = "RemoteHost";
	GetOutboundPinholeTimeoutArgs[0].val = remoteHost;
	GetOutboundPinholeTimeoutArgs[1].elt = "RemotePort";
	GetOutboundPinholeTimeoutArgs[1].val = remotePort;
	GetOutboundPinholeTimeoutArgs[2].elt = "Protocol";
	GetOutboundPinholeTimeoutArgs[2].val = proto;
	GetOutboundPinholeTimeoutArgs[3].elt = "InternalPort";
	GetOutboundPinholeTimeoutArgs[3].val = intPort;
	GetOutboundPinholeTimeoutArgs[4].elt = "InternalClient";
	GetOutboundPinholeTimeoutArgs[4].val = intClient;
	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                           "GetOutboundPinholeTimeout", GetOutboundPinholeTimeoutArgs, &bufsize);
	free(GetOutboundPinholeTimeoutArgs);
	if(!buffer)
		return UPNPCOMMAND_HTTP_ERROR;
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal)
	{
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(resVal, "%d", &ret);
	}
	else
	{
		ret = UPNPCOMMAND_SUCCESS;
		p = GetValueFromNameValueList(&pdata, "OutboundPinholeTimeout");
		if(p)
			*opTimeout = my_atoui(p);
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
	struct UPNParg * AddPinholeArgs;
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	char * p;
	int ret;

	if(!intPort || !intClient || !proto || !remoteHost || !remotePort || !leaseTime)
		return UPNPCOMMAND_INVALID_ARGS;

	AddPinholeArgs = calloc(7, sizeof(struct UPNParg));
	if(AddPinholeArgs == NULL)
		return UPNPCOMMAND_MEM_ALLOC_ERROR;
	/* RemoteHost can be wilcarded */
	if(strncmp(remoteHost, "empty", 5)==0)
	{
		AddPinholeArgs[0].elt = "RemoteHost";
		AddPinholeArgs[0].val = "";
	}
	else
	{
		AddPinholeArgs[0].elt = "RemoteHost";
		AddPinholeArgs[0].val = remoteHost;
	}
	AddPinholeArgs[1].elt = "RemotePort";
	AddPinholeArgs[1].val = remotePort;
	AddPinholeArgs[2].elt = "Protocol";
	AddPinholeArgs[2].val = proto;
	AddPinholeArgs[3].elt = "InternalPort";
	AddPinholeArgs[3].val = intPort;
	if(strncmp(intClient, "empty", 5)==0)
	{
		AddPinholeArgs[4].elt = "InternalClient";
		AddPinholeArgs[4].val = "";
	}
	else
	{
		AddPinholeArgs[4].elt = "InternalClient";
		AddPinholeArgs[4].val = intClient;
	}
	AddPinholeArgs[5].elt = "LeaseTime";
	AddPinholeArgs[5].val = leaseTime;
	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                           "AddPinhole", AddPinholeArgs, &bufsize);
	free(AddPinholeArgs);
	if(!buffer)
		return UPNPCOMMAND_HTTP_ERROR;
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;
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
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(resVal, "%d", &ret);
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
	struct UPNParg * UpdatePinholeArgs;
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	int ret;

	if(!uniqueID || !leaseTime)
		return UPNPCOMMAND_INVALID_ARGS;

	UpdatePinholeArgs = calloc(3, sizeof(struct UPNParg));
	if(UpdatePinholeArgs == NULL)
		return UPNPCOMMAND_MEM_ALLOC_ERROR;
	UpdatePinholeArgs[0].elt = "UniqueID";
	UpdatePinholeArgs[0].val = uniqueID;
	UpdatePinholeArgs[1].elt = "NewLeaseTime";
	UpdatePinholeArgs[1].val = leaseTime;
	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                           "UpdatePinhole", UpdatePinholeArgs, &bufsize);
	free(UpdatePinholeArgs);
	if(!buffer)
		return UPNPCOMMAND_HTTP_ERROR;
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal)
	{
		/*printf("AddPortMapping errorCode = '%s'\n", resVal); */
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(resVal, "%d", &ret);
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
	struct UPNParg * DeletePinholeArgs;
	char * buffer;
	int bufsize;
	struct NameValueParserData pdata;
	const char * resVal;
	int ret;

	if(!uniqueID)
		return UPNPCOMMAND_INVALID_ARGS;

	DeletePinholeArgs = calloc(2, sizeof(struct UPNParg));
	if(DeletePinholeArgs == NULL)
		return UPNPCOMMAND_MEM_ALLOC_ERROR;
	DeletePinholeArgs[0].elt = "UniqueID";
	DeletePinholeArgs[0].val = uniqueID;
	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                           "DeletePinhole", DeletePinholeArgs, &bufsize);
	free(DeletePinholeArgs);
	if(!buffer)
		return UPNPCOMMAND_HTTP_ERROR;
	/*DisplayNameValueList(buffer, bufsize);*/
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;
	resVal = GetValueFromNameValueList(&pdata, "errorCode");
	if(resVal)
	{
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(resVal, "%d", &ret);
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
	struct UPNParg * CheckPinholeWorkingArgs;
	char * buffer;
	int bufsize;
	char * p;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;

	if(!uniqueID)
		return UPNPCOMMAND_INVALID_ARGS;

	CheckPinholeWorkingArgs = calloc(4, sizeof(struct UPNParg));
	if(CheckPinholeWorkingArgs == NULL)
		return UPNPCOMMAND_MEM_ALLOC_ERROR;
	CheckPinholeWorkingArgs[0].elt = "UniqueID";
	CheckPinholeWorkingArgs[0].val = uniqueID;
	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                           "CheckPinholeWorking", CheckPinholeWorkingArgs, &bufsize);
	free(CheckPinholeWorkingArgs);
	if(!buffer)
	{
		return UPNPCOMMAND_HTTP_ERROR;
	}
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;

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
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(p, "%d", &ret);
	}

	ClearNameValueList(&pdata);
	return ret;
}

MINIUPNP_LIBSPEC int
UPNP_GetPinholePackets(const char * controlURL, const char * servicetype,
                                 const char * uniqueID, int * packets)
{
	struct NameValueParserData pdata;
	struct UPNParg * GetPinholePacketsArgs;
	char * buffer;
	int bufsize;
	char * p;
	int ret = UPNPCOMMAND_UNKNOWN_ERROR;

	if(!uniqueID)
		return UPNPCOMMAND_INVALID_ARGS;

	GetPinholePacketsArgs = calloc(4, sizeof(struct UPNParg));
	if(GetPinholePacketsArgs == NULL)
		return UPNPCOMMAND_MEM_ALLOC_ERROR;
	GetPinholePacketsArgs[0].elt = "UniqueID";
	GetPinholePacketsArgs[0].val = uniqueID;
	buffer = simpleUPnPcommand(-1, controlURL, servicetype,
	                           "GetPinholePackets", GetPinholePacketsArgs, &bufsize);
	free(GetPinholePacketsArgs);
	if(!buffer)
		return UPNPCOMMAND_HTTP_ERROR;
	ParseNameValue(buffer, bufsize, &pdata);
	free(buffer); buffer = NULL;

	p = GetValueFromNameValueList(&pdata, "PinholePackets");
	if(p)
	{
		*packets=my_atoui(p);
		ret = UPNPCOMMAND_SUCCESS;
	}

	p = GetValueFromNameValueList(&pdata, "errorCode");
	if(p)
	{
		ret = UPNPCOMMAND_UNKNOWN_ERROR;
		sscanf(p, "%d", &ret);
	}

	ClearNameValueList(&pdata);
	return ret;
}


