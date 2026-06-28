/* $Id: igd_desc_parse.c,v 1.17 2015/09/15 13:30:04 nanard Exp $ */
/* Project : miniupnp
 * http://miniupnp.free.fr/
 * Author : Thomas Bernard
 * Copyright (c) 2005-2015 Thomas Bernard
 * This software is subject to the conditions detailed in the
 * LICENCE file provided in this distribution. */

#include "igd_desc_parse.h"
#include <stdio.h>
#include <string.h>

/* Start element handler :
 * update nesting level counter and copy element name */
void IGDstartelt(void * d, const char * name, int l)
{
	struct IGDdatas * datas = (struct IGDdatas *)d;
	if(l >= MINIUPNPC_URL_MAXSIZE)
		l = MINIUPNPC_URL_MAXSIZE-1;
	memcpy(datas->cureltname, name, l);
	datas->cureltname[l] = '\0';
	datas->level++;
	if( (l==7) && !memcmp(name, "service", l) ) {
		datas->tmp.controlurl[0] = '\0';
		datas->tmp.eventsuburl[0] = '\0';
		datas->tmp.scpdurl[0] = '\0';
		datas->tmp.servicetype[0] = '\0';
	}
}

#define COMPARE(str, cstr) (0==memcmp(str, cstr, sizeof(cstr) - 1))

/* End element handler :
 * update nesting level counter and update parser state if
 * service element is parsed */
void IGDendelt(void * d, const char * name, int l)
{
	struct IGDdatas * datas = (struct IGDdatas *)d;
	datas->level--;
	/*printf("endelt %2d %.*s\n", datas->level, l, name);*/
	if( (l==7) && !memcmp(name, "service", l) )
	{
		if(COMPARE(datas->tmp.servicetype,
		           "urn:schemas-upnp-org:service:WANCommonInterfaceConfig:")) {
			memcpy(&datas->CIF, &datas->tmp, sizeof(struct IGDdatas_service));
		} else if(COMPARE(datas->tmp.servicetype,
			                "urn:schemas-upnp-org:service:WANIPv6FirewallControl:")) {
			memcpy(&datas->IPv6FC, &datas->tmp, sizeof(struct IGDdatas_service));
		} else if(COMPARE(datas->tmp.servicetype,
		                  "urn:schemas-upnp-org:service:WANIPConnection:")
		         || COMPARE(datas->tmp.servicetype,
		                    "urn:schemas-upnp-org:service:WANPPPConnection:") ) {
			if(datas->first.servicetype[0] == '\0') {
				memcpy(&datas->first, &datas->tmp, sizeof(struct IGDdatas_service));
			} else {
				memcpy(&datas->second, &datas->tmp, sizeof(struct IGDdatas_service));
			}
		}
	}
}

/* Data handler :
 * copy data depending on the current element name and state */
void IGDdata(void * d, const char * data, int l)
{
	struct IGDdatas * datas = (struct IGDdatas *)d;
	char * dstmember = 0;
	/*printf("%2d %s : %.*s\n",
           datas->level, datas->cureltname, l, data);	*/
	if( !strcmp(datas->cureltname, "URLBase") )
		dstmember = datas->urlbase;
	else if( !strcmp(datas->cureltname, "presentationURL") )
		dstmember = datas->presentationurl;
	else if( !strcmp(datas->cureltname, "serviceType") )
		dstmember = datas->tmp.servicetype;
	else if( !strcmp(datas->cureltname, "controlURL") )
		dstmember = datas->tmp.controlurl;
	else if( !strcmp(datas->cureltname, "eventSubURL") )
		dstmember = datas->tmp.eventsuburl;
	else if( !strcmp(datas->cureltname, "SCPDURL") )
		dstmember = datas->tmp.scpdurl;
/*	else if( !strcmp(datas->cureltname, "deviceType") )
		dstmember = datas->devicetype_tmp;*/
	if(dstmember)
	{
		if(l>=MINIUPNPC_URL_MAXSIZE)
			l = MINIUPNPC_URL_MAXSIZE-1;
		memcpy(dstmember, data, l);
		dstmember[l] = '\0';
	}
}

#ifdef DEBUG
void printIGD(struct IGDdatas * d)
{
	printf("urlbase = '%s'\n", d->urlbase);
	printf("WAN Device (Common interface config) :\n");
	/*printf(" deviceType = '%s'\n", d->CIF.devicetype);*/
	printf(" serviceType = '%s'\n", d->CIF.servicetype);
	printf(" controlURL = '%s'\n", d->CIF.controlurl);
	printf(" eventSubURL = '%s'\n", d->CIF.eventsuburl);
	printf(" SCPDURL = '%s'\n", d->CIF.scpdurl);
	printf("primary WAN Connection Device (IP or PPP Connection):\n");
	/*printf(" deviceType = '%s'\n", d->first.devicetype);*/
	printf(" servicetype = '%s'\n", d->first.servicetype);
	printf(" controlURL = '%s'\n", d->first.controlurl);
	printf(" eventSubURL = '%s'\n", d->first.eventsuburl);
	printf(" SCPDURL = '%s'\n", d->first.scpdurl);
	printf("secondary WAN Connection Device (IP or PPP Connection):\n");
	/*printf(" deviceType = '%s'\n", d->second.devicetype);*/
	printf(" servicetype = '%s'\n", d->second.servicetype);
	printf(" controlURL = '%s'\n", d->second.controlurl);
	printf(" eventSubURL = '%s'\n", d->second.eventsuburl);
	printf(" SCPDURL = '%s'\n", d->second.scpdurl);
	printf("WAN IPv6 Firewall Control :\n");
	/*printf(" deviceType = '%s'\n", d->IPv6FC.devicetype);*/
	printf(" servicetype = '%s'\n", d->IPv6FC.servicetype);
	printf(" controlURL = '%s'\n", d->IPv6FC.controlurl);
	printf(" eventSubURL = '%s'\n", d->IPv6FC.eventsuburl);
	printf(" SCPDURL = '%s'\n", d->IPv6FC.scpdurl);
}
#endif /* DEBUG */

