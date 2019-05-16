/* $Id: upnpc.c,v 1.119 2018/03/13 23:34:46 nanard Exp $ */
/* Project : miniupnp
 * Author : Thomas Bernard
 * Copyright (c) 2005-2018 Thomas Bernard
 * This software is subject to the conditions detailed in the
 * LICENCE file provided in this distribution. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <winsock2.h>
#define snprintf _snprintf
#else
/* for IPPROTO_TCP / IPPROTO_UDP */
#include <netinet/in.h>
#endif
#include <ctype.h>
#include "miniwget.h"
#include "miniupnpc.h"
#include "upnpcommands.h"
#include "portlistingparse.h"
#include "upnperrors.h"
#include "miniupnpcstrings.h"

/* protofix() checks if protocol is "UDP" or "TCP"
 * returns NULL if not */
const char * protofix(const char * proto)
{
	static const char proto_tcp[4] = { 'T', 'C', 'P', 0};
	static const char proto_udp[4] = { 'U', 'D', 'P', 0};
	int i, b;
	for(i=0, b=1; i<4; i++)
		b = b && (   (proto[i] == proto_tcp[i])
		          || (proto[i] == (proto_tcp[i] | 32)) );
	if(b)
		return proto_tcp;
	for(i=0, b=1; i<4; i++)
		b = b && (   (proto[i] == proto_udp[i])
		          || (proto[i] == (proto_udp[i] | 32)) );
	if(b)
		return proto_udp;
	return 0;
}

/* is_int() checks if parameter is an integer or not
 * 1 for integer
 * 0 for not an integer */
int is_int(char const* s)
{
	if(s == NULL)
		return 0;
	while(*s) {
		/* #define isdigit(c) ((c) >= '0' && (c) <= '9') */
		if(!isdigit(*s))
			return 0;
		s++;
	}
	return 1;
}

static void DisplayInfos(struct UPNPUrls * urls,
                         struct IGDdatas * data)
{
	char externalIPAddress[40];
	char connectionType[64];
	char status[64];
	char lastconnerr[64];
	unsigned int uptime = 0;
	unsigned int brUp, brDown;
	time_t timenow, timestarted;
	int r;
	if(UPNP_GetConnectionTypeInfo(urls->controlURL,
	                              data->first.servicetype,
	                              connectionType) != UPNPCOMMAND_SUCCESS)
		printf("GetConnectionTypeInfo failed.\n");
	else
		printf("Connection Type : %s\n", connectionType);
	if(UPNP_GetStatusInfo(urls->controlURL, data->first.servicetype,
	                      status, &uptime, lastconnerr) != UPNPCOMMAND_SUCCESS)
		printf("GetStatusInfo failed.\n");
	else
		printf("Status : %s, uptime=%us, LastConnectionError : %s\n",
		       status, uptime, lastconnerr);
	if(uptime > 0) {
		timenow = time(NULL);
		timestarted = timenow - uptime;
		printf("  Time started : %s", ctime(&timestarted));
	}
	if(UPNP_GetLinkLayerMaxBitRates(urls->controlURL_CIF, data->CIF.servicetype,
	                                &brDown, &brUp) != UPNPCOMMAND_SUCCESS) {
		printf("GetLinkLayerMaxBitRates failed.\n");
	} else {
		printf("MaxBitRateDown : %u bps", brDown);
		if(brDown >= 1000000) {
			printf(" (%u.%u Mbps)", brDown / 1000000, (brDown / 100000) % 10);
		} else if(brDown >= 1000) {
			printf(" (%u Kbps)", brDown / 1000);
		}
		printf("   MaxBitRateUp %u bps", brUp);
		if(brUp >= 1000000) {
			printf(" (%u.%u Mbps)", brUp / 1000000, (brUp / 100000) % 10);
		} else if(brUp >= 1000) {
			printf(" (%u Kbps)", brUp / 1000);
		}
		printf("\n");
	}
	r = UPNP_GetExternalIPAddress(urls->controlURL,
	                          data->first.servicetype,
							  externalIPAddress);
	if(r != UPNPCOMMAND_SUCCESS) {
		printf("GetExternalIPAddress failed. (errorcode=%d)\n", r);
	} else {
		printf("ExternalIPAddress = %s\n", externalIPAddress);
	}
}

static void GetConnectionStatus(struct UPNPUrls * urls,
                               struct IGDdatas * data)
{
	unsigned int bytessent, bytesreceived, packetsreceived, packetssent;
	DisplayInfos(urls, data);
	bytessent = UPNP_GetTotalBytesSent(urls->controlURL_CIF, data->CIF.servicetype);
	bytesreceived = UPNP_GetTotalBytesReceived(urls->controlURL_CIF, data->CIF.servicetype);
	packetssent = UPNP_GetTotalPacketsSent(urls->controlURL_CIF, data->CIF.servicetype);
	packetsreceived = UPNP_GetTotalPacketsReceived(urls->controlURL_CIF, data->CIF.servicetype);
	printf("Bytes:   Sent: %8u\tRecv: %8u\n", bytessent, bytesreceived);
	printf("Packets: Sent: %8u\tRecv: %8u\n", packetssent, packetsreceived);
}

static void ListRedirections(struct UPNPUrls * urls,
                             struct IGDdatas * data)
{
	int r;
	int i = 0;
	char index[6];
	char intClient[40];
	char intPort[6];
	char extPort[6];
	char protocol[4];
	char desc[80];
	char enabled[6];
	char rHost[64];
	char duration[16];
	/*unsigned int num=0;
	UPNP_GetPortMappingNumberOfEntries(urls->controlURL, data->servicetype, &num);
	printf("PortMappingNumberOfEntries : %u\n", num);*/
	printf(" i protocol exPort->inAddr:inPort description remoteHost leaseTime\n");
	do {
		snprintf(index, 6, "%d", i);
		rHost[0] = '\0'; enabled[0] = '\0';
		duration[0] = '\0'; desc[0] = '\0';
		extPort[0] = '\0'; intPort[0] = '\0'; intClient[0] = '\0';
		r = UPNP_GetGenericPortMappingEntry(urls->controlURL,
		                               data->first.servicetype,
		                               index,
		                               extPort, intClient, intPort,
									   protocol, desc, enabled,
									   rHost, duration);
		if(r==0)
		/*
			printf("%02d - %s %s->%s:%s\tenabled=%s leaseDuration=%s\n"
			       "     desc='%s' rHost='%s'\n",
			       i, protocol, extPort, intClient, intPort,
				   enabled, duration,
				   desc, rHost);
				   */
			printf("%2d %s %5s->%s:%-5s '%s' '%s' %s\n",
			       i, protocol, extPort, intClient, intPort,
			       desc, rHost, duration);
		else
			printf("GetGenericPortMappingEntry() returned %d (%s)\n",
			       r, strupnperror(r));
		i++;
	} while(r==0);
}

static void NewListRedirections(struct UPNPUrls * urls,
                                struct IGDdatas * data)
{
	int r;
	int i = 0;
	struct PortMappingParserData pdata;
	struct PortMapping * pm;

	memset(&pdata, 0, sizeof(struct PortMappingParserData));
	r = UPNP_GetListOfPortMappings(urls->controlURL,
                                   data->first.servicetype,
	                               "0",
	                               "65535",
	                               "TCP",
	                               "1000",
	                               &pdata);
	if(r == UPNPCOMMAND_SUCCESS)
	{
		printf(" i protocol exPort->inAddr:inPort description remoteHost leaseTime\n");
		for(pm = pdata.l_head; pm != NULL; pm = pm->l_next)
		{
			printf("%2d %s %5hu->%s:%-5hu '%s' '%s' %u\n",
			       i, pm->protocol, pm->externalPort, pm->internalClient,
			       pm->internalPort,
			       pm->description, pm->remoteHost,
			       (unsigned)pm->leaseTime);
			i++;
		}
		FreePortListing(&pdata);
	}
	else
	{
		printf("GetListOfPortMappings() returned %d (%s)\n",
		       r, strupnperror(r));
	}
	r = UPNP_GetListOfPortMappings(urls->controlURL,
                                   data->first.servicetype,
	                               "0",
	                               "65535",
	                               "UDP",
	                               "1000",
	                               &pdata);
	if(r == UPNPCOMMAND_SUCCESS)
	{
		for(pm = pdata.l_head; pm != NULL; pm = pm->l_next)
		{
			printf("%2d %s %5hu->%s:%-5hu '%s' '%s' %u\n",
			       i, pm->protocol, pm->externalPort, pm->internalClient,
			       pm->internalPort,
			       pm->description, pm->remoteHost,
			       (unsigned)pm->leaseTime);
			i++;
		}
		FreePortListing(&pdata);
	}
	else
	{
		printf("GetListOfPortMappings() returned %d (%s)\n",
		       r, strupnperror(r));
	}
}

/* Test function
 * 1 - get connection type
 * 2 - get extenal ip address
 * 3 - Add port mapping
 * 4 - get this port mapping from the IGD */
static int SetRedirectAndTest(struct UPNPUrls * urls,
			       struct IGDdatas * data,
			       const char * iaddr,
			       const char * iport,
			       const char * eport,
			       const char * proto,
			       const char * leaseDuration,
			       const char * description,
			       int addAny)
{
	char externalIPAddress[40];
	char intClient[40];
	char intPort[6];
	char reservedPort[6];
	char duration[16];
	int r;

	if(!iaddr || !iport || !eport || !proto)
	{
		fprintf(stderr, "Wrong arguments\n");
		return -1;
	}
	proto = protofix(proto);
	if(!proto)
	{
		fprintf(stderr, "invalid protocol\n");
		return -1;
	}

	r = UPNP_GetExternalIPAddress(urls->controlURL,
				      data->first.servicetype,
				      externalIPAddress);
	if(r!=UPNPCOMMAND_SUCCESS)
		printf("GetExternalIPAddress failed.\n");
	else
		printf("ExternalIPAddress = %s\n", externalIPAddress);

	if (addAny) {
		r = UPNP_AddAnyPortMapping(urls->controlURL, data->first.servicetype,
					   eport, iport, iaddr, description,
					   proto, 0, leaseDuration, reservedPort);
		if(r==UPNPCOMMAND_SUCCESS)
			eport = reservedPort;
		else
			printf("AddAnyPortMapping(%s, %s, %s) failed with code %d (%s)\n",
			       eport, iport, iaddr, r, strupnperror(r));
	} else {
		r = UPNP_AddPortMapping(urls->controlURL, data->first.servicetype,
					eport, iport, iaddr, description,
					proto, 0, leaseDuration);
		if(r!=UPNPCOMMAND_SUCCESS) {
			printf("AddPortMapping(%s, %s, %s) failed with code %d (%s)\n",
			       eport, iport, iaddr, r, strupnperror(r));
			return -2;
	}
	}

	r = UPNP_GetSpecificPortMappingEntry(urls->controlURL,
					     data->first.servicetype,
					     eport, proto, NULL/*remoteHost*/,
					     intClient, intPort, NULL/*desc*/,
					     NULL/*enabled*/, duration);
	if(r!=UPNPCOMMAND_SUCCESS) {
		printf("GetSpecificPortMappingEntry() failed with code %d (%s)\n",
		       r, strupnperror(r));
		return -2;
	} else {
		printf("InternalIP:Port = %s:%s\n", intClient, intPort);
		printf("external %s:%s %s is redirected to internal %s:%s (duration=%s)\n",
		       externalIPAddress, eport, proto, intClient, intPort, duration);
	}
	return 0;
}

static int
RemoveRedirect(struct UPNPUrls * urls,
               struct IGDdatas * data,
               const char * eport,
               const char * proto,
               const char * remoteHost)
{
	int r;
	if(!proto || !eport)
	{
		fprintf(stderr, "invalid arguments\n");
		return -1;
	}
	proto = protofix(proto);
	if(!proto)
	{
		fprintf(stderr, "protocol invalid\n");
		return -1;
	}
	r = UPNP_DeletePortMapping(urls->controlURL, data->first.servicetype, eport, proto, remoteHost);
	if(r!=UPNPCOMMAND_SUCCESS) {
		printf("UPNP_DeletePortMapping() failed with code : %d\n", r);
		return -2;
	}else {
		printf("UPNP_DeletePortMapping() returned : %d\n", r);
	}
	return 0;
}

static int
RemoveRedirectRange(struct UPNPUrls * urls,
		    struct IGDdatas * data,
		    const char * ePortStart, char const * ePortEnd,
		    const char * proto, const char * manage)
{
	int r;

	if (!manage)
		manage = "0";

	if(!proto || !ePortStart || !ePortEnd)
	{
		fprintf(stderr, "invalid arguments\n");
		return -1;
	}
	proto = protofix(proto);
	if(!proto)
	{
		fprintf(stderr, "protocol invalid\n");
		return -1;
	}
	r = UPNP_DeletePortMappingRange(urls->controlURL, data->first.servicetype, ePortStart, ePortEnd, proto, manage);
	if(r!=UPNPCOMMAND_SUCCESS) {
		printf("UPNP_DeletePortMappingRange() failed with code : %d\n", r);
		return -2;
	}else {
		printf("UPNP_DeletePortMappingRange() returned : %d\n", r);
	}
	return 0;
}

/* IGD:2, functions for service WANIPv6FirewallControl:1 */
static void GetFirewallStatus(struct UPNPUrls * urls, struct IGDdatas * data)
{
	unsigned int bytessent, bytesreceived, packetsreceived, packetssent;
	int firewallEnabled = 0, inboundPinholeAllowed = 0;

	UPNP_GetFirewallStatus(urls->controlURL_6FC, data->IPv6FC.servicetype, &firewallEnabled, &inboundPinholeAllowed);
	printf("FirewallEnabled: %d & Inbound Pinhole Allowed: %d\n", firewallEnabled, inboundPinholeAllowed);
	printf("GetFirewallStatus:\n   Firewall Enabled: %s\n   Inbound Pinhole Allowed: %s\n", (firewallEnabled)? "Yes":"No", (inboundPinholeAllowed)? "Yes":"No");

	bytessent = UPNP_GetTotalBytesSent(urls->controlURL_CIF, data->CIF.servicetype);
	bytesreceived = UPNP_GetTotalBytesReceived(urls->controlURL_CIF, data->CIF.servicetype);
	packetssent = UPNP_GetTotalPacketsSent(urls->controlURL_CIF, data->CIF.servicetype);
	packetsreceived = UPNP_GetTotalPacketsReceived(urls->controlURL_CIF, data->CIF.servicetype);
	printf("Bytes:   Sent: %8u\tRecv: %8u\n", bytessent, bytesreceived);
	printf("Packets: Sent: %8u\tRecv: %8u\n", packetssent, packetsreceived);
}

/* Test function
 * 1 - Add pinhole
 * 2 - Check if pinhole is working from the IGD side */
static void SetPinholeAndTest(struct UPNPUrls * urls, struct IGDdatas * data,
					const char * remoteaddr, const char * eport,
					const char * intaddr, const char * iport,
					const char * proto, const char * lease_time)
{
	char uniqueID[8];
	/*int isWorking = 0;*/
	int r;
	char proto_tmp[8];

	if(!intaddr || !remoteaddr || !iport || !eport || !proto || !lease_time)
	{
		fprintf(stderr, "Wrong arguments\n");
		return;
	}
	if(atoi(proto) == 0)
	{
		const char * protocol;
		protocol = protofix(proto);
		if(protocol && (strcmp("TCP", protocol) == 0))
		{
			snprintf(proto_tmp, sizeof(proto_tmp), "%d", IPPROTO_TCP);
			proto = proto_tmp;
		}
		else if(protocol && (strcmp("UDP", protocol) == 0))
		{
			snprintf(proto_tmp, sizeof(proto_tmp), "%d", IPPROTO_UDP);
			proto = proto_tmp;
		}
		else
		{
			fprintf(stderr, "invalid protocol\n");
			return;
		}
	}
	r = UPNP_AddPinhole(urls->controlURL_6FC, data->IPv6FC.servicetype, remoteaddr, eport, intaddr, iport, proto, lease_time, uniqueID);
	if(r!=UPNPCOMMAND_SUCCESS)
		printf("AddPinhole([%s]:%s -> [%s]:%s) failed with code %d (%s)\n",
		       remoteaddr, eport, intaddr, iport, r, strupnperror(r));
	else
	{
		printf("AddPinhole: ([%s]:%s -> [%s]:%s) / Pinhole ID = %s\n",
		       remoteaddr, eport, intaddr, iport, uniqueID);
		/*r = UPNP_CheckPinholeWorking(urls->controlURL_6FC, data->servicetype_6FC, uniqueID, &isWorking);
		if(r!=UPNPCOMMAND_SUCCESS)
			printf("CheckPinholeWorking() failed with code %d (%s)\n", r, strupnperror(r));
		printf("CheckPinholeWorking: Pinhole ID = %s / IsWorking = %s\n", uniqueID, (isWorking)? "Yes":"No");*/
	}
}

/* Test function
 * 1 - Check if pinhole is working from the IGD side
 * 2 - Update pinhole */
static void GetPinholeAndUpdate(struct UPNPUrls * urls, struct IGDdatas * data,
					const char * uniqueID, const char * lease_time)
{
	int isWorking = 0;
	int r;

	if(!uniqueID || !lease_time)
	{
		fprintf(stderr, "Wrong arguments\n");
		return;
	}
	r = UPNP_CheckPinholeWorking(urls->controlURL_6FC, data->IPv6FC.servicetype, uniqueID, &isWorking);
	printf("CheckPinholeWorking: Pinhole ID = %s / IsWorking = %s\n", uniqueID, (isWorking)? "Yes":"No");
	if(r!=UPNPCOMMAND_SUCCESS)
		printf("CheckPinholeWorking() failed with code %d (%s)\n", r, strupnperror(r));
	if(isWorking || r==709)
	{
		r = UPNP_UpdatePinhole(urls->controlURL_6FC, data->IPv6FC.servicetype, uniqueID, lease_time);
		printf("UpdatePinhole: Pinhole ID = %s with Lease Time: %s\n", uniqueID, lease_time);
		if(r!=UPNPCOMMAND_SUCCESS)
			printf("UpdatePinhole: ID (%s) failed with code %d (%s)\n", uniqueID, r, strupnperror(r));
	}
}

/* Test function
 * Get pinhole timeout
 */
static void GetPinholeOutboundTimeout(struct UPNPUrls * urls, struct IGDdatas * data,
					const char * remoteaddr, const char * eport,
					const char * intaddr, const char * iport,
					const char * proto)
{
	int timeout = 0;
	int r;

	if(!intaddr || !remoteaddr || !iport || !eport || !proto)
	{
		fprintf(stderr, "Wrong arguments\n");
		return;
	}

	r = UPNP_GetOutboundPinholeTimeout(urls->controlURL_6FC, data->IPv6FC.servicetype, remoteaddr, eport, intaddr, iport, proto, &timeout);
	if(r!=UPNPCOMMAND_SUCCESS)
		printf("GetOutboundPinholeTimeout([%s]:%s -> [%s]:%s) failed with code %d (%s)\n",
		       intaddr, iport, remoteaddr, eport, r, strupnperror(r));
	else
		printf("GetOutboundPinholeTimeout: ([%s]:%s -> [%s]:%s) / Timeout = %d\n", intaddr, iport, remoteaddr, eport, timeout);
}

static void
GetPinholePackets(struct UPNPUrls * urls,
               struct IGDdatas * data, const char * uniqueID)
{
	int r, pinholePackets = 0;
	if(!uniqueID)
	{
		fprintf(stderr, "invalid arguments\n");
		return;
	}
	r = UPNP_GetPinholePackets(urls->controlURL_6FC, data->IPv6FC.servicetype, uniqueID, &pinholePackets);
	if(r!=UPNPCOMMAND_SUCCESS)
		printf("GetPinholePackets() failed with code %d (%s)\n", r, strupnperror(r));
	else
		printf("GetPinholePackets: Pinhole ID = %s / PinholePackets = %d\n", uniqueID, pinholePackets);
}

static void
CheckPinhole(struct UPNPUrls * urls,
               struct IGDdatas * data, const char * uniqueID)
{
	int r, isWorking = 0;
	if(!uniqueID)
	{
		fprintf(stderr, "invalid arguments\n");
		return;
	}
	r = UPNP_CheckPinholeWorking(urls->controlURL_6FC, data->IPv6FC.servicetype, uniqueID, &isWorking);
	if(r!=UPNPCOMMAND_SUCCESS)
		printf("CheckPinholeWorking() failed with code %d (%s)\n", r, strupnperror(r));
	else
		printf("CheckPinholeWorking: Pinhole ID = %s / IsWorking = %s\n", uniqueID, (isWorking)? "Yes":"No");
}

static void
RemovePinhole(struct UPNPUrls * urls,
               struct IGDdatas * data, const char * uniqueID)
{
	int r;
	if(!uniqueID)
	{
		fprintf(stderr, "invalid arguments\n");
		return;
	}
	r = UPNP_DeletePinhole(urls->controlURL_6FC, data->IPv6FC.servicetype, uniqueID);
	printf("UPNP_DeletePinhole() returned : %d\n", r);
}


/* sample upnp client program */
int main(int argc, char ** argv)
{
	char command = 0;
	char ** commandargv = 0;
	int commandargc = 0;
	struct UPNPDev * devlist = 0;
	char lanaddr[64] = "unset";	/* my ip address on the LAN */
	int i;
	const char * rootdescurl = 0;
	const char * multicastif = 0;
	const char * minissdpdpath = 0;
	int localport = UPNP_LOCAL_PORT_ANY;
	int retcode = 0;
	int error = 0;
	int ipv6 = 0;
	unsigned char ttl = 2;	/* defaulting to 2 */
	const char * description = 0;

#ifdef _WIN32
	WSADATA wsaData;
	int nResult = WSAStartup(MAKEWORD(2,2), &wsaData);
	if(nResult != NO_ERROR)
	{
		fprintf(stderr, "WSAStartup() failed.\n");
		return -1;
	}
#endif
    printf("upnpc : miniupnpc library test client, version %s.\n", MINIUPNPC_VERSION_STRING);
	printf(" (c) 2005-2018 Thomas Bernard.\n");
    printf("Go to http://miniupnp.free.fr/ or https://miniupnp.tuxfamily.org/\n"
	       "for more information.\n");
	/* command line processing */
	for(i=1; i<argc; i++)
	{
		if(0 == strcmp(argv[i], "--help") || 0 == strcmp(argv[i], "-h"))
		{
			command = 0;
			break;
		}
		if(argv[i][0] == '-')
		{
			if(argv[i][1] == 'u')
				rootdescurl = argv[++i];
			else if(argv[i][1] == 'm')
			{
				multicastif = argv[++i];
				minissdpdpath = "";	/* Disable usage of minissdpd */
			}
			else if(argv[i][1] == 'z')
			{
				char junk;
				if(sscanf(argv[++i], "%d%c", &localport, &junk)!=1 ||
					localport<0 || localport>65535 ||
				   (localport >1 && localport < 1024))
				{
					fprintf(stderr, "Invalid localport '%s'\n", argv[i]);
					localport = UPNP_LOCAL_PORT_ANY;
					break;
				}
			}
			else if(argv[i][1] == 'p')
				minissdpdpath = argv[++i];
			else if(argv[i][1] == '6')
				ipv6 = 1;
			else if(argv[i][1] == 'e')
				description = argv[++i];
			else if(argv[i][1] == 't')
				ttl = (unsigned char)atoi(argv[++i]);
			else
			{
				command = argv[i][1];
				i++;
				commandargv = argv + i;
				commandargc = argc - i;
				break;
			}
		}
		else
		{
			fprintf(stderr, "option '%s' invalid\n", argv[i]);
		}
	}

	if(!command
	   || (command == 'a' && commandargc<4)
	   || (command == 'd' && argc<2)
	   || (command == 'r' && argc<2)
	   || (command == 'A' && commandargc<6)
	   || (command == 'U' && commandargc<2)
	   || (command == 'D' && commandargc<1))
	{
		fprintf(stderr, "Usage :\t%s [options] -a ip port external_port protocol [duration]\n\t\tAdd port redirection\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -d external_port protocol <remote host>\n\t\tDelete port redirection\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -s\n\t\tGet Connection status\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -l\n\t\tList redirections\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -L\n\t\tList redirections (using GetListOfPortMappings (for IGD:2 only)\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -n ip port external_port protocol [duration]\n\t\tAdd (any) port redirection allowing IGD to use alternative external_port (for IGD:2 only)\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -N external_port_start external_port_end protocol [manage]\n\t\tDelete range of port redirections (for IGD:2 only)\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -r port1 [external_port1] protocol1 [port2 [external_port2] protocol2] [...]\n\t\tAdd all redirections to the current host\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -A remote_ip remote_port internal_ip internal_port protocol lease_time\n\t\tAdd Pinhole (for IGD:2 only)\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -U uniqueID new_lease_time\n\t\tUpdate Pinhole (for IGD:2 only)\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -C uniqueID\n\t\tCheck if Pinhole is Working (for IGD:2 only)\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -K uniqueID\n\t\tGet Number of packets going through the rule (for IGD:2 only)\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -D uniqueID\n\t\tDelete Pinhole (for IGD:2 only)\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -S\n\t\tGet Firewall status (for IGD:2 only)\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -G remote_ip remote_port internal_ip internal_port protocol\n\t\tGet Outbound Pinhole Timeout (for IGD:2 only)\n", argv[0]);
		fprintf(stderr, "       \t%s [options] -P\n\t\tGet Presentation url\n", argv[0]);
		fprintf(stderr, "\nprotocol is UDP or TCP\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  -e description : set description for port mapping.\n");
		fprintf(stderr, "  -6 : use ip v6 instead of ip v4.\n");
		fprintf(stderr, "  -u url : bypass discovery process by providing the XML root description url.\n");
		fprintf(stderr, "  -m address/interface : provide ip address (ip v4) or interface name (ip v4 or v6) to use for sending SSDP multicast packets.\n");
		fprintf(stderr, "  -z localport : SSDP packets local (source) port (1024-65535).\n");
		fprintf(stderr, "  -p path : use this path for MiniSSDPd socket.\n");
		fprintf(stderr, "  -t ttl : set multicast TTL. Default value is 2.\n");
		return 1;
	}

	if( rootdescurl
	  || (devlist = upnpDiscover(2000, multicastif, minissdpdpath,
	                             localport, ipv6, ttl, &error)))
	{
		struct UPNPDev * device;
		struct UPNPUrls urls;
		struct IGDdatas data;
		if(devlist)
		{
			printf("List of UPNP devices found on the network :\n");
			for(device = devlist; device; device = device->pNext)
			{
				printf(" desc: %s\n st: %s\n\n",
					   device->descURL, device->st);
			}
		}
		else if(!rootdescurl)
		{
			printf("upnpDiscover() error code=%d\n", error);
		}
		i = 1;
		if( (rootdescurl && UPNP_GetIGDFromUrl(rootdescurl, &urls, &data, lanaddr, sizeof(lanaddr)))
		  || (i = UPNP_GetValidIGD(devlist, &urls, &data, lanaddr, sizeof(lanaddr))))
		{
			switch(i) {
			case 1:
				printf("Found valid IGD : %s\n", urls.controlURL);
				break;
			case 2:
				printf("Found a (not connected?) IGD : %s\n", urls.controlURL);
				printf("Trying to continue anyway\n");
				break;
			case 3:
				printf("UPnP device found. Is it an IGD ? : %s\n", urls.controlURL);
				printf("Trying to continue anyway\n");
				break;
			default:
				printf("Found device (igd ?) : %s\n", urls.controlURL);
				printf("Trying to continue anyway\n");
			}
			printf("Local LAN ip address : %s\n", lanaddr);
			#if 0
			printf("getting \"%s\"\n", urls.ipcondescURL);
			descXML = miniwget(urls.ipcondescURL, &descXMLsize);
			if(descXML)
			{
				/*fwrite(descXML, 1, descXMLsize, stdout);*/
				free(descXML); descXML = NULL;
			}
			#endif

			switch(command)
			{
			case 'l':
				DisplayInfos(&urls, &data);
				ListRedirections(&urls, &data);
				break;
			case 'L':
				NewListRedirections(&urls, &data);
				break;
			case 'a':
				if (SetRedirectAndTest(&urls, &data,
						   commandargv[0], commandargv[1],
						   commandargv[2], commandargv[3],
						   (commandargc > 4)?commandargv[4]:"0",
						   description, 0) < 0)
					retcode = 2;
				break;
			case 'd':
				if (RemoveRedirect(&urls, &data, commandargv[0], commandargv[1],
				               commandargc > 2 ? commandargv[2] : NULL) < 0)
					retcode = 2;
				break;
			case 'n':	/* aNy */
				if (SetRedirectAndTest(&urls, &data,
						   commandargv[0], commandargv[1],
						   commandargv[2], commandargv[3],
						   (commandargc > 4)?commandargv[4]:"0",
						   description, 1) < 0)
					retcode = 2;
				break;
			case 'N':
				if (commandargc < 3)
					fprintf(stderr, "too few arguments\n");

				if (RemoveRedirectRange(&urls, &data, commandargv[0], commandargv[1], commandargv[2],
						    commandargc > 3 ? commandargv[3] : NULL) < 0)
					retcode = 2;
				break;
			case 's':
				GetConnectionStatus(&urls, &data);
				break;
			case 'r':
				i = 0;
				while(i<commandargc)
				{
					if(!is_int(commandargv[i])) {
						/* 1st parameter not an integer : error */
						fprintf(stderr, "command -r : %s is not an port number\n", commandargv[i]);
						retcode = 1;
						break;
					} else if(is_int(commandargv[i+1])){
						/* 2nd parameter is an integer : <port> <external_port> <protocol> */
						if (SetRedirectAndTest(&urls, &data,
								   lanaddr, commandargv[i],
								   commandargv[i+1], commandargv[i+2], "0",
								   description, 0) < 0)
							retcode = 2;
						i+=3;	/* 3 parameters parsed */
					} else {
						/* 2nd parameter not an integer : <port> <protocol> */
						if (SetRedirectAndTest(&urls, &data,
								   lanaddr, commandargv[i],
								   commandargv[i], commandargv[i+1], "0",
								   description, 0) < 0)
							retcode = 2;
						i+=2;	/* 2 parameters parsed */
					}
				}
				break;
			case 'A':
				SetPinholeAndTest(&urls, &data,
				                  commandargv[0], commandargv[1],
				                  commandargv[2], commandargv[3],
				                  commandargv[4], commandargv[5]);
				break;
			case 'U':
				GetPinholeAndUpdate(&urls, &data,
				                   commandargv[0], commandargv[1]);
				break;
			case 'C':
				for(i=0; i<commandargc; i++)
				{
					CheckPinhole(&urls, &data, commandargv[i]);
				}
				break;
			case 'K':
				for(i=0; i<commandargc; i++)
				{
					GetPinholePackets(&urls, &data, commandargv[i]);
				}
				break;
			case 'D':
				for(i=0; i<commandargc; i++)
				{
					RemovePinhole(&urls, &data, commandargv[i]);
				}
				break;
			case 'S':
				GetFirewallStatus(&urls, &data);
				break;
			case 'G':
				GetPinholeOutboundTimeout(&urls, &data,
							commandargv[0], commandargv[1],
							commandargv[2], commandargv[3],
							commandargv[4]);
				break;
			case 'P':
				printf("Presentation URL found:\n");
				printf("            %s\n", data.presentationurl);
				break;
			default:
				fprintf(stderr, "Unknown switch -%c\n", command);
				retcode = 1;
			}

			FreeUPNPUrls(&urls);
		}
		else
		{
			fprintf(stderr, "No valid UPNP Internet Gateway Device found.\n");
			retcode = 1;
		}
		freeUPNPDevlist(devlist); devlist = 0;
	}
	else
	{
		fprintf(stderr, "No IGD UPnP Device found on the network !\n");
		retcode = 1;
	}
#ifdef _WIN32
	nResult = WSACleanup();
	if(nResult != NO_ERROR) {
		fprintf(stderr, "WSACleanup() failed.\n");
	}
#endif /* _WIN32 */
	return retcode;
}

