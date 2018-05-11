/* $Id: minissdpc.c,v 1.32 2016/10/07 09:04:36 nanard Exp $ */
/* vim: tabstop=4 shiftwidth=4 noexpandtab
 * Project : miniupnp
 * Web : http://miniupnp.free.fr/
 * Author : Thomas BERNARD
 * copyright (c) 2005-2018 Thomas Bernard
 * This software is subjet to the conditions detailed in the
 * provided LICENCE file. */
/*#include <syslog.h>*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#if defined (__NetBSD__)
#include <net/if.h>
#endif
#if defined(_WIN32) || defined(__amigaos__) || defined(__amigaos4__)
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <io.h>
#include <iphlpapi.h>
#define snprintf _snprintf
#if !defined(_MSC_VER)
#include <stdint.h>
#else /* !defined(_MSC_VER) */
typedef unsigned short uint16_t;
#endif /* !defined(_MSC_VER) */
#ifndef strncasecmp
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
#define strncasecmp _memicmp
#else /* defined(_MSC_VER) && (_MSC_VER >= 1400) */
#define strncasecmp memicmp
#endif /* defined(_MSC_VER) && (_MSC_VER >= 1400) */
#endif /* #ifndef strncasecmp */
#endif /* _WIN32 */
#if defined(__amigaos__) || defined(__amigaos4__)
#include <sys/socket.h>
#endif /* defined(__amigaos__) || defined(__amigaos4__) */
#if defined(__amigaos__)
#define uint16_t unsigned short
#endif /* defined(__amigaos__) */
/* Hack */
#define UNIX_PATH_LEN   108
struct sockaddr_un {
  uint16_t sun_family;
  char     sun_path[UNIX_PATH_LEN];
};
#else /* defined(_WIN32) || defined(__amigaos__) || defined(__amigaos4__) */
#include <strings.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/param.h>
#include <sys/time.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <net/if.h>
#define closesocket close
#endif

#include "miniupnpc_socketdef.h"

#if !defined(__DragonFly__) && !defined(__OpenBSD__) && !defined(__NetBSD__) && !defined(__APPLE__) && !defined(_WIN32) && !defined(__CYGWIN__) && !defined(__sun) && !defined(__GNU__) && !defined(__FreeBSD_kernel__)
#define HAS_IP_MREQN
#endif

#if !defined(HAS_IP_MREQN) && !defined(_WIN32)
#include <sys/ioctl.h>
#if defined(__sun)
#include <sys/sockio.h>
#endif
#endif

#if defined(HAS_IP_MREQN) && defined(NEED_STRUCT_IP_MREQN)
/* Several versions of glibc don't define this structure,
 * define it here and compile with CFLAGS NEED_STRUCT_IP_MREQN */
struct ip_mreqn
{
	struct in_addr	imr_multiaddr;		/* IP multicast address of group */
	struct in_addr	imr_address;		/* local IP address of interface */
	int		imr_ifindex;		/* Interface index */
};
#endif

#if defined(__amigaos__) || defined(__amigaos4__)
/* Amiga OS specific stuff */
#define TIMEVAL struct timeval
#endif

#include "minissdpc.h"
#include "miniupnpc.h"
#include "receivedata.h"

#if !(defined(_WIN32) || defined(__amigaos__) || defined(__amigaos4__))

#include "codelength.h"

struct UPNPDev *
getDevicesFromMiniSSDPD(const char * devtype, const char * socketpath, int * error)
{
	struct UPNPDev * devlist = NULL;
	int s;
	int res;

	s = connectToMiniSSDPD(socketpath);
	if (s < 0) {
		if (error)
			*error = s;
		return NULL;
	}
	res = requestDevicesFromMiniSSDPD(s, devtype);
	if (res < 0) {
		if (error)
			*error = res;
	} else {
		devlist = receiveDevicesFromMiniSSDPD(s, error);
	}
	disconnectFromMiniSSDPD(s);
	return devlist;
}

/* macros used to read from unix socket */
#define READ_BYTE_BUFFER(c) \
	if((int)bufferindex >= n) { \
		n = read(s, buffer, sizeof(buffer)); \
		if(n<=0) break; \
		bufferindex = 0; \
	} \
	c = buffer[bufferindex++];

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif /* MIN */

#define READ_COPY_BUFFER(dst, len) \
	for(l = len, p = (unsigned char *)dst; l > 0; ) { \
		unsigned int lcopy; \
		if((int)bufferindex >= n) { \
			n = read(s, buffer, sizeof(buffer)); \
			if(n<=0) break; \
			bufferindex = 0; \
		} \
		lcopy = MIN(l, (n - bufferindex)); \
		memcpy(p, buffer + bufferindex, lcopy); \
		l -= lcopy; \
		p += lcopy; \
		bufferindex += lcopy; \
	}

#define READ_DISCARD_BUFFER(len) \
	for(l = len; l > 0; ) { \
		unsigned int lcopy; \
		if(bufferindex >= n) { \
			n = read(s, buffer, sizeof(buffer)); \
			if(n<=0) break; \
			bufferindex = 0; \
		} \
		lcopy = MIN(l, (n - bufferindex)); \
		l -= lcopy; \
		bufferindex += lcopy; \
	}

int
connectToMiniSSDPD(const char * socketpath)
{
	int s;
	struct sockaddr_un addr;
#if defined(MINIUPNPC_SET_SOCKET_TIMEOUT) && !defined(__sun)
	struct timeval timeout;
#endif /* #ifdef MINIUPNPC_SET_SOCKET_TIMEOUT */

	s = socket(AF_UNIX, SOCK_STREAM, 0);
	if(s < 0)
	{
		/*syslog(LOG_ERR, "socket(unix): %m");*/
		perror("socket(unix)");
		return MINISSDPC_SOCKET_ERROR;
	}
#if defined(MINIUPNPC_SET_SOCKET_TIMEOUT) && !defined(__sun)
	/* setting a 3 seconds timeout */
	/* not supported for AF_UNIX sockets under Solaris */
	timeout.tv_sec = 3;
	timeout.tv_usec = 0;
	if(setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(struct timeval)) < 0)
	{
		perror("setsockopt SO_RCVTIMEO unix");
	}
	timeout.tv_sec = 3;
	timeout.tv_usec = 0;
	if(setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(struct timeval)) < 0)
	{
		perror("setsockopt SO_SNDTIMEO unix");
	}
#endif /* #ifdef MINIUPNPC_SET_SOCKET_TIMEOUT */
	if(!socketpath)
		socketpath = "/var/run/minissdpd.sock";
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, socketpath, sizeof(addr.sun_path));
	/* TODO : check if we need to handle the EINTR */
	if(connect(s, (struct sockaddr *)&addr, sizeof(struct sockaddr_un)) < 0)
	{
		/*syslog(LOG_WARNING, "connect(\"%s\"): %m", socketpath);*/
		close(s);
		return MINISSDPC_SOCKET_ERROR;
	}
	return s;
}

int
disconnectFromMiniSSDPD(int s)
{
	if (close(s) < 0)
		return MINISSDPC_SOCKET_ERROR;
	return MINISSDPC_SUCCESS;
}

int
requestDevicesFromMiniSSDPD(int s, const char * devtype)
{
	unsigned char buffer[256];
	unsigned char * p;
	unsigned int stsize, l;

	stsize = strlen(devtype);
	if(stsize == 8 && 0 == memcmp(devtype, "ssdp:all", 8))
	{
		buffer[0] = 3;	/* request type 3 : everything */
	}
	else
	{
		buffer[0] = 1; /* request type 1 : request devices/services by type */
	}
	p = buffer + 1;
	l = stsize;	CODELENGTH(l, p);
	if(p + stsize > buffer + sizeof(buffer))
	{
		/* devtype is too long ! */
#ifdef DEBUG
		fprintf(stderr, "devtype is too long ! stsize=%u sizeof(buffer)=%u\n",
		        stsize, (unsigned)sizeof(buffer));
#endif /* DEBUG */
		return MINISSDPC_INVALID_INPUT;
	}
	memcpy(p, devtype, stsize);
	p += stsize;
	if(write(s, buffer, p - buffer) < 0)
	{
		/*syslog(LOG_ERR, "write(): %m");*/
		perror("minissdpc.c: write()");
		return MINISSDPC_SOCKET_ERROR;
	}
	return MINISSDPC_SUCCESS;
}

struct UPNPDev *
receiveDevicesFromMiniSSDPD(int s, int * error)
{
	struct UPNPDev * tmp;
	struct UPNPDev * devlist = NULL;
	unsigned char buffer[256];
	ssize_t n;
	unsigned char * p;
	unsigned char * url;
	unsigned char * st;
	unsigned int bufferindex;
	unsigned int i, ndev;
	unsigned int urlsize, stsize, usnsize, l;

	n = read(s, buffer, sizeof(buffer));
	if(n<=0)
	{
		perror("minissdpc.c: read()");
		if (error)
			*error = MINISSDPC_SOCKET_ERROR;
		return NULL;
	}
	ndev = buffer[0];
	bufferindex = 1;
	for(i = 0; i < ndev; i++)
	{
		DECODELENGTH_READ(urlsize, READ_BYTE_BUFFER);
		if(n<=0) {
			if (error)
				*error = MINISSDPC_INVALID_SERVER_REPLY;
			return devlist;
		}
#ifdef DEBUG
		printf("  urlsize=%u", urlsize);
#endif /* DEBUG */
		url = malloc(urlsize);
		if(url == NULL) {
			if (error)
				*error = MINISSDPC_MEMORY_ERROR;
			return devlist;
		}
		READ_COPY_BUFFER(url, urlsize);
		if(n<=0) {
			if (error)
				*error = MINISSDPC_INVALID_SERVER_REPLY;
			goto free_url_and_return;
		}
		DECODELENGTH_READ(stsize, READ_BYTE_BUFFER);
		if(n<=0) {
			if (error)
				*error = MINISSDPC_INVALID_SERVER_REPLY;
			goto free_url_and_return;
		}
#ifdef DEBUG
		printf("   stsize=%u", stsize);
#endif /* DEBUG */
		st = malloc(stsize);
		if (st == NULL) {
			if (error)
				*error = MINISSDPC_MEMORY_ERROR;
			goto free_url_and_return;
		}
		READ_COPY_BUFFER(st, stsize);
		if(n<=0) {
			if (error)
				*error = MINISSDPC_INVALID_SERVER_REPLY;
			goto free_url_and_st_and_return;
		}
		DECODELENGTH_READ(usnsize, READ_BYTE_BUFFER);
		if(n<=0) {
			if (error)
				*error = MINISSDPC_INVALID_SERVER_REPLY;
			goto free_url_and_st_and_return;
		}
#ifdef DEBUG
		printf("   usnsize=%u\n", usnsize);
#endif /* DEBUG */
		tmp = (struct UPNPDev *)malloc(sizeof(struct UPNPDev)+urlsize+stsize+usnsize);
		if(tmp == NULL) {
			if (error)
				*error = MINISSDPC_MEMORY_ERROR;
			goto free_url_and_st_and_return;
		}
		tmp->pNext = devlist;
		tmp->descURL = tmp->buffer;
		tmp->st = tmp->buffer + 1 + urlsize;
		memcpy(tmp->buffer, url, urlsize);
		tmp->buffer[urlsize] = '\0';
		memcpy(tmp->st, st, stsize);
		tmp->buffer[urlsize+1+stsize] = '\0';
		free(url);
		free(st);
		url = NULL;
		st = NULL;
		tmp->usn = tmp->buffer + 1 + urlsize + 1 + stsize;
		READ_COPY_BUFFER(tmp->usn, usnsize);
		if(n<=0) {
			if (error)
				*error = MINISSDPC_INVALID_SERVER_REPLY;
			goto free_tmp_and_return;
		}
		tmp->buffer[urlsize+1+stsize+1+usnsize] = '\0';
		tmp->scope_id = 0;	/* default value. scope_id is not available with MiniSSDPd */
		devlist = tmp;
	}
	if (error)
		*error = MINISSDPC_SUCCESS;
	return devlist;

free_url_and_st_and_return:
	free(st);
free_url_and_return:
	free(url);
	return devlist;

free_tmp_and_return:
	free(tmp);
	return devlist;
}

#endif /* !(defined(_WIN32) || defined(__amigaos__) || defined(__amigaos4__)) */

/* parseMSEARCHReply()
 * the last 4 arguments are filled during the parsing :
 *    - location/locationsize : "location:" field of the SSDP reply packet
 *    - st/stsize : "st:" field of the SSDP reply packet.
 * The strings are NOT null terminated */
static void
parseMSEARCHReply(const char * reply, int size,
                  const char * * location, int * locationsize,
			      const char * * st, int * stsize,
			      const char * * usn, int * usnsize)
{
	int a, b, i;
	i = 0;
	a = i;	/* start of the line */
	b = 0;	/* end of the "header" (position of the colon) */
	while(i<size)
	{
		switch(reply[i])
		{
		case ':':
				if(b==0)
				{
					b = i; /* end of the "header" */
					/*for(j=a; j<b; j++)
					{
						putchar(reply[j]);
					}
					*/
				}
				break;
		case '\x0a':
		case '\x0d':
				if(b!=0)
				{
					/*for(j=b+1; j<i; j++)
					{
						putchar(reply[j]);
					}
					putchar('\n');*/
					/* skip the colon and white spaces */
					do { b++; } while(reply[b]==' ');
					if(0==strncasecmp(reply+a, "location", 8))
					{
						*location = reply+b;
						*locationsize = i-b;
					}
					else if(0==strncasecmp(reply+a, "st", 2))
					{
						*st = reply+b;
						*stsize = i-b;
					}
					else if(0==strncasecmp(reply+a, "usn", 3))
					{
						*usn = reply+b;
						*usnsize = i-b;
					}
					b = 0;
				}
				a = i+1;
				break;
		default:
				break;
		}
		i++;
	}
}

/* port upnp discover : SSDP protocol */
#define SSDP_PORT 1900
#define XSTR(s) STR(s)
#define STR(s) #s
#define UPNP_MCAST_ADDR "239.255.255.250"
/* for IPv6 */
#define UPNP_MCAST_LL_ADDR "FF02::C" /* link-local */
#define UPNP_MCAST_SL_ADDR "FF05::C" /* site-local */

/* direct discovery if minissdpd responses are not sufficient */
/* ssdpDiscoverDevices() :
 * return a chained list of all devices found or NULL if
 * no devices was found.
 * It is up to the caller to free the chained list
 * delay is in millisecond (poll).
 * UDA v1.1 says :
 *   The TTL for the IP packet SHOULD default to 2 and
 *   SHOULD be configurable. */
struct UPNPDev *
ssdpDiscoverDevices(const char * const deviceTypes[],
                    int delay, const char * multicastif,
                    int localport,
                    int ipv6, unsigned char ttl,
                    int * error,
                    int searchalltypes)
{
	struct UPNPDev * tmp;
	struct UPNPDev * devlist = 0;
	unsigned int scope_id = 0;
	int opt = 1;
	static const char MSearchMsgFmt[] =
	"M-SEARCH * HTTP/1.1\r\n"
	"HOST: %s:" XSTR(SSDP_PORT) "\r\n"
	"ST: %s\r\n"
	"MAN: \"ssdp:discover\"\r\n"
	"MX: %u\r\n"
	"\r\n";
	int deviceIndex;
	char bufr[1536];	/* reception and emission buffer */
	SOCKET sudp;
	int n;
	struct sockaddr_storage sockudp_r;
	unsigned int mx;
#ifdef NO_GETADDRINFO
	struct sockaddr_storage sockudp_w;
#else
	int rv;
	struct addrinfo hints, *servinfo, *p;
#endif
#ifdef _WIN32
	MIB_IPFORWARDROW ip_forward;
	unsigned long _ttl = (unsigned long)ttl;
#endif
	int linklocal = 1;
	int sentok;

	if(error)
		*error = MINISSDPC_UNKNOWN_ERROR;

	if(localport==UPNP_LOCAL_PORT_SAME)
		localport = SSDP_PORT;

#ifdef _WIN32
	sudp = socket(ipv6 ? PF_INET6 : PF_INET, SOCK_DGRAM, IPPROTO_UDP);
#else
	sudp = socket(ipv6 ? PF_INET6 : PF_INET, SOCK_DGRAM, 0);
#endif
	if(ISINVALID(sudp))
	{
		if(error)
			*error = MINISSDPC_SOCKET_ERROR;
		PRINT_SOCKET_ERROR("socket");
		return NULL;
	}
	/* reception */
	memset(&sockudp_r, 0, sizeof(struct sockaddr_storage));
	if(ipv6) {
		struct sockaddr_in6 * p = (struct sockaddr_in6 *)&sockudp_r;
		p->sin6_family = AF_INET6;
		if(localport > 0 && localport < 65536)
			p->sin6_port = htons((unsigned short)localport);
		p->sin6_addr = in6addr_any; /* in6addr_any is not available with MinGW32 3.4.2 */
	} else {
		struct sockaddr_in * p = (struct sockaddr_in *)&sockudp_r;
		p->sin_family = AF_INET;
		if(localport > 0 && localport < 65536)
			p->sin_port = htons((unsigned short)localport);
		p->sin_addr.s_addr = INADDR_ANY;
	}
#ifdef _WIN32
/* This code could help us to use the right Network interface for
 * SSDP multicast traffic */
/* Get IP associated with the index given in the ip_forward struct
 * in order to give this ip to setsockopt(sudp, IPPROTO_IP, IP_MULTICAST_IF) */
	if(!ipv6
	   && (GetBestRoute(inet_addr("223.255.255.255"), 0, &ip_forward) == NO_ERROR)) {
		DWORD dwRetVal = 0;
		PMIB_IPADDRTABLE pIPAddrTable;
		DWORD dwSize = 0;
#ifdef DEBUG
		IN_ADDR IPAddr;
#endif
		int i;
#ifdef DEBUG
		printf("ifIndex=%lu nextHop=%lx \n", ip_forward.dwForwardIfIndex, ip_forward.dwForwardNextHop);
#endif
		pIPAddrTable = (MIB_IPADDRTABLE *) malloc(sizeof (MIB_IPADDRTABLE));
		if(pIPAddrTable) {
			if (GetIpAddrTable(pIPAddrTable, &dwSize, 0) == ERROR_INSUFFICIENT_BUFFER) {
				free(pIPAddrTable);
				pIPAddrTable = (MIB_IPADDRTABLE *) malloc(dwSize);
			}
		}
		if(pIPAddrTable) {
			dwRetVal = GetIpAddrTable( pIPAddrTable, &dwSize, 0 );
			if (dwRetVal == NO_ERROR) {
#ifdef DEBUG
				printf("\tNum Entries: %ld\n", pIPAddrTable->dwNumEntries);
#endif
				for (i=0; i < (int) pIPAddrTable->dwNumEntries; i++) {
#ifdef DEBUG
					printf("\n\tInterface Index[%d]:\t%ld\n", i, pIPAddrTable->table[i].dwIndex);
					IPAddr.S_un.S_addr = (u_long) pIPAddrTable->table[i].dwAddr;
					printf("\tIP Address[%d]:     \t%s\n", i, inet_ntoa(IPAddr) );
					IPAddr.S_un.S_addr = (u_long) pIPAddrTable->table[i].dwMask;
					printf("\tSubnet Mask[%d]:    \t%s\n", i, inet_ntoa(IPAddr) );
					IPAddr.S_un.S_addr = (u_long) pIPAddrTable->table[i].dwBCastAddr;
					printf("\tBroadCast[%d]:      \t%s (%ld)\n", i, inet_ntoa(IPAddr), pIPAddrTable->table[i].dwBCastAddr);
					printf("\tReassembly size[%d]:\t%ld\n", i, pIPAddrTable->table[i].dwReasmSize);
					printf("\tType and State[%d]:", i);
					printf("\n");
#endif
					if (pIPAddrTable->table[i].dwIndex == ip_forward.dwForwardIfIndex) {
						/* Set the address of this interface to be used */
						struct in_addr mc_if;
						memset(&mc_if, 0, sizeof(mc_if));
						mc_if.s_addr = pIPAddrTable->table[i].dwAddr;
						if(setsockopt(sudp, IPPROTO_IP, IP_MULTICAST_IF, (const char *)&mc_if, sizeof(mc_if)) < 0) {
							PRINT_SOCKET_ERROR("setsockopt");
						}
						((struct sockaddr_in *)&sockudp_r)->sin_addr.s_addr = pIPAddrTable->table[i].dwAddr;
#ifndef DEBUG
						break;
#endif
					}
				}
			}
			free(pIPAddrTable);
			pIPAddrTable = NULL;
		}
	}
#endif	/* _WIN32 */

#ifdef _WIN32
	if (setsockopt(sudp, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof (opt)) < 0)
#else
	if (setsockopt(sudp, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof (opt)) < 0)
#endif
	{
		if(error)
			*error = MINISSDPC_SOCKET_ERROR;
		PRINT_SOCKET_ERROR("setsockopt(SO_REUSEADDR,...)");
		return NULL;
	}

	if(ipv6) {
#ifdef _WIN32
		DWORD mcastHops = ttl;
		if(setsockopt(sudp, IPPROTO_IPV6, IPV6_MULTICAST_HOPS, (const char *)&mcastHops, sizeof(mcastHops)) < 0)
#else  /* _WIN32 */
		int mcastHops = ttl;
		if(setsockopt(sudp, IPPROTO_IPV6, IPV6_MULTICAST_HOPS, &mcastHops, sizeof(mcastHops)) < 0)
#endif /* _WIN32 */
		{
			PRINT_SOCKET_ERROR("setsockopt(IPV6_MULTICAST_HOPS,...)");
		}
	} else {
#ifdef _WIN32
		if(setsockopt(sudp, IPPROTO_IP, IP_MULTICAST_TTL, (const char *)&_ttl, sizeof(_ttl)) < 0)
#else  /* _WIN32 */
		if(setsockopt(sudp, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)) < 0)
#endif /* _WIN32 */
		{
			/* not a fatal error */
			PRINT_SOCKET_ERROR("setsockopt(IP_MULTICAST_TTL,...)");
		}
	}

	if(multicastif)
	{
		if(ipv6) {
#if !defined(_WIN32)
			/* according to MSDN, if_nametoindex() is supported since
			 * MS Windows Vista and MS Windows Server 2008.
			 * http://msdn.microsoft.com/en-us/library/bb408409%28v=vs.85%29.aspx */
			unsigned int ifindex = if_nametoindex(multicastif); /* eth0, etc. */
			if(setsockopt(sudp, IPPROTO_IPV6, IPV6_MULTICAST_IF, &ifindex, sizeof(ifindex)) < 0)
			{
				PRINT_SOCKET_ERROR("setsockopt IPV6_MULTICAST_IF");
			}
#else
#ifdef DEBUG
			printf("Setting of multicast interface not supported in IPv6 under Windows.\n");
#endif
#endif
		} else {
			struct in_addr mc_if;
			mc_if.s_addr = inet_addr(multicastif); /* ex: 192.168.x.x */
			if(mc_if.s_addr != INADDR_NONE)
			{
				((struct sockaddr_in *)&sockudp_r)->sin_addr.s_addr = mc_if.s_addr;
				if(setsockopt(sudp, IPPROTO_IP, IP_MULTICAST_IF, (const char *)&mc_if, sizeof(mc_if)) < 0)
				{
					PRINT_SOCKET_ERROR("setsockopt IP_MULTICAST_IF");
				}
			} else {
#ifdef HAS_IP_MREQN
				/* was not an ip address, try with an interface name */
				struct ip_mreqn reqn;	/* only defined with -D_BSD_SOURCE or -D_GNU_SOURCE */
				memset(&reqn, 0, sizeof(struct ip_mreqn));
				reqn.imr_ifindex = if_nametoindex(multicastif);
				if(setsockopt(sudp, IPPROTO_IP, IP_MULTICAST_IF, (const char *)&reqn, sizeof(reqn)) < 0)
				{
					PRINT_SOCKET_ERROR("setsockopt IP_MULTICAST_IF");
				}
#elif !defined(_WIN32)
				struct ifreq ifr;
				int ifrlen = sizeof(ifr);
				strncpy(ifr.ifr_name, multicastif, IFNAMSIZ);
				ifr.ifr_name[IFNAMSIZ-1] = '\0';
				if(ioctl(sudp, SIOCGIFADDR, &ifr, &ifrlen) < 0)
				{
					PRINT_SOCKET_ERROR("ioctl(...SIOCGIFADDR...)");
				}
				mc_if.s_addr = ((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr.s_addr;
				if(setsockopt(sudp, IPPROTO_IP, IP_MULTICAST_IF, (const char *)&mc_if, sizeof(mc_if)) < 0)
				{
					PRINT_SOCKET_ERROR("setsockopt IP_MULTICAST_IF");
				}
#else /* _WIN32 */
#ifdef DEBUG
				printf("Setting of multicast interface not supported with interface name.\n");
#endif
#endif /* #ifdef HAS_IP_MREQN / !defined(_WIN32) */
			}
		}
	}

	/* Before sending the packed, we first "bind" in order to be able
	 * to receive the response */
	if (bind(sudp, (const struct sockaddr *)&sockudp_r,
	         ipv6 ? sizeof(struct sockaddr_in6) : sizeof(struct sockaddr_in)) != 0)
	{
		if(error)
			*error = MINISSDPC_SOCKET_ERROR;
		PRINT_SOCKET_ERROR("bind");
		closesocket(sudp);
		return NULL;
	}

	if(error)
		*error = MINISSDPC_SUCCESS;
	/* Calculating maximum response time in seconds */
	mx = ((unsigned int)delay) / 1000u;
	if(mx == 0) {
		mx = 1;
		delay = 1000;
	}
	/* receiving SSDP response packet */
	for(deviceIndex = 0; deviceTypes[deviceIndex]; deviceIndex++) {
		sentok = 0;
		/* sending the SSDP M-SEARCH packet */
		n = snprintf(bufr, sizeof(bufr),
		             MSearchMsgFmt,
		             ipv6 ?
		             (linklocal ? "[" UPNP_MCAST_LL_ADDR "]" :  "[" UPNP_MCAST_SL_ADDR "]")
		             : UPNP_MCAST_ADDR,
		             deviceTypes[deviceIndex], mx);
		if ((unsigned int)n >= sizeof(bufr)) {
			if(error)
				*error = MINISSDPC_MEMORY_ERROR;
			goto error;
		}
#ifdef DEBUG
		/*printf("Sending %s", bufr);*/
		printf("Sending M-SEARCH request to %s with ST: %s\n",
		       ipv6 ?
		       (linklocal ? "[" UPNP_MCAST_LL_ADDR "]" :  "[" UPNP_MCAST_SL_ADDR "]")
		       : UPNP_MCAST_ADDR,
		       deviceTypes[deviceIndex]);
#endif
#ifdef NO_GETADDRINFO
		/* the following code is not using getaddrinfo */
		/* emission */
		memset(&sockudp_w, 0, sizeof(struct sockaddr_storage));
		if(ipv6) {
			struct sockaddr_in6 * p = (struct sockaddr_in6 *)&sockudp_w;
			p->sin6_family = AF_INET6;
			p->sin6_port = htons(SSDP_PORT);
			inet_pton(AF_INET6,
			          linklocal ? UPNP_MCAST_LL_ADDR : UPNP_MCAST_SL_ADDR,
			          &(p->sin6_addr));
		} else {
			struct sockaddr_in * p = (struct sockaddr_in *)&sockudp_w;
			p->sin_family = AF_INET;
			p->sin_port = htons(SSDP_PORT);
			p->sin_addr.s_addr = inet_addr(UPNP_MCAST_ADDR);
		}
		n = sendto(sudp, bufr, n, 0, &sockudp_w,
		           ipv6 ? sizeof(struct sockaddr_in6) : sizeof(struct sockaddr_in));
		if (n < 0) {
			if(error)
				*error = MINISSDPC_SOCKET_ERROR;
			PRINT_SOCKET_ERROR("sendto");
		} else {
			sentok = 1;
		}
#else /* #ifdef NO_GETADDRINFO */
		memset(&hints, 0, sizeof(hints));
		hints.ai_family = AF_UNSPEC; /* AF_INET6 or AF_INET */
		hints.ai_socktype = SOCK_DGRAM;
		/*hints.ai_flags = */
		if ((rv = getaddrinfo(ipv6
		                      ? (linklocal ? UPNP_MCAST_LL_ADDR : UPNP_MCAST_SL_ADDR)
		                      : UPNP_MCAST_ADDR,
		                      XSTR(SSDP_PORT), &hints, &servinfo)) != 0) {
			if(error)
				*error = MINISSDPC_SOCKET_ERROR;
#ifdef _WIN32
			fprintf(stderr, "getaddrinfo() failed: %d\n", rv);
#else
			fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
#endif
			break;
		}
		for(p = servinfo; p; p = p->ai_next) {
			n = sendto(sudp, bufr, n, 0, p->ai_addr, p->ai_addrlen);
			if (n < 0) {
#ifdef DEBUG
				char hbuf[NI_MAXHOST], sbuf[NI_MAXSERV];
				if (getnameinfo(p->ai_addr, p->ai_addrlen, hbuf, sizeof(hbuf), sbuf,
				                sizeof(sbuf), NI_NUMERICHOST | NI_NUMERICSERV) == 0) {
					fprintf(stderr, "host:%s port:%s\n", hbuf, sbuf);
				}
#endif
				PRINT_SOCKET_ERROR("sendto");
				continue;
			} else {
				sentok = 1;
			}
		}
		freeaddrinfo(servinfo);
		if(!sentok) {
			if(error)
				*error = MINISSDPC_SOCKET_ERROR;
		}
#endif /* #ifdef NO_GETADDRINFO */
		/* Waiting for SSDP REPLY packet to M-SEARCH
		 * if searchalltypes is set, enter the loop only
		 * when the last deviceType is reached */
		if((sentok && !searchalltypes) || !deviceTypes[deviceIndex + 1]) do {
			n = receivedata(sudp, bufr, sizeof(bufr), delay, &scope_id);
			if (n < 0) {
				/* error */
				if(error)
					*error = MINISSDPC_SOCKET_ERROR;
				goto error;
			} else if (n == 0) {
				/* no data or Time Out */
#ifdef DEBUG
				printf("NODATA or TIMEOUT\n");
#endif /* DEBUG */
				if (devlist && !searchalltypes) {
					/* found some devices, stop now*/
					if(error)
						*error = MINISSDPC_SUCCESS;
					goto error;
				}
			} else {
				const char * descURL=NULL;
				int urlsize=0;
				const char * st=NULL;
				int stsize=0;
				const char * usn=NULL;
				int usnsize=0;
				parseMSEARCHReply(bufr, n, &descURL, &urlsize, &st, &stsize, &usn, &usnsize);
				if(st&&descURL) {
#ifdef DEBUG
					printf("M-SEARCH Reply:\n  ST: %.*s\n  USN: %.*s\n  Location: %.*s\n",
					       stsize, st, usnsize, (usn?usn:""), urlsize, descURL);
#endif /* DEBUG */
					for(tmp=devlist; tmp; tmp = tmp->pNext) {
						if(memcmp(tmp->descURL, descURL, urlsize) == 0 &&
						   tmp->descURL[urlsize] == '\0' &&
						   memcmp(tmp->st, st, stsize) == 0 &&
						   tmp->st[stsize] == '\0' &&
						   (usnsize == 0 || memcmp(tmp->usn, usn, usnsize) == 0) &&
						   tmp->usn[usnsize] == '\0')
							break;
					}
					/* at the exit of the loop above, tmp is null if
					 * no duplicate device was found */
					if(tmp)
						continue;
					tmp = (struct UPNPDev *)malloc(sizeof(struct UPNPDev)+urlsize+stsize+usnsize);
					if(!tmp) {
						/* memory allocation error */
						if(error)
							*error = MINISSDPC_MEMORY_ERROR;
						goto error;
					}
					tmp->pNext = devlist;
					tmp->descURL = tmp->buffer;
					tmp->st = tmp->buffer + 1 + urlsize;
					tmp->usn = tmp->st + 1 + stsize;
					memcpy(tmp->buffer, descURL, urlsize);
					tmp->buffer[urlsize] = '\0';
					memcpy(tmp->st, st, stsize);
					tmp->buffer[urlsize+1+stsize] = '\0';
					if(usn != NULL)
						memcpy(tmp->usn, usn, usnsize);
					tmp->buffer[urlsize+1+stsize+1+usnsize] = '\0';
					tmp->scope_id = scope_id;
					devlist = tmp;
				}
			}
		} while(n > 0);
		if(ipv6) {
			/* switch linklocal flag */
			if(linklocal) {
				linklocal = 0;
				--deviceIndex;
			} else {
				linklocal = 1;
			}
		}
	}
error:
	closesocket(sudp);
	return devlist;
}

