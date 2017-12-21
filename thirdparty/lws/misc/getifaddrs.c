/*
 * Copyright (c) 2000 - 2001 Kungliga Tekniska H�gskolan
 * (Royal Institute of Technology, Stockholm, Sweden).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Institute nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE INSTITUTE AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE INSTITUTE OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * originally downloaded from
 *
 * http://ftp.uninett.no/pub/OpenBSD/src/kerberosV/src/lib/roken/getifaddrs.c
 */

#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <net/if.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "private-libwebsockets.h"

#ifdef LWS_HAVE_SYS_SOCKIO_H
#include <sys/sockio.h>
#endif

#ifdef LWS_HAVE_NETINET_IN6_VAR_H
#include <netinet/in6_var.h>
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#include "getifaddrs.h"

static int
getifaddrs2(struct ifaddrs **ifap, int af, int siocgifconf, int siocgifflags,
	    size_t ifreq_sz)
{
	int ret;
	int fd;
	size_t buf_size;
	char *buf;
	struct ifconf ifconf;
	char *p;
	size_t sz;
	struct sockaddr sa_zero;
	struct ifreq *ifr;
	struct ifaddrs *start,  **end = &start;

	buf = NULL;

	memset(&sa_zero, 0, sizeof(sa_zero));
	fd = socket(af, SOCK_DGRAM, 0);
	if (fd < 0)
		return -1;

	buf_size = 8192;
	for (;;) {
		buf = lws_zalloc(buf_size, "getifaddrs2");
		if (buf == NULL) {
			ret = ENOMEM;
			goto error_out;
		}
		ifconf.ifc_len = buf_size;
		ifconf.ifc_buf = buf;

		/*
		 * Solaris returns EINVAL when the buffer is too small.
		 */
		if (ioctl(fd, siocgifconf, &ifconf) < 0 && errno != EINVAL) {
			ret = errno;
			goto error_out;
		}
		/*
		 * Can the difference between a full and a overfull buf
		 * be determined?
		 */

		if (ifconf.ifc_len < (int)buf_size)
			break;
		lws_free(buf);
		buf_size *= 2;
	}

	for (p = ifconf.ifc_buf; p < ifconf.ifc_buf + ifconf.ifc_len; p += sz) {
		struct ifreq ifreq;
		struct sockaddr *sa;
		size_t salen;

		ifr = (struct ifreq *)p;
		sa  = &ifr->ifr_addr;

		sz = ifreq_sz;
		salen = sizeof(struct sockaddr);
#ifdef LWS_HAVE_STRUCT_SOCKADDR_SA_LEN
		salen = sa->sa_len;
		sz = max(sz, sizeof(ifr->ifr_name) + sa->sa_len);
#endif
#ifdef SA_LEN
		salen = SA_LEN(sa);
		sz = max(sz, sizeof(ifr->ifr_name) + SA_LEN(sa));
#endif
		memset(&ifreq, 0, sizeof(ifreq));
		memcpy(ifreq.ifr_name, ifr->ifr_name, sizeof(ifr->ifr_name));

		if (ioctl(fd, siocgifflags, &ifreq) < 0) {
			ret = errno;
			goto error_out;
		}

		*end = lws_malloc(sizeof(**end), "getifaddrs");

		(*end)->ifa_next = NULL;
		(*end)->ifa_name = strdup(ifr->ifr_name);
		(*end)->ifa_flags = ifreq.ifr_flags;
		(*end)->ifa_addr = lws_malloc(salen, "getifaddrs");
		memcpy((*end)->ifa_addr, sa, salen);
		(*end)->ifa_netmask = NULL;

#if 0
		/* fix these when we actually need them */
		if (ifreq.ifr_flags & IFF_BROADCAST) {
			(*end)->ifa_broadaddr =
				lws_malloc(sizeof(ifr->ifr_broadaddr), "getifaddrs");
			memcpy((*end)->ifa_broadaddr, &ifr->ifr_broadaddr,
						    sizeof(ifr->ifr_broadaddr));
		} else if (ifreq.ifr_flags & IFF_POINTOPOINT) {
			(*end)->ifa_dstaddr =
				lws_malloc(sizeof(ifr->ifr_dstaddr), "getifaddrs");
			memcpy((*end)->ifa_dstaddr, &ifr->ifr_dstaddr,
						      sizeof(ifr->ifr_dstaddr));
		} else
			(*end)->ifa_dstaddr = NULL;
#else
		(*end)->ifa_dstaddr = NULL;
#endif
		(*end)->ifa_data = NULL;

		end = &(*end)->ifa_next;

	}
	*ifap = start;
	close(fd);
	lws_free(buf);
	return 0;

error_out:
	close(fd);
	lws_free(buf);
	errno = ret;

	return -1;
}

int
getifaddrs(struct ifaddrs **ifap)
{
	int ret = -1;
	errno = ENXIO;
#if defined(AF_INET6) && defined(SIOCGIF6CONF) && defined(SIOCGIF6FLAGS)
	if (ret)
		ret = getifaddrs2(ifap, AF_INET6, SIOCGIF6CONF, SIOCGIF6FLAGS,
			   sizeof(struct in6_ifreq));
#endif
#if defined(LWS_HAVE_IPV6) && defined(SIOCGIFCONF)
	if (ret)
		ret = getifaddrs2(ifap, AF_INET6, SIOCGIFCONF, SIOCGIFFLAGS,
			   sizeof(struct ifreq));
#endif
#if defined(AF_INET) && defined(SIOCGIFCONF) && defined(SIOCGIFFLAGS)
	if (ret)
		ret = getifaddrs2(ifap, AF_INET, SIOCGIFCONF, SIOCGIFFLAGS,
			   sizeof(struct ifreq));
#endif
	return ret;
}

void
freeifaddrs(struct ifaddrs *ifp)
{
	struct ifaddrs *p, *q;

	for (p = ifp; p; ) {
		lws_free(p->ifa_name);
		lws_free(p->ifa_addr);
		lws_free(p->ifa_dstaddr);
		lws_free(p->ifa_netmask);
		lws_free(p->ifa_data);
		q = p;
		p = p->ifa_next;
		lws_free(q);
	}
}

#ifdef TEST

void
print_addr(const char *s, struct sockaddr *sa)
{
	int i;
	printf("  %s=%d/", s, sa->sa_family);
#ifdef LWS_HAVE_STRUCT_SOCKADDR_SA_LEN
	for (i = 0;
	       i < sa->sa_len - ((lws_intptr_t)sa->sa_data - (lws_intptr_t)&sa->sa_family); i++)
		printf("%02x", ((unsigned char *)sa->sa_data)[i]);
#else
	for (i = 0; i < sizeof(sa->sa_data); i++)
		printf("%02x", ((unsigned char *)sa->sa_data)[i]);
#endif
	printf("\n");
}

void
print_ifaddrs(struct ifaddrs *x)
{
	struct ifaddrs *p;

	for (p = x; p; p = p->ifa_next) {
		printf("%s\n", p->ifa_name);
		printf("  flags=%x\n", p->ifa_flags);
		if (p->ifa_addr)
			print_addr("addr", p->ifa_addr);
		if (p->ifa_dstaddr)
			print_addr("dstaddr", p->ifa_dstaddr);
		if (p->ifa_netmask)
			print_addr("netmask", p->ifa_netmask);
		printf("  %p\n", p->ifa_data);
	}
}

int
main()
{
	struct ifaddrs *a = NULL, *b;
	getifaddrs2(&a, AF_INET, SIOCGIFCONF, SIOCGIFFLAGS,
		    sizeof(struct ifreq));
	print_ifaddrs(a);
	printf("---\n");
	getifaddrs(&b);
	print_ifaddrs(b);
	return 0;
}
#endif
