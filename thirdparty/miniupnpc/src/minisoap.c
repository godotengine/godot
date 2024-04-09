/* $Id: minisoap.c,v 1.32 2023/07/05 22:43:50 nanard Exp $ */
/* vim: tabstop=4 shiftwidth=4 noexpandtab
 * Project : miniupnp
 * Author : Thomas Bernard
 * Copyright (c) 2005-2023 Thomas Bernard
 * This software is subject to the conditions detailed in the
 * LICENCE file provided in this distribution.
 *
 * Minimal SOAP implementation for UPnP protocol.
 */
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#include <winsock2.h>
#include "win32_snprintf.h"
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#endif
#include "minisoap.h"
#include "miniupnpcstrings.h"

/* only for malloc */
#include <stdlib.h>

/* httpWrite sends the headers and the body to the socket
 * and returns the number of bytes sent */
static int
httpWrite(SOCKET fd, const char * body, int bodysize,
          const char * headers, int headerssize)
{
	int n = 0;
	/*n = write(fd, headers, headerssize);*/
	/*if(bodysize>0)
		n += write(fd, body, bodysize);*/
	/* Note : my old linksys router only took into account
	 * soap request that are sent into only one packet */
	char * p;
	/* TODO: AVOID MALLOC, we could use writev() for that */
	p = malloc(headerssize+bodysize);
	if(!p)
	  return -1;
	memcpy(p, headers, headerssize);
	memcpy(p+headerssize, body, bodysize);
	/*n = write(fd, p, headerssize+bodysize);*/
	n = send(fd, p, headerssize+bodysize, 0);
	if(n<0) {
	  PRINT_SOCKET_ERROR("send");
	}
	/* disable send on the socket */
	/* draytek routers don't seem to like that... */
#if 0
#ifdef _WIN32
	if(shutdown(fd, SD_SEND)<0) {
#else
	if(shutdown(fd, SHUT_WR)<0)	{ /*SD_SEND*/
#endif
		PRINT_SOCKET_ERROR("shutdown");
	}
#endif
	free(p);
	return n;
}

/* self explanatory  */
int soapPostSubmit(SOCKET fd,
                   const char * url,
				   const char * host,
				   unsigned short port,
				   const char * action,
				   const char * body,
				   const char * httpversion)
{
	char headerbuf[512];
	int headerssize;
	char portstr[8];
	int bodysize = (int)strlen(body);
	/* We are not using keep-alive HTTP connections.
	 * HTTP/1.1 needs the header Connection: close to do that.
	 * This is the default with HTTP/1.0
	 * Using HTTP/1.1 means we need to support chunked transfer-encoding :
	 * When using HTTP/1.1, the router "BiPAC 7404VNOX" always use chunked
	 * transfer encoding. */
    /* Connection: Close is normally there only in HTTP/1.1 but who knows */
	portstr[0] = '\0';
	if(port != 80)
		snprintf(portstr, sizeof(portstr), ":%hu", port);
	headerssize = snprintf(headerbuf, sizeof(headerbuf),
                       "POST %s HTTP/%s\r\n"
	                   "Host: %s%s\r\n"
					   "User-Agent: " OS_STRING " " UPNP_VERSION_STRING " MiniUPnPc/" MINIUPNPC_VERSION_STRING "\r\n"
	                   "Content-Length: %d\r\n"
#if (UPNP_VERSION_MAJOR == 1) && (UPNP_VERSION_MINOR == 0)
					   "Content-Type: text/xml\r\n"
#else
					   "Content-Type: text/xml; charset=\"utf-8\"\r\n"
#endif
					   "SOAPAction: \"%s\"\r\n"
					   "Connection: Close\r\n"
					   "Cache-Control: no-cache\r\n"	/* ??? */
					   "Pragma: no-cache\r\n"
					   "\r\n",
					   url, httpversion, host, portstr, bodysize, action);
	if ((unsigned int)headerssize >= sizeof(headerbuf))
		return -1;
#ifdef DEBUG
	/*printf("SOAP request : headersize=%d bodysize=%d\n",
	       headerssize, bodysize);
	*/
	printf("SOAP request : POST %s HTTP/%s - Host: %s%s\n",
	        url, httpversion, host, portstr);
	printf("SOAPAction: \"%s\" - Content-Length: %d\n", action, bodysize);
	printf("Headers :\n%s", headerbuf);
	printf("Body :\n%s\n", body);
#endif
	return httpWrite(fd, body, bodysize, headerbuf, headerssize);
}


