/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2019 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/

#include "curl_setup.h"
#include "socketpair.h"

#if !defined(HAVE_SOCKETPAIR) && !defined(CURL_DISABLE_SOCKETPAIR)
#ifdef WIN32
/*
 * This is a socketpair() implementation for Windows.
 */
#include <string.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <io.h>
#else
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h> /* IPPROTO_TCP */
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifndef INADDR_LOOPBACK
#define INADDR_LOOPBACK 0x7f000001
#endif /* !INADDR_LOOPBACK */
#endif /* !WIN32 */

#include "nonblock.h" /* for curlx_nonblock */
#include "timeval.h"  /* needed before select.h */
#include "select.h"   /* for Curl_poll */

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

int Curl_socketpair(int domain, int type, int protocol,
                    curl_socket_t socks[2])
{
  union {
    struct sockaddr_in inaddr;
    struct sockaddr addr;
  } a, a2;
  curl_socket_t listener;
  curl_socklen_t addrlen = sizeof(a.inaddr);
  int reuse = 1;
  struct pollfd pfd[1];
  (void)domain;
  (void)type;
  (void)protocol;

  listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if(listener == CURL_SOCKET_BAD)
    return -1;

  memset(&a, 0, sizeof(a));
  a.inaddr.sin_family = AF_INET;
  a.inaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  a.inaddr.sin_port = 0;

  socks[0] = socks[1] = CURL_SOCKET_BAD;

  if(setsockopt(listener, SOL_SOCKET, SO_REUSEADDR,
                (char *)&reuse, (curl_socklen_t)sizeof(reuse)) == -1)
    goto error;
  if(bind(listener, &a.addr, sizeof(a.inaddr)) == -1)
    goto error;
  if(getsockname(listener, &a.addr, &addrlen) == -1 ||
     addrlen < (int)sizeof(a.inaddr))
    goto error;
  if(listen(listener, 1) == -1)
    goto error;
  socks[0] = socket(AF_INET, SOCK_STREAM, 0);
  if(socks[0] == CURL_SOCKET_BAD)
    goto error;
  if(connect(socks[0], &a.addr, sizeof(a.inaddr)) == -1)
    goto error;

  /* use non-blocking accept to make sure we don't block forever */
  if(curlx_nonblock(listener, TRUE) < 0)
    goto error;
  pfd[0].fd = listener;
  pfd[0].events = POLLIN;
  pfd[0].revents = 0;
  (void)Curl_poll(pfd, 1, 10*1000); /* 10 seconds */
  socks[1] = accept(listener, NULL, NULL);
  if(socks[1] == CURL_SOCKET_BAD)
    goto error;

  /* verify that nothing else connected */
  addrlen = sizeof(a.inaddr);
  if(getsockname(socks[0], &a.addr, &addrlen) == -1 ||
     addrlen < (int)sizeof(a.inaddr))
    goto error;
  addrlen = sizeof(a2.inaddr);
  if(getpeername(socks[1], &a2.addr, &addrlen) == -1 ||
     addrlen < (int)sizeof(a2.inaddr))
    goto error;
  if(a.inaddr.sin_family != a2.inaddr.sin_family ||
     a.inaddr.sin_addr.s_addr != a2.inaddr.sin_addr.s_addr ||
     a.inaddr.sin_port != a2.inaddr.sin_port)
    goto error;

  sclose(listener);
  return 0;

  error:
  sclose(listener);
  sclose(socks[0]);
  sclose(socks[1]);
  return -1;
}

#endif /* ! HAVE_SOCKETPAIR */
