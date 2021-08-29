#ifndef HEADER_CURL_SELECT_H
#define HEADER_CURL_SELECT_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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

#ifdef HAVE_POLL_H
#include <poll.h>
#elif defined(HAVE_SYS_POLL_H)
#include <sys/poll.h>
#endif

/*
 * Definition of pollfd struct and constants for platforms lacking them.
 */

#if !defined(HAVE_STRUCT_POLLFD) && \
    !defined(HAVE_SYS_POLL_H) && \
    !defined(HAVE_POLL_H) && \
    !defined(POLLIN)

#define POLLIN      0x01
#define POLLPRI     0x02
#define POLLOUT     0x04
#define POLLERR     0x08
#define POLLHUP     0x10
#define POLLNVAL    0x20

struct pollfd
{
    curl_socket_t fd;
    short   events;
    short   revents;
};

#endif

#ifndef POLLRDNORM
#define POLLRDNORM POLLIN
#endif

#ifndef POLLWRNORM
#define POLLWRNORM POLLOUT
#endif

#ifndef POLLRDBAND
#define POLLRDBAND POLLPRI
#endif

/* there are three CSELECT defines that are defined in the public header that
   are exposed to users, but this *IN2 bit is only ever used internally and
   therefore defined here */
#define CURL_CSELECT_IN2 (CURL_CSELECT_ERR << 1)

int Curl_socket_check(curl_socket_t readfd, curl_socket_t readfd2,
                      curl_socket_t writefd,
                      timediff_t timeout_ms);
#define SOCKET_READABLE(x,z) \
  Curl_socket_check(x, CURL_SOCKET_BAD, CURL_SOCKET_BAD, z)
#define SOCKET_WRITABLE(x,z) \
  Curl_socket_check(CURL_SOCKET_BAD, CURL_SOCKET_BAD, x, z)

int Curl_poll(struct pollfd ufds[], unsigned int nfds, timediff_t timeout_ms);
int Curl_wait_ms(timediff_t timeout_ms);

#ifdef TPF
int tpf_select_libcurl(int maxfds, fd_set* reads, fd_set* writes,
                       fd_set* excepts, struct timeval *tv);
#endif

/* TPF sockets are not in range [0..FD_SETSIZE-1], which
   unfortunately makes it impossible for us to easily check if they're valid

   With Winsock the valid range is [0..INVALID_SOCKET-1] according to
   https://docs.microsoft.com/en-us/windows/win32/winsock/socket-data-type-2
*/
#if defined(TPF)
#define VALID_SOCK(x) 1
#define VERIFY_SOCK(x) Curl_nop_stmt
#define FDSET_SOCK(x) 1
#elif defined(USE_WINSOCK)
#define VALID_SOCK(s) ((s) < INVALID_SOCKET)
#define FDSET_SOCK(x) 1
#define VERIFY_SOCK(x) do { \
  if(!VALID_SOCK(x)) { \
    SET_SOCKERRNO(WSAEINVAL); \
    return -1; \
  } \
} while(0)
#else
#define VALID_SOCK(s) ((s) >= 0)

/* If the socket is small enough to get set or read from an fdset */
#define FDSET_SOCK(s) ((s) < FD_SETSIZE)

#define VERIFY_SOCK(x) do {                     \
    if(!VALID_SOCK(x) || !FDSET_SOCK(x)) {      \
      SET_SOCKERRNO(EINVAL);                    \
      return -1;                                \
    }                                           \
  } while(0)
#endif

#endif /* HEADER_CURL_SELECT_H */
