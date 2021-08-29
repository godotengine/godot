#ifndef HEADER_CURL_CONNECT_H
#define HEADER_CURL_CONNECT_H
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

#include "nonblock.h" /* for curlx_nonblock(), formerly Curl_nonblock() */
#include "sockaddr.h"
#include "timeval.h"

CURLcode Curl_is_connected(struct Curl_easy *data,
                           struct connectdata *conn,
                           int sockindex,
                           bool *connected);

CURLcode Curl_connecthost(struct Curl_easy *data,
                          struct connectdata *conn,
                          const struct Curl_dns_entry *host);

/* generic function that returns how much time there's left to run, according
   to the timeouts set */
timediff_t Curl_timeleft(struct Curl_easy *data,
                         struct curltime *nowp,
                         bool duringconnect);

#define DEFAULT_CONNECT_TIMEOUT 300000 /* milliseconds == five minutes */

/*
 * Used to extract socket and connectdata struct for the most recent
 * transfer on the given Curl_easy.
 *
 * The returned socket will be CURL_SOCKET_BAD in case of failure!
 */
curl_socket_t Curl_getconnectinfo(struct Curl_easy *data,
                                  struct connectdata **connp);

bool Curl_addr2string(struct sockaddr *sa, curl_socklen_t salen,
                      char *addr, int *port);

/*
 * Check if a connection seems to be alive.
 */
bool Curl_connalive(struct connectdata *conn);

#ifdef USE_WINSOCK
/* When you run a program that uses the Windows Sockets API, you may
   experience slow performance when you copy data to a TCP server.

   https://support.microsoft.com/kb/823764

   Work-around: Make the Socket Send Buffer Size Larger Than the Program Send
   Buffer Size

*/
void Curl_sndbufset(curl_socket_t sockfd);
#else
#define Curl_sndbufset(y) Curl_nop_stmt
#endif

void Curl_updateconninfo(struct Curl_easy *data, struct connectdata *conn,
                         curl_socket_t sockfd);
void Curl_conninfo_remote(struct Curl_easy *data, struct connectdata *conn,
                          curl_socket_t sockfd);
void Curl_conninfo_local(struct Curl_easy *data, curl_socket_t sockfd,
                         char *local_ip, int *local_port);
void Curl_persistconninfo(struct Curl_easy *data, struct connectdata *conn,
                          char *local_ip, int local_port);
int Curl_closesocket(struct Curl_easy *data, struct connectdata *conn,
                     curl_socket_t sock);

/*
 * The Curl_sockaddr_ex structure is basically libcurl's external API
 * curl_sockaddr structure with enough space available to directly hold any
 * protocol-specific address structures. The variable declared here will be
 * used to pass / receive data to/from the fopensocket callback if this has
 * been set, before that, it is initialized from parameters.
 */
struct Curl_sockaddr_ex {
  int family;
  int socktype;
  int protocol;
  unsigned int addrlen;
  union {
    struct sockaddr addr;
    struct Curl_sockaddr_storage buff;
  } _sa_ex_u;
};
#define sa_addr _sa_ex_u.addr

/*
 * Create a socket based on info from 'conn' and 'ai'.
 *
 * Fill in 'addr' and 'sockfd' accordingly if OK is returned. If the open
 * socket callback is set, used that!
 *
 */
CURLcode Curl_socket(struct Curl_easy *data,
                     const struct Curl_addrinfo *ai,
                     struct Curl_sockaddr_ex *addr,
                     curl_socket_t *sockfd);

/*
 * Curl_conncontrol() marks the end of a connection/stream. The 'closeit'
 * argument specifies if it is the end of a connection or a stream.
 *
 * For stream-based protocols (such as HTTP/2), a stream close will not cause
 * a connection close. Other protocols will close the connection for both
 * cases.
 *
 * It sets the bit.close bit to TRUE (with an explanation for debug builds),
 * when the connection will close.
 */

#define CONNCTRL_KEEP 0 /* undo a marked closure */
#define CONNCTRL_CONNECTION 1
#define CONNCTRL_STREAM 2

void Curl_conncontrol(struct connectdata *conn,
                      int closeit
#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
                      , const char *reason
#endif
  );

#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
#define streamclose(x,y) Curl_conncontrol(x, CONNCTRL_STREAM, y)
#define connclose(x,y) Curl_conncontrol(x, CONNCTRL_CONNECTION, y)
#define connkeep(x,y) Curl_conncontrol(x, CONNCTRL_KEEP, y)
#else /* if !DEBUGBUILD || CURL_DISABLE_VERBOSE_STRINGS */
#define streamclose(x,y) Curl_conncontrol(x, CONNCTRL_STREAM)
#define connclose(x,y) Curl_conncontrol(x, CONNCTRL_CONNECTION)
#define connkeep(x,y) Curl_conncontrol(x, CONNCTRL_KEEP)
#endif

bool Curl_conn_data_pending(struct connectdata *conn, int sockindex);

#endif /* HEADER_CURL_CONNECT_H */
