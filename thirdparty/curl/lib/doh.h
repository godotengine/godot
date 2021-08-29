#ifndef HEADER_CURL_DOH_H
#define HEADER_CURL_DOH_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2018 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
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

#include "urldata.h"
#include "curl_addrinfo.h"

#ifndef CURL_DISABLE_DOH

/*
 * Curl_doh() resolve a name using DoH (DNS-over-HTTPS). It resolves a name
 * and returns a 'Curl_addrinfo *' with the address information.
 */

struct Curl_addrinfo *Curl_doh(struct Curl_easy *data,
                               const char *hostname,
                               int port,
                               int *waitp);

CURLcode Curl_doh_is_resolved(struct Curl_easy *data,
                              struct Curl_dns_entry **dns);

int Curl_doh_getsock(struct connectdata *conn, curl_socket_t *socks);

typedef enum {
  DOH_OK,
  DOH_DNS_BAD_LABEL,    /* 1 */
  DOH_DNS_OUT_OF_RANGE, /* 2 */
  DOH_DNS_LABEL_LOOP,   /* 3 */
  DOH_TOO_SMALL_BUFFER, /* 4 */
  DOH_OUT_OF_MEM,       /* 5 */
  DOH_DNS_RDATA_LEN,    /* 6 */
  DOH_DNS_MALFORMAT,    /* 7 */
  DOH_DNS_BAD_RCODE,    /* 8 - no such name */
  DOH_DNS_UNEXPECTED_TYPE,  /* 9 */
  DOH_DNS_UNEXPECTED_CLASS, /* 10 */
  DOH_NO_CONTENT,           /* 11 */
  DOH_DNS_BAD_ID,           /* 12 */
  DOH_DNS_NAME_TOO_LONG     /* 13 */
} DOHcode;

typedef enum {
  DNS_TYPE_A = 1,
  DNS_TYPE_NS = 2,
  DNS_TYPE_CNAME = 5,
  DNS_TYPE_AAAA = 28,
  DNS_TYPE_DNAME = 39           /* RFC6672 */
} DNStype;

#define DOH_MAX_ADDR 24
#define DOH_MAX_CNAME 4

struct dohaddr {
  int type;
  union {
    unsigned char v4[4]; /* network byte order */
    unsigned char v6[16];
  } ip;
};

struct dohentry {
  struct dynbuf cname[DOH_MAX_CNAME];
  struct dohaddr addr[DOH_MAX_ADDR];
  int numaddr;
  unsigned int ttl;
  int numcname;
};


#ifdef DEBUGBUILD
DOHcode doh_encode(const char *host,
                   DNStype dnstype,
                   unsigned char *dnsp, /* buffer */
                   size_t len,  /* buffer size */
                   size_t *olen); /* output length */
DOHcode doh_decode(const unsigned char *doh,
                   size_t dohlen,
                   DNStype dnstype,
                   struct dohentry *d);
void de_init(struct dohentry *d);
void de_cleanup(struct dohentry *d);
#endif

#else /* if DoH is disabled */
#define Curl_doh(a,b,c,d) NULL
#define Curl_doh_is_resolved(x,y) CURLE_COULDNT_RESOLVE_HOST
#endif

#endif /* HEADER_CURL_DOH_H */
