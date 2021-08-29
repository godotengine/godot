#ifndef HEADER_CURL_HOSTIP_H
#define HEADER_CURL_HOSTIP_H
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
#include "hash.h"
#include "curl_addrinfo.h"
#include "timeval.h" /* for timediff_t */
#include "asyn.h"

#ifdef HAVE_SETJMP_H
#include <setjmp.h>
#endif

#ifdef NETWARE
#undef in_addr_t
#define in_addr_t unsigned long
#endif

/* Allocate enough memory to hold the full name information structs and
 * everything. OSF1 is known to require at least 8872 bytes. The buffer
 * required for storing all possible aliases and IP numbers is according to
 * Stevens' Unix Network Programming 2nd edition, p. 304: 8192 bytes!
 */
#define CURL_HOSTENT_SIZE 9000

#define CURL_TIMEOUT_RESOLVE 300 /* when using asynch methods, we allow this
                                    many seconds for a name resolve */

#define CURL_ASYNC_SUCCESS CURLE_OK

struct addrinfo;
struct hostent;
struct Curl_easy;
struct connectdata;

/*
 * Curl_global_host_cache_init() initializes and sets up a global DNS cache.
 * Global DNS cache is general badness. Do not use. This will be removed in
 * a future version. Use the share interface instead!
 *
 * Returns a struct Curl_hash pointer on success, NULL on failure.
 */
struct Curl_hash *Curl_global_host_cache_init(void);

struct Curl_dns_entry {
  struct Curl_addrinfo *addr;
  /* timestamp == 0 -- permanent CURLOPT_RESOLVE entry (doesn't time out) */
  time_t timestamp;
  /* use-counter, use Curl_resolv_unlock to release reference */
  long inuse;
};

bool Curl_host_is_ipnum(const char *hostname);

/*
 * Curl_resolv() returns an entry with the info for the specified host
 * and port.
 *
 * The returned data *MUST* be "unlocked" with Curl_resolv_unlock() after
 * use, or we'll leak memory!
 */
/* return codes */
enum resolve_t {
  CURLRESOLV_TIMEDOUT = -2,
  CURLRESOLV_ERROR    = -1,
  CURLRESOLV_RESOLVED =  0,
  CURLRESOLV_PENDING  =  1
};
enum resolve_t Curl_resolv(struct Curl_easy *data,
                           const char *hostname,
                           int port,
                           bool allowDOH,
                           struct Curl_dns_entry **dnsentry);
enum resolve_t Curl_resolv_timeout(struct Curl_easy *data,
                                   const char *hostname, int port,
                                   struct Curl_dns_entry **dnsentry,
                                   timediff_t timeoutms);

#ifdef ENABLE_IPV6
/*
 * Curl_ipv6works() returns TRUE if IPv6 seems to work.
 */
bool Curl_ipv6works(struct Curl_easy *data);
#else
#define Curl_ipv6works(x) FALSE
#endif

/*
 * Curl_ipvalid() checks what CURL_IPRESOLVE_* requirements that might've
 * been set and returns TRUE if they are OK.
 */
bool Curl_ipvalid(struct Curl_easy *data, struct connectdata *conn);


/*
 * Curl_getaddrinfo() is the generic low-level name resolve API within this
 * source file. There are several versions of this function - for different
 * name resolve layers (selected at build-time). They all take this same set
 * of arguments
 */
struct Curl_addrinfo *Curl_getaddrinfo(struct Curl_easy *data,
                                       const char *hostname,
                                       int port,
                                       int *waitp);


/* unlock a previously resolved dns entry */
void Curl_resolv_unlock(struct Curl_easy *data,
                        struct Curl_dns_entry *dns);

/* init a new dns cache and return success */
int Curl_mk_dnscache(struct Curl_hash *hash);

/* prune old entries from the DNS cache */
void Curl_hostcache_prune(struct Curl_easy *data);

/* Return # of addresses in a Curl_addrinfo struct */
int Curl_num_addresses(const struct Curl_addrinfo *addr);

/* IPv4 threadsafe resolve function used for synch and asynch builds */
struct Curl_addrinfo *Curl_ipv4_resolve_r(const char *hostname, int port);

CURLcode Curl_once_resolved(struct Curl_easy *data, bool *protocol_connect);

/*
 * Curl_addrinfo_callback() is used when we build with any asynch specialty.
 * Handles end of async request processing. Inserts ai into hostcache when
 * status is CURL_ASYNC_SUCCESS. Twiddles fields in conn to indicate async
 * request completed whether successful or failed.
 */
CURLcode Curl_addrinfo_callback(struct Curl_easy *data,
                                int status,
                                struct Curl_addrinfo *ai);

/*
 * Curl_printable_address() returns a printable version of the 1st address
 * given in the 'ip' argument. The result will be stored in the buf that is
 * bufsize bytes big.
 */
void Curl_printable_address(const struct Curl_addrinfo *ip,
                            char *buf, size_t bufsize);

/*
 * Curl_fetch_addr() fetches a 'Curl_dns_entry' already in the DNS cache.
 *
 * Returns the Curl_dns_entry entry pointer or NULL if not in the cache.
 *
 * The returned data *MUST* be "unlocked" with Curl_resolv_unlock() after
 * use, or we'll leak memory!
 */
struct Curl_dns_entry *
Curl_fetch_addr(struct Curl_easy *data,
                const char *hostname,
                int port);

/*
 * Curl_cache_addr() stores a 'Curl_addrinfo' struct in the DNS cache.
 *
 * Returns the Curl_dns_entry entry pointer or NULL if the storage failed.
 */
struct Curl_dns_entry *
Curl_cache_addr(struct Curl_easy *data, struct Curl_addrinfo *addr,
                const char *hostname, int port);

#ifndef INADDR_NONE
#define CURL_INADDR_NONE (in_addr_t) ~0
#else
#define CURL_INADDR_NONE INADDR_NONE
#endif

#ifdef HAVE_SIGSETJMP
/* Forward-declaration of variable defined in hostip.c. Beware this
 * is a global and unique instance. This is used to store the return
 * address that we can jump back to from inside a signal handler.
 * This is not thread-safe stuff.
 */
extern sigjmp_buf curl_jmpenv;
#endif

/*
 * Function provided by the resolver backend to set DNS servers to use.
 */
CURLcode Curl_set_dns_servers(struct Curl_easy *data, char *servers);

/*
 * Function provided by the resolver backend to set
 * outgoing interface to use for DNS requests
 */
CURLcode Curl_set_dns_interface(struct Curl_easy *data,
                                const char *interf);

/*
 * Function provided by the resolver backend to set
 * local IPv4 address to use as source address for DNS requests
 */
CURLcode Curl_set_dns_local_ip4(struct Curl_easy *data,
                                const char *local_ip4);

/*
 * Function provided by the resolver backend to set
 * local IPv6 address to use as source address for DNS requests
 */
CURLcode Curl_set_dns_local_ip6(struct Curl_easy *data,
                                const char *local_ip6);

/*
 * Clean off entries from the cache
 */
void Curl_hostcache_clean(struct Curl_easy *data, struct Curl_hash *hash);

/*
 * Populate the cache with specified entries from CURLOPT_RESOLVE.
 */
CURLcode Curl_loadhostpairs(struct Curl_easy *data);
CURLcode Curl_resolv_check(struct Curl_easy *data,
                           struct Curl_dns_entry **dns);
int Curl_resolv_getsock(struct Curl_easy *data,
                        curl_socket_t *socks);

CURLcode Curl_resolver_error(struct Curl_easy *data);
#endif /* HEADER_CURL_HOSTIP_H */
