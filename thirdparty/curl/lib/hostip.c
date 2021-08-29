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

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETINET_IN6_H
#include <netinet/in6.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef __VMS
#include <in.h>
#include <inet.h>
#endif

#ifdef HAVE_SETJMP_H
#include <setjmp.h>
#endif
#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

#ifdef HAVE_PROCESS_H
#include <process.h>
#endif

#include "urldata.h"
#include "sendf.h"
#include "hostip.h"
#include "hash.h"
#include "rand.h"
#include "share.h"
#include "url.h"
#include "inet_ntop.h"
#include "inet_pton.h"
#include "multiif.h"
#include "doh.h"
#include "warnless.h"
#include "strcase.h"
/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

#if defined(ENABLE_IPV6) && defined(CURL_OSX_CALL_COPYPROXIES)
#include <SystemConfiguration/SCDynamicStoreCopySpecific.h>
#endif

#if defined(CURLRES_SYNCH) && \
    defined(HAVE_ALARM) && defined(SIGALRM) && defined(HAVE_SIGSETJMP)
/* alarm-based timeouts can only be used with all the dependencies satisfied */
#define USE_ALARM_TIMEOUT
#endif

#define MAX_HOSTCACHE_LEN (255 + 7) /* max FQDN + colon + port number + zero */

/*
 * hostip.c explained
 * ==================
 *
 * The main COMPILE-TIME DEFINES to keep in mind when reading the host*.c
 * source file are these:
 *
 * CURLRES_IPV6 - this host has getaddrinfo() and family, and thus we use
 * that. The host may not be able to resolve IPv6, but we don't really have to
 * take that into account. Hosts that aren't IPv6-enabled have CURLRES_IPV4
 * defined.
 *
 * CURLRES_ARES - is defined if libcurl is built to use c-ares for
 * asynchronous name resolves. This can be Windows or *nix.
 *
 * CURLRES_THREADED - is defined if libcurl is built to run under (native)
 * Windows, and then the name resolve will be done in a new thread, and the
 * supported API will be the same as for ares-builds.
 *
 * If any of the two previous are defined, CURLRES_ASYNCH is defined too. If
 * libcurl is not built to use an asynchronous resolver, CURLRES_SYNCH is
 * defined.
 *
 * The host*.c sources files are split up like this:
 *
 * hostip.c   - method-independent resolver functions and utility functions
 * hostasyn.c - functions for asynchronous name resolves
 * hostsyn.c  - functions for synchronous name resolves
 * hostip4.c  - IPv4 specific functions
 * hostip6.c  - IPv6 specific functions
 *
 * The two asynchronous name resolver backends are implemented in:
 * asyn-ares.c   - functions for ares-using name resolves
 * asyn-thread.c - functions for threaded name resolves

 * The hostip.h is the united header file for all this. It defines the
 * CURLRES_* defines based on the config*.h and curl_setup.h defines.
 */

static void freednsentry(void *freethis);

/*
 * Return # of addresses in a Curl_addrinfo struct
 */
int Curl_num_addresses(const struct Curl_addrinfo *addr)
{
  int i = 0;
  while(addr) {
    addr = addr->ai_next;
    i++;
  }
  return i;
}

/*
 * Curl_printable_address() stores a printable version of the 1st address
 * given in the 'ai' argument. The result will be stored in the buf that is
 * bufsize bytes big.
 *
 * If the conversion fails, the target buffer is empty.
 */
void Curl_printable_address(const struct Curl_addrinfo *ai, char *buf,
                            size_t bufsize)
{
  DEBUGASSERT(bufsize);
  buf[0] = 0;

  switch(ai->ai_family) {
  case AF_INET: {
    const struct sockaddr_in *sa4 = (const void *)ai->ai_addr;
    const struct in_addr *ipaddr4 = &sa4->sin_addr;
    (void)Curl_inet_ntop(ai->ai_family, (const void *)ipaddr4, buf, bufsize);
    break;
  }
#ifdef ENABLE_IPV6
  case AF_INET6: {
    const struct sockaddr_in6 *sa6 = (const void *)ai->ai_addr;
    const struct in6_addr *ipaddr6 = &sa6->sin6_addr;
    (void)Curl_inet_ntop(ai->ai_family, (const void *)ipaddr6, buf, bufsize);
    break;
  }
#endif
  default:
    break;
  }
}

/*
 * Create a hostcache id string for the provided host + port, to be used by
 * the DNS caching. Without alloc.
 */
static void
create_hostcache_id(const char *name, int port, char *ptr, size_t buflen)
{
  size_t len = strlen(name);
  if(len > (buflen - 7))
    len = buflen - 7;
  /* store and lower case the name */
  while(len--)
    *ptr++ = (char)TOLOWER(*name++);
  msnprintf(ptr, 7, ":%u", port);
}

struct hostcache_prune_data {
  long cache_timeout;
  time_t now;
};

/*
 * This function is set as a callback to be called for every entry in the DNS
 * cache when we want to prune old unused entries.
 *
 * Returning non-zero means remove the entry, return 0 to keep it in the
 * cache.
 */
static int
hostcache_timestamp_remove(void *datap, void *hc)
{
  struct hostcache_prune_data *data =
    (struct hostcache_prune_data *) datap;
  struct Curl_dns_entry *c = (struct Curl_dns_entry *) hc;

  return (0 != c->timestamp)
    && (data->now - c->timestamp >= data->cache_timeout);
}

/*
 * Prune the DNS cache. This assumes that a lock has already been taken.
 */
static void
hostcache_prune(struct Curl_hash *hostcache, long cache_timeout, time_t now)
{
  struct hostcache_prune_data user;

  user.cache_timeout = cache_timeout;
  user.now = now;

  Curl_hash_clean_with_criterium(hostcache,
                                 (void *) &user,
                                 hostcache_timestamp_remove);
}

/*
 * Library-wide function for pruning the DNS cache. This function takes and
 * returns the appropriate locks.
 */
void Curl_hostcache_prune(struct Curl_easy *data)
{
  time_t now;

  if((data->set.dns_cache_timeout == -1) || !data->dns.hostcache)
    /* cache forever means never prune, and NULL hostcache means
       we can't do it */
    return;

  if(data->share)
    Curl_share_lock(data, CURL_LOCK_DATA_DNS, CURL_LOCK_ACCESS_SINGLE);

  time(&now);

  /* Remove outdated and unused entries from the hostcache */
  hostcache_prune(data->dns.hostcache,
                  data->set.dns_cache_timeout,
                  now);

  if(data->share)
    Curl_share_unlock(data, CURL_LOCK_DATA_DNS);
}

#ifdef HAVE_SIGSETJMP
/* Beware this is a global and unique instance. This is used to store the
   return address that we can jump back to from inside a signal handler. This
   is not thread-safe stuff. */
sigjmp_buf curl_jmpenv;
#endif

/* lookup address, returns entry if found and not stale */
static struct Curl_dns_entry *fetch_addr(struct Curl_easy *data,
                                         const char *hostname,
                                         int port)
{
  struct Curl_dns_entry *dns = NULL;
  size_t entry_len;
  char entry_id[MAX_HOSTCACHE_LEN];

  /* Create an entry id, based upon the hostname and port */
  create_hostcache_id(hostname, port, entry_id, sizeof(entry_id));
  entry_len = strlen(entry_id);

  /* See if its already in our dns cache */
  dns = Curl_hash_pick(data->dns.hostcache, entry_id, entry_len + 1);

  /* No entry found in cache, check if we might have a wildcard entry */
  if(!dns && data->state.wildcard_resolve) {
    create_hostcache_id("*", port, entry_id, sizeof(entry_id));
    entry_len = strlen(entry_id);

    /* See if it's already in our dns cache */
    dns = Curl_hash_pick(data->dns.hostcache, entry_id, entry_len + 1);
  }

  if(dns && (data->set.dns_cache_timeout != -1)) {
    /* See whether the returned entry is stale. Done before we release lock */
    struct hostcache_prune_data user;

    time(&user.now);
    user.cache_timeout = data->set.dns_cache_timeout;

    if(hostcache_timestamp_remove(&user, dns)) {
      infof(data, "Hostname in DNS cache was stale, zapped");
      dns = NULL; /* the memory deallocation is being handled by the hash */
      Curl_hash_delete(data->dns.hostcache, entry_id, entry_len + 1);
    }
  }

  return dns;
}

/*
 * Curl_fetch_addr() fetches a 'Curl_dns_entry' already in the DNS cache.
 *
 * Curl_resolv() checks initially and multi_runsingle() checks each time
 * it discovers the handle in the state WAITRESOLVE whether the hostname
 * has already been resolved and the address has already been stored in
 * the DNS cache. This short circuits waiting for a lot of pending
 * lookups for the same hostname requested by different handles.
 *
 * Returns the Curl_dns_entry entry pointer or NULL if not in the cache.
 *
 * The returned data *MUST* be "unlocked" with Curl_resolv_unlock() after
 * use, or we'll leak memory!
 */
struct Curl_dns_entry *
Curl_fetch_addr(struct Curl_easy *data,
                const char *hostname,
                int port)
{
  struct Curl_dns_entry *dns = NULL;

  if(data->share)
    Curl_share_lock(data, CURL_LOCK_DATA_DNS, CURL_LOCK_ACCESS_SINGLE);

  dns = fetch_addr(data, hostname, port);

  if(dns)
    dns->inuse++; /* we use it! */

  if(data->share)
    Curl_share_unlock(data, CURL_LOCK_DATA_DNS);

  return dns;
}

#ifndef CURL_DISABLE_SHUFFLE_DNS
UNITTEST CURLcode Curl_shuffle_addr(struct Curl_easy *data,
                                    struct Curl_addrinfo **addr);
/*
 * Curl_shuffle_addr() shuffles the order of addresses in a 'Curl_addrinfo'
 * struct by re-linking its linked list.
 *
 * The addr argument should be the address of a pointer to the head node of a
 * `Curl_addrinfo` list and it will be modified to point to the new head after
 * shuffling.
 *
 * Not declared static only to make it easy to use in a unit test!
 *
 * @unittest: 1608
 */
UNITTEST CURLcode Curl_shuffle_addr(struct Curl_easy *data,
                                    struct Curl_addrinfo **addr)
{
  CURLcode result = CURLE_OK;
  const int num_addrs = Curl_num_addresses(*addr);

  if(num_addrs > 1) {
    struct Curl_addrinfo **nodes;
    infof(data, "Shuffling %i addresses", num_addrs);

    nodes = malloc(num_addrs*sizeof(*nodes));
    if(nodes) {
      int i;
      unsigned int *rnd;
      const size_t rnd_size = num_addrs * sizeof(*rnd);

      /* build a plain array of Curl_addrinfo pointers */
      nodes[0] = *addr;
      for(i = 1; i < num_addrs; i++) {
        nodes[i] = nodes[i-1]->ai_next;
      }

      rnd = malloc(rnd_size);
      if(rnd) {
        /* Fisher-Yates shuffle */
        if(Curl_rand(data, (unsigned char *)rnd, rnd_size) == CURLE_OK) {
          struct Curl_addrinfo *swap_tmp;
          for(i = num_addrs - 1; i > 0; i--) {
            swap_tmp = nodes[rnd[i] % (i + 1)];
            nodes[rnd[i] % (i + 1)] = nodes[i];
            nodes[i] = swap_tmp;
          }

          /* relink list in the new order */
          for(i = 1; i < num_addrs; i++) {
            nodes[i-1]->ai_next = nodes[i];
          }

          nodes[num_addrs-1]->ai_next = NULL;
          *addr = nodes[0];
        }
        free(rnd);
      }
      else
        result = CURLE_OUT_OF_MEMORY;
      free(nodes);
    }
    else
      result = CURLE_OUT_OF_MEMORY;
  }
  return result;
}
#endif

/*
 * Curl_cache_addr() stores a 'Curl_addrinfo' struct in the DNS cache.
 *
 * When calling Curl_resolv() has resulted in a response with a returned
 * address, we call this function to store the information in the dns
 * cache etc
 *
 * Returns the Curl_dns_entry entry pointer or NULL if the storage failed.
 */
struct Curl_dns_entry *
Curl_cache_addr(struct Curl_easy *data,
                struct Curl_addrinfo *addr,
                const char *hostname,
                int port)
{
  char entry_id[MAX_HOSTCACHE_LEN];
  size_t entry_len;
  struct Curl_dns_entry *dns;
  struct Curl_dns_entry *dns2;

#ifndef CURL_DISABLE_SHUFFLE_DNS
  /* shuffle addresses if requested */
  if(data->set.dns_shuffle_addresses) {
    CURLcode result = Curl_shuffle_addr(data, &addr);
    if(result)
      return NULL;
  }
#endif

  /* Create a new cache entry */
  dns = calloc(1, sizeof(struct Curl_dns_entry));
  if(!dns) {
    return NULL;
  }

  /* Create an entry id, based upon the hostname and port */
  create_hostcache_id(hostname, port, entry_id, sizeof(entry_id));
  entry_len = strlen(entry_id);

  dns->inuse = 1;   /* the cache has the first reference */
  dns->addr = addr; /* this is the address(es) */
  time(&dns->timestamp);
  if(dns->timestamp == 0)
    dns->timestamp = 1;   /* zero indicates permanent CURLOPT_RESOLVE entry */

  /* Store the resolved data in our DNS cache. */
  dns2 = Curl_hash_add(data->dns.hostcache, entry_id, entry_len + 1,
                       (void *)dns);
  if(!dns2) {
    free(dns);
    return NULL;
  }

  dns = dns2;
  dns->inuse++;         /* mark entry as in-use */
  return dns;
}

#ifdef ENABLE_IPV6
/* return a static IPv6 resolve for 'localhost' */
static struct Curl_addrinfo *get_localhost6(int port)
{
  struct Curl_addrinfo *ca;
  const size_t ss_size = sizeof(struct sockaddr_in6);
  const size_t hostlen = strlen("localhost");
  struct sockaddr_in6 sa6;
  unsigned char ipv6[16];
  unsigned short port16 = (unsigned short)(port & 0xffff);
  ca = calloc(sizeof(struct Curl_addrinfo) + ss_size + hostlen + 1, 1);
  if(!ca)
    return NULL;

  sa6.sin6_family = AF_INET6;
  sa6.sin6_port = htons(port16);
  sa6.sin6_flowinfo = 0;
  sa6.sin6_scope_id = 0;
  if(Curl_inet_pton(AF_INET6, "::1", ipv6) < 1)
    return NULL;
  memcpy(&sa6.sin6_addr, ipv6, sizeof(ipv6));

  ca->ai_flags     = 0;
  ca->ai_family    = AF_INET6;
  ca->ai_socktype  = SOCK_STREAM;
  ca->ai_protocol  = IPPROTO_TCP;
  ca->ai_addrlen   = (curl_socklen_t)ss_size;
  ca->ai_next      = NULL;
  ca->ai_addr = (void *)((char *)ca + sizeof(struct Curl_addrinfo));
  memcpy(ca->ai_addr, &sa6, ss_size);
  ca->ai_canonname = (char *)ca->ai_addr + ss_size;
  strcpy(ca->ai_canonname, "localhost");
  return ca;
}
#else
#define get_localhost6(x) NULL
#endif

/* return a static IPv4 resolve for 'localhost' */
static struct Curl_addrinfo *get_localhost(int port)
{
  struct Curl_addrinfo *ca;
  const size_t ss_size = sizeof(struct sockaddr_in);
  const size_t hostlen = strlen("localhost");
  struct sockaddr_in sa;
  unsigned int ipv4;
  unsigned short port16 = (unsigned short)(port & 0xffff);

  /* memset to clear the sa.sin_zero field */
  memset(&sa, 0, sizeof(sa));
  sa.sin_family = AF_INET;
  sa.sin_port = htons(port16);
  if(Curl_inet_pton(AF_INET, "127.0.0.1", (char *)&ipv4) < 1)
    return NULL;
  memcpy(&sa.sin_addr, &ipv4, sizeof(ipv4));

  ca = calloc(sizeof(struct Curl_addrinfo) + ss_size + hostlen + 1, 1);
  if(!ca)
    return NULL;
  ca->ai_flags     = 0;
  ca->ai_family    = AF_INET;
  ca->ai_socktype  = SOCK_STREAM;
  ca->ai_protocol  = IPPROTO_TCP;
  ca->ai_addrlen   = (curl_socklen_t)ss_size;
  ca->ai_addr = (void *)((char *)ca + sizeof(struct Curl_addrinfo));
  memcpy(ca->ai_addr, &sa, ss_size);
  ca->ai_canonname = (char *)ca->ai_addr + ss_size;
  strcpy(ca->ai_canonname, "localhost");
  ca->ai_next = get_localhost6(port);
  return ca;
}

#ifdef ENABLE_IPV6
/*
 * Curl_ipv6works() returns TRUE if IPv6 seems to work.
 */
bool Curl_ipv6works(struct Curl_easy *data)
{
  if(data) {
    /* the nature of most system is that IPv6 status doesn't come and go
       during a program's lifetime so we only probe the first time and then we
       have the info kept for fast re-use */
    DEBUGASSERT(data);
    DEBUGASSERT(data->multi);
    return data->multi->ipv6_works;
  }
  else {
    int ipv6_works = -1;
    /* probe to see if we have a working IPv6 stack */
    curl_socket_t s = socket(PF_INET6, SOCK_DGRAM, 0);
    if(s == CURL_SOCKET_BAD)
      /* an IPv6 address was requested but we can't get/use one */
      ipv6_works = 0;
    else {
      ipv6_works = 1;
      sclose(s);
    }
    return (ipv6_works>0)?TRUE:FALSE;
  }
}
#endif /* ENABLE_IPV6 */

/*
 * Curl_host_is_ipnum() returns TRUE if the given string is a numerical IPv4
 * (or IPv6 if supported) address.
 */
bool Curl_host_is_ipnum(const char *hostname)
{
  struct in_addr in;
#ifdef ENABLE_IPV6
  struct in6_addr in6;
#endif
  if(Curl_inet_pton(AF_INET, hostname, &in) > 0
#ifdef ENABLE_IPV6
     || Curl_inet_pton(AF_INET6, hostname, &in6) > 0
#endif
    )
    return TRUE;
  return FALSE;
}

/*
 * Curl_resolv() is the main name resolve function within libcurl. It resolves
 * a name and returns a pointer to the entry in the 'entry' argument (if one
 * is provided). This function might return immediately if we're using asynch
 * resolves. See the return codes.
 *
 * The cache entry we return will get its 'inuse' counter increased when this
 * function is used. You MUST call Curl_resolv_unlock() later (when you're
 * done using this struct) to decrease the counter again.
 *
 * Return codes:
 *
 * CURLRESOLV_ERROR   (-1) = error, no pointer
 * CURLRESOLV_RESOLVED (0) = OK, pointer provided
 * CURLRESOLV_PENDING  (1) = waiting for response, no pointer
 */

enum resolve_t Curl_resolv(struct Curl_easy *data,
                           const char *hostname,
                           int port,
                           bool allowDOH,
                           struct Curl_dns_entry **entry)
{
  struct Curl_dns_entry *dns = NULL;
  CURLcode result;
  enum resolve_t rc = CURLRESOLV_ERROR; /* default to failure */
  struct connectdata *conn = data->conn;
  *entry = NULL;
  conn->bits.doh = FALSE; /* default is not */

  if(data->share)
    Curl_share_lock(data, CURL_LOCK_DATA_DNS, CURL_LOCK_ACCESS_SINGLE);

  dns = fetch_addr(data, hostname, port);

  if(dns) {
    infof(data, "Hostname %s was found in DNS cache", hostname);
    dns->inuse++; /* we use it! */
    rc = CURLRESOLV_RESOLVED;
  }

  if(data->share)
    Curl_share_unlock(data, CURL_LOCK_DATA_DNS);

  if(!dns) {
    /* The entry was not in the cache. Resolve it to IP address */

    struct Curl_addrinfo *addr = NULL;
    int respwait = 0;
    struct in_addr in;
#ifndef USE_RESOLVE_ON_IPS
    const
#endif
      bool ipnum = FALSE;

    /* notify the resolver start callback */
    if(data->set.resolver_start) {
      int st;
      Curl_set_in_callback(data, true);
      st = data->set.resolver_start(
#ifdef USE_CURL_ASYNC
        data->state.async.resolver,
#else
        NULL,
#endif
        NULL,
        data->set.resolver_start_client);
      Curl_set_in_callback(data, false);
      if(st)
        return CURLRESOLV_ERROR;
    }

#if defined(ENABLE_IPV6) && defined(CURL_OSX_CALL_COPYPROXIES)
    {
      /*
       * The automagic conversion from IPv4 literals to IPv6 literals only
       * works if the SCDynamicStoreCopyProxies system function gets called
       * first. As Curl currently doesn't support system-wide HTTP proxies, we
       * therefore don't use any value this function might return.
       *
       * This function is only available on a macOS and is not needed for
       * IPv4-only builds, hence the conditions above.
       */
      CFDictionaryRef dict = SCDynamicStoreCopyProxies(NULL);
      if(dict)
        CFRelease(dict);
    }
#endif

#ifndef USE_RESOLVE_ON_IPS
    /* First check if this is an IPv4 address string */
    if(Curl_inet_pton(AF_INET, hostname, &in) > 0)
      /* This is a dotted IP address 123.123.123.123-style */
      addr = Curl_ip2addr(AF_INET, &in, hostname, port);
#ifdef ENABLE_IPV6
    if(!addr) {
      struct in6_addr in6;
      /* check if this is an IPv6 address string */
      if(Curl_inet_pton(AF_INET6, hostname, &in6) > 0)
        /* This is an IPv6 address literal */
        addr = Curl_ip2addr(AF_INET6, &in6, hostname, port);
    }
#endif /* ENABLE_IPV6 */

#else /* if USE_RESOLVE_ON_IPS */
    /* First check if this is an IPv4 address string */
    if(Curl_inet_pton(AF_INET, hostname, &in) > 0)
      /* This is a dotted IP address 123.123.123.123-style */
      ipnum = TRUE;
#ifdef ENABLE_IPV6
    else {
      struct in6_addr in6;
      /* check if this is an IPv6 address string */
      if(Curl_inet_pton(AF_INET6, hostname, &in6) > 0)
        /* This is an IPv6 address literal */
        ipnum = TRUE;
    }
#endif /* ENABLE_IPV6 */

#endif /* !USE_RESOLVE_ON_IPS */

    if(!addr) {
      if(conn->ip_version == CURL_IPRESOLVE_V6 && !Curl_ipv6works(data))
        return CURLRESOLV_ERROR;

      if(strcasecompare(hostname, "localhost"))
        addr = get_localhost(port);
      else if(allowDOH && data->set.doh && !ipnum)
        addr = Curl_doh(data, hostname, port, &respwait);
      else {
        /* Check what IP specifics the app has requested and if we can provide
         * it. If not, bail out. */
        if(!Curl_ipvalid(data, conn))
          return CURLRESOLV_ERROR;
        /* If Curl_getaddrinfo() returns NULL, 'respwait' might be set to a
           non-zero value indicating that we need to wait for the response to
           the resolve call */
        addr = Curl_getaddrinfo(data, hostname, port, &respwait);
      }
    }
    if(!addr) {
      if(respwait) {
        /* the response to our resolve call will come asynchronously at
           a later time, good or bad */
        /* First, check that we haven't received the info by now */
        result = Curl_resolv_check(data, &dns);
        if(result) /* error detected */
          return CURLRESOLV_ERROR;
        if(dns)
          rc = CURLRESOLV_RESOLVED; /* pointer provided */
        else
          rc = CURLRESOLV_PENDING; /* no info yet */
      }
    }
    else {
      if(data->share)
        Curl_share_lock(data, CURL_LOCK_DATA_DNS, CURL_LOCK_ACCESS_SINGLE);

      /* we got a response, store it in the cache */
      dns = Curl_cache_addr(data, addr, hostname, port);

      if(data->share)
        Curl_share_unlock(data, CURL_LOCK_DATA_DNS);

      if(!dns)
        /* returned failure, bail out nicely */
        Curl_freeaddrinfo(addr);
      else
        rc = CURLRESOLV_RESOLVED;
    }
  }

  *entry = dns;

  return rc;
}

#ifdef USE_ALARM_TIMEOUT
/*
 * This signal handler jumps back into the main libcurl code and continues
 * execution.  This effectively causes the remainder of the application to run
 * within a signal handler which is nonportable and could lead to problems.
 */
static
void alarmfunc(int sig)
{
  /* this is for "-ansi -Wall -pedantic" to stop complaining!   (rabe) */
  (void)sig;
  siglongjmp(curl_jmpenv, 1);
}
#endif /* USE_ALARM_TIMEOUT */

/*
 * Curl_resolv_timeout() is the same as Curl_resolv() but specifies a
 * timeout.  This function might return immediately if we're using asynch
 * resolves. See the return codes.
 *
 * The cache entry we return will get its 'inuse' counter increased when this
 * function is used. You MUST call Curl_resolv_unlock() later (when you're
 * done using this struct) to decrease the counter again.
 *
 * If built with a synchronous resolver and use of signals is not
 * disabled by the application, then a nonzero timeout will cause a
 * timeout after the specified number of milliseconds. Otherwise, timeout
 * is ignored.
 *
 * Return codes:
 *
 * CURLRESOLV_TIMEDOUT(-2) = warning, time too short or previous alarm expired
 * CURLRESOLV_ERROR   (-1) = error, no pointer
 * CURLRESOLV_RESOLVED (0) = OK, pointer provided
 * CURLRESOLV_PENDING  (1) = waiting for response, no pointer
 */

enum resolve_t Curl_resolv_timeout(struct Curl_easy *data,
                                   const char *hostname,
                                   int port,
                                   struct Curl_dns_entry **entry,
                                   timediff_t timeoutms)
{
#ifdef USE_ALARM_TIMEOUT
#ifdef HAVE_SIGACTION
  struct sigaction keep_sigact;   /* store the old struct here */
  volatile bool keep_copysig = FALSE; /* whether old sigact has been saved */
  struct sigaction sigact;
#else
#ifdef HAVE_SIGNAL
  void (*keep_sigact)(int);       /* store the old handler here */
#endif /* HAVE_SIGNAL */
#endif /* HAVE_SIGACTION */
  volatile long timeout;
  volatile unsigned int prev_alarm = 0;
#endif /* USE_ALARM_TIMEOUT */
  enum resolve_t rc;

  *entry = NULL;

  if(timeoutms < 0)
    /* got an already expired timeout */
    return CURLRESOLV_TIMEDOUT;

#ifdef USE_ALARM_TIMEOUT
  if(data->set.no_signal)
    /* Ignore the timeout when signals are disabled */
    timeout = 0;
  else
    timeout = (timeoutms > LONG_MAX) ? LONG_MAX : (long)timeoutms;

  if(!timeout)
    /* USE_ALARM_TIMEOUT defined, but no timeout actually requested */
    return Curl_resolv(data, hostname, port, TRUE, entry);

  if(timeout < 1000) {
    /* The alarm() function only provides integer second resolution, so if
       we want to wait less than one second we must bail out already now. */
    failf(data,
        "remaining timeout of %ld too small to resolve via SIGALRM method",
        timeout);
    return CURLRESOLV_TIMEDOUT;
  }
  /* This allows us to time-out from the name resolver, as the timeout
     will generate a signal and we will siglongjmp() from that here.
     This technique has problems (see alarmfunc).
     This should be the last thing we do before calling Curl_resolv(),
     as otherwise we'd have to worry about variables that get modified
     before we invoke Curl_resolv() (and thus use "volatile"). */
  if(sigsetjmp(curl_jmpenv, 1)) {
    /* this is coming from a siglongjmp() after an alarm signal */
    failf(data, "name lookup timed out");
    rc = CURLRESOLV_ERROR;
    goto clean_up;
  }
  else {
    /*************************************************************
     * Set signal handler to catch SIGALRM
     * Store the old value to be able to set it back later!
     *************************************************************/
#ifdef HAVE_SIGACTION
    sigaction(SIGALRM, NULL, &sigact);
    keep_sigact = sigact;
    keep_copysig = TRUE; /* yes, we have a copy */
    sigact.sa_handler = alarmfunc;
#ifdef SA_RESTART
    /* HPUX doesn't have SA_RESTART but defaults to that behavior! */
    sigact.sa_flags &= ~SA_RESTART;
#endif
    /* now set the new struct */
    sigaction(SIGALRM, &sigact, NULL);
#else /* HAVE_SIGACTION */
    /* no sigaction(), revert to the much lamer signal() */
#ifdef HAVE_SIGNAL
    keep_sigact = signal(SIGALRM, alarmfunc);
#endif
#endif /* HAVE_SIGACTION */

    /* alarm() makes a signal get sent when the timeout fires off, and that
       will abort system calls */
    prev_alarm = alarm(curlx_sltoui(timeout/1000L));
  }

#else
#ifndef CURLRES_ASYNCH
  if(timeoutms)
    infof(data, "timeout on name lookup is not supported");
#else
  (void)timeoutms; /* timeoutms not used with an async resolver */
#endif
#endif /* USE_ALARM_TIMEOUT */

  /* Perform the actual name resolution. This might be interrupted by an
   * alarm if it takes too long.
   */
  rc = Curl_resolv(data, hostname, port, TRUE, entry);

#ifdef USE_ALARM_TIMEOUT
clean_up:

  if(!prev_alarm)
    /* deactivate a possibly active alarm before uninstalling the handler */
    alarm(0);

#ifdef HAVE_SIGACTION
  if(keep_copysig) {
    /* we got a struct as it looked before, now put that one back nice
       and clean */
    sigaction(SIGALRM, &keep_sigact, NULL); /* put it back */
  }
#else
#ifdef HAVE_SIGNAL
  /* restore the previous SIGALRM handler */
  signal(SIGALRM, keep_sigact);
#endif
#endif /* HAVE_SIGACTION */

  /* switch back the alarm() to either zero or to what it was before minus
     the time we spent until now! */
  if(prev_alarm) {
    /* there was an alarm() set before us, now put it back */
    timediff_t elapsed_secs = Curl_timediff(Curl_now(),
                                            data->conn->created) / 1000;

    /* the alarm period is counted in even number of seconds */
    unsigned long alarm_set = (unsigned long)(prev_alarm - elapsed_secs);

    if(!alarm_set ||
       ((alarm_set >= 0x80000000) && (prev_alarm < 0x80000000)) ) {
      /* if the alarm time-left reached zero or turned "negative" (counted
         with unsigned values), we should fire off a SIGALRM here, but we
         won't, and zero would be to switch it off so we never set it to
         less than 1! */
      alarm(1);
      rc = CURLRESOLV_TIMEDOUT;
      failf(data, "Previous alarm fired off!");
    }
    else
      alarm((unsigned int)alarm_set);
  }
#endif /* USE_ALARM_TIMEOUT */

  return rc;
}

/*
 * Curl_resolv_unlock() unlocks the given cached DNS entry. When this has been
 * made, the struct may be destroyed due to pruning. It is important that only
 * one unlock is made for each Curl_resolv() call.
 *
 * May be called with 'data' == NULL for global cache.
 */
void Curl_resolv_unlock(struct Curl_easy *data, struct Curl_dns_entry *dns)
{
  if(data && data->share)
    Curl_share_lock(data, CURL_LOCK_DATA_DNS, CURL_LOCK_ACCESS_SINGLE);

  freednsentry(dns);

  if(data && data->share)
    Curl_share_unlock(data, CURL_LOCK_DATA_DNS);
}

/*
 * File-internal: release cache dns entry reference, free if inuse drops to 0
 */
static void freednsentry(void *freethis)
{
  struct Curl_dns_entry *dns = (struct Curl_dns_entry *) freethis;
  DEBUGASSERT(dns && (dns->inuse>0));

  dns->inuse--;
  if(dns->inuse == 0) {
    Curl_freeaddrinfo(dns->addr);
    free(dns);
  }
}

/*
 * Curl_mk_dnscache() inits a new DNS cache and returns success/failure.
 */
int Curl_mk_dnscache(struct Curl_hash *hash)
{
  return Curl_hash_init(hash, 7, Curl_hash_str, Curl_str_key_compare,
                        freednsentry);
}

/*
 * Curl_hostcache_clean()
 *
 * This _can_ be called with 'data' == NULL but then of course no locking
 * can be done!
 */

void Curl_hostcache_clean(struct Curl_easy *data,
                          struct Curl_hash *hash)
{
  if(data && data->share)
    Curl_share_lock(data, CURL_LOCK_DATA_DNS, CURL_LOCK_ACCESS_SINGLE);

  Curl_hash_clean(hash);

  if(data && data->share)
    Curl_share_unlock(data, CURL_LOCK_DATA_DNS);
}


CURLcode Curl_loadhostpairs(struct Curl_easy *data)
{
  struct curl_slist *hostp;
  char hostname[256];
  int port = 0;

  /* Default is no wildcard found */
  data->state.wildcard_resolve = false;

  for(hostp = data->state.resolve; hostp; hostp = hostp->next) {
    char entry_id[MAX_HOSTCACHE_LEN];
    if(!hostp->data)
      continue;
    if(hostp->data[0] == '-') {
      size_t entry_len;

      if(2 != sscanf(hostp->data + 1, "%255[^:]:%d", hostname, &port)) {
        infof(data, "Couldn't parse CURLOPT_RESOLVE removal entry '%s'",
              hostp->data);
        continue;
      }

      /* Create an entry id, based upon the hostname and port */
      create_hostcache_id(hostname, port, entry_id, sizeof(entry_id));
      entry_len = strlen(entry_id);

      if(data->share)
        Curl_share_lock(data, CURL_LOCK_DATA_DNS, CURL_LOCK_ACCESS_SINGLE);

      /* delete entry, ignore if it didn't exist */
      Curl_hash_delete(data->dns.hostcache, entry_id, entry_len + 1);

      if(data->share)
        Curl_share_unlock(data, CURL_LOCK_DATA_DNS);
    }
    else {
      struct Curl_dns_entry *dns;
      struct Curl_addrinfo *head = NULL, *tail = NULL;
      size_t entry_len;
      char address[64];
#if !defined(CURL_DISABLE_VERBOSE_STRINGS)
      char *addresses = NULL;
#endif
      char *addr_begin;
      char *addr_end;
      char *port_ptr;
      char *end_ptr;
      bool permanent = TRUE;
      char *host_begin;
      char *host_end;
      unsigned long tmp_port;
      bool error = true;

      host_begin = hostp->data;
      if(host_begin[0] == '+') {
        host_begin++;
        permanent = FALSE;
      }
      host_end = strchr(host_begin, ':');
      if(!host_end ||
         ((host_end - host_begin) >= (ptrdiff_t)sizeof(hostname)))
        goto err;

      memcpy(hostname, host_begin, host_end - host_begin);
      hostname[host_end - host_begin] = '\0';

      port_ptr = host_end + 1;
      tmp_port = strtoul(port_ptr, &end_ptr, 10);
      if(tmp_port > USHRT_MAX || end_ptr == port_ptr || *end_ptr != ':')
        goto err;

      port = (int)tmp_port;
#if !defined(CURL_DISABLE_VERBOSE_STRINGS)
      addresses = end_ptr + 1;
#endif

      while(*end_ptr) {
        size_t alen;
        struct Curl_addrinfo *ai;

        addr_begin = end_ptr + 1;
        addr_end = strchr(addr_begin, ',');
        if(!addr_end)
          addr_end = addr_begin + strlen(addr_begin);
        end_ptr = addr_end;

        /* allow IP(v6) address within [brackets] */
        if(*addr_begin == '[') {
          if(addr_end == addr_begin || *(addr_end - 1) != ']')
            goto err;
          ++addr_begin;
          --addr_end;
        }

        alen = addr_end - addr_begin;
        if(!alen)
          continue;

        if(alen >= sizeof(address))
          goto err;

        memcpy(address, addr_begin, alen);
        address[alen] = '\0';

#ifndef ENABLE_IPV6
        if(strchr(address, ':')) {
          infof(data, "Ignoring resolve address '%s', missing IPv6 support.",
                address);
          continue;
        }
#endif

        ai = Curl_str2addr(address, port);
        if(!ai) {
          infof(data, "Resolve address '%s' found illegal!", address);
          goto err;
        }

        if(tail) {
          tail->ai_next = ai;
          tail = tail->ai_next;
        }
        else {
          head = tail = ai;
        }
      }

      if(!head)
        goto err;

      error = false;
   err:
      if(error) {
        failf(data, "Couldn't parse CURLOPT_RESOLVE entry '%s'!",
              hostp->data);
        Curl_freeaddrinfo(head);
        return CURLE_SETOPT_OPTION_SYNTAX;
      }

      /* Create an entry id, based upon the hostname and port */
      create_hostcache_id(hostname, port, entry_id, sizeof(entry_id));
      entry_len = strlen(entry_id);

      if(data->share)
        Curl_share_lock(data, CURL_LOCK_DATA_DNS, CURL_LOCK_ACCESS_SINGLE);

      /* See if it's already in our dns cache */
      dns = Curl_hash_pick(data->dns.hostcache, entry_id, entry_len + 1);

      if(dns) {
        infof(data, "RESOLVE %s:%d is - old addresses discarded!",
                hostname, port);
        /* delete old entry, there are two reasons for this
         1. old entry may have different addresses.
         2. even if entry with correct addresses is already in the cache,
            but if it is close to expire, then by the time next http
            request is made, it can get expired and pruned because old
            entry is not necessarily marked as permanent.
         3. when adding a non-permanent entry, we want it to remove and
            replace an existing permanent entry.
         4. when adding a non-permanent entry, we want it to get a "fresh"
            timeout that starts _now_. */

        Curl_hash_delete(data->dns.hostcache, entry_id, entry_len + 1);
      }

      /* put this new host in the cache */
      dns = Curl_cache_addr(data, head, hostname, port);
      if(dns) {
        if(permanent)
          dns->timestamp = 0; /* mark as permanent */
        /* release the returned reference; the cache itself will keep the
         * entry alive: */
        dns->inuse--;
      }

      if(data->share)
        Curl_share_unlock(data, CURL_LOCK_DATA_DNS);

      if(!dns) {
        Curl_freeaddrinfo(head);
        return CURLE_OUT_OF_MEMORY;
      }
      infof(data, "Added %s:%d:%s to DNS cache%s",
            hostname, port, addresses, permanent ? "" : " (non-permanent)");

      /* Wildcard hostname */
      if(hostname[0] == '*' && hostname[1] == '\0') {
        infof(data, "RESOLVE %s:%d is wildcard, enabling wildcard checks",
              hostname, port);
        data->state.wildcard_resolve = true;
      }
    }
  }
  data->state.resolve = NULL; /* dealt with now */

  return CURLE_OK;
}

CURLcode Curl_resolv_check(struct Curl_easy *data,
                           struct Curl_dns_entry **dns)
{
#if defined(CURL_DISABLE_DOH) && !defined(CURLRES_ASYNCH)
  (void)dns;
#endif

  if(data->conn->bits.doh)
    return Curl_doh_is_resolved(data, dns);
  return Curl_resolver_is_resolved(data, dns);
}

int Curl_resolv_getsock(struct Curl_easy *data,
                        curl_socket_t *socks)
{
#ifdef CURLRES_ASYNCH
  if(data->conn->bits.doh)
    /* nothing to wait for during DoH resolve, those handles have their own
       sockets */
    return GETSOCK_BLANK;
  return Curl_resolver_getsock(data, socks);
#else
  (void)data;
  (void)socks;
  return GETSOCK_BLANK;
#endif
}

/* Call this function after Curl_connect() has returned async=TRUE and
   then a successful name resolve has been received.

   Note: this function disconnects and frees the conn data in case of
   resolve failure */
CURLcode Curl_once_resolved(struct Curl_easy *data, bool *protocol_done)
{
  CURLcode result;
  struct connectdata *conn = data->conn;

#ifdef USE_CURL_ASYNC
  if(data->state.async.dns) {
    conn->dns_entry = data->state.async.dns;
    data->state.async.dns = NULL;
  }
#endif

  result = Curl_setup_conn(data, protocol_done);

  if(result) {
    Curl_detach_connnection(data);
    Curl_conncache_remove_conn(data, conn, TRUE);
    Curl_disconnect(data, conn, TRUE);
  }
  return result;
}

/*
 * Curl_resolver_error() calls failf() with the appropriate message after a
 * resolve error
 */

#ifdef USE_CURL_ASYNC
CURLcode Curl_resolver_error(struct Curl_easy *data)
{
  const char *host_or_proxy;
  CURLcode result;

#ifndef CURL_DISABLE_PROXY
  struct connectdata *conn = data->conn;
  if(conn->bits.httpproxy) {
    host_or_proxy = "proxy";
    result = CURLE_COULDNT_RESOLVE_PROXY;
  }
  else
#endif
  {
    host_or_proxy = "host";
    result = CURLE_COULDNT_RESOLVE_HOST;
  }

  failf(data, "Could not resolve %s: %s", host_or_proxy,
        data->state.async.hostname);

  return result;
}
#endif /* USE_CURL_ASYNC */
