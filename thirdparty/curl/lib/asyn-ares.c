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

/***********************************************************************
 * Only for ares-enabled builds
 * And only for functions that fulfill the asynch resolver backend API
 * as defined in asyn.h, nothing else belongs in this file!
 **********************************************************************/

#ifdef CURLRES_ARES

#include <limits.h>
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
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

#ifdef HAVE_PROCESS_H
#include <process.h>
#endif

#if (defined(NETWARE) && defined(__NOVELL_LIBC__))
#undef in_addr_t
#define in_addr_t unsigned long
#endif

#include "urldata.h"
#include "sendf.h"
#include "hostip.h"
#include "hash.h"
#include "share.h"
#include "url.h"
#include "multiif.h"
#include "inet_pton.h"
#include "connect.h"
#include "select.h"
#include "progress.h"

#  if defined(CURL_STATICLIB) && !defined(CARES_STATICLIB) &&   \
  defined(WIN32)
#    define CARES_STATICLIB
#  endif
#  include <ares.h>
#  include <ares_version.h> /* really old c-ares didn't include this by
                               itself */

#if ARES_VERSION >= 0x010500
/* c-ares 1.5.0 or later, the callback proto is modified */
#define HAVE_CARES_CALLBACK_TIMEOUTS 1
#endif

#if ARES_VERSION >= 0x010601
/* IPv6 supported since 1.6.1 */
#define HAVE_CARES_IPV6 1
#endif

#if ARES_VERSION >= 0x010704
#define HAVE_CARES_SERVERS_CSV 1
#define HAVE_CARES_LOCAL_DEV 1
#define HAVE_CARES_SET_LOCAL 1
#endif

#if ARES_VERSION >= 0x010b00
#define HAVE_CARES_PORTS_CSV 1
#endif

#if ARES_VERSION >= 0x011000
/* 1.16.0 or later has ares_getaddrinfo */
#define HAVE_CARES_GETADDRINFO 1
#endif

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

struct thread_data {
  int num_pending; /* number of outstanding c-ares requests */
  struct Curl_addrinfo *temp_ai; /* intermediary result while fetching c-ares
                                    parts */
  int last_status;
  struct curltime happy_eyeballs_dns_time; /* when this timer started, or 0 */
};

/* How long we are willing to wait for additional parallel responses after
   obtaining a "definitive" one.

   This is intended to equal the c-ares default timeout.  cURL always uses that
   default value.  Unfortunately, c-ares doesn't expose its default timeout in
   its API, but it is officially documented as 5 seconds.

   See query_completed_cb() for an explanation of how this is used.
 */
#define HAPPY_EYEBALLS_DNS_TIMEOUT 5000

/*
 * Curl_resolver_global_init() - the generic low-level asynchronous name
 * resolve API.  Called from curl_global_init() to initialize global resolver
 * environment.  Initializes ares library.
 */
int Curl_resolver_global_init(void)
{
#ifdef CARES_HAVE_ARES_LIBRARY_INIT
  if(ares_library_init(ARES_LIB_INIT_ALL)) {
    return CURLE_FAILED_INIT;
  }
#endif
  return CURLE_OK;
}

/*
 * Curl_resolver_global_cleanup()
 *
 * Called from curl_global_cleanup() to destroy global resolver environment.
 * Deinitializes ares library.
 */
void Curl_resolver_global_cleanup(void)
{
#ifdef CARES_HAVE_ARES_LIBRARY_CLEANUP
  ares_library_cleanup();
#endif
}


static void sock_state_cb(void *data, ares_socket_t socket_fd,
                          int readable, int writable)
{
  struct Curl_easy *easy = data;
  if(!readable && !writable) {
    DEBUGASSERT(easy);
    Curl_multi_closed(easy, socket_fd);
  }
}

/*
 * Curl_resolver_init()
 *
 * Called from curl_easy_init() -> Curl_open() to initialize resolver
 * URL-state specific environment ('resolver' member of the UrlState
 * structure).  Fills the passed pointer by the initialized ares_channel.
 */
CURLcode Curl_resolver_init(struct Curl_easy *easy, void **resolver)
{
  int status;
  struct ares_options options;
  int optmask = ARES_OPT_SOCK_STATE_CB;
  options.sock_state_cb = sock_state_cb;
  options.sock_state_cb_data = easy;
  status = ares_init_options((ares_channel*)resolver, &options, optmask);
  if(status != ARES_SUCCESS) {
    if(status == ARES_ENOMEM)
      return CURLE_OUT_OF_MEMORY;
    else
      return CURLE_FAILED_INIT;
  }
  return CURLE_OK;
  /* make sure that all other returns from this function should destroy the
     ares channel before returning error! */
}

/*
 * Curl_resolver_cleanup()
 *
 * Called from curl_easy_cleanup() -> Curl_close() to cleanup resolver
 * URL-state specific environment ('resolver' member of the UrlState
 * structure).  Destroys the ares channel.
 */
void Curl_resolver_cleanup(void *resolver)
{
  ares_destroy((ares_channel)resolver);
}

/*
 * Curl_resolver_duphandle()
 *
 * Called from curl_easy_duphandle() to duplicate resolver URL-state specific
 * environment ('resolver' member of the UrlState structure).  Duplicates the
 * 'from' ares channel and passes the resulting channel to the 'to' pointer.
 */
CURLcode Curl_resolver_duphandle(struct Curl_easy *easy, void **to, void *from)
{
  (void)from;
  /*
   * it would be better to call ares_dup instead, but right now
   * it is not possible to set 'sock_state_cb_data' outside of
   * ares_init_options
   */
  return Curl_resolver_init(easy, to);
}

static void destroy_async_data(struct Curl_async *async);

/*
 * Cancel all possibly still on-going resolves for this connection.
 */
void Curl_resolver_cancel(struct Curl_easy *data)
{
  DEBUGASSERT(data);
  if(data->state.async.resolver)
    ares_cancel((ares_channel)data->state.async.resolver);
  destroy_async_data(&data->state.async);
}

/*
 * We're equivalent to Curl_resolver_cancel() for the c-ares resolver.  We
 * never block.
 */
void Curl_resolver_kill(struct Curl_easy *data)
{
  /* We don't need to check the resolver state because we can be called safely
     at any time and we always do the same thing. */
  Curl_resolver_cancel(data);
}

/*
 * destroy_async_data() cleans up async resolver data.
 */
static void destroy_async_data(struct Curl_async *async)
{
  free(async->hostname);

  if(async->tdata) {
    struct thread_data *res = async->tdata;
    if(res) {
      if(res->temp_ai) {
        Curl_freeaddrinfo(res->temp_ai);
        res->temp_ai = NULL;
      }
      free(res);
    }
    async->tdata = NULL;
  }

  async->hostname = NULL;
}

/*
 * Curl_resolver_getsock() is called when someone from the outside world
 * (using curl_multi_fdset()) wants to get our fd_set setup and we're talking
 * with ares. The caller must make sure that this function is only called when
 * we have a working ares channel.
 *
 * Returns: sockets-in-use-bitmap
 */

int Curl_resolver_getsock(struct Curl_easy *data,
                          curl_socket_t *socks)
{
  struct timeval maxtime;
  struct timeval timebuf;
  struct timeval *timeout;
  long milli;
  int max = ares_getsock((ares_channel)data->state.async.resolver,
                         (ares_socket_t *)socks, MAX_SOCKSPEREASYHANDLE);

  maxtime.tv_sec = CURL_TIMEOUT_RESOLVE;
  maxtime.tv_usec = 0;

  timeout = ares_timeout((ares_channel)data->state.async.resolver, &maxtime,
                         &timebuf);
  milli = (timeout->tv_sec * 1000) + (timeout->tv_usec/1000);
  if(milli == 0)
    milli += 10;
  Curl_expire(data, milli, EXPIRE_ASYNC_NAME);

  return max;
}

/*
 * waitperform()
 *
 * 1) Ask ares what sockets it currently plays with, then
 * 2) wait for the timeout period to check for action on ares' sockets.
 * 3) tell ares to act on all the sockets marked as "with action"
 *
 * return number of sockets it worked on
 */

static int waitperform(struct Curl_easy *data, timediff_t timeout_ms)
{
  int nfds;
  int bitmask;
  ares_socket_t socks[ARES_GETSOCK_MAXNUM];
  struct pollfd pfd[ARES_GETSOCK_MAXNUM];
  int i;
  int num = 0;

  bitmask = ares_getsock((ares_channel)data->state.async.resolver, socks,
                         ARES_GETSOCK_MAXNUM);

  for(i = 0; i < ARES_GETSOCK_MAXNUM; i++) {
    pfd[i].events = 0;
    pfd[i].revents = 0;
    if(ARES_GETSOCK_READABLE(bitmask, i)) {
      pfd[i].fd = socks[i];
      pfd[i].events |= POLLRDNORM|POLLIN;
    }
    if(ARES_GETSOCK_WRITABLE(bitmask, i)) {
      pfd[i].fd = socks[i];
      pfd[i].events |= POLLWRNORM|POLLOUT;
    }
    if(pfd[i].events)
      num++;
    else
      break;
  }

  if(num)
    nfds = Curl_poll(pfd, num, timeout_ms);
  else
    nfds = 0;

  if(!nfds)
    /* Call ares_process() unconditionally here, even if we simply timed out
       above, as otherwise the ares name resolve won't timeout! */
    ares_process_fd((ares_channel)data->state.async.resolver, ARES_SOCKET_BAD,
                    ARES_SOCKET_BAD);
  else {
    /* move through the descriptors and ask for processing on them */
    for(i = 0; i < num; i++)
      ares_process_fd((ares_channel)data->state.async.resolver,
                      (pfd[i].revents & (POLLRDNORM|POLLIN))?
                      pfd[i].fd:ARES_SOCKET_BAD,
                      (pfd[i].revents & (POLLWRNORM|POLLOUT))?
                      pfd[i].fd:ARES_SOCKET_BAD);
  }
  return nfds;
}

/*
 * Curl_resolver_is_resolved() is called repeatedly to check if a previous
 * name resolve request has completed. It should also make sure to time-out if
 * the operation seems to take too long.
 *
 * Returns normal CURLcode errors.
 */
CURLcode Curl_resolver_is_resolved(struct Curl_easy *data,
                                   struct Curl_dns_entry **dns)
{
  struct thread_data *res = data->state.async.tdata;
  CURLcode result = CURLE_OK;

  DEBUGASSERT(dns);
  *dns = NULL;

  waitperform(data, 0);

  /* Now that we've checked for any last minute results above, see if there are
     any responses still pending when the EXPIRE_HAPPY_EYEBALLS_DNS timer
     expires. */
  if(res
     && res->num_pending
     /* This is only set to non-zero if the timer was started. */
     && (res->happy_eyeballs_dns_time.tv_sec
         || res->happy_eyeballs_dns_time.tv_usec)
     && (Curl_timediff(Curl_now(), res->happy_eyeballs_dns_time)
         >= HAPPY_EYEBALLS_DNS_TIMEOUT)) {
    /* Remember that the EXPIRE_HAPPY_EYEBALLS_DNS timer is no longer
       running. */
    memset(
      &res->happy_eyeballs_dns_time, 0, sizeof(res->happy_eyeballs_dns_time));

    /* Cancel the raw c-ares request, which will fire query_completed_cb() with
       ARES_ECANCELLED synchronously for all pending responses.  This will
       leave us with res->num_pending == 0, which is perfect for the next
       block. */
    ares_cancel((ares_channel)data->state.async.resolver);
    DEBUGASSERT(res->num_pending == 0);
  }

  if(res && !res->num_pending) {
    (void)Curl_addrinfo_callback(data, res->last_status, res->temp_ai);
    /* temp_ai ownership is moved to the connection, so we need not free-up
       them */
    res->temp_ai = NULL;

    if(!data->state.async.dns)
      result = Curl_resolver_error(data);
    else
      *dns = data->state.async.dns;

    destroy_async_data(&data->state.async);
  }

  return result;
}

/*
 * Curl_resolver_wait_resolv()
 *
 * Waits for a resolve to finish. This function should be avoided since using
 * this risk getting the multi interface to "hang".
 *
 * 'entry' MUST be non-NULL.
 *
 * Returns CURLE_COULDNT_RESOLVE_HOST if the host was not resolved,
 * CURLE_OPERATION_TIMEDOUT if a time-out occurred, or other errors.
 */
CURLcode Curl_resolver_wait_resolv(struct Curl_easy *data,
                                   struct Curl_dns_entry **entry)
{
  CURLcode result = CURLE_OK;
  timediff_t timeout;
  struct curltime now = Curl_now();

  DEBUGASSERT(entry);
  *entry = NULL; /* clear on entry */

  timeout = Curl_timeleft(data, &now, TRUE);
  if(timeout < 0) {
    /* already expired! */
    connclose(data->conn, "Timed out before name resolve started");
    return CURLE_OPERATION_TIMEDOUT;
  }
  if(!timeout)
    timeout = CURL_TIMEOUT_RESOLVE * 1000; /* default name resolve timeout */

  /* Wait for the name resolve query to complete. */
  while(!result) {
    struct timeval *tvp, tv, store;
    int itimeout;
    timediff_t timeout_ms;

#if TIMEDIFF_T_MAX > INT_MAX
    itimeout = (timeout > INT_MAX) ? INT_MAX : (int)timeout;
#else
    itimeout = (int)timeout;
#endif

    store.tv_sec = itimeout/1000;
    store.tv_usec = (itimeout%1000)*1000;

    tvp = ares_timeout((ares_channel)data->state.async.resolver, &store, &tv);

    /* use the timeout period ares returned to us above if less than one
       second is left, otherwise just use 1000ms to make sure the progress
       callback gets called frequent enough */
    if(!tvp->tv_sec)
      timeout_ms = (timediff_t)(tvp->tv_usec/1000);
    else
      timeout_ms = 1000;

    waitperform(data, timeout_ms);
    result = Curl_resolver_is_resolved(data, entry);

    if(result || data->state.async.done)
      break;

    if(Curl_pgrsUpdate(data))
      result = CURLE_ABORTED_BY_CALLBACK;
    else {
      struct curltime now2 = Curl_now();
      timediff_t timediff = Curl_timediff(now2, now); /* spent time */
      if(timediff <= 0)
        timeout -= 1; /* always deduct at least 1 */
      else if(timediff > timeout)
        timeout = -1;
      else
        timeout -= timediff;
      now = now2; /* for next loop */
    }
    if(timeout < 0)
      result = CURLE_OPERATION_TIMEDOUT;
  }
  if(result)
    /* failure, so we cancel the ares operation */
    ares_cancel((ares_channel)data->state.async.resolver);

  /* Operation complete, if the lookup was successful we now have the entry
     in the cache. */
  if(entry)
    *entry = data->state.async.dns;

  if(result)
    /* close the connection, since we can't return failure here without
       cleaning up this connection properly. */
    connclose(data->conn, "c-ares resolve failed");

  return result;
}

#ifndef HAVE_CARES_GETADDRINFO

/* Connects results to the list */
static void compound_results(struct thread_data *res,
                             struct Curl_addrinfo *ai)
{
  if(!ai)
    return;

#ifdef ENABLE_IPV6 /* CURLRES_IPV6 */
  if(res->temp_ai && res->temp_ai->ai_family == PF_INET6) {
    /* We have results already, put the new IPv6 entries at the head of the
       list. */
    struct Curl_addrinfo *temp_ai_tail = res->temp_ai;

    while(temp_ai_tail->ai_next)
      temp_ai_tail = temp_ai_tail->ai_next;

    temp_ai_tail->ai_next = ai;
  }
  else
#endif /* CURLRES_IPV6 */
  {
    /* Add the new results to the list of old results. */
    struct Curl_addrinfo *ai_tail = ai;
    while(ai_tail->ai_next)
      ai_tail = ai_tail->ai_next;

    ai_tail->ai_next = res->temp_ai;
    res->temp_ai = ai;
  }
}

/*
 * ares_query_completed_cb() is the callback that ares will call when
 * the host query initiated by ares_gethostbyname() from Curl_getaddrinfo(),
 * when using ares, is completed either successfully or with failure.
 */
static void query_completed_cb(void *arg,  /* (struct connectdata *) */
                               int status,
#ifdef HAVE_CARES_CALLBACK_TIMEOUTS
                               int timeouts,
#endif
                               struct hostent *hostent)
{
  struct Curl_easy *data = (struct Curl_easy *)arg;
  struct thread_data *res;

#ifdef HAVE_CARES_CALLBACK_TIMEOUTS
  (void)timeouts; /* ignored */
#endif

  if(ARES_EDESTRUCTION == status)
    /* when this ares handle is getting destroyed, the 'arg' pointer may not
       be valid so only defer it when we know the 'status' says its fine! */
    return;

  res = data->state.async.tdata;
  if(res) {
    res->num_pending--;

    if(CURL_ASYNC_SUCCESS == status) {
      struct Curl_addrinfo *ai = Curl_he2ai(hostent, data->state.async.port);
      if(ai) {
        compound_results(res, ai);
      }
    }
    /* A successful result overwrites any previous error */
    if(res->last_status != ARES_SUCCESS)
      res->last_status = status;

    /* If there are responses still pending, we presume they must be the
       complementary IPv4 or IPv6 lookups that we started in parallel in
       Curl_resolver_getaddrinfo() (for Happy Eyeballs).  If we've got a
       "definitive" response from one of a set of parallel queries, we need to
       think about how long we're willing to wait for more responses. */
    if(res->num_pending
       /* Only these c-ares status values count as "definitive" for these
          purposes.  For example, ARES_ENODATA is what we expect when there is
          no IPv6 entry for a domain name, and that's not a reason to get more
          aggressive in our timeouts for the other response.  Other errors are
          either a result of bad input (which should affect all parallel
          requests), local or network conditions, non-definitive server
          responses, or us cancelling the request. */
       && (status == ARES_SUCCESS || status == ARES_ENOTFOUND)) {
      /* Right now, there can only be up to two parallel queries, so don't
         bother handling any other cases. */
      DEBUGASSERT(res->num_pending == 1);

      /* It's possible that one of these parallel queries could succeed
         quickly, but the other could always fail or timeout (when we're
         talking to a pool of DNS servers that can only successfully resolve
         IPv4 address, for example).

         It's also possible that the other request could always just take
         longer because it needs more time or only the second DNS server can
         fulfill it successfully.  But, to align with the philosophy of Happy
         Eyeballs, we don't want to wait _too_ long or users will think
         requests are slow when IPv6 lookups don't actually work (but IPv4 ones
         do).

         So, now that we have a usable answer (some IPv4 addresses, some IPv6
         addresses, or "no such domain"), we start a timeout for the remaining
         pending responses.  Even though it is typical that this resolved
         request came back quickly, that needn't be the case.  It might be that
         this completing request didn't get a result from the first DNS server
         or even the first round of the whole DNS server pool.  So it could
         already be quite some time after we issued the DNS queries in the
         first place.  Without modifying c-ares, we can't know exactly where in
         its retry cycle we are.  We could guess based on how much time has
         gone by, but it doesn't really matter.  Happy Eyeballs tells us that,
         given usable information in hand, we simply don't want to wait "too
         much longer" after we get a result.

         We simply wait an additional amount of time equal to the default
         c-ares query timeout.  That is enough time for a typical parallel
         response to arrive without being "too long".  Even on a network
         where one of the two types of queries is failing or timing out
         constantly, this will usually mean we wait a total of the default
         c-ares timeout (5 seconds) plus the round trip time for the successful
         request, which seems bearable.  The downside is that c-ares might race
         with us to issue one more retry just before we give up, but it seems
         better to "waste" that request instead of trying to guess the perfect
         timeout to prevent it.  After all, we don't even know where in the
         c-ares retry cycle each request is.
      */
      res->happy_eyeballs_dns_time = Curl_now();
      Curl_expire(data, HAPPY_EYEBALLS_DNS_TIMEOUT,
                  EXPIRE_HAPPY_EYEBALLS_DNS);
    }
  }
}
#else
/* c-ares 1.16.0 or later */

/*
 * ares2addr() converts an address list provided by c-ares to an internal
 * libcurl compatible list
 */
static struct Curl_addrinfo *ares2addr(struct ares_addrinfo_node *node)
{
  /* traverse the ares_addrinfo_node list */
  struct ares_addrinfo_node *ai;
  struct Curl_addrinfo *cafirst = NULL;
  struct Curl_addrinfo *calast = NULL;
  int error = 0;

  for(ai = node; ai != NULL; ai = ai->ai_next) {
    size_t ss_size;
    struct Curl_addrinfo *ca;
    /* ignore elements with unsupported address family, */
    /* settle family-specific sockaddr structure size.  */
    if(ai->ai_family == AF_INET)
      ss_size = sizeof(struct sockaddr_in);
#ifdef ENABLE_IPV6
    else if(ai->ai_family == AF_INET6)
      ss_size = sizeof(struct sockaddr_in6);
#endif
    else
      continue;

    /* ignore elements without required address info */
    if(!ai->ai_addr || !(ai->ai_addrlen > 0))
      continue;

    /* ignore elements with bogus address size */
    if((size_t)ai->ai_addrlen < ss_size)
      continue;

    ca = malloc(sizeof(struct Curl_addrinfo) + ss_size);
    if(!ca) {
      error = EAI_MEMORY;
      break;
    }

    /* copy each structure member individually, member ordering, */
    /* size, or padding might be different for each platform.    */

    ca->ai_flags     = ai->ai_flags;
    ca->ai_family    = ai->ai_family;
    ca->ai_socktype  = ai->ai_socktype;
    ca->ai_protocol  = ai->ai_protocol;
    ca->ai_addrlen   = (curl_socklen_t)ss_size;
    ca->ai_addr      = NULL;
    ca->ai_canonname = NULL;
    ca->ai_next      = NULL;

    ca->ai_addr = (void *)((char *)ca + sizeof(struct Curl_addrinfo));
    memcpy(ca->ai_addr, ai->ai_addr, ss_size);

    /* if the return list is empty, this becomes the first element */
    if(!cafirst)
      cafirst = ca;

    /* add this element last in the return list */
    if(calast)
      calast->ai_next = ca;
    calast = ca;
  }

  /* if we failed, destroy the Curl_addrinfo list */
  if(error) {
    Curl_freeaddrinfo(cafirst);
    cafirst = NULL;
  }

  return cafirst;
}

static void addrinfo_cb(void *arg, int status, int timeouts,
                        struct ares_addrinfo *result)
{
  struct Curl_easy *data = (struct Curl_easy *)arg;
  struct thread_data *res = data->state.async.tdata;
  (void)timeouts;
  if(ARES_SUCCESS == status) {
    res->temp_ai = ares2addr(result->nodes);
    res->last_status = CURL_ASYNC_SUCCESS;
    ares_freeaddrinfo(result);
  }
  res->num_pending--;
}

#endif
/*
 * Curl_resolver_getaddrinfo() - when using ares
 *
 * Returns name information about the given hostname and port number. If
 * successful, the 'hostent' is returned and the forth argument will point to
 * memory we need to free after use. That memory *MUST* be freed with
 * Curl_freeaddrinfo(), nothing else.
 */
struct Curl_addrinfo *Curl_resolver_getaddrinfo(struct Curl_easy *data,
                                                const char *hostname,
                                                int port,
                                                int *waitp)
{
  char *bufp;

  *waitp = 0; /* default to synchronous response */

  bufp = strdup(hostname);
  if(bufp) {
    struct thread_data *res = NULL;
    free(data->state.async.hostname);
    data->state.async.hostname = bufp;
    data->state.async.port = port;
    data->state.async.done = FALSE;   /* not done */
    data->state.async.status = 0;     /* clear */
    data->state.async.dns = NULL;     /* clear */
    res = calloc(sizeof(struct thread_data), 1);
    if(!res) {
      free(data->state.async.hostname);
      data->state.async.hostname = NULL;
      return NULL;
    }
    data->state.async.tdata = res;

    /* initial status - failed */
    res->last_status = ARES_ENOTFOUND;

#ifdef HAVE_CARES_GETADDRINFO
    {
      struct ares_addrinfo_hints hints;
      char service[12];
      int pf = PF_INET;
      memset(&hints, 0, sizeof(hints));
#ifdef CURLRES_IPV6
      if(Curl_ipv6works(data))
        /* The stack seems to be IPv6-enabled */
        pf = PF_UNSPEC;
#endif /* CURLRES_IPV6 */
      hints.ai_family = pf;
      hints.ai_socktype = (data->conn->transport == TRNSPRT_TCP)?
        SOCK_STREAM : SOCK_DGRAM;
      msnprintf(service, sizeof(service), "%d", port);
      res->num_pending = 1;
      ares_getaddrinfo((ares_channel)data->state.async.resolver, hostname,
                       service, &hints, addrinfo_cb, data);
    }
#else

#ifdef HAVE_CARES_IPV6
    if(Curl_ipv6works(data)) {
      /* The stack seems to be IPv6-enabled */
      res->num_pending = 2;

      /* areschannel is already setup in the Curl_open() function */
      ares_gethostbyname((ares_channel)data->state.async.resolver, hostname,
                          PF_INET, query_completed_cb, data);
      ares_gethostbyname((ares_channel)data->state.async.resolver, hostname,
                          PF_INET6, query_completed_cb, data);
    }
    else
#endif
    {
      res->num_pending = 1;

      /* areschannel is already setup in the Curl_open() function */
      ares_gethostbyname((ares_channel)data->state.async.resolver,
                         hostname, PF_INET,
                         query_completed_cb, data);
    }
#endif
    *waitp = 1; /* expect asynchronous response */
  }
  return NULL; /* no struct yet */
}

CURLcode Curl_set_dns_servers(struct Curl_easy *data,
                              char *servers)
{
  CURLcode result = CURLE_NOT_BUILT_IN;
  int ares_result;

  /* If server is NULL or empty, this would purge all DNS servers
   * from ares library, which will cause any and all queries to fail.
   * So, just return OK if none are configured and don't actually make
   * any changes to c-ares.  This lets c-ares use it's defaults, which
   * it gets from the OS (for instance from /etc/resolv.conf on Linux).
   */
  if(!(servers && servers[0]))
    return CURLE_OK;

#ifdef HAVE_CARES_SERVERS_CSV
#ifdef HAVE_CARES_PORTS_CSV
  ares_result = ares_set_servers_ports_csv(data->state.async.resolver,
                                           servers);
#else
  ares_result = ares_set_servers_csv(data->state.async.resolver, servers);
#endif
  switch(ares_result) {
  case ARES_SUCCESS:
    result = CURLE_OK;
    break;
  case ARES_ENOMEM:
    result = CURLE_OUT_OF_MEMORY;
    break;
  case ARES_ENOTINITIALIZED:
  case ARES_ENODATA:
  case ARES_EBADSTR:
  default:
    result = CURLE_BAD_FUNCTION_ARGUMENT;
    break;
  }
#else /* too old c-ares version! */
  (void)data;
  (void)(ares_result);
#endif
  return result;
}

CURLcode Curl_set_dns_interface(struct Curl_easy *data,
                                const char *interf)
{
#ifdef HAVE_CARES_LOCAL_DEV
  if(!interf)
    interf = "";

  ares_set_local_dev((ares_channel)data->state.async.resolver, interf);

  return CURLE_OK;
#else /* c-ares version too old! */
  (void)data;
  (void)interf;
  return CURLE_NOT_BUILT_IN;
#endif
}

CURLcode Curl_set_dns_local_ip4(struct Curl_easy *data,
                                const char *local_ip4)
{
#ifdef HAVE_CARES_SET_LOCAL
  struct in_addr a4;

  if((!local_ip4) || (local_ip4[0] == 0)) {
    a4.s_addr = 0; /* disabled: do not bind to a specific address */
  }
  else {
    if(Curl_inet_pton(AF_INET, local_ip4, &a4) != 1) {
      return CURLE_BAD_FUNCTION_ARGUMENT;
    }
  }

  ares_set_local_ip4((ares_channel)data->state.async.resolver,
                     ntohl(a4.s_addr));

  return CURLE_OK;
#else /* c-ares version too old! */
  (void)data;
  (void)local_ip4;
  return CURLE_NOT_BUILT_IN;
#endif
}

CURLcode Curl_set_dns_local_ip6(struct Curl_easy *data,
                                const char *local_ip6)
{
#if defined(HAVE_CARES_SET_LOCAL) && defined(ENABLE_IPV6)
  unsigned char a6[INET6_ADDRSTRLEN];

  if((!local_ip6) || (local_ip6[0] == 0)) {
    /* disabled: do not bind to a specific address */
    memset(a6, 0, sizeof(a6));
  }
  else {
    if(Curl_inet_pton(AF_INET6, local_ip6, a6) != 1) {
      return CURLE_BAD_FUNCTION_ARGUMENT;
    }
  }

  ares_set_local_ip6((ares_channel)data->state.async.resolver, a6);

  return CURLE_OK;
#else /* c-ares version too old! */
  (void)data;
  (void)local_ip6;
  return CURLE_NOT_BUILT_IN;
#endif
}
#endif /* CURLRES_ARES */
