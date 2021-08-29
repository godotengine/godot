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

#include <curl/curl.h>

#include "urldata.h"
#include "transfer.h"
#include "url.h"
#include "connect.h"
#include "progress.h"
#include "easyif.h"
#include "share.h"
#include "psl.h"
#include "multiif.h"
#include "sendf.h"
#include "timeval.h"
#include "http.h"
#include "select.h"
#include "warnless.h"
#include "speedcheck.h"
#include "conncache.h"
#include "multihandle.h"
#include "sigpipe.h"
#include "vtls/vtls.h"
#include "connect.h"
#include "http_proxy.h"
#include "http2.h"
#include "socketpair.h"
#include "socks.h"
/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

/*
  CURL_SOCKET_HASH_TABLE_SIZE should be a prime number. Increasing it from 97
  to 911 takes on a 32-bit machine 4 x 804 = 3211 more bytes.  Still, every
  CURL handle takes 45-50 K memory, therefore this 3K are not significant.
*/
#ifndef CURL_SOCKET_HASH_TABLE_SIZE
#define CURL_SOCKET_HASH_TABLE_SIZE 911
#endif

#ifndef CURL_CONNECTION_HASH_SIZE
#define CURL_CONNECTION_HASH_SIZE 97
#endif

#define CURL_MULTI_HANDLE 0x000bab1e

#define GOOD_MULTI_HANDLE(x) \
  ((x) && (x)->magic == CURL_MULTI_HANDLE)

static CURLMcode singlesocket(struct Curl_multi *multi,
                              struct Curl_easy *data);
static CURLMcode add_next_timeout(struct curltime now,
                                  struct Curl_multi *multi,
                                  struct Curl_easy *d);
static CURLMcode multi_timeout(struct Curl_multi *multi,
                               long *timeout_ms);
static void process_pending_handles(struct Curl_multi *multi);

#ifdef DEBUGBUILD
static const char * const statename[]={
  "INIT",
  "PENDING",
  "CONNECT",
  "RESOLVING",
  "CONNECTING",
  "TUNNELING",
  "PROTOCONNECT",
  "PROTOCONNECTING",
  "DO",
  "DOING",
  "DOING_MORE",
  "DID",
  "PERFORMING",
  "RATELIMITING",
  "DONE",
  "COMPLETED",
  "MSGSENT",
};
#endif

/* function pointer called once when switching TO a state */
typedef void (*init_multistate_func)(struct Curl_easy *data);

/* called in DID state, before PERFORMING state */
static void before_perform(struct Curl_easy *data)
{
  data->req.chunk = FALSE;
  Curl_pgrsTime(data, TIMER_PRETRANSFER);
}

static void init_completed(struct Curl_easy *data)
{
  /* this is a completed transfer */

  /* Important: reset the conn pointer so that we don't point to memory
     that could be freed anytime */
  Curl_detach_connnection(data);
  Curl_expire_clear(data); /* stop all timers */
}

/* always use this function to change state, to make debugging easier */
static void mstate(struct Curl_easy *data, CURLMstate state
#ifdef DEBUGBUILD
                   , int lineno
#endif
)
{
  CURLMstate oldstate = data->mstate;
  static const init_multistate_func finit[MSTATE_LAST] = {
    NULL,              /* INIT */
    NULL,              /* PENDING */
    Curl_init_CONNECT, /* CONNECT */
    NULL,              /* RESOLVING */
    NULL,              /* CONNECTING */
    NULL,              /* TUNNELING */
    NULL,              /* PROTOCONNECT */
    NULL,              /* PROTOCONNECTING */
    Curl_connect_free, /* DO */
    NULL,              /* DOING */
    NULL,              /* DOING_MORE */
    before_perform,    /* DID */
    NULL,              /* PERFORMING */
    NULL,              /* RATELIMITING */
    NULL,              /* DONE */
    init_completed,    /* COMPLETED */
    NULL               /* MSGSENT */
  };

#if defined(DEBUGBUILD) && defined(CURL_DISABLE_VERBOSE_STRINGS)
  (void) lineno;
#endif

  if(oldstate == state)
    /* don't bother when the new state is the same as the old state */
    return;

  data->mstate = state;

#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
  if(data->mstate >= MSTATE_PENDING &&
     data->mstate < MSTATE_COMPLETED) {
    long connection_id = -5000;

    if(data->conn)
      connection_id = data->conn->connection_id;

    infof(data,
          "STATE: %s => %s handle %p; line %d (connection #%ld)",
          statename[oldstate], statename[data->mstate],
          (void *)data, lineno, connection_id);
  }
#endif

  if(state == MSTATE_COMPLETED) {
    /* changing to COMPLETED means there's one less easy handle 'alive' */
    DEBUGASSERT(data->multi->num_alive > 0);
    data->multi->num_alive--;
  }

  /* if this state has an init-function, run it */
  if(finit[state])
    finit[state](data);
}

#ifndef DEBUGBUILD
#define multistate(x,y) mstate(x,y)
#else
#define multistate(x,y) mstate(x,y, __LINE__)
#endif

/*
 * We add one of these structs to the sockhash for each socket
 */

struct Curl_sh_entry {
  struct Curl_hash transfers; /* hash of transfers using this socket */
  unsigned int action;  /* what combined action READ/WRITE this socket waits
                           for */
  unsigned int users; /* number of transfers using this */
  void *socketp; /* settable by users with curl_multi_assign() */
  unsigned int readers; /* this many transfers want to read */
  unsigned int writers; /* this many transfers want to write */
};
/* bits for 'action' having no bits means this socket is not expecting any
   action */
#define SH_READ  1
#define SH_WRITE 2

/* look up a given socket in the socket hash, skip invalid sockets */
static struct Curl_sh_entry *sh_getentry(struct Curl_hash *sh,
                                         curl_socket_t s)
{
  if(s != CURL_SOCKET_BAD) {
    /* only look for proper sockets */
    return Curl_hash_pick(sh, (char *)&s, sizeof(curl_socket_t));
  }
  return NULL;
}

#define TRHASH_SIZE 13
static size_t trhash(void *key, size_t key_length, size_t slots_num)
{
  size_t keyval = (size_t)*(struct Curl_easy **)key;
  (void) key_length;

  return (keyval % slots_num);
}

static size_t trhash_compare(void *k1, size_t k1_len, void *k2, size_t k2_len)
{
  (void)k1_len;
  (void)k2_len;

  return *(struct Curl_easy **)k1 == *(struct Curl_easy **)k2;
}

static void trhash_dtor(void *nada)
{
  (void)nada;
}


/* make sure this socket is present in the hash for this handle */
static struct Curl_sh_entry *sh_addentry(struct Curl_hash *sh,
                                         curl_socket_t s)
{
  struct Curl_sh_entry *there = sh_getentry(sh, s);
  struct Curl_sh_entry *check;

  if(there) {
    /* it is present, return fine */
    return there;
  }

  /* not present, add it */
  check = calloc(1, sizeof(struct Curl_sh_entry));
  if(!check)
    return NULL; /* major failure */

  if(Curl_hash_init(&check->transfers, TRHASH_SIZE, trhash,
                    trhash_compare, trhash_dtor)) {
    free(check);
    return NULL;
  }

  /* make/add new hash entry */
  if(!Curl_hash_add(sh, (char *)&s, sizeof(curl_socket_t), check)) {
    Curl_hash_destroy(&check->transfers);
    free(check);
    return NULL; /* major failure */
  }

  return check; /* things are good in sockhash land */
}


/* delete the given socket + handle from the hash */
static void sh_delentry(struct Curl_sh_entry *entry,
                        struct Curl_hash *sh, curl_socket_t s)
{
  Curl_hash_destroy(&entry->transfers);

  /* We remove the hash entry. This will end up in a call to
     sh_freeentry(). */
  Curl_hash_delete(sh, (char *)&s, sizeof(curl_socket_t));
}

/*
 * free a sockhash entry
 */
static void sh_freeentry(void *freethis)
{
  struct Curl_sh_entry *p = (struct Curl_sh_entry *) freethis;

  free(p);
}

static size_t fd_key_compare(void *k1, size_t k1_len, void *k2, size_t k2_len)
{
  (void) k1_len; (void) k2_len;

  return (*((curl_socket_t *) k1)) == (*((curl_socket_t *) k2));
}

static size_t hash_fd(void *key, size_t key_length, size_t slots_num)
{
  curl_socket_t fd = *((curl_socket_t *) key);
  (void) key_length;

  return (fd % slots_num);
}

/*
 * sh_init() creates a new socket hash and returns the handle for it.
 *
 * Quote from README.multi_socket:
 *
 * "Some tests at 7000 and 9000 connections showed that the socket hash lookup
 * is somewhat of a bottle neck. Its current implementation may be a bit too
 * limiting. It simply has a fixed-size array, and on each entry in the array
 * it has a linked list with entries. So the hash only checks which list to
 * scan through. The code I had used so for used a list with merely 7 slots
 * (as that is what the DNS hash uses) but with 7000 connections that would
 * make an average of 1000 nodes in each list to run through. I upped that to
 * 97 slots (I believe a prime is suitable) and noticed a significant speed
 * increase.  I need to reconsider the hash implementation or use a rather
 * large default value like this. At 9000 connections I was still below 10us
 * per call."
 *
 */
static int sh_init(struct Curl_hash *hash, int hashsize)
{
  return Curl_hash_init(hash, hashsize, hash_fd, fd_key_compare,
                        sh_freeentry);
}

/*
 * multi_addmsg()
 *
 * Called when a transfer is completed. Adds the given msg pointer to
 * the list kept in the multi handle.
 */
static CURLMcode multi_addmsg(struct Curl_multi *multi,
                              struct Curl_message *msg)
{
  Curl_llist_insert_next(&multi->msglist, multi->msglist.tail, msg,
                         &msg->list);
  return CURLM_OK;
}

struct Curl_multi *Curl_multi_handle(int hashsize, /* socket hash */
                                     int chashsize) /* connection hash */
{
  struct Curl_multi *multi = calloc(1, sizeof(struct Curl_multi));

  if(!multi)
    return NULL;

  multi->magic = CURL_MULTI_HANDLE;

  if(Curl_mk_dnscache(&multi->hostcache))
    goto error;

  if(sh_init(&multi->sockhash, hashsize))
    goto error;

  if(Curl_conncache_init(&multi->conn_cache, chashsize))
    goto error;

  Curl_llist_init(&multi->msglist, NULL);
  Curl_llist_init(&multi->pending, NULL);

  multi->multiplexing = TRUE;

  /* -1 means it not set by user, use the default value */
  multi->maxconnects = -1;
  multi->max_concurrent_streams = 100;
  multi->ipv6_works = Curl_ipv6works(NULL);

#ifdef USE_WINSOCK
  multi->wsa_event = WSACreateEvent();
  if(multi->wsa_event == WSA_INVALID_EVENT)
    goto error;
#else
#ifdef ENABLE_WAKEUP
  if(Curl_socketpair(AF_UNIX, SOCK_STREAM, 0, multi->wakeup_pair) < 0) {
    multi->wakeup_pair[0] = CURL_SOCKET_BAD;
    multi->wakeup_pair[1] = CURL_SOCKET_BAD;
  }
  else if(curlx_nonblock(multi->wakeup_pair[0], TRUE) < 0 ||
          curlx_nonblock(multi->wakeup_pair[1], TRUE) < 0) {
    sclose(multi->wakeup_pair[0]);
    sclose(multi->wakeup_pair[1]);
    multi->wakeup_pair[0] = CURL_SOCKET_BAD;
    multi->wakeup_pair[1] = CURL_SOCKET_BAD;
  }
#endif
#endif

  return multi;

  error:

  Curl_hash_destroy(&multi->sockhash);
  Curl_hash_destroy(&multi->hostcache);
  Curl_conncache_destroy(&multi->conn_cache);
  Curl_llist_destroy(&multi->msglist, NULL);
  Curl_llist_destroy(&multi->pending, NULL);

  free(multi);
  return NULL;
}

struct Curl_multi *curl_multi_init(void)
{
  return Curl_multi_handle(CURL_SOCKET_HASH_TABLE_SIZE,
                           CURL_CONNECTION_HASH_SIZE);
}

CURLMcode curl_multi_add_handle(struct Curl_multi *multi,
                                struct Curl_easy *data)
{
  CURLMcode rc;
  /* First, make some basic checks that the CURLM handle is a good handle */
  if(!GOOD_MULTI_HANDLE(multi))
    return CURLM_BAD_HANDLE;

  /* Verify that we got a somewhat good easy handle too */
  if(!GOOD_EASY_HANDLE(data))
    return CURLM_BAD_EASY_HANDLE;

  /* Prevent users from adding same easy handle more than once and prevent
     adding to more than one multi stack */
  if(data->multi)
    return CURLM_ADDED_ALREADY;

  if(multi->in_callback)
    return CURLM_RECURSIVE_API_CALL;

  if(multi->dead) {
    /* a "dead" handle cannot get added transfers while any existing easy
       handles are still alive - but if there are none alive anymore, it is
       fine to start over and unmark the "deadness" of this handle */
    if(multi->num_alive)
      return CURLM_ABORTED_BY_CALLBACK;
    multi->dead = FALSE;
  }

  /* Initialize timeout list for this handle */
  Curl_llist_init(&data->state.timeoutlist, NULL);

  /*
   * No failure allowed in this function beyond this point. And no
   * modification of easy nor multi handle allowed before this except for
   * potential multi's connection cache growing which won't be undone in this
   * function no matter what.
   */
  if(data->set.errorbuffer)
    data->set.errorbuffer[0] = 0;

  /* make the Curl_easy refer back to this multi handle - before Curl_expire()
     is called. */
  data->multi = multi;

  /* Set the timeout for this handle to expire really soon so that it will
     be taken care of even when this handle is added in the midst of operation
     when only the curl_multi_socket() API is used. During that flow, only
     sockets that time-out or have actions will be dealt with. Since this
     handle has no action yet, we make sure it times out to get things to
     happen. */
  Curl_expire(data, 0, EXPIRE_RUN_NOW);

  /* A somewhat crude work-around for a little glitch in Curl_update_timer()
     that happens if the lastcall time is set to the same time when the handle
     is removed as when the next handle is added, as then the check in
     Curl_update_timer() that prevents calling the application multiple times
     with the same timer info will not trigger and then the new handle's
     timeout will not be notified to the app.

     The work-around is thus simply to clear the 'lastcall' variable to force
     Curl_update_timer() to always trigger a callback to the app when a new
     easy handle is added */
  memset(&multi->timer_lastcall, 0, sizeof(multi->timer_lastcall));

  rc = Curl_update_timer(multi);
  if(rc)
    return rc;

  /* set the easy handle */
  multistate(data, MSTATE_INIT);

  /* for multi interface connections, we share DNS cache automatically if the
     easy handle's one is currently not set. */
  if(!data->dns.hostcache ||
     (data->dns.hostcachetype == HCACHE_NONE)) {
    data->dns.hostcache = &multi->hostcache;
    data->dns.hostcachetype = HCACHE_MULTI;
  }

  /* Point to the shared or multi handle connection cache */
  if(data->share && (data->share->specifier & (1<< CURL_LOCK_DATA_CONNECT)))
    data->state.conn_cache = &data->share->conn_cache;
  else
    data->state.conn_cache = &multi->conn_cache;
  data->state.lastconnect_id = -1;

#ifdef USE_LIBPSL
  /* Do the same for PSL. */
  if(data->share && (data->share->specifier & (1 << CURL_LOCK_DATA_PSL)))
    data->psl = &data->share->psl;
  else
    data->psl = &multi->psl;
#endif

  /* We add the new entry last in the list. */
  data->next = NULL; /* end of the line */
  if(multi->easyp) {
    struct Curl_easy *last = multi->easylp;
    last->next = data;
    data->prev = last;
    multi->easylp = data; /* the new last node */
  }
  else {
    /* first node, make prev NULL! */
    data->prev = NULL;
    multi->easylp = multi->easyp = data; /* both first and last */
  }

  /* increase the node-counter */
  multi->num_easy++;

  /* increase the alive-counter */
  multi->num_alive++;

  CONNCACHE_LOCK(data);
  /* The closure handle only ever has default timeouts set. To improve the
     state somewhat we clone the timeouts from each added handle so that the
     closure handle always has the same timeouts as the most recently added
     easy handle. */
  data->state.conn_cache->closure_handle->set.timeout = data->set.timeout;
  data->state.conn_cache->closure_handle->set.server_response_timeout =
    data->set.server_response_timeout;
  data->state.conn_cache->closure_handle->set.no_signal =
    data->set.no_signal;
  CONNCACHE_UNLOCK(data);

  return CURLM_OK;
}

#if 0
/* Debug-function, used like this:
 *
 * Curl_hash_print(multi->sockhash, debug_print_sock_hash);
 *
 * Enable the hash print function first by editing hash.c
 */
static void debug_print_sock_hash(void *p)
{
  struct Curl_sh_entry *sh = (struct Curl_sh_entry *)p;

  fprintf(stderr, " [easy %p/magic %x/socket %d]",
          (void *)sh->data, sh->data->magic, (int)sh->socket);
}
#endif

static CURLcode multi_done(struct Curl_easy *data,
                           CURLcode status,  /* an error if this is called
                                                after an error was detected */
                           bool premature)
{
  CURLcode result;
  struct connectdata *conn = data->conn;
  unsigned int i;

  DEBUGF(infof(data, "multi_done"));

  if(data->state.done)
    /* Stop if multi_done() has already been called */
    return CURLE_OK;

  /* Stop the resolver and free its own resources (but not dns_entry yet). */
  Curl_resolver_kill(data);

  /* Cleanup possible redirect junk */
  Curl_safefree(data->req.newurl);
  Curl_safefree(data->req.location);

  switch(status) {
  case CURLE_ABORTED_BY_CALLBACK:
  case CURLE_READ_ERROR:
  case CURLE_WRITE_ERROR:
    /* When we're aborted due to a callback return code it basically have to
       be counted as premature as there is trouble ahead if we don't. We have
       many callbacks and protocols work differently, we could potentially do
       this more fine-grained in the future. */
    premature = TRUE;
  default:
    break;
  }

  /* this calls the protocol-specific function pointer previously set */
  if(conn->handler->done)
    result = conn->handler->done(data, status, premature);
  else
    result = status;

  if(CURLE_ABORTED_BY_CALLBACK != result) {
    /* avoid this if we already aborted by callback to avoid this calling
       another callback */
    CURLcode rc = Curl_pgrsDone(data);
    if(!result && rc)
      result = CURLE_ABORTED_BY_CALLBACK;
  }

  process_pending_handles(data->multi); /* connection / multiplex */

  CONNCACHE_LOCK(data);
  Curl_detach_connnection(data);
  if(CONN_INUSE(conn)) {
    /* Stop if still used. */
    CONNCACHE_UNLOCK(data);
    DEBUGF(infof(data, "Connection still in use %zu, "
                 "no more multi_done now!",
                 conn->easyq.size));
    return CURLE_OK;
  }

  data->state.done = TRUE; /* called just now! */

  if(conn->dns_entry) {
    Curl_resolv_unlock(data, conn->dns_entry); /* done with this */
    conn->dns_entry = NULL;
  }
  Curl_hostcache_prune(data);
  Curl_safefree(data->state.ulbuf);

  /* if the transfer was completed in a paused state there can be buffered
     data left to free */
  for(i = 0; i < data->state.tempcount; i++) {
    Curl_dyn_free(&data->state.tempwrite[i].b);
  }
  data->state.tempcount = 0;

  /* if data->set.reuse_forbid is TRUE, it means the libcurl client has
     forced us to close this connection. This is ignored for requests taking
     place in a NTLM/NEGOTIATE authentication handshake

     if conn->bits.close is TRUE, it means that the connection should be
     closed in spite of all our efforts to be nice, due to protocol
     restrictions in our or the server's end

     if premature is TRUE, it means this connection was said to be DONE before
     the entire request operation is complete and thus we can't know in what
     state it is for re-using, so we're forced to close it. In a perfect world
     we can add code that keep track of if we really must close it here or not,
     but currently we have no such detail knowledge.
  */

  if((data->set.reuse_forbid
#if defined(USE_NTLM)
      && !(conn->http_ntlm_state == NTLMSTATE_TYPE2 ||
           conn->proxy_ntlm_state == NTLMSTATE_TYPE2)
#endif
#if defined(USE_SPNEGO)
      && !(conn->http_negotiate_state == GSS_AUTHRECV ||
           conn->proxy_negotiate_state == GSS_AUTHRECV)
#endif
     ) || conn->bits.close
       || (premature && !(conn->handler->flags & PROTOPT_STREAM))) {
    CURLcode res2;
    connclose(conn, "disconnecting");
    Curl_conncache_remove_conn(data, conn, FALSE);
    CONNCACHE_UNLOCK(data);
    res2 = Curl_disconnect(data, conn, premature);

    /* If we had an error already, make sure we return that one. But
       if we got a new error, return that. */
    if(!result && res2)
      result = res2;
  }
  else {
    char buffer[256];
    const char *host =
#ifndef CURL_DISABLE_PROXY
      conn->bits.socksproxy ?
      conn->socks_proxy.host.dispname :
      conn->bits.httpproxy ? conn->http_proxy.host.dispname :
#endif
      conn->bits.conn_to_host ? conn->conn_to_host.dispname :
      conn->host.dispname;
    /* create string before returning the connection */
    msnprintf(buffer, sizeof(buffer),
              "Connection #%ld to host %s left intact",
              conn->connection_id, host);
    /* the connection is no longer in use by this transfer */
    CONNCACHE_UNLOCK(data);
    if(Curl_conncache_return_conn(data, conn)) {
      /* remember the most recently used connection */
      data->state.lastconnect_id = conn->connection_id;
      infof(data, "%s", buffer);
    }
    else
      data->state.lastconnect_id = -1;
  }

  Curl_safefree(data->state.buffer);
  Curl_free_request_state(data);
  return result;
}

static int close_connect_only(struct Curl_easy *data,
                              struct connectdata *conn, void *param)
{
  (void)param;
  if(data->state.lastconnect_id != conn->connection_id)
    return 0;

  if(!conn->bits.connect_only)
    return 1;

  connclose(conn, "Removing connect-only easy handle");

  return 1;
}

CURLMcode curl_multi_remove_handle(struct Curl_multi *multi,
                                   struct Curl_easy *data)
{
  struct Curl_easy *easy = data;
  bool premature;
  struct Curl_llist_element *e;
  CURLMcode rc;

  /* First, make some basic checks that the CURLM handle is a good handle */
  if(!GOOD_MULTI_HANDLE(multi))
    return CURLM_BAD_HANDLE;

  /* Verify that we got a somewhat good easy handle too */
  if(!GOOD_EASY_HANDLE(data))
    return CURLM_BAD_EASY_HANDLE;

  /* Prevent users from trying to remove same easy handle more than once */
  if(!data->multi)
    return CURLM_OK; /* it is already removed so let's say it is fine! */

  /* Prevent users from trying to remove an easy handle from the wrong multi */
  if(data->multi != multi)
    return CURLM_BAD_EASY_HANDLE;

  if(multi->in_callback)
    return CURLM_RECURSIVE_API_CALL;

  premature = (data->mstate < MSTATE_COMPLETED) ? TRUE : FALSE;

  /* If the 'state' is not INIT or COMPLETED, we might need to do something
     nice to put the easy_handle in a good known state when this returns. */
  if(premature) {
    /* this handle is "alive" so we need to count down the total number of
       alive connections when this is removed */
    multi->num_alive--;
  }

  if(data->conn &&
     data->mstate > MSTATE_DO &&
     data->mstate < MSTATE_COMPLETED) {
    /* Set connection owner so that the DONE function closes it.  We can
       safely do this here since connection is killed. */
    streamclose(data->conn, "Removed with partial response");
  }

  if(data->conn) {
    /* multi_done() clears the association between the easy handle and the
       connection.

       Note that this ignores the return code simply because there's
       nothing really useful to do with it anyway! */
    (void)multi_done(data, data->result, premature);
  }

  /* The timer must be shut down before data->multi is set to NULL, else the
     timenode will remain in the splay tree after curl_easy_cleanup is
     called. Do it after multi_done() in case that sets another time! */
  Curl_expire_clear(data);

  if(data->connect_queue.ptr)
    /* the handle was in the pending list waiting for an available connection,
       so go ahead and remove it */
    Curl_llist_remove(&multi->pending, &data->connect_queue, NULL);

  if(data->dns.hostcachetype == HCACHE_MULTI) {
    /* stop using the multi handle's DNS cache, *after* the possible
       multi_done() call above */
    data->dns.hostcache = NULL;
    data->dns.hostcachetype = HCACHE_NONE;
  }

  Curl_wildcard_dtor(&data->wildcard);

  /* destroy the timeout list that is held in the easy handle, do this *after*
     multi_done() as that may actually call Curl_expire that uses this */
  Curl_llist_destroy(&data->state.timeoutlist, NULL);

  /* change state without using multistate(), only to make singlesocket() do
     what we want */
  data->mstate = MSTATE_COMPLETED;

  /* This ignores the return code even in case of problems because there's
     nothing more to do about that, here */
  (void)singlesocket(multi, easy); /* to let the application know what sockets
                                      that vanish with this handle */

  /* Remove the association between the connection and the handle */
  Curl_detach_connnection(data);

  if(data->state.lastconnect_id != -1) {
    /* Mark any connect-only connection for closure */
    Curl_conncache_foreach(data, data->state.conn_cache,
                           NULL, close_connect_only);
  }

#ifdef USE_LIBPSL
  /* Remove the PSL association. */
  if(data->psl == &multi->psl)
    data->psl = NULL;
#endif

  /* as this was using a shared connection cache we clear the pointer to that
     since we're not part of that multi handle anymore */
  data->state.conn_cache = NULL;

  data->multi = NULL; /* clear the association to this multi handle */

  /* make sure there's no pending message in the queue sent from this easy
     handle */

  for(e = multi->msglist.head; e; e = e->next) {
    struct Curl_message *msg = e->ptr;

    if(msg->extmsg.easy_handle == easy) {
      Curl_llist_remove(&multi->msglist, e, NULL);
      /* there can only be one from this specific handle */
      break;
    }
  }

  /* Remove from the pending list if it is there. Otherwise this will
     remain on the pending list forever due to the state change. */
  for(e = multi->pending.head; e; e = e->next) {
    struct Curl_easy *curr_data = e->ptr;

    if(curr_data == data) {
      Curl_llist_remove(&multi->pending, e, NULL);
      break;
    }
  }

  /* make the previous node point to our next */
  if(data->prev)
    data->prev->next = data->next;
  else
    multi->easyp = data->next; /* point to first node */

  /* make our next point to our previous node */
  if(data->next)
    data->next->prev = data->prev;
  else
    multi->easylp = data->prev; /* point to last node */

  /* NOTE NOTE NOTE
     We do not touch the easy handle here! */
  multi->num_easy--; /* one less to care about now */

  process_pending_handles(multi);

  rc = Curl_update_timer(multi);
  if(rc)
    return rc;
  return CURLM_OK;
}

/* Return TRUE if the application asked for multiplexing */
bool Curl_multiplex_wanted(const struct Curl_multi *multi)
{
  return (multi && (multi->multiplexing));
}

/*
 * Curl_detach_connnection() removes the given transfer from the connection.
 *
 * This is the only function that should clear data->conn. This will
 * occasionally be called with the data->conn pointer already cleared.
 */
void Curl_detach_connnection(struct Curl_easy *data)
{
  struct connectdata *conn = data->conn;
  if(conn) {
    Curl_connect_done(data); /* if mid-CONNECT, shut it down */
    Curl_llist_remove(&conn->easyq, &data->conn_queue, NULL);
    Curl_ssl_detach_conn(data, conn);
  }
  data->conn = NULL;
}

/*
 * Curl_attach_connnection() attaches this transfer to this connection.
 *
 * This is the only function that should assign data->conn
 */
void Curl_attach_connnection(struct Curl_easy *data,
                             struct connectdata *conn)
{
  DEBUGASSERT(!data->conn);
  DEBUGASSERT(conn);
  data->conn = conn;
  Curl_llist_insert_next(&conn->easyq, conn->easyq.tail, data,
                         &data->conn_queue);
  if(conn->handler->attach)
    conn->handler->attach(data, conn);
  Curl_ssl_associate_conn(data, conn);
}

static int waitconnect_getsock(struct connectdata *conn,
                               curl_socket_t *sock)
{
  int i;
  int s = 0;
  int rc = 0;

#ifdef USE_SSL
#ifndef CURL_DISABLE_PROXY
  if(CONNECT_FIRSTSOCKET_PROXY_SSL())
    return Curl_ssl->getsock(conn, sock);
#endif
#endif

  if(SOCKS_STATE(conn->cnnct.state))
    return Curl_SOCKS_getsock(conn, sock, FIRSTSOCKET);

  for(i = 0; i<2; i++) {
    if(conn->tempsock[i] != CURL_SOCKET_BAD) {
      sock[s] = conn->tempsock[i];
      rc |= GETSOCK_WRITESOCK(s);
#ifdef ENABLE_QUIC
      if(conn->transport == TRNSPRT_QUIC)
        /* when connecting QUIC, we want to read the socket too */
        rc |= GETSOCK_READSOCK(s);
#endif
      s++;
    }
  }

  return rc;
}

static int waitproxyconnect_getsock(struct connectdata *conn,
                                    curl_socket_t *sock)
{
  sock[0] = conn->sock[FIRSTSOCKET];

  if(conn->connect_state)
    return Curl_connect_getsock(conn);

  return GETSOCK_WRITESOCK(0);
}

static int domore_getsock(struct Curl_easy *data,
                          struct connectdata *conn,
                          curl_socket_t *socks)
{
  if(conn && conn->handler->domore_getsock)
    return conn->handler->domore_getsock(data, conn, socks);
  return GETSOCK_BLANK;
}

static int doing_getsock(struct Curl_easy *data,
                         struct connectdata *conn,
                         curl_socket_t *socks)
{
  if(conn && conn->handler->doing_getsock)
    return conn->handler->doing_getsock(data, conn, socks);
  return GETSOCK_BLANK;
}

static int protocol_getsock(struct Curl_easy *data,
                            struct connectdata *conn,
                            curl_socket_t *socks)
{
  if(conn->handler->proto_getsock)
    return conn->handler->proto_getsock(data, conn, socks);
  /* Backup getsock logic. Since there is a live socket in use, we must wait
     for it or it will be removed from watching when the multi_socket API is
     used. */
  socks[0] = conn->sock[FIRSTSOCKET];
  return GETSOCK_READSOCK(0) | GETSOCK_WRITESOCK(0);
}

/* returns bitmapped flags for this handle and its sockets. The 'socks[]'
   array contains MAX_SOCKSPEREASYHANDLE entries. */
static int multi_getsock(struct Curl_easy *data,
                         curl_socket_t *socks)
{
  struct connectdata *conn = data->conn;
  /* The no connection case can happen when this is called from
     curl_multi_remove_handle() => singlesocket() => multi_getsock().
  */
  if(!conn)
    return 0;

  switch(data->mstate) {
  default:
    return 0;

  case MSTATE_RESOLVING:
    return Curl_resolv_getsock(data, socks);

  case MSTATE_PROTOCONNECTING:
  case MSTATE_PROTOCONNECT:
    return protocol_getsock(data, conn, socks);

  case MSTATE_DO:
  case MSTATE_DOING:
    return doing_getsock(data, conn, socks);

  case MSTATE_TUNNELING:
    return waitproxyconnect_getsock(conn, socks);

  case MSTATE_CONNECTING:
    return waitconnect_getsock(conn, socks);

  case MSTATE_DOING_MORE:
    return domore_getsock(data, conn, socks);

  case MSTATE_DID: /* since is set after DO is completed, we switch to
                        waiting for the same as the PERFORMING state */
  case MSTATE_PERFORMING:
    return Curl_single_getsock(data, conn, socks);
  }

}

CURLMcode curl_multi_fdset(struct Curl_multi *multi,
                           fd_set *read_fd_set, fd_set *write_fd_set,
                           fd_set *exc_fd_set, int *max_fd)
{
  /* Scan through all the easy handles to get the file descriptors set.
     Some easy handles may not have connected to the remote host yet,
     and then we must make sure that is done. */
  struct Curl_easy *data;
  int this_max_fd = -1;
  curl_socket_t sockbunch[MAX_SOCKSPEREASYHANDLE];
  int i;
  (void)exc_fd_set; /* not used */

  if(!GOOD_MULTI_HANDLE(multi))
    return CURLM_BAD_HANDLE;

  if(multi->in_callback)
    return CURLM_RECURSIVE_API_CALL;

  data = multi->easyp;
  while(data) {
    int bitmap;
#ifdef __clang_analyzer_
    /* to prevent "The left operand of '>=' is a garbage value" warnings */
    memset(sockbunch, 0, sizeof(sockbunch));
#endif
    bitmap = multi_getsock(data, sockbunch);

    for(i = 0; i< MAX_SOCKSPEREASYHANDLE; i++) {
      curl_socket_t s = CURL_SOCKET_BAD;

      if((bitmap & GETSOCK_READSOCK(i)) && VALID_SOCK(sockbunch[i])) {
        if(!FDSET_SOCK(sockbunch[i]))
          /* pretend it doesn't exist */
          continue;
        FD_SET(sockbunch[i], read_fd_set);
        s = sockbunch[i];
      }
      if((bitmap & GETSOCK_WRITESOCK(i)) && VALID_SOCK(sockbunch[i])) {
        if(!FDSET_SOCK(sockbunch[i]))
          /* pretend it doesn't exist */
          continue;
        FD_SET(sockbunch[i], write_fd_set);
        s = sockbunch[i];
      }
      if(s == CURL_SOCKET_BAD)
        /* this socket is unused, break out of loop */
        break;
      if((int)s > this_max_fd)
        this_max_fd = (int)s;
    }

    data = data->next; /* check next handle */
  }

  *max_fd = this_max_fd;

  return CURLM_OK;
}

#define NUM_POLLS_ON_STACK 10

static CURLMcode multi_wait(struct Curl_multi *multi,
                            struct curl_waitfd extra_fds[],
                            unsigned int extra_nfds,
                            int timeout_ms,
                            int *ret,
                            bool extrawait, /* when no socket, wait */
                            bool use_wakeup)
{
  struct Curl_easy *data;
  curl_socket_t sockbunch[MAX_SOCKSPEREASYHANDLE];
  int bitmap;
  unsigned int i;
  unsigned int nfds = 0;
  unsigned int curlfds;
  long timeout_internal;
  int retcode = 0;
  struct pollfd a_few_on_stack[NUM_POLLS_ON_STACK];
  struct pollfd *ufds = &a_few_on_stack[0];
  bool ufds_malloc = FALSE;
#ifdef USE_WINSOCK
  WSANETWORKEVENTS wsa_events;
  DEBUGASSERT(multi->wsa_event != WSA_INVALID_EVENT);
#endif
#ifndef ENABLE_WAKEUP
  (void)use_wakeup;
#endif

  if(!GOOD_MULTI_HANDLE(multi))
    return CURLM_BAD_HANDLE;

  if(multi->in_callback)
    return CURLM_RECURSIVE_API_CALL;

  if(timeout_ms < 0)
    return CURLM_BAD_FUNCTION_ARGUMENT;

  /* Count up how many fds we have from the multi handle */
  data = multi->easyp;
  while(data) {
    bitmap = multi_getsock(data, sockbunch);

    for(i = 0; i< MAX_SOCKSPEREASYHANDLE; i++) {
      curl_socket_t s = CURL_SOCKET_BAD;

      if((bitmap & GETSOCK_READSOCK(i)) && VALID_SOCK((sockbunch[i]))) {
        ++nfds;
        s = sockbunch[i];
      }
      if((bitmap & GETSOCK_WRITESOCK(i)) && VALID_SOCK((sockbunch[i]))) {
        ++nfds;
        s = sockbunch[i];
      }
      if(s == CURL_SOCKET_BAD) {
        break;
      }
    }

    data = data->next; /* check next handle */
  }

  /* If the internally desired timeout is actually shorter than requested from
     the outside, then use the shorter time! But only if the internal timer
     is actually larger than -1! */
  (void)multi_timeout(multi, &timeout_internal);
  if((timeout_internal >= 0) && (timeout_internal < (long)timeout_ms))
    timeout_ms = (int)timeout_internal;

  curlfds = nfds; /* number of internal file descriptors */
  nfds += extra_nfds; /* add the externally provided ones */

#ifdef ENABLE_WAKEUP
#ifdef USE_WINSOCK
  if(use_wakeup) {
#else
  if(use_wakeup && multi->wakeup_pair[0] != CURL_SOCKET_BAD) {
#endif
    ++nfds;
  }
#endif

  if(nfds > NUM_POLLS_ON_STACK) {
    /* 'nfds' is a 32 bit value and 'struct pollfd' is typically 8 bytes
       big, so at 2^29 sockets this value might wrap. When a process gets
       the capability to actually handle over 500 million sockets this
       calculation needs a integer overflow check. */
    ufds = malloc(nfds * sizeof(struct pollfd));
    if(!ufds)
      return CURLM_OUT_OF_MEMORY;
    ufds_malloc = TRUE;
  }
  nfds = 0;

  /* only do the second loop if we found descriptors in the first stage run
     above */

  if(curlfds) {
    /* Add the curl handles to our pollfds first */
    data = multi->easyp;
    while(data) {
      bitmap = multi_getsock(data, sockbunch);

      for(i = 0; i < MAX_SOCKSPEREASYHANDLE; i++) {
        curl_socket_t s = CURL_SOCKET_BAD;
#ifdef USE_WINSOCK
        long mask = 0;
#endif
        if((bitmap & GETSOCK_READSOCK(i)) && VALID_SOCK((sockbunch[i]))) {
          s = sockbunch[i];
#ifdef USE_WINSOCK
          mask |= FD_READ|FD_ACCEPT|FD_CLOSE;
#endif
          ufds[nfds].fd = s;
          ufds[nfds].events = POLLIN;
          ++nfds;
        }
        if((bitmap & GETSOCK_WRITESOCK(i)) && VALID_SOCK((sockbunch[i]))) {
          s = sockbunch[i];
#ifdef USE_WINSOCK
          mask |= FD_WRITE|FD_CONNECT|FD_CLOSE;
          send(s, NULL, 0, 0); /* reset FD_WRITE */
#endif
          ufds[nfds].fd = s;
          ufds[nfds].events = POLLOUT;
          ++nfds;
        }
        /* s is only set if either being readable or writable is checked */
        if(s == CURL_SOCKET_BAD) {
          /* break on entry not checked for being readable or writable */
          break;
        }
#ifdef USE_WINSOCK
        if(WSAEventSelect(s, multi->wsa_event, mask) != 0) {
          if(ufds_malloc)
            free(ufds);
          return CURLM_INTERNAL_ERROR;
        }
#endif
      }

      data = data->next; /* check next handle */
    }
  }

  /* Add external file descriptions from poll-like struct curl_waitfd */
  for(i = 0; i < extra_nfds; i++) {
#ifdef USE_WINSOCK
    long mask = 0;
    if(extra_fds[i].events & CURL_WAIT_POLLIN)
      mask |= FD_READ|FD_ACCEPT|FD_CLOSE;
    if(extra_fds[i].events & CURL_WAIT_POLLPRI)
      mask |= FD_OOB;
    if(extra_fds[i].events & CURL_WAIT_POLLOUT) {
      mask |= FD_WRITE|FD_CONNECT|FD_CLOSE;
      send(extra_fds[i].fd, NULL, 0, 0); /* reset FD_WRITE */
    }
    if(WSAEventSelect(extra_fds[i].fd, multi->wsa_event, mask) != 0) {
      if(ufds_malloc)
        free(ufds);
      return CURLM_INTERNAL_ERROR;
    }
#endif
    ufds[nfds].fd = extra_fds[i].fd;
    ufds[nfds].events = 0;
    if(extra_fds[i].events & CURL_WAIT_POLLIN)
      ufds[nfds].events |= POLLIN;
    if(extra_fds[i].events & CURL_WAIT_POLLPRI)
      ufds[nfds].events |= POLLPRI;
    if(extra_fds[i].events & CURL_WAIT_POLLOUT)
      ufds[nfds].events |= POLLOUT;
    ++nfds;
  }

#ifdef ENABLE_WAKEUP
#ifndef USE_WINSOCK
  if(use_wakeup && multi->wakeup_pair[0] != CURL_SOCKET_BAD) {
    ufds[nfds].fd = multi->wakeup_pair[0];
    ufds[nfds].events = POLLIN;
    ++nfds;
  }
#endif
#endif

#if defined(ENABLE_WAKEUP) && defined(USE_WINSOCK)
  if(nfds || use_wakeup) {
#else
  if(nfds) {
#endif
    int pollrc;
#ifdef USE_WINSOCK
    if(nfds)
      pollrc = Curl_poll(ufds, nfds, 0); /* just pre-check with WinSock */
    else
      pollrc = 0;
    if(pollrc <= 0) /* now wait... if not ready during the pre-check above */
      WSAWaitForMultipleEvents(1, &multi->wsa_event, FALSE, timeout_ms, FALSE);
#else
    pollrc = Curl_poll(ufds, nfds, timeout_ms); /* wait... */
#endif

    if(pollrc > 0) {
      retcode = pollrc;
#ifdef USE_WINSOCK
    }
    /* With WinSock, we have to run the following section unconditionally
       to call WSAEventSelect(fd, event, 0) on all the sockets */
    {
#endif
      /* copy revents results from the poll to the curl_multi_wait poll
         struct, the bit values of the actual underlying poll() implementation
         may not be the same as the ones in the public libcurl API! */
      for(i = 0; i < extra_nfds; i++) {
        unsigned r = ufds[curlfds + i].revents;
        unsigned short mask = 0;
#ifdef USE_WINSOCK
        wsa_events.lNetworkEvents = 0;
        if(WSAEnumNetworkEvents(extra_fds[i].fd, NULL, &wsa_events) == 0) {
          if(wsa_events.lNetworkEvents & (FD_READ|FD_ACCEPT|FD_CLOSE))
            mask |= CURL_WAIT_POLLIN;
          if(wsa_events.lNetworkEvents & (FD_WRITE|FD_CONNECT|FD_CLOSE))
            mask |= CURL_WAIT_POLLOUT;
          if(wsa_events.lNetworkEvents & FD_OOB)
            mask |= CURL_WAIT_POLLPRI;
          if(ret && pollrc <= 0 && wsa_events.lNetworkEvents)
            retcode++;
        }
        WSAEventSelect(extra_fds[i].fd, multi->wsa_event, 0);
        if(pollrc <= 0)
          continue;
#endif
        if(r & POLLIN)
          mask |= CURL_WAIT_POLLIN;
        if(r & POLLOUT)
          mask |= CURL_WAIT_POLLOUT;
        if(r & POLLPRI)
          mask |= CURL_WAIT_POLLPRI;
        extra_fds[i].revents = mask;
      }

#ifdef USE_WINSOCK
      /* Count up all our own sockets that had activity,
         and remove them from the event. */
      if(curlfds) {
        data = multi->easyp;
        while(data) {
          bitmap = multi_getsock(data, sockbunch);

          for(i = 0; i < MAX_SOCKSPEREASYHANDLE; i++) {
            if(bitmap & (GETSOCK_READSOCK(i) | GETSOCK_WRITESOCK(i))) {
              wsa_events.lNetworkEvents = 0;
              if(WSAEnumNetworkEvents(sockbunch[i], NULL, &wsa_events) == 0) {
                if(ret && pollrc <= 0 && wsa_events.lNetworkEvents)
                  retcode++;
              }
              WSAEventSelect(sockbunch[i], multi->wsa_event, 0);
            }
            else {
              /* break on entry not checked for being readable or writable */
              break;
            }
          }

          data = data->next;
        }
      }

      WSAResetEvent(multi->wsa_event);
#else
#ifdef ENABLE_WAKEUP
      if(use_wakeup && multi->wakeup_pair[0] != CURL_SOCKET_BAD) {
        if(ufds[curlfds + extra_nfds].revents & POLLIN) {
          char buf[64];
          ssize_t nread;
          while(1) {
            /* the reading socket is non-blocking, try to read
               data from it until it receives an error (except EINTR).
               In normal cases it will get EAGAIN or EWOULDBLOCK
               when there is no more data, breaking the loop. */
            nread = sread(multi->wakeup_pair[0], buf, sizeof(buf));
            if(nread <= 0) {
              if(nread < 0 && EINTR == SOCKERRNO)
                continue;
              break;
            }
          }
          /* do not count the wakeup socket into the returned value */
          retcode--;
        }
      }
#endif
#endif
    }
  }

  if(ufds_malloc)
    free(ufds);
  if(ret)
    *ret = retcode;
#if defined(ENABLE_WAKEUP) && defined(USE_WINSOCK)
  if(extrawait && !nfds && !use_wakeup) {
#else
  if(extrawait && !nfds) {
#endif
    long sleep_ms = 0;

    /* Avoid busy-looping when there's nothing particular to wait for */
    if(!curl_multi_timeout(multi, &sleep_ms) && sleep_ms) {
      if(sleep_ms > timeout_ms)
        sleep_ms = timeout_ms;
      /* when there are no easy handles in the multi, this holds a -1
         timeout */
      else if(sleep_ms < 0)
        sleep_ms = timeout_ms;
      Curl_wait_ms(sleep_ms);
    }
  }

  return CURLM_OK;
}

CURLMcode curl_multi_wait(struct Curl_multi *multi,
                          struct curl_waitfd extra_fds[],
                          unsigned int extra_nfds,
                          int timeout_ms,
                          int *ret)
{
  return multi_wait(multi, extra_fds, extra_nfds, timeout_ms, ret, FALSE,
                    FALSE);
}

CURLMcode curl_multi_poll(struct Curl_multi *multi,
                          struct curl_waitfd extra_fds[],
                          unsigned int extra_nfds,
                          int timeout_ms,
                          int *ret)
{
  return multi_wait(multi, extra_fds, extra_nfds, timeout_ms, ret, TRUE,
                    TRUE);
}

CURLMcode curl_multi_wakeup(struct Curl_multi *multi)
{
  /* this function is usually called from another thread,
     it has to be careful only to access parts of the
     Curl_multi struct that are constant */

  /* GOOD_MULTI_HANDLE can be safely called */
  if(!GOOD_MULTI_HANDLE(multi))
    return CURLM_BAD_HANDLE;

#ifdef ENABLE_WAKEUP
#ifdef USE_WINSOCK
  if(WSASetEvent(multi->wsa_event))
    return CURLM_OK;
#else
  /* the wakeup_pair variable is only written during init and cleanup,
     making it safe to access from another thread after the init part
     and before cleanup */
  if(multi->wakeup_pair[1] != CURL_SOCKET_BAD) {
    char buf[1];
    buf[0] = 1;
    while(1) {
      /* swrite() is not thread-safe in general, because concurrent calls
         can have their messages interleaved, but in this case the content
         of the messages does not matter, which makes it ok to call.

         The write socket is set to non-blocking, this way this function
         cannot block, making it safe to call even from the same thread
         that will call curl_multi_wait(). If swrite() returns that it
         would block, it's considered successful because it means that
         previous calls to this function will wake up the poll(). */
      if(swrite(multi->wakeup_pair[1], buf, sizeof(buf)) < 0) {
        int err = SOCKERRNO;
        int return_success;
#ifdef USE_WINSOCK
        return_success = WSAEWOULDBLOCK == err;
#else
        if(EINTR == err)
          continue;
        return_success = EWOULDBLOCK == err || EAGAIN == err;
#endif
        if(!return_success)
          return CURLM_WAKEUP_FAILURE;
      }
      return CURLM_OK;
    }
  }
#endif
#endif
  return CURLM_WAKEUP_FAILURE;
}

/*
 * multi_ischanged() is called
 *
 * Returns TRUE/FALSE whether the state is changed to trigger a CONNECT_PEND
 * => CONNECT action.
 *
 * Set 'clear' to TRUE to have it also clear the state variable.
 */
static bool multi_ischanged(struct Curl_multi *multi, bool clear)
{
  bool retval = multi->recheckstate;
  if(clear)
    multi->recheckstate = FALSE;
  return retval;
}

CURLMcode Curl_multi_add_perform(struct Curl_multi *multi,
                                 struct Curl_easy *data,
                                 struct connectdata *conn)
{
  CURLMcode rc;

  if(multi->in_callback)
    return CURLM_RECURSIVE_API_CALL;

  rc = curl_multi_add_handle(multi, data);
  if(!rc) {
    struct SingleRequest *k = &data->req;

    /* pass in NULL for 'conn' here since we don't want to init the
       connection, only this transfer */
    Curl_init_do(data, NULL);

    /* take this handle to the perform state right away */
    multistate(data, MSTATE_PERFORMING);
    Curl_attach_connnection(data, conn);
    k->keepon |= KEEP_RECV; /* setup to receive! */
  }
  return rc;
}

static CURLcode multi_do(struct Curl_easy *data, bool *done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;

  DEBUGASSERT(conn);
  DEBUGASSERT(conn->handler);

  if(conn->handler->do_it)
    /* generic protocol-specific function pointer set in curl_connect() */
    result = conn->handler->do_it(data, done);

  return result;
}

/*
 * multi_do_more() is called during the DO_MORE multi state. It is basically a
 * second stage DO state which (wrongly) was introduced to support FTP's
 * second connection.
 *
 * 'complete' can return 0 for incomplete, 1 for done and -1 for go back to
 * DOING state there's more work to do!
 */

static CURLcode multi_do_more(struct Curl_easy *data, int *complete)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;

  *complete = 0;

  if(conn->handler->do_more)
    result = conn->handler->do_more(data, complete);

  return result;
}

/*
 * Check whether a timeout occurred, and handle it if it did
 */
static bool multi_handle_timeout(struct Curl_easy *data,
                                 struct curltime *now,
                                 bool *stream_error,
                                 CURLcode *result,
                                 bool connect_timeout)
{
  timediff_t timeout_ms;
  timeout_ms = Curl_timeleft(data, now, connect_timeout);

  if(timeout_ms < 0) {
    /* Handle timed out */
    if(data->mstate == MSTATE_RESOLVING)
      failf(data, "Resolving timed out after %" CURL_FORMAT_TIMEDIFF_T
            " milliseconds",
            Curl_timediff(*now, data->progress.t_startsingle));
    else if(data->mstate == MSTATE_CONNECTING)
      failf(data, "Connection timed out after %" CURL_FORMAT_TIMEDIFF_T
            " milliseconds",
            Curl_timediff(*now, data->progress.t_startsingle));
    else {
      struct SingleRequest *k = &data->req;
      if(k->size != -1) {
        failf(data, "Operation timed out after %" CURL_FORMAT_TIMEDIFF_T
              " milliseconds with %" CURL_FORMAT_CURL_OFF_T " out of %"
              CURL_FORMAT_CURL_OFF_T " bytes received",
              Curl_timediff(*now, data->progress.t_startsingle),
              k->bytecount, k->size);
      }
      else {
        failf(data, "Operation timed out after %" CURL_FORMAT_TIMEDIFF_T
              " milliseconds with %" CURL_FORMAT_CURL_OFF_T
              " bytes received",
              Curl_timediff(*now, data->progress.t_startsingle),
              k->bytecount);
      }
    }

    /* Force connection closed if the connection has indeed been used */
    if(data->mstate > MSTATE_DO) {
      streamclose(data->conn, "Disconnected with pending data");
      *stream_error = TRUE;
    }
    *result = CURLE_OPERATION_TIMEDOUT;
    (void)multi_done(data, *result, TRUE);
  }

  return (timeout_ms < 0);
}

/*
 * We are doing protocol-specific connecting and this is being called over and
 * over from the multi interface until the connection phase is done on
 * protocol layer.
 */

static CURLcode protocol_connecting(struct Curl_easy *data, bool *done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;

  if(conn && conn->handler->connecting) {
    *done = FALSE;
    result = conn->handler->connecting(data, done);
  }
  else
    *done = TRUE;

  return result;
}

/*
 * We are DOING this is being called over and over from the multi interface
 * until the DOING phase is done on protocol layer.
 */

static CURLcode protocol_doing(struct Curl_easy *data, bool *done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;

  if(conn && conn->handler->doing) {
    *done = FALSE;
    result = conn->handler->doing(data, done);
  }
  else
    *done = TRUE;

  return result;
}

/*
 * We have discovered that the TCP connection has been successful, we can now
 * proceed with some action.
 *
 */
static CURLcode protocol_connect(struct Curl_easy *data,
                                 bool *protocol_done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;
  DEBUGASSERT(conn);
  DEBUGASSERT(protocol_done);

  *protocol_done = FALSE;

  if(conn->bits.tcpconnect[FIRSTSOCKET] && conn->bits.protoconnstart) {
    /* We already are connected, get back. This may happen when the connect
       worked fine in the first call, like when we connect to a local server
       or proxy. Note that we don't know if the protocol is actually done.

       Unless this protocol doesn't have any protocol-connect callback, as
       then we know we're done. */
    if(!conn->handler->connecting)
      *protocol_done = TRUE;

    return CURLE_OK;
  }

  if(!conn->bits.protoconnstart) {
#ifndef CURL_DISABLE_PROXY
    result = Curl_proxy_connect(data, FIRSTSOCKET);
    if(result)
      return result;

    if(CONNECT_FIRSTSOCKET_PROXY_SSL())
      /* wait for HTTPS proxy SSL initialization to complete */
      return CURLE_OK;

    if(conn->bits.tunnel_proxy && conn->bits.httpproxy &&
       Curl_connect_ongoing(conn))
      /* when using an HTTP tunnel proxy, await complete tunnel establishment
         before proceeding further. Return CURLE_OK so we'll be called again */
      return CURLE_OK;
#endif
    if(conn->handler->connect_it) {
      /* is there a protocol-specific connect() procedure? */

      /* Call the protocol-specific connect function */
      result = conn->handler->connect_it(data, protocol_done);
    }
    else
      *protocol_done = TRUE;

    /* it has started, possibly even completed but that knowledge isn't stored
       in this bit! */
    if(!result)
      conn->bits.protoconnstart = TRUE;
  }

  return result; /* pass back status */
}

/*
 * Curl_preconnect() is called immediately before a connect starts. When a
 * redirect is followed, this is then called multiple times during a single
 * transfer.
 */
CURLcode Curl_preconnect(struct Curl_easy *data)
{
  if(!data->state.buffer) {
    data->state.buffer = malloc(data->set.buffer_size + 1);
    if(!data->state.buffer)
      return CURLE_OUT_OF_MEMORY;
  }
  return CURLE_OK;
}


static CURLMcode multi_runsingle(struct Curl_multi *multi,
                                 struct curltime *nowp,
                                 struct Curl_easy *data)
{
  struct Curl_message *msg = NULL;
  bool connected;
  bool async;
  bool protocol_connected = FALSE;
  bool dophase_done = FALSE;
  bool done = FALSE;
  CURLMcode rc;
  CURLcode result = CURLE_OK;
  timediff_t recv_timeout_ms;
  timediff_t send_timeout_ms;
  int control;

  if(!GOOD_EASY_HANDLE(data))
    return CURLM_BAD_EASY_HANDLE;

  if(multi->dead) {
    /* a multi-level callback returned error before, meaning every individual
     transfer now has failed */
    result = CURLE_ABORTED_BY_CALLBACK;
    Curl_posttransfer(data);
    multi_done(data, result, FALSE);
    multistate(data, MSTATE_COMPLETED);
  }

  do {
    /* A "stream" here is a logical stream if the protocol can handle that
       (HTTP/2), or the full connection for older protocols */
    bool stream_error = FALSE;
    rc = CURLM_OK;

    if(multi_ischanged(multi, TRUE)) {
      DEBUGF(infof(data, "multi changed, check CONNECT_PEND queue!"));
      process_pending_handles(multi); /* multiplexed */
    }

    if(data->mstate > MSTATE_CONNECT &&
       data->mstate < MSTATE_COMPLETED) {
      /* Make sure we set the connection's current owner */
      DEBUGASSERT(data->conn);
      if(!data->conn)
        return CURLM_INTERNAL_ERROR;
    }

    if(data->conn &&
       (data->mstate >= MSTATE_CONNECT) &&
       (data->mstate < MSTATE_COMPLETED)) {
      /* Check for overall operation timeout here but defer handling the
       * connection timeout to later, to allow for a connection to be set up
       * in the window since we last checked timeout. This prevents us
       * tearing down a completed connection in the case where we were slow
       * to check the timeout (e.g. process descheduled during this loop).
       * We set connect_timeout=FALSE to do this. */

      /* we need to wait for the connect state as only then is the start time
         stored, but we must not check already completed handles */
      if(multi_handle_timeout(data, nowp, &stream_error, &result, FALSE)) {
        /* Skip the statemachine and go directly to error handling section. */
        goto statemachine_end;
      }
    }

    switch(data->mstate) {
    case MSTATE_INIT:
      /* init this transfer. */
      result = Curl_pretransfer(data);

      if(!result) {
        /* after init, go CONNECT */
        multistate(data, MSTATE_CONNECT);
        *nowp = Curl_pgrsTime(data, TIMER_STARTOP);
        rc = CURLM_CALL_MULTI_PERFORM;
      }
      break;

    case MSTATE_PENDING:
      /* We will stay here until there is a connection available. Then
         we try again in the MSTATE_CONNECT state. */
      break;

    case MSTATE_CONNECT:
      /* Connect. We want to get a connection identifier filled in. */
      /* init this transfer. */
      result = Curl_preconnect(data);
      if(result)
        break;

      *nowp = Curl_pgrsTime(data, TIMER_STARTSINGLE);
      if(data->set.timeout)
        Curl_expire(data, data->set.timeout, EXPIRE_TIMEOUT);

      if(data->set.connecttimeout)
        Curl_expire(data, data->set.connecttimeout, EXPIRE_CONNECTTIMEOUT);

      result = Curl_connect(data, &async, &protocol_connected);
      if(CURLE_NO_CONNECTION_AVAILABLE == result) {
        /* There was no connection available. We will go to the pending
           state and wait for an available connection. */
        multistate(data, MSTATE_PENDING);

        /* add this handle to the list of connect-pending handles */
        Curl_llist_insert_next(&multi->pending, multi->pending.tail, data,
                               &data->connect_queue);
        result = CURLE_OK;
        break;
      }
      else if(data->state.previouslypending) {
        /* this transfer comes from the pending queue so try move another */
        infof(data, "Transfer was pending, now try another");
        process_pending_handles(data->multi);
      }

      if(!result) {
        if(async)
          /* We're now waiting for an asynchronous name lookup */
          multistate(data, MSTATE_RESOLVING);
        else {
          /* after the connect has been sent off, go WAITCONNECT unless the
             protocol connect is already done and we can go directly to
             WAITDO or DO! */
          rc = CURLM_CALL_MULTI_PERFORM;

          if(protocol_connected)
            multistate(data, MSTATE_DO);
          else {
#ifndef CURL_DISABLE_HTTP
            if(Curl_connect_ongoing(data->conn))
              multistate(data, MSTATE_TUNNELING);
            else
#endif
              multistate(data, MSTATE_CONNECTING);
          }
        }
      }
      break;

    case MSTATE_RESOLVING:
      /* awaiting an asynch name resolve to complete */
    {
      struct Curl_dns_entry *dns = NULL;
      struct connectdata *conn = data->conn;
      const char *hostname;

      DEBUGASSERT(conn);
#ifndef CURL_DISABLE_PROXY
      if(conn->bits.httpproxy)
        hostname = conn->http_proxy.host.name;
      else
#endif
        if(conn->bits.conn_to_host)
        hostname = conn->conn_to_host.name;
      else
        hostname = conn->host.name;

      /* check if we have the name resolved by now */
      dns = Curl_fetch_addr(data, hostname, (int)conn->port);

      if(dns) {
#ifdef CURLRES_ASYNCH
        data->state.async.dns = dns;
        data->state.async.done = TRUE;
#endif
        result = CURLE_OK;
        infof(data, "Hostname '%s' was found in DNS cache", hostname);
      }

      if(!dns)
        result = Curl_resolv_check(data, &dns);

      /* Update sockets here, because the socket(s) may have been
         closed and the application thus needs to be told, even if it
         is likely that the same socket(s) will again be used further
         down.  If the name has not yet been resolved, it is likely
         that new sockets have been opened in an attempt to contact
         another resolver. */
      rc = singlesocket(multi, data);
      if(rc)
        return rc;

      if(dns) {
        /* Perform the next step in the connection phase, and then move on
           to the WAITCONNECT state */
        result = Curl_once_resolved(data, &protocol_connected);

        if(result)
          /* if Curl_once_resolved() returns failure, the connection struct
             is already freed and gone */
          data->conn = NULL; /* no more connection */
        else {
          /* call again please so that we get the next socket setup */
          rc = CURLM_CALL_MULTI_PERFORM;
          if(protocol_connected)
            multistate(data, MSTATE_DO);
          else {
#ifndef CURL_DISABLE_HTTP
            if(Curl_connect_ongoing(data->conn))
              multistate(data, MSTATE_TUNNELING);
            else
#endif
              multistate(data, MSTATE_CONNECTING);
          }
        }
      }

      if(result) {
        /* failure detected */
        stream_error = TRUE;
        break;
      }
    }
    break;

#ifndef CURL_DISABLE_HTTP
    case MSTATE_TUNNELING:
      /* this is HTTP-specific, but sending CONNECT to a proxy is HTTP... */
      DEBUGASSERT(data->conn);
      result = Curl_http_connect(data, &protocol_connected);
#ifndef CURL_DISABLE_PROXY
      if(data->conn->bits.proxy_connect_closed) {
        rc = CURLM_CALL_MULTI_PERFORM;
        /* connect back to proxy again */
        result = CURLE_OK;
        multi_done(data, CURLE_OK, FALSE);
        multistate(data, MSTATE_CONNECT);
      }
      else
#endif
        if(!result) {
          if(
#ifndef CURL_DISABLE_PROXY
            (data->conn->http_proxy.proxytype != CURLPROXY_HTTPS ||
             data->conn->bits.proxy_ssl_connected[FIRSTSOCKET]) &&
#endif
            Curl_connect_complete(data->conn)) {
            rc = CURLM_CALL_MULTI_PERFORM;
            /* initiate protocol connect phase */
            multistate(data, MSTATE_PROTOCONNECT);
          }
        }
      else
        stream_error = TRUE;
      break;
#endif

    case MSTATE_CONNECTING:
      /* awaiting a completion of an asynch TCP connect */
      DEBUGASSERT(data->conn);
      result = Curl_is_connected(data, data->conn, FIRSTSOCKET, &connected);
      if(connected && !result) {
#ifndef CURL_DISABLE_HTTP
        if(
#ifndef CURL_DISABLE_PROXY
          (data->conn->http_proxy.proxytype == CURLPROXY_HTTPS &&
           !data->conn->bits.proxy_ssl_connected[FIRSTSOCKET]) ||
#endif
          Curl_connect_ongoing(data->conn)) {
          multistate(data, MSTATE_TUNNELING);
          break;
        }
#endif
        rc = CURLM_CALL_MULTI_PERFORM;
#ifndef CURL_DISABLE_PROXY
        multistate(data,
                   data->conn->bits.tunnel_proxy?
                   MSTATE_TUNNELING : MSTATE_PROTOCONNECT);
#else
        multistate(data, MSTATE_PROTOCONNECT);
#endif
      }
      else if(result) {
        /* failure detected */
        Curl_posttransfer(data);
        multi_done(data, result, TRUE);
        stream_error = TRUE;
        break;
      }
      break;

    case MSTATE_PROTOCONNECT:
      result = protocol_connect(data, &protocol_connected);
      if(!result && !protocol_connected)
        /* switch to waiting state */
        multistate(data, MSTATE_PROTOCONNECTING);
      else if(!result) {
        /* protocol connect has completed, go WAITDO or DO */
        multistate(data, MSTATE_DO);
        rc = CURLM_CALL_MULTI_PERFORM;
      }
      else {
        /* failure detected */
        Curl_posttransfer(data);
        multi_done(data, result, TRUE);
        stream_error = TRUE;
      }
      break;

    case MSTATE_PROTOCONNECTING:
      /* protocol-specific connect phase */
      result = protocol_connecting(data, &protocol_connected);
      if(!result && protocol_connected) {
        /* after the connect has completed, go WAITDO or DO */
        multistate(data, MSTATE_DO);
        rc = CURLM_CALL_MULTI_PERFORM;
      }
      else if(result) {
        /* failure detected */
        Curl_posttransfer(data);
        multi_done(data, result, TRUE);
        stream_error = TRUE;
      }
      break;

    case MSTATE_DO:
      if(data->set.fprereq) {
        int prereq_rc;

        /* call the prerequest callback function */
        Curl_set_in_callback(data, true);
        prereq_rc = data->set.fprereq(data->set.prereq_userp,
                                      data->info.conn_primary_ip,
                                      data->info.conn_local_ip,
                                      data->info.conn_primary_port,
                                      data->info.conn_local_port);
        Curl_set_in_callback(data, false);
        if(prereq_rc != CURL_PREREQFUNC_OK) {
          failf(data, "operation aborted by pre-request callback");
          /* failure in pre-request callback - don't do any other processing */
          result = CURLE_ABORTED_BY_CALLBACK;
          Curl_posttransfer(data);
          multi_done(data, result, FALSE);
          stream_error = TRUE;
          break;
        }
      }

      if(data->set.connect_only) {
        /* keep connection open for application to use the socket */
        connkeep(data->conn, "CONNECT_ONLY");
        multistate(data, MSTATE_DONE);
        result = CURLE_OK;
        rc = CURLM_CALL_MULTI_PERFORM;
      }
      else {
        /* Perform the protocol's DO action */
        result = multi_do(data, &dophase_done);

        /* When multi_do() returns failure, data->conn might be NULL! */

        if(!result) {
          if(!dophase_done) {
#ifndef CURL_DISABLE_FTP
            /* some steps needed for wildcard matching */
            if(data->state.wildcardmatch) {
              struct WildcardData *wc = &data->wildcard;
              if(wc->state == CURLWC_DONE || wc->state == CURLWC_SKIP) {
                /* skip some states if it is important */
                multi_done(data, CURLE_OK, FALSE);

                /* if there's no connection left, skip the DONE state */
                multistate(data, data->conn ?
                           MSTATE_DONE : MSTATE_COMPLETED);
                rc = CURLM_CALL_MULTI_PERFORM;
                break;
              }
            }
#endif
            /* DO was not completed in one function call, we must continue
               DOING... */
            multistate(data, MSTATE_DOING);
            rc = CURLM_OK;
          }

          /* after DO, go DO_DONE... or DO_MORE */
          else if(data->conn->bits.do_more) {
            /* we're supposed to do more, but we need to sit down, relax
               and wait a little while first */
            multistate(data, MSTATE_DOING_MORE);
            rc = CURLM_OK;
          }
          else {
            /* we're done with the DO, now DID */
            multistate(data, MSTATE_DID);
            rc = CURLM_CALL_MULTI_PERFORM;
          }
        }
        else if((CURLE_SEND_ERROR == result) &&
                data->conn->bits.reuse) {
          /*
           * In this situation, a connection that we were trying to use
           * may have unexpectedly died.  If possible, send the connection
           * back to the CONNECT phase so we can try again.
           */
          char *newurl = NULL;
          followtype follow = FOLLOW_NONE;
          CURLcode drc;

          drc = Curl_retry_request(data, &newurl);
          if(drc) {
            /* a failure here pretty much implies an out of memory */
            result = drc;
            stream_error = TRUE;
          }

          Curl_posttransfer(data);
          drc = multi_done(data, result, FALSE);

          /* When set to retry the connection, we must to go back to
           * the CONNECT state */
          if(newurl) {
            if(!drc || (drc == CURLE_SEND_ERROR)) {
              follow = FOLLOW_RETRY;
              drc = Curl_follow(data, newurl, follow);
              if(!drc) {
                multistate(data, MSTATE_CONNECT);
                rc = CURLM_CALL_MULTI_PERFORM;
                result = CURLE_OK;
              }
              else {
                /* Follow failed */
                result = drc;
              }
            }
            else {
              /* done didn't return OK or SEND_ERROR */
              result = drc;
            }
          }
          else {
            /* Have error handler disconnect conn if we can't retry */
            stream_error = TRUE;
          }
          free(newurl);
        }
        else {
          /* failure detected */
          Curl_posttransfer(data);
          if(data->conn)
            multi_done(data, result, FALSE);
          stream_error = TRUE;
        }
      }
      break;

    case MSTATE_DOING:
      /* we continue DOING until the DO phase is complete */
      DEBUGASSERT(data->conn);
      result = protocol_doing(data, &dophase_done);
      if(!result) {
        if(dophase_done) {
          /* after DO, go DO_DONE or DO_MORE */
          multistate(data, data->conn->bits.do_more?
                     MSTATE_DOING_MORE : MSTATE_DID);
          rc = CURLM_CALL_MULTI_PERFORM;
        } /* dophase_done */
      }
      else {
        /* failure detected */
        Curl_posttransfer(data);
        multi_done(data, result, FALSE);
        stream_error = TRUE;
      }
      break;

    case MSTATE_DOING_MORE:
      /*
       * When we are connected, DOING MORE and then go DID
       */
      DEBUGASSERT(data->conn);
      result = multi_do_more(data, &control);

      if(!result) {
        if(control) {
          /* if positive, advance to DO_DONE
             if negative, go back to DOING */
          multistate(data, control == 1?
                     MSTATE_DID : MSTATE_DOING);
          rc = CURLM_CALL_MULTI_PERFORM;
        }
        else
          /* stay in DO_MORE */
          rc = CURLM_OK;
      }
      else {
        /* failure detected */
        Curl_posttransfer(data);
        multi_done(data, result, FALSE);
        stream_error = TRUE;
      }
      break;

    case MSTATE_DID:
      DEBUGASSERT(data->conn);
      if(data->conn->bits.multiplex)
        /* Check if we can move pending requests to send pipe */
        process_pending_handles(multi); /*  multiplexed */

      /* Only perform the transfer if there's a good socket to work with.
         Having both BAD is a signal to skip immediately to DONE */
      if((data->conn->sockfd != CURL_SOCKET_BAD) ||
         (data->conn->writesockfd != CURL_SOCKET_BAD))
        multistate(data, MSTATE_PERFORMING);
      else {
#ifndef CURL_DISABLE_FTP
        if(data->state.wildcardmatch &&
           ((data->conn->handler->flags & PROTOPT_WILDCARD) == 0)) {
          data->wildcard.state = CURLWC_DONE;
        }
#endif
        multistate(data, MSTATE_DONE);
      }
      rc = CURLM_CALL_MULTI_PERFORM;
      break;

    case MSTATE_RATELIMITING: /* limit-rate exceeded in either direction */
      DEBUGASSERT(data->conn);
      /* if both rates are within spec, resume transfer */
      if(Curl_pgrsUpdate(data))
        result = CURLE_ABORTED_BY_CALLBACK;
      else
        result = Curl_speedcheck(data, *nowp);

      if(result) {
        if(!(data->conn->handler->flags & PROTOPT_DUAL) &&
           result != CURLE_HTTP2_STREAM)
          streamclose(data->conn, "Transfer returned error");

        Curl_posttransfer(data);
        multi_done(data, result, TRUE);
      }
      else {
        send_timeout_ms = 0;
        if(data->set.max_send_speed)
          send_timeout_ms =
            Curl_pgrsLimitWaitTime(data->progress.uploaded,
                                   data->progress.ul_limit_size,
                                   data->set.max_send_speed,
                                   data->progress.ul_limit_start,
                                   *nowp);

        recv_timeout_ms = 0;
        if(data->set.max_recv_speed)
          recv_timeout_ms =
            Curl_pgrsLimitWaitTime(data->progress.downloaded,
                                   data->progress.dl_limit_size,
                                   data->set.max_recv_speed,
                                   data->progress.dl_limit_start,
                                   *nowp);

        if(!send_timeout_ms && !recv_timeout_ms) {
          multistate(data, MSTATE_PERFORMING);
          Curl_ratelimit(data, *nowp);
        }
        else if(send_timeout_ms >= recv_timeout_ms)
          Curl_expire(data, send_timeout_ms, EXPIRE_TOOFAST);
        else
          Curl_expire(data, recv_timeout_ms, EXPIRE_TOOFAST);
      }
      break;

    case MSTATE_PERFORMING:
    {
      char *newurl = NULL;
      bool retry = FALSE;
      bool comeback = FALSE;
      DEBUGASSERT(data->state.buffer);
      /* check if over send speed */
      send_timeout_ms = 0;
      if(data->set.max_send_speed)
        send_timeout_ms = Curl_pgrsLimitWaitTime(data->progress.uploaded,
                                                 data->progress.ul_limit_size,
                                                 data->set.max_send_speed,
                                                 data->progress.ul_limit_start,
                                                 *nowp);

      /* check if over recv speed */
      recv_timeout_ms = 0;
      if(data->set.max_recv_speed)
        recv_timeout_ms = Curl_pgrsLimitWaitTime(data->progress.downloaded,
                                                 data->progress.dl_limit_size,
                                                 data->set.max_recv_speed,
                                                 data->progress.dl_limit_start,
                                                 *nowp);

      if(send_timeout_ms || recv_timeout_ms) {
        Curl_ratelimit(data, *nowp);
        multistate(data, MSTATE_RATELIMITING);
        if(send_timeout_ms >= recv_timeout_ms)
          Curl_expire(data, send_timeout_ms, EXPIRE_TOOFAST);
        else
          Curl_expire(data, recv_timeout_ms, EXPIRE_TOOFAST);
        break;
      }

      /* read/write data if it is ready to do so */
      result = Curl_readwrite(data->conn, data, &done, &comeback);

      if(done || (result == CURLE_RECV_ERROR)) {
        /* If CURLE_RECV_ERROR happens early enough, we assume it was a race
         * condition and the server closed the re-used connection exactly when
         * we wanted to use it, so figure out if that is indeed the case.
         */
        CURLcode ret = Curl_retry_request(data, &newurl);
        if(!ret)
          retry = (newurl)?TRUE:FALSE;
        else if(!result)
          result = ret;

        if(retry) {
          /* if we are to retry, set the result to OK and consider the
             request as done */
          result = CURLE_OK;
          done = TRUE;
        }
      }
      else if((CURLE_HTTP2_STREAM == result) &&
              Curl_h2_http_1_1_error(data)) {
        CURLcode ret = Curl_retry_request(data, &newurl);

        if(!ret) {
          infof(data, "Downgrades to HTTP/1.1!");
          streamclose(data->conn, "Disconnect HTTP/2 for HTTP/1");
          data->state.httpwant = CURL_HTTP_VERSION_1_1;
          /* clear the error message bit too as we ignore the one we got */
          data->state.errorbuf = FALSE;
          if(!newurl)
            /* typically for HTTP_1_1_REQUIRED error on first flight */
            newurl = strdup(data->state.url);
          /* if we are to retry, set the result to OK and consider the request
             as done */
          retry = TRUE;
          result = CURLE_OK;
          done = TRUE;
        }
        else
          result = ret;
      }

      if(result) {
        /*
         * The transfer phase returned error, we mark the connection to get
         * closed to prevent being re-used. This is because we can't possibly
         * know if the connection is in a good shape or not now.  Unless it is
         * a protocol which uses two "channels" like FTP, as then the error
         * happened in the data connection.
         */

        if(!(data->conn->handler->flags & PROTOPT_DUAL) &&
           result != CURLE_HTTP2_STREAM)
          streamclose(data->conn, "Transfer returned error");

        Curl_posttransfer(data);
        multi_done(data, result, TRUE);
      }
      else if(done) {

        /* call this even if the readwrite function returned error */
        Curl_posttransfer(data);

        /* When we follow redirects or is set to retry the connection, we must
           to go back to the CONNECT state */
        if(data->req.newurl || retry) {
          followtype follow = FOLLOW_NONE;
          if(!retry) {
            /* if the URL is a follow-location and not just a retried request
               then figure out the URL here */
            free(newurl);
            newurl = data->req.newurl;
            data->req.newurl = NULL;
            follow = FOLLOW_REDIR;
          }
          else
            follow = FOLLOW_RETRY;
          (void)multi_done(data, CURLE_OK, FALSE);
          /* multi_done() might return CURLE_GOT_NOTHING */
          result = Curl_follow(data, newurl, follow);
          if(!result) {
            multistate(data, MSTATE_CONNECT);
            rc = CURLM_CALL_MULTI_PERFORM;
          }
          free(newurl);
        }
        else {
          /* after the transfer is done, go DONE */

          /* but first check to see if we got a location info even though we're
             not following redirects */
          if(data->req.location) {
            free(newurl);
            newurl = data->req.location;
            data->req.location = NULL;
            result = Curl_follow(data, newurl, FOLLOW_FAKE);
            free(newurl);
            if(result) {
              stream_error = TRUE;
              result = multi_done(data, result, TRUE);
            }
          }

          if(!result) {
            multistate(data, MSTATE_DONE);
            rc = CURLM_CALL_MULTI_PERFORM;
          }
        }
      }
      else if(comeback) {
        /* This avoids CURLM_CALL_MULTI_PERFORM so that a very fast transfer
           won't get stuck on this transfer at the expense of other concurrent
           transfers */
        Curl_expire(data, 0, EXPIRE_RUN_NOW);
        rc = CURLM_OK;
      }
      break;
    }

    case MSTATE_DONE:
      /* this state is highly transient, so run another loop after this */
      rc = CURLM_CALL_MULTI_PERFORM;

      if(data->conn) {
        CURLcode res;

        if(data->conn->bits.multiplex)
          /* Check if we can move pending requests to connection */
          process_pending_handles(multi); /* multiplexing */

        /* post-transfer command */
        res = multi_done(data, result, FALSE);

        /* allow a previously set error code take precedence */
        if(!result)
          result = res;
      }

#ifndef CURL_DISABLE_FTP
      if(data->state.wildcardmatch) {
        if(data->wildcard.state != CURLWC_DONE) {
          /* if a wildcard is set and we are not ending -> lets start again
             with MSTATE_INIT */
          multistate(data, MSTATE_INIT);
          break;
        }
      }
#endif
      /* after we have DONE what we're supposed to do, go COMPLETED, and
         it doesn't matter what the multi_done() returned! */
      multistate(data, MSTATE_COMPLETED);
      break;

    case MSTATE_COMPLETED:
      break;

    case MSTATE_MSGSENT:
      data->result = result;
      return CURLM_OK; /* do nothing */

    default:
      return CURLM_INTERNAL_ERROR;
    }

    if(data->conn &&
       data->mstate >= MSTATE_CONNECT &&
       data->mstate < MSTATE_DO &&
       rc != CURLM_CALL_MULTI_PERFORM &&
       !multi_ischanged(multi, false)) {
      /* We now handle stream timeouts if and only if this will be the last
       * loop iteration. We only check this on the last iteration to ensure
       * that if we know we have additional work to do immediately
       * (i.e. CURLM_CALL_MULTI_PERFORM == TRUE) then we should do that before
       * declaring the connection timed out as we may almost have a completed
       * connection. */
      multi_handle_timeout(data, nowp, &stream_error, &result, TRUE);
    }

    statemachine_end:

    if(data->mstate < MSTATE_COMPLETED) {
      if(result) {
        /*
         * If an error was returned, and we aren't in completed state now,
         * then we go to completed and consider this transfer aborted.
         */

        /* NOTE: no attempt to disconnect connections must be made
           in the case blocks above - cleanup happens only here */

        /* Check if we can move pending requests to send pipe */
        process_pending_handles(multi); /* connection */

        if(data->conn) {
          if(stream_error) {
            /* Don't attempt to send data over a connection that timed out */
            bool dead_connection = result == CURLE_OPERATION_TIMEDOUT;
            struct connectdata *conn = data->conn;

            /* This is where we make sure that the conn pointer is reset.
               We don't have to do this in every case block above where a
               failure is detected */
            Curl_detach_connnection(data);

            /* remove connection from cache */
            Curl_conncache_remove_conn(data, conn, TRUE);

            /* disconnect properly */
            Curl_disconnect(data, conn, dead_connection);
          }
        }
        else if(data->mstate == MSTATE_CONNECT) {
          /* Curl_connect() failed */
          (void)Curl_posttransfer(data);
        }

        multistate(data, MSTATE_COMPLETED);
        rc = CURLM_CALL_MULTI_PERFORM;
      }
      /* if there's still a connection to use, call the progress function */
      else if(data->conn && Curl_pgrsUpdate(data)) {
        /* aborted due to progress callback return code must close the
           connection */
        result = CURLE_ABORTED_BY_CALLBACK;
        streamclose(data->conn, "Aborted by callback");

        /* if not yet in DONE state, go there, otherwise COMPLETED */
        multistate(data, (data->mstate < MSTATE_DONE)?
                   MSTATE_DONE: MSTATE_COMPLETED);
        rc = CURLM_CALL_MULTI_PERFORM;
      }
    }

    if(MSTATE_COMPLETED == data->mstate) {
      if(data->set.fmultidone) {
        /* signal via callback instead */
        data->set.fmultidone(data, result);
      }
      else {
        /* now fill in the Curl_message with this info */
        msg = &data->msg;

        msg->extmsg.msg = CURLMSG_DONE;
        msg->extmsg.easy_handle = data;
        msg->extmsg.data.result = result;

        rc = multi_addmsg(multi, msg);
        DEBUGASSERT(!data->conn);
      }
      multistate(data, MSTATE_MSGSENT);
    }
  } while((rc == CURLM_CALL_MULTI_PERFORM) || multi_ischanged(multi, FALSE));

  data->result = result;
  return rc;
}


CURLMcode curl_multi_perform(struct Curl_multi *multi, int *running_handles)
{
  struct Curl_easy *data;
  CURLMcode returncode = CURLM_OK;
  struct Curl_tree *t;
  struct curltime now = Curl_now();

  if(!GOOD_MULTI_HANDLE(multi))
    return CURLM_BAD_HANDLE;

  if(multi->in_callback)
    return CURLM_RECURSIVE_API_CALL;

  data = multi->easyp;
  while(data) {
    CURLMcode result;
    SIGPIPE_VARIABLE(pipe_st);

    sigpipe_ignore(data, &pipe_st);
    result = multi_runsingle(multi, &now, data);
    sigpipe_restore(&pipe_st);

    if(result)
      returncode = result;

    data = data->next; /* operate on next handle */
  }

  /*
   * Simply remove all expired timers from the splay since handles are dealt
   * with unconditionally by this function and curl_multi_timeout() requires
   * that already passed/handled expire times are removed from the splay.
   *
   * It is important that the 'now' value is set at the entry of this function
   * and not for the current time as it may have ticked a little while since
   * then and then we risk this loop to remove timers that actually have not
   * been handled!
   */
  do {
    multi->timetree = Curl_splaygetbest(now, multi->timetree, &t);
    if(t)
      /* the removed may have another timeout in queue */
      (void)add_next_timeout(now, multi, t->payload);

  } while(t);

  *running_handles = multi->num_alive;

  if(CURLM_OK >= returncode)
    returncode = Curl_update_timer(multi);

  return returncode;
}

CURLMcode curl_multi_cleanup(struct Curl_multi *multi)
{
  struct Curl_easy *data;
  struct Curl_easy *nextdata;

  if(GOOD_MULTI_HANDLE(multi)) {
    if(multi->in_callback)
      return CURLM_RECURSIVE_API_CALL;

    multi->magic = 0; /* not good anymore */

    /* First remove all remaining easy handles */
    data = multi->easyp;
    while(data) {
      nextdata = data->next;
      if(!data->state.done && data->conn)
        /* if DONE was never called for this handle */
        (void)multi_done(data, CURLE_OK, TRUE);
      if(data->dns.hostcachetype == HCACHE_MULTI) {
        /* clear out the usage of the shared DNS cache */
        Curl_hostcache_clean(data, data->dns.hostcache);
        data->dns.hostcache = NULL;
        data->dns.hostcachetype = HCACHE_NONE;
      }

      /* Clear the pointer to the connection cache */
      data->state.conn_cache = NULL;
      data->multi = NULL; /* clear the association */

#ifdef USE_LIBPSL
      if(data->psl == &multi->psl)
        data->psl = NULL;
#endif

      data = nextdata;
    }

    /* Close all the connections in the connection cache */
    Curl_conncache_close_all_connections(&multi->conn_cache);

    Curl_hash_destroy(&multi->sockhash);
    Curl_conncache_destroy(&multi->conn_cache);
    Curl_llist_destroy(&multi->msglist, NULL);
    Curl_llist_destroy(&multi->pending, NULL);

    Curl_hash_destroy(&multi->hostcache);
    Curl_psl_destroy(&multi->psl);

#ifdef USE_WINSOCK
    WSACloseEvent(multi->wsa_event);
#else
#ifdef ENABLE_WAKEUP
    sclose(multi->wakeup_pair[0]);
    sclose(multi->wakeup_pair[1]);
#endif
#endif
    free(multi);

    return CURLM_OK;
  }
  return CURLM_BAD_HANDLE;
}

/*
 * curl_multi_info_read()
 *
 * This function is the primary way for a multi/multi_socket application to
 * figure out if a transfer has ended. We MUST make this function as fast as
 * possible as it will be polled frequently and we MUST NOT scan any lists in
 * here to figure out things. We must scale fine to thousands of handles and
 * beyond. The current design is fully O(1).
 */

CURLMsg *curl_multi_info_read(struct Curl_multi *multi, int *msgs_in_queue)
{
  struct Curl_message *msg;

  *msgs_in_queue = 0; /* default to none */

  if(GOOD_MULTI_HANDLE(multi) &&
     !multi->in_callback &&
     Curl_llist_count(&multi->msglist)) {
    /* there is one or more messages in the list */
    struct Curl_llist_element *e;

    /* extract the head of the list to return */
    e = multi->msglist.head;

    msg = e->ptr;

    /* remove the extracted entry */
    Curl_llist_remove(&multi->msglist, e, NULL);

    *msgs_in_queue = curlx_uztosi(Curl_llist_count(&multi->msglist));

    return &msg->extmsg;
  }
  return NULL;
}

/*
 * singlesocket() checks what sockets we deal with and their "action state"
 * and if we have a different state in any of those sockets from last time we
 * call the callback accordingly.
 */
static CURLMcode singlesocket(struct Curl_multi *multi,
                              struct Curl_easy *data)
{
  curl_socket_t socks[MAX_SOCKSPEREASYHANDLE];
  int i;
  struct Curl_sh_entry *entry;
  curl_socket_t s;
  int num;
  unsigned int curraction;
  unsigned char actions[MAX_SOCKSPEREASYHANDLE];
  int rc;

  for(i = 0; i< MAX_SOCKSPEREASYHANDLE; i++)
    socks[i] = CURL_SOCKET_BAD;

  /* Fill in the 'current' struct with the state as it is now: what sockets to
     supervise and for what actions */
  curraction = multi_getsock(data, socks);

  /* We have 0 .. N sockets already and we get to know about the 0 .. M
     sockets we should have from now on. Detect the differences, remove no
     longer supervised ones and add new ones */

  /* walk over the sockets we got right now */
  for(i = 0; (i< MAX_SOCKSPEREASYHANDLE) &&
        (curraction & (GETSOCK_READSOCK(i) | GETSOCK_WRITESOCK(i)));
      i++) {
    unsigned char action = CURL_POLL_NONE;
    unsigned char prevaction = 0;
    int comboaction;
    bool sincebefore = FALSE;

    s = socks[i];

    /* get it from the hash */
    entry = sh_getentry(&multi->sockhash, s);

    if(curraction & GETSOCK_READSOCK(i))
      action |= CURL_POLL_IN;
    if(curraction & GETSOCK_WRITESOCK(i))
      action |= CURL_POLL_OUT;

    actions[i] = action;
    if(entry) {
      /* check if new for this transfer */
      int j;
      for(j = 0; j< data->numsocks; j++) {
        if(s == data->sockets[j]) {
          prevaction = data->actions[j];
          sincebefore = TRUE;
          break;
        }
      }
    }
    else {
      /* this is a socket we didn't have before, add it to the hash! */
      entry = sh_addentry(&multi->sockhash, s);
      if(!entry)
        /* fatal */
        return CURLM_OUT_OF_MEMORY;
    }
    if(sincebefore && (prevaction != action)) {
      /* Socket was used already, but different action now */
      if(prevaction & CURL_POLL_IN)
        entry->readers--;
      if(prevaction & CURL_POLL_OUT)
        entry->writers--;
      if(action & CURL_POLL_IN)
        entry->readers++;
      if(action & CURL_POLL_OUT)
        entry->writers++;
    }
    else if(!sincebefore) {
      /* a new user */
      entry->users++;
      if(action & CURL_POLL_IN)
        entry->readers++;
      if(action & CURL_POLL_OUT)
        entry->writers++;

      /* add 'data' to the transfer hash on this socket! */
      if(!Curl_hash_add(&entry->transfers, (char *)&data, /* hash key */
                        sizeof(struct Curl_easy *), data)) {
        Curl_hash_destroy(&entry->transfers);
        return CURLM_OUT_OF_MEMORY;
      }
    }

    comboaction = (entry->writers? CURL_POLL_OUT : 0) |
                   (entry->readers ? CURL_POLL_IN : 0);

    /* socket existed before and has the same action set as before */
    if(sincebefore && ((int)entry->action == comboaction))
      /* same, continue */
      continue;

    if(multi->socket_cb) {
      rc = multi->socket_cb(data, s, comboaction, multi->socket_userp,
                            entry->socketp);
      if(rc == -1) {
        multi->dead = TRUE;
        return CURLM_ABORTED_BY_CALLBACK;
      }
    }

    entry->action = comboaction; /* store the current action state */
  }

  num = i; /* number of sockets */

  /* when we've walked over all the sockets we should have right now, we must
     make sure to detect sockets that are removed */
  for(i = 0; i< data->numsocks; i++) {
    int j;
    bool stillused = FALSE;
    s = data->sockets[i];
    for(j = 0; j < num; j++) {
      if(s == socks[j]) {
        /* this is still supervised */
        stillused = TRUE;
        break;
      }
    }
    if(stillused)
      continue;

    entry = sh_getentry(&multi->sockhash, s);
    /* if this is NULL here, the socket has been closed and notified so
       already by Curl_multi_closed() */
    if(entry) {
      unsigned char oldactions = data->actions[i];
      /* this socket has been removed. Decrease user count */
      entry->users--;
      if(oldactions & CURL_POLL_OUT)
        entry->writers--;
      if(oldactions & CURL_POLL_IN)
        entry->readers--;
      if(!entry->users) {
        if(multi->socket_cb) {
          rc = multi->socket_cb(data, s, CURL_POLL_REMOVE,
                                multi->socket_userp, entry->socketp);
          if(rc == -1) {
            multi->dead = TRUE;
            return CURLM_ABORTED_BY_CALLBACK;
          }
        }
        sh_delentry(entry, &multi->sockhash, s);
      }
      else {
        /* still users, but remove this handle as a user of this socket */
        if(Curl_hash_delete(&entry->transfers, (char *)&data,
                            sizeof(struct Curl_easy *))) {
          DEBUGASSERT(NULL);
        }
      }
    }
  } /* for loop over numsocks */

  memcpy(data->sockets, socks, num*sizeof(curl_socket_t));
  memcpy(data->actions, actions, num*sizeof(char));
  data->numsocks = num;
  return CURLM_OK;
}

CURLcode Curl_updatesocket(struct Curl_easy *data)
{
  if(singlesocket(data->multi, data))
    return CURLE_ABORTED_BY_CALLBACK;
  return CURLE_OK;
}


/*
 * Curl_multi_closed()
 *
 * Used by the connect code to tell the multi_socket code that one of the
 * sockets we were using is about to be closed.  This function will then
 * remove it from the sockethash for this handle to make the multi_socket API
 * behave properly, especially for the case when libcurl will create another
 * socket again and it gets the same file descriptor number.
 */

void Curl_multi_closed(struct Curl_easy *data, curl_socket_t s)
{
  if(data) {
    /* if there's still an easy handle associated with this connection */
    struct Curl_multi *multi = data->multi;
    if(multi) {
      /* this is set if this connection is part of a handle that is added to
         a multi handle, and only then this is necessary */
      struct Curl_sh_entry *entry = sh_getentry(&multi->sockhash, s);

      if(entry) {
        int rc = 0;
        if(multi->socket_cb)
          rc = multi->socket_cb(data, s, CURL_POLL_REMOVE,
                                multi->socket_userp, entry->socketp);

        /* now remove it from the socket hash */
        sh_delentry(entry, &multi->sockhash, s);
        if(rc == -1)
          /* This just marks the multi handle as "dead" without returning an
             error code primarily because this function is used from many
             places where propagating an error back is tricky. */
          multi->dead = TRUE;
      }
    }
  }
}

/*
 * add_next_timeout()
 *
 * Each Curl_easy has a list of timeouts. The add_next_timeout() is called
 * when it has just been removed from the splay tree because the timeout has
 * expired. This function is then to advance in the list to pick the next
 * timeout to use (skip the already expired ones) and add this node back to
 * the splay tree again.
 *
 * The splay tree only has each sessionhandle as a single node and the nearest
 * timeout is used to sort it on.
 */
static CURLMcode add_next_timeout(struct curltime now,
                                  struct Curl_multi *multi,
                                  struct Curl_easy *d)
{
  struct curltime *tv = &d->state.expiretime;
  struct Curl_llist *list = &d->state.timeoutlist;
  struct Curl_llist_element *e;
  struct time_node *node = NULL;

  /* move over the timeout list for this specific handle and remove all
     timeouts that are now passed tense and store the next pending
     timeout in *tv */
  for(e = list->head; e;) {
    struct Curl_llist_element *n = e->next;
    timediff_t diff;
    node = (struct time_node *)e->ptr;
    diff = Curl_timediff(node->time, now);
    if(diff <= 0)
      /* remove outdated entry */
      Curl_llist_remove(list, e, NULL);
    else
      /* the list is sorted so get out on the first mismatch */
      break;
    e = n;
  }
  e = list->head;
  if(!e) {
    /* clear the expire times within the handles that we remove from the
       splay tree */
    tv->tv_sec = 0;
    tv->tv_usec = 0;
  }
  else {
    /* copy the first entry to 'tv' */
    memcpy(tv, &node->time, sizeof(*tv));

    /* Insert this node again into the splay.  Keep the timer in the list in
       case we need to recompute future timers. */
    multi->timetree = Curl_splayinsert(*tv, multi->timetree,
                                       &d->state.timenode);
  }
  return CURLM_OK;
}

static CURLMcode multi_socket(struct Curl_multi *multi,
                              bool checkall,
                              curl_socket_t s,
                              int ev_bitmask,
                              int *running_handles)
{
  CURLMcode result = CURLM_OK;
  struct Curl_easy *data = NULL;
  struct Curl_tree *t;
  struct curltime now = Curl_now();

  if(checkall) {
    /* *perform() deals with running_handles on its own */
    result = curl_multi_perform(multi, running_handles);

    /* walk through each easy handle and do the socket state change magic
       and callbacks */
    if(result != CURLM_BAD_HANDLE) {
      data = multi->easyp;
      while(data && !result) {
        result = singlesocket(multi, data);
        data = data->next;
      }
    }

    /* or should we fall-through and do the timer-based stuff? */
    return result;
  }
  if(s != CURL_SOCKET_TIMEOUT) {
    struct Curl_sh_entry *entry = sh_getentry(&multi->sockhash, s);

    if(!entry)
      /* Unmatched socket, we can't act on it but we ignore this fact.  In
         real-world tests it has been proved that libevent can in fact give
         the application actions even though the socket was just previously
         asked to get removed, so thus we better survive stray socket actions
         and just move on. */
      ;
    else {
      struct Curl_hash_iterator iter;
      struct Curl_hash_element *he;

      /* the socket can be shared by many transfers, iterate */
      Curl_hash_start_iterate(&entry->transfers, &iter);
      for(he = Curl_hash_next_element(&iter); he;
          he = Curl_hash_next_element(&iter)) {
        data = (struct Curl_easy *)he->ptr;
        DEBUGASSERT(data);
        DEBUGASSERT(data->magic == CURLEASY_MAGIC_NUMBER);

        if(data->conn && !(data->conn->handler->flags & PROTOPT_DIRLOCK))
          /* set socket event bitmask if they're not locked */
          data->conn->cselect_bits = ev_bitmask;

        Curl_expire(data, 0, EXPIRE_RUN_NOW);
      }

      /* Now we fall-through and do the timer-based stuff, since we don't want
         to force the user to have to deal with timeouts as long as at least
         one connection in fact has traffic. */

      data = NULL; /* set data to NULL again to avoid calling
                      multi_runsingle() in case there's no need to */
      now = Curl_now(); /* get a newer time since the multi_runsingle() loop
                           may have taken some time */
    }
  }
  else {
    /* Asked to run due to time-out. Clear the 'lastcall' variable to force
       Curl_update_timer() to trigger a callback to the app again even if the
       same timeout is still the one to run after this call. That handles the
       case when the application asks libcurl to run the timeout
       prematurely. */
    memset(&multi->timer_lastcall, 0, sizeof(multi->timer_lastcall));
  }

  /*
   * The loop following here will go on as long as there are expire-times left
   * to process in the splay and 'data' will be re-assigned for every expired
   * handle we deal with.
   */
  do {
    /* the first loop lap 'data' can be NULL */
    if(data) {
      SIGPIPE_VARIABLE(pipe_st);

      sigpipe_ignore(data, &pipe_st);
      result = multi_runsingle(multi, &now, data);
      sigpipe_restore(&pipe_st);

      if(CURLM_OK >= result) {
        /* get the socket(s) and check if the state has been changed since
           last */
        result = singlesocket(multi, data);
        if(result)
          return result;
      }
    }

    /* Check if there's one (more) expired timer to deal with! This function
       extracts a matching node if there is one */

    multi->timetree = Curl_splaygetbest(now, multi->timetree, &t);
    if(t) {
      data = t->payload; /* assign this for next loop */
      (void)add_next_timeout(now, multi, t->payload);
    }

  } while(t);

  *running_handles = multi->num_alive;
  return result;
}

#undef curl_multi_setopt
CURLMcode curl_multi_setopt(struct Curl_multi *multi,
                            CURLMoption option, ...)
{
  CURLMcode res = CURLM_OK;
  va_list param;

  if(!GOOD_MULTI_HANDLE(multi))
    return CURLM_BAD_HANDLE;

  if(multi->in_callback)
    return CURLM_RECURSIVE_API_CALL;

  va_start(param, option);

  switch(option) {
  case CURLMOPT_SOCKETFUNCTION:
    multi->socket_cb = va_arg(param, curl_socket_callback);
    break;
  case CURLMOPT_SOCKETDATA:
    multi->socket_userp = va_arg(param, void *);
    break;
  case CURLMOPT_PUSHFUNCTION:
    multi->push_cb = va_arg(param, curl_push_callback);
    break;
  case CURLMOPT_PUSHDATA:
    multi->push_userp = va_arg(param, void *);
    break;
  case CURLMOPT_PIPELINING:
    multi->multiplexing = va_arg(param, long) & CURLPIPE_MULTIPLEX;
    break;
  case CURLMOPT_TIMERFUNCTION:
    multi->timer_cb = va_arg(param, curl_multi_timer_callback);
    break;
  case CURLMOPT_TIMERDATA:
    multi->timer_userp = va_arg(param, void *);
    break;
  case CURLMOPT_MAXCONNECTS:
    multi->maxconnects = va_arg(param, long);
    break;
  case CURLMOPT_MAX_HOST_CONNECTIONS:
    multi->max_host_connections = va_arg(param, long);
    break;
  case CURLMOPT_MAX_TOTAL_CONNECTIONS:
    multi->max_total_connections = va_arg(param, long);
    break;
    /* options formerly used for pipelining */
  case CURLMOPT_MAX_PIPELINE_LENGTH:
    break;
  case CURLMOPT_CONTENT_LENGTH_PENALTY_SIZE:
    break;
  case CURLMOPT_CHUNK_LENGTH_PENALTY_SIZE:
    break;
  case CURLMOPT_PIPELINING_SITE_BL:
    break;
  case CURLMOPT_PIPELINING_SERVER_BL:
    break;
  case CURLMOPT_MAX_CONCURRENT_STREAMS:
    {
      long streams = va_arg(param, long);
      if(streams < 1)
        streams = 100;
      multi->max_concurrent_streams = curlx_sltoui(streams);
    }
    break;
  default:
    res = CURLM_UNKNOWN_OPTION;
    break;
  }
  va_end(param);
  return res;
}

/* we define curl_multi_socket() in the public multi.h header */
#undef curl_multi_socket

CURLMcode curl_multi_socket(struct Curl_multi *multi, curl_socket_t s,
                            int *running_handles)
{
  CURLMcode result;
  if(multi->in_callback)
    return CURLM_RECURSIVE_API_CALL;
  result = multi_socket(multi, FALSE, s, 0, running_handles);
  if(CURLM_OK >= result)
    result = Curl_update_timer(multi);
  return result;
}

CURLMcode curl_multi_socket_action(struct Curl_multi *multi, curl_socket_t s,
                                   int ev_bitmask, int *running_handles)
{
  CURLMcode result;
  if(multi->in_callback)
    return CURLM_RECURSIVE_API_CALL;
  result = multi_socket(multi, FALSE, s, ev_bitmask, running_handles);
  if(CURLM_OK >= result)
    result = Curl_update_timer(multi);
  return result;
}

CURLMcode curl_multi_socket_all(struct Curl_multi *multi, int *running_handles)
{
  CURLMcode result;
  if(multi->in_callback)
    return CURLM_RECURSIVE_API_CALL;
  result = multi_socket(multi, TRUE, CURL_SOCKET_BAD, 0, running_handles);
  if(CURLM_OK >= result)
    result = Curl_update_timer(multi);
  return result;
}

static CURLMcode multi_timeout(struct Curl_multi *multi,
                               long *timeout_ms)
{
  static const struct curltime tv_zero = {0, 0};

  if(multi->dead) {
    *timeout_ms = 0;
    return CURLM_OK;
  }

  if(multi->timetree) {
    /* we have a tree of expire times */
    struct curltime now = Curl_now();

    /* splay the lowest to the bottom */
    multi->timetree = Curl_splay(tv_zero, multi->timetree);

    if(Curl_splaycomparekeys(multi->timetree->key, now) > 0) {
      /* some time left before expiration */
      timediff_t diff = Curl_timediff(multi->timetree->key, now);
      if(diff <= 0)
        /*
         * Since we only provide millisecond resolution on the returned value
         * and the diff might be less than one millisecond here, we don't
         * return zero as that may cause short bursts of busyloops on fast
         * processors while the diff is still present but less than one
         * millisecond! instead we return 1 until the time is ripe.
         */
        *timeout_ms = 1;
      else
        /* this should be safe even on 64 bit archs, as we don't use that
           overly long timeouts */
        *timeout_ms = (long)diff;
    }
    else
      /* 0 means immediately */
      *timeout_ms = 0;
  }
  else
    *timeout_ms = -1;

  return CURLM_OK;
}

CURLMcode curl_multi_timeout(struct Curl_multi *multi,
                             long *timeout_ms)
{
  /* First, make some basic checks that the CURLM handle is a good handle */
  if(!GOOD_MULTI_HANDLE(multi))
    return CURLM_BAD_HANDLE;

  if(multi->in_callback)
    return CURLM_RECURSIVE_API_CALL;

  return multi_timeout(multi, timeout_ms);
}

/*
 * Tell the application it should update its timers, if it subscribes to the
 * update timer callback.
 */
CURLMcode Curl_update_timer(struct Curl_multi *multi)
{
  long timeout_ms;
  int rc;

  if(!multi->timer_cb || multi->dead)
    return CURLM_OK;
  if(multi_timeout(multi, &timeout_ms)) {
    return CURLM_OK;
  }
  if(timeout_ms < 0) {
    static const struct curltime none = {0, 0};
    if(Curl_splaycomparekeys(none, multi->timer_lastcall)) {
      multi->timer_lastcall = none;
      /* there's no timeout now but there was one previously, tell the app to
         disable it */
      rc = multi->timer_cb(multi, -1, multi->timer_userp);
      if(rc == -1) {
        multi->dead = TRUE;
        return CURLM_ABORTED_BY_CALLBACK;
      }
      return CURLM_OK;
    }
    return CURLM_OK;
  }

  /* When multi_timeout() is done, multi->timetree points to the node with the
   * timeout we got the (relative) time-out time for. We can thus easily check
   * if this is the same (fixed) time as we got in a previous call and then
   * avoid calling the callback again. */
  if(Curl_splaycomparekeys(multi->timetree->key, multi->timer_lastcall) == 0)
    return CURLM_OK;

  multi->timer_lastcall = multi->timetree->key;

  rc = multi->timer_cb(multi, timeout_ms, multi->timer_userp);
  if(rc == -1) {
    multi->dead = TRUE;
    return CURLM_ABORTED_BY_CALLBACK;
  }
  return CURLM_OK;
}

/*
 * multi_deltimeout()
 *
 * Remove a given timestamp from the list of timeouts.
 */
static void
multi_deltimeout(struct Curl_easy *data, expire_id eid)
{
  struct Curl_llist_element *e;
  struct Curl_llist *timeoutlist = &data->state.timeoutlist;
  /* find and remove the specific node from the list */
  for(e = timeoutlist->head; e; e = e->next) {
    struct time_node *n = (struct time_node *)e->ptr;
    if(n->eid == eid) {
      Curl_llist_remove(timeoutlist, e, NULL);
      return;
    }
  }
}

/*
 * multi_addtimeout()
 *
 * Add a timestamp to the list of timeouts. Keep the list sorted so that head
 * of list is always the timeout nearest in time.
 *
 */
static CURLMcode
multi_addtimeout(struct Curl_easy *data,
                 struct curltime *stamp,
                 expire_id eid)
{
  struct Curl_llist_element *e;
  struct time_node *node;
  struct Curl_llist_element *prev = NULL;
  size_t n;
  struct Curl_llist *timeoutlist = &data->state.timeoutlist;

  node = &data->state.expires[eid];

  /* copy the timestamp and id */
  memcpy(&node->time, stamp, sizeof(*stamp));
  node->eid = eid; /* also marks it as in use */

  n = Curl_llist_count(timeoutlist);
  if(n) {
    /* find the correct spot in the list */
    for(e = timeoutlist->head; e; e = e->next) {
      struct time_node *check = (struct time_node *)e->ptr;
      timediff_t diff = Curl_timediff(check->time, node->time);
      if(diff > 0)
        break;
      prev = e;
    }

  }
  /* else
     this is the first timeout on the list */

  Curl_llist_insert_next(timeoutlist, prev, node, &node->list);
  return CURLM_OK;
}

/*
 * Curl_expire()
 *
 * given a number of milliseconds from now to use to set the 'act before
 * this'-time for the transfer, to be extracted by curl_multi_timeout()
 *
 * The timeout will be added to a queue of timeouts if it defines a moment in
 * time that is later than the current head of queue.
 *
 * Expire replaces a former timeout using the same id if already set.
 */
void Curl_expire(struct Curl_easy *data, timediff_t milli, expire_id id)
{
  struct Curl_multi *multi = data->multi;
  struct curltime *nowp = &data->state.expiretime;
  struct curltime set;

  /* this is only interesting while there is still an associated multi struct
     remaining! */
  if(!multi)
    return;

  DEBUGASSERT(id < EXPIRE_LAST);

  set = Curl_now();
  set.tv_sec += (time_t)(milli/1000); /* might be a 64 to 32 bit conversion */
  set.tv_usec += (unsigned int)(milli%1000)*1000;

  if(set.tv_usec >= 1000000) {
    set.tv_sec++;
    set.tv_usec -= 1000000;
  }

  /* Remove any timer with the same id just in case. */
  multi_deltimeout(data, id);

  /* Add it to the timer list.  It must stay in the list until it has expired
     in case we need to recompute the minimum timer later. */
  multi_addtimeout(data, &set, id);

  if(nowp->tv_sec || nowp->tv_usec) {
    /* This means that the struct is added as a node in the splay tree.
       Compare if the new time is earlier, and only remove-old/add-new if it
       is. */
    timediff_t diff = Curl_timediff(set, *nowp);
    int rc;

    if(diff > 0) {
      /* The current splay tree entry is sooner than this new expiry time.
         We don't need to update our splay tree entry. */
      return;
    }

    /* Since this is an updated time, we must remove the previous entry from
       the splay tree first and then re-add the new value */
    rc = Curl_splayremove(multi->timetree, &data->state.timenode,
                          &multi->timetree);
    if(rc)
      infof(data, "Internal error removing splay node = %d", rc);
  }

  /* Indicate that we are in the splay tree and insert the new timer expiry
     value since it is our local minimum. */
  *nowp = set;
  data->state.timenode.payload = data;
  multi->timetree = Curl_splayinsert(*nowp, multi->timetree,
                                     &data->state.timenode);
}

/*
 * Curl_expire_done()
 *
 * Removes the expire timer. Marks it as done.
 *
 */
void Curl_expire_done(struct Curl_easy *data, expire_id id)
{
  /* remove the timer, if there */
  multi_deltimeout(data, id);
}

/*
 * Curl_expire_clear()
 *
 * Clear ALL timeout values for this handle.
 */
void Curl_expire_clear(struct Curl_easy *data)
{
  struct Curl_multi *multi = data->multi;
  struct curltime *nowp = &data->state.expiretime;

  /* this is only interesting while there is still an associated multi struct
     remaining! */
  if(!multi)
    return;

  if(nowp->tv_sec || nowp->tv_usec) {
    /* Since this is an cleared time, we must remove the previous entry from
       the splay tree */
    struct Curl_llist *list = &data->state.timeoutlist;
    int rc;

    rc = Curl_splayremove(multi->timetree, &data->state.timenode,
                          &multi->timetree);
    if(rc)
      infof(data, "Internal error clearing splay node = %d", rc);

    /* flush the timeout list too */
    while(list->size > 0) {
      Curl_llist_remove(list, list->tail, NULL);
    }

#ifdef DEBUGBUILD
    infof(data, "Expire cleared (transfer %p)", data);
#endif
    nowp->tv_sec = 0;
    nowp->tv_usec = 0;
  }
}




CURLMcode curl_multi_assign(struct Curl_multi *multi, curl_socket_t s,
                            void *hashp)
{
  struct Curl_sh_entry *there = NULL;

  if(multi->in_callback)
    return CURLM_RECURSIVE_API_CALL;

  there = sh_getentry(&multi->sockhash, s);

  if(!there)
    return CURLM_BAD_SOCKET;

  there->socketp = hashp;

  return CURLM_OK;
}

size_t Curl_multi_max_host_connections(struct Curl_multi *multi)
{
  return multi ? multi->max_host_connections : 0;
}

size_t Curl_multi_max_total_connections(struct Curl_multi *multi)
{
  return multi ? multi->max_total_connections : 0;
}

/*
 * When information about a connection has appeared, call this!
 */

void Curl_multiuse_state(struct Curl_easy *data,
                         int bundlestate) /* use BUNDLE_* defines */
{
  struct connectdata *conn;
  DEBUGASSERT(data);
  DEBUGASSERT(data->multi);
  conn = data->conn;
  DEBUGASSERT(conn);
  DEBUGASSERT(conn->bundle);

  conn->bundle->multiuse = bundlestate;
  process_pending_handles(data->multi);
}

static void process_pending_handles(struct Curl_multi *multi)
{
  struct Curl_llist_element *e = multi->pending.head;
  if(e) {
    struct Curl_easy *data = e->ptr;

    DEBUGASSERT(data->mstate == MSTATE_PENDING);

    multistate(data, MSTATE_CONNECT);

    /* Remove this node from the list */
    Curl_llist_remove(&multi->pending, e, NULL);

    /* Make sure that the handle will be processed soonish. */
    Curl_expire(data, 0, EXPIRE_RUN_NOW);

    /* mark this as having been in the pending queue */
    data->state.previouslypending = TRUE;
  }
}

void Curl_set_in_callback(struct Curl_easy *data, bool value)
{
  /* might get called when there is no data pointer! */
  if(data) {
    if(data->multi_easy)
      data->multi_easy->in_callback = value;
    else if(data->multi)
      data->multi->in_callback = value;
  }
}

bool Curl_is_in_callback(struct Curl_easy *easy)
{
  return ((easy->multi && easy->multi->in_callback) ||
          (easy->multi_easy && easy->multi_easy->in_callback));
}

#ifdef DEBUGBUILD
void Curl_multi_dump(struct Curl_multi *multi)
{
  struct Curl_easy *data;
  int i;
  fprintf(stderr, "* Multi status: %d handles, %d alive\n",
          multi->num_easy, multi->num_alive);
  for(data = multi->easyp; data; data = data->next) {
    if(data->mstate < MSTATE_COMPLETED) {
      /* only display handles that are not completed */
      fprintf(stderr, "handle %p, state %s, %d sockets\n",
              (void *)data,
              statename[data->mstate], data->numsocks);
      for(i = 0; i < data->numsocks; i++) {
        curl_socket_t s = data->sockets[i];
        struct Curl_sh_entry *entry = sh_getentry(&multi->sockhash, s);

        fprintf(stderr, "%d ", (int)s);
        if(!entry) {
          fprintf(stderr, "INTERNAL CONFUSION\n");
          continue;
        }
        fprintf(stderr, "[%s %s] ",
                (entry->action&CURL_POLL_IN)?"RECVING":"",
                (entry->action&CURL_POLL_OUT)?"SENDING":"");
      }
      if(data->numsocks)
        fprintf(stderr, "\n");
    }
  }
}
#endif

unsigned int Curl_multi_max_concurrent_streams(struct Curl_multi *multi)
{
  DEBUGASSERT(multi);
  return multi->max_concurrent_streams;
}
