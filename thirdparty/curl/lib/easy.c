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

/*
 * See comment in curl_memory.h for the explanation of this sanity check.
 */

#ifdef CURLX_NO_MEMORY_CALLBACKS
#error "libcurl shall not ever be built with CURLX_NO_MEMORY_CALLBACKS defined"
#endif

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef HAVE_NET_IF_H
#include <net/if.h>
#endif
#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#include "urldata.h"
#include <curl/curl.h>
#include "transfer.h"
#include "vtls/vtls.h"
#include "url.h"
#include "getinfo.h"
#include "hostip.h"
#include "share.h"
#include "strdup.h"
#include "progress.h"
#include "easyif.h"
#include "multiif.h"
#include "select.h"
#include "sendf.h" /* for failf function prototype */
#include "connect.h" /* for Curl_getconnectinfo */
#include "slist.h"
#include "mime.h"
#include "amigaos.h"
#include "non-ascii.h"
#include "warnless.h"
#include "multiif.h"
#include "sigpipe.h"
#include "vssh/ssh.h"
#include "setopt.h"
#include "http_digest.h"
#include "system_win32.h"
#include "http2.h"
#include "dynbuf.h"
#include "altsvc.h"
#include "hsts.h"

/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

/* true globals -- for curl_global_init() and curl_global_cleanup() */
static unsigned int  initialized;
static long          init_flags;

/*
 * strdup (and other memory functions) is redefined in complicated
 * ways, but at this point it must be defined as the system-supplied strdup
 * so the callback pointer is initialized correctly.
 */
#if defined(_WIN32_WCE)
#define system_strdup _strdup
#elif !defined(HAVE_STRDUP)
#define system_strdup curlx_strdup
#else
#define system_strdup strdup
#endif

#if defined(_MSC_VER) && defined(_DLL) && !defined(__POCC__)
#  pragma warning(disable:4232) /* MSVC extension, dllimport identity */
#endif

/*
 * If a memory-using function (like curl_getenv) is used before
 * curl_global_init() is called, we need to have these pointers set already.
 */
curl_malloc_callback Curl_cmalloc = (curl_malloc_callback)malloc;
curl_free_callback Curl_cfree = (curl_free_callback)free;
curl_realloc_callback Curl_crealloc = (curl_realloc_callback)realloc;
curl_strdup_callback Curl_cstrdup = (curl_strdup_callback)system_strdup;
curl_calloc_callback Curl_ccalloc = (curl_calloc_callback)calloc;
#if defined(WIN32) && defined(UNICODE)
curl_wcsdup_callback Curl_cwcsdup = Curl_wcsdup;
#endif

#if defined(_MSC_VER) && defined(_DLL) && !defined(__POCC__)
#  pragma warning(default:4232) /* MSVC extension, dllimport identity */
#endif

#ifdef DEBUGBUILD
static char *leakpointer;
#endif

/**
 * curl_global_init() globally initializes curl given a bitwise set of the
 * different features of what to initialize.
 */
static CURLcode global_init(long flags, bool memoryfuncs)
{
  if(initialized++)
    return CURLE_OK;

  if(memoryfuncs) {
    /* Setup the default memory functions here (again) */
    Curl_cmalloc = (curl_malloc_callback)malloc;
    Curl_cfree = (curl_free_callback)free;
    Curl_crealloc = (curl_realloc_callback)realloc;
    Curl_cstrdup = (curl_strdup_callback)system_strdup;
    Curl_ccalloc = (curl_calloc_callback)calloc;
#if defined(WIN32) && defined(UNICODE)
    Curl_cwcsdup = (curl_wcsdup_callback)_wcsdup;
#endif
  }

  if(!Curl_ssl_init()) {
    DEBUGF(fprintf(stderr, "Error: Curl_ssl_init failed\n"));
    goto fail;
  }

#ifdef WIN32
  if(Curl_win32_init(flags)) {
    DEBUGF(fprintf(stderr, "Error: win32_init failed\n"));
    goto fail;
  }
#endif

#ifdef __AMIGA__
  if(!Curl_amiga_init()) {
    DEBUGF(fprintf(stderr, "Error: Curl_amiga_init failed\n"));
    goto fail;
  }
#endif

#ifdef NETWARE
  if(netware_init()) {
    DEBUGF(fprintf(stderr, "Warning: LONG namespace not available\n"));
  }
#endif

  if(Curl_resolver_global_init()) {
    DEBUGF(fprintf(stderr, "Error: resolver_global_init failed\n"));
    goto fail;
  }

#if defined(USE_SSH)
  if(Curl_ssh_init()) {
    goto fail;
  }
#endif

#ifdef USE_WOLFSSH
  if(WS_SUCCESS != wolfSSH_Init()) {
    DEBUGF(fprintf(stderr, "Error: wolfSSH_Init failed\n"));
    return CURLE_FAILED_INIT;
  }
#endif

  init_flags = flags;

#ifdef DEBUGBUILD
  if(getenv("CURL_GLOBAL_INIT"))
    /* alloc data that will leak if *cleanup() is not called! */
    leakpointer = malloc(1);
#endif

  return CURLE_OK;

  fail:
  initialized--; /* undo the increase */
  return CURLE_FAILED_INIT;
}


/**
 * curl_global_init() globally initializes curl given a bitwise set of the
 * different features of what to initialize.
 */
CURLcode curl_global_init(long flags)
{
  return global_init(flags, TRUE);
}

/*
 * curl_global_init_mem() globally initializes curl and also registers the
 * user provided callback routines.
 */
CURLcode curl_global_init_mem(long flags, curl_malloc_callback m,
                              curl_free_callback f, curl_realloc_callback r,
                              curl_strdup_callback s, curl_calloc_callback c)
{
  /* Invalid input, return immediately */
  if(!m || !f || !r || !s || !c)
    return CURLE_FAILED_INIT;

  if(initialized) {
    /* Already initialized, don't do it again, but bump the variable anyway to
       work like curl_global_init() and require the same amount of cleanup
       calls. */
    initialized++;
    return CURLE_OK;
  }

  /* set memory functions before global_init() in case it wants memory
     functions */
  Curl_cmalloc = m;
  Curl_cfree = f;
  Curl_cstrdup = s;
  Curl_crealloc = r;
  Curl_ccalloc = c;

  /* Call the actual init function, but without setting */
  return global_init(flags, FALSE);
}

/**
 * curl_global_cleanup() globally cleanups curl, uses the value of
 * "init_flags" to determine what needs to be cleaned up and what doesn't.
 */
void curl_global_cleanup(void)
{
  if(!initialized)
    return;

  if(--initialized)
    return;

  Curl_ssl_cleanup();
  Curl_resolver_global_cleanup();

#ifdef WIN32
  Curl_win32_cleanup(init_flags);
#endif

  Curl_amiga_cleanup();

  Curl_ssh_cleanup();

#ifdef USE_WOLFSSH
  (void)wolfSSH_Cleanup();
#endif
#ifdef DEBUGBUILD
  free(leakpointer);
#endif

  init_flags  = 0;
}

/*
 * curl_easy_init() is the external interface to alloc, setup and init an
 * easy handle that is returned. If anything goes wrong, NULL is returned.
 */
struct Curl_easy *curl_easy_init(void)
{
  CURLcode result;
  struct Curl_easy *data;

  /* Make sure we inited the global SSL stuff */
  if(!initialized) {
    result = curl_global_init(CURL_GLOBAL_DEFAULT);
    if(result) {
      /* something in the global init failed, return nothing */
      DEBUGF(fprintf(stderr, "Error: curl_global_init failed\n"));
      return NULL;
    }
  }

  /* We use curl_open() with undefined URL so far */
  result = Curl_open(&data);
  if(result) {
    DEBUGF(fprintf(stderr, "Error: Curl_open failed\n"));
    return NULL;
  }

  return data;
}

#ifdef CURLDEBUG

struct socketmonitor {
  struct socketmonitor *next; /* the next node in the list or NULL */
  struct pollfd socket; /* socket info of what to monitor */
};

struct events {
  long ms;              /* timeout, run the timeout function when reached */
  bool msbump;          /* set TRUE when timeout is set by callback */
  int num_sockets;      /* number of nodes in the monitor list */
  struct socketmonitor *list; /* list of sockets to monitor */
  int running_handles;  /* store the returned number */
};

/* events_timer
 *
 * Callback that gets called with a new value when the timeout should be
 * updated.
 */

static int events_timer(struct Curl_multi *multi,    /* multi handle */
                        long timeout_ms, /* see above */
                        void *userp)    /* private callback pointer */
{
  struct events *ev = userp;
  (void)multi;
  if(timeout_ms == -1)
    /* timeout removed */
    timeout_ms = 0;
  else if(timeout_ms == 0)
    /* timeout is already reached! */
    timeout_ms = 1; /* trigger asap */

  ev->ms = timeout_ms;
  ev->msbump = TRUE;
  return 0;
}


/* poll2cselect
 *
 * convert from poll() bit definitions to libcurl's CURL_CSELECT_* ones
 */
static int poll2cselect(int pollmask)
{
  int omask = 0;
  if(pollmask & POLLIN)
    omask |= CURL_CSELECT_IN;
  if(pollmask & POLLOUT)
    omask |= CURL_CSELECT_OUT;
  if(pollmask & POLLERR)
    omask |= CURL_CSELECT_ERR;
  return omask;
}


/* socketcb2poll
 *
 * convert from libcurl' CURL_POLL_* bit definitions to poll()'s
 */
static short socketcb2poll(int pollmask)
{
  short omask = 0;
  if(pollmask & CURL_POLL_IN)
    omask |= POLLIN;
  if(pollmask & CURL_POLL_OUT)
    omask |= POLLOUT;
  return omask;
}

/* events_socket
 *
 * Callback that gets called with information about socket activity to
 * monitor.
 */
static int events_socket(struct Curl_easy *easy,      /* easy handle */
                         curl_socket_t s, /* socket */
                         int what,        /* see above */
                         void *userp,     /* private callback
                                             pointer */
                         void *socketp)   /* private socket
                                             pointer */
{
  struct events *ev = userp;
  struct socketmonitor *m;
  struct socketmonitor *prev = NULL;

#if defined(CURL_DISABLE_VERBOSE_STRINGS)
  (void) easy;
#endif
  (void)socketp;

  m = ev->list;
  while(m) {
    if(m->socket.fd == s) {

      if(what == CURL_POLL_REMOVE) {
        struct socketmonitor *nxt = m->next;
        /* remove this node from the list of monitored sockets */
        if(prev)
          prev->next = nxt;
        else
          ev->list = nxt;
        free(m);
        m = nxt;
        infof(easy, "socket cb: socket %d REMOVED", s);
      }
      else {
        /* The socket 's' is already being monitored, update the activity
           mask. Convert from libcurl bitmask to the poll one. */
        m->socket.events = socketcb2poll(what);
        infof(easy, "socket cb: socket %d UPDATED as %s%s", s,
              (what&CURL_POLL_IN)?"IN":"",
              (what&CURL_POLL_OUT)?"OUT":"");
      }
      break;
    }
    prev = m;
    m = m->next; /* move to next node */
  }
  if(!m) {
    if(what == CURL_POLL_REMOVE) {
      /* this happens a bit too often, libcurl fix perhaps? */
      /* fprintf(stderr,
         "%s: socket %d asked to be REMOVED but not present!\n",
                 __func__, s); */
    }
    else {
      m = malloc(sizeof(struct socketmonitor));
      if(m) {
        m->next = ev->list;
        m->socket.fd = s;
        m->socket.events = socketcb2poll(what);
        m->socket.revents = 0;
        ev->list = m;
        infof(easy, "socket cb: socket %d ADDED as %s%s", s,
              (what&CURL_POLL_IN)?"IN":"",
              (what&CURL_POLL_OUT)?"OUT":"");
      }
      else
        return CURLE_OUT_OF_MEMORY;
    }
  }

  return 0;
}


/*
 * events_setup()
 *
 * Do the multi handle setups that only event-based transfers need.
 */
static void events_setup(struct Curl_multi *multi, struct events *ev)
{
  /* timer callback */
  curl_multi_setopt(multi, CURLMOPT_TIMERFUNCTION, events_timer);
  curl_multi_setopt(multi, CURLMOPT_TIMERDATA, ev);

  /* socket callback */
  curl_multi_setopt(multi, CURLMOPT_SOCKETFUNCTION, events_socket);
  curl_multi_setopt(multi, CURLMOPT_SOCKETDATA, ev);
}


/* wait_or_timeout()
 *
 * waits for activity on any of the given sockets, or the timeout to trigger.
 */

static CURLcode wait_or_timeout(struct Curl_multi *multi, struct events *ev)
{
  bool done = FALSE;
  CURLMcode mcode = CURLM_OK;
  CURLcode result = CURLE_OK;

  while(!done) {
    CURLMsg *msg;
    struct socketmonitor *m;
    struct pollfd *f;
    struct pollfd fds[4];
    int numfds = 0;
    int pollrc;
    int i;
    struct curltime before;
    struct curltime after;

    /* populate the fds[] array */
    for(m = ev->list, f = &fds[0]; m; m = m->next) {
      f->fd = m->socket.fd;
      f->events = m->socket.events;
      f->revents = 0;
      /* fprintf(stderr, "poll() %d check socket %d\n", numfds, f->fd); */
      f++;
      numfds++;
    }

    /* get the time stamp to use to figure out how long poll takes */
    before = Curl_now();

    /* wait for activity or timeout */
    pollrc = Curl_poll(fds, numfds, ev->ms);

    after = Curl_now();

    ev->msbump = FALSE; /* reset here */

    if(0 == pollrc) {
      /* timeout! */
      ev->ms = 0;
      /* fprintf(stderr, "call curl_multi_socket_action(TIMEOUT)\n"); */
      mcode = curl_multi_socket_action(multi, CURL_SOCKET_TIMEOUT, 0,
                                       &ev->running_handles);
    }
    else if(pollrc > 0) {
      /* loop over the monitored sockets to see which ones had activity */
      for(i = 0; i< numfds; i++) {
        if(fds[i].revents) {
          /* socket activity, tell libcurl */
          int act = poll2cselect(fds[i].revents); /* convert */
          infof(multi->easyp, "call curl_multi_socket_action(socket %d)",
                fds[i].fd);
          mcode = curl_multi_socket_action(multi, fds[i].fd, act,
                                           &ev->running_handles);
        }
      }

      if(!ev->msbump) {
        /* If nothing updated the timeout, we decrease it by the spent time.
         * If it was updated, it has the new timeout time stored already.
         */
        timediff_t timediff = Curl_timediff(after, before);
        if(timediff > 0) {
          if(timediff > ev->ms)
            ev->ms = 0;
          else
            ev->ms -= (long)timediff;
        }
      }
    }
    else
      return CURLE_RECV_ERROR;

    if(mcode)
      return CURLE_URL_MALFORMAT;

    /* we don't really care about the "msgs_in_queue" value returned in the
       second argument */
    msg = curl_multi_info_read(multi, &pollrc);
    if(msg) {
      result = msg->data.result;
      done = TRUE;
    }
  }

  return result;
}


/* easy_events()
 *
 * Runs a transfer in a blocking manner using the events-based API
 */
static CURLcode easy_events(struct Curl_multi *multi)
{
  /* this struct is made static to allow it to be used after this function
     returns and curl_multi_remove_handle() is called */
  static struct events evs = {2, FALSE, 0, NULL, 0};

  /* if running event-based, do some further multi inits */
  events_setup(multi, &evs);

  return wait_or_timeout(multi, &evs);
}
#else /* CURLDEBUG */
/* when not built with debug, this function doesn't exist */
#define easy_events(x) CURLE_NOT_BUILT_IN
#endif

static CURLcode easy_transfer(struct Curl_multi *multi)
{
  bool done = FALSE;
  CURLMcode mcode = CURLM_OK;
  CURLcode result = CURLE_OK;

  while(!done && !mcode) {
    int still_running = 0;

    mcode = curl_multi_poll(multi, NULL, 0, 1000, NULL);

    if(!mcode)
      mcode = curl_multi_perform(multi, &still_running);

    /* only read 'still_running' if curl_multi_perform() return OK */
    if(!mcode && !still_running) {
      int rc;
      CURLMsg *msg = curl_multi_info_read(multi, &rc);
      if(msg) {
        result = msg->data.result;
        done = TRUE;
      }
    }
  }

  /* Make sure to return some kind of error if there was a multi problem */
  if(mcode) {
    result = (mcode == CURLM_OUT_OF_MEMORY) ? CURLE_OUT_OF_MEMORY :
              /* The other multi errors should never happen, so return
                 something suitably generic */
              CURLE_BAD_FUNCTION_ARGUMENT;
  }

  return result;
}


/*
 * easy_perform() is the external interface that performs a blocking
 * transfer as previously setup.
 *
 * CONCEPT: This function creates a multi handle, adds the easy handle to it,
 * runs curl_multi_perform() until the transfer is done, then detaches the
 * easy handle, destroys the multi handle and returns the easy handle's return
 * code.
 *
 * REALITY: it can't just create and destroy the multi handle that easily. It
 * needs to keep it around since if this easy handle is used again by this
 * function, the same multi handle must be re-used so that the same pools and
 * caches can be used.
 *
 * DEBUG: if 'events' is set TRUE, this function will use a replacement engine
 * instead of curl_multi_perform() and use curl_multi_socket_action().
 */
static CURLcode easy_perform(struct Curl_easy *data, bool events)
{
  struct Curl_multi *multi;
  CURLMcode mcode;
  CURLcode result = CURLE_OK;
  SIGPIPE_VARIABLE(pipe_st);

  if(!data)
    return CURLE_BAD_FUNCTION_ARGUMENT;

  if(data->set.errorbuffer)
    /* clear this as early as possible */
    data->set.errorbuffer[0] = 0;

  if(data->multi) {
    failf(data, "easy handle already used in multi handle");
    return CURLE_FAILED_INIT;
  }

  if(data->multi_easy)
    multi = data->multi_easy;
  else {
    /* this multi handle will only ever have a single easy handled attached
       to it, so make it use minimal hashes */
    multi = Curl_multi_handle(1, 3);
    if(!multi)
      return CURLE_OUT_OF_MEMORY;
    data->multi_easy = multi;
  }

  if(multi->in_callback)
    return CURLE_RECURSIVE_API_CALL;

  /* Copy the MAXCONNECTS option to the multi handle */
  curl_multi_setopt(multi, CURLMOPT_MAXCONNECTS, data->set.maxconnects);

  mcode = curl_multi_add_handle(multi, data);
  if(mcode) {
    curl_multi_cleanup(multi);
    data->multi_easy = NULL;
    if(mcode == CURLM_OUT_OF_MEMORY)
      return CURLE_OUT_OF_MEMORY;
    return CURLE_FAILED_INIT;
  }

  sigpipe_ignore(data, &pipe_st);

  /* run the transfer */
  result = events ? easy_events(multi) : easy_transfer(multi);

  /* ignoring the return code isn't nice, but atm we can't really handle
     a failure here, room for future improvement! */
  (void)curl_multi_remove_handle(multi, data);

  sigpipe_restore(&pipe_st);

  /* The multi handle is kept alive, owned by the easy handle */
  return result;
}


/*
 * curl_easy_perform() is the external interface that performs a blocking
 * transfer as previously setup.
 */
CURLcode curl_easy_perform(struct Curl_easy *data)
{
  return easy_perform(data, FALSE);
}

#ifdef CURLDEBUG
/*
 * curl_easy_perform_ev() is the external interface that performs a blocking
 * transfer using the event-based API internally.
 */
CURLcode curl_easy_perform_ev(struct Curl_easy *data)
{
  return easy_perform(data, TRUE);
}

#endif

/*
 * curl_easy_cleanup() is the external interface to cleaning/freeing the given
 * easy handle.
 */
void curl_easy_cleanup(struct Curl_easy *data)
{
  SIGPIPE_VARIABLE(pipe_st);

  if(!data)
    return;

  sigpipe_ignore(data, &pipe_st);
  Curl_close(&data);
  sigpipe_restore(&pipe_st);
}

/*
 * curl_easy_getinfo() is an external interface that allows an app to retrieve
 * information from a performed transfer and similar.
 */
#undef curl_easy_getinfo
CURLcode curl_easy_getinfo(struct Curl_easy *data, CURLINFO info, ...)
{
  va_list arg;
  void *paramp;
  CURLcode result;

  va_start(arg, info);
  paramp = va_arg(arg, void *);

  result = Curl_getinfo(data, info, paramp);

  va_end(arg);
  return result;
}

static CURLcode dupset(struct Curl_easy *dst, struct Curl_easy *src)
{
  CURLcode result = CURLE_OK;
  enum dupstring i;
  enum dupblob j;

  /* Copy src->set into dst->set first, then deal with the strings
     afterwards */
  dst->set = src->set;
  Curl_mime_initpart(&dst->set.mimepost, dst);

  /* clear all string pointers first */
  memset(dst->set.str, 0, STRING_LAST * sizeof(char *));

  /* duplicate all strings */
  for(i = (enum dupstring)0; i< STRING_LASTZEROTERMINATED; i++) {
    result = Curl_setstropt(&dst->set.str[i], src->set.str[i]);
    if(result)
      return result;
  }

  /* clear all blob pointers first */
  memset(dst->set.blobs, 0, BLOB_LAST * sizeof(struct curl_blob *));
  /* duplicate all blobs */
  for(j = (enum dupblob)0; j < BLOB_LAST; j++) {
    result = Curl_setblobopt(&dst->set.blobs[j], src->set.blobs[j]);
    if(result)
      return result;
  }

  /* duplicate memory areas pointed to */
  i = STRING_COPYPOSTFIELDS;
  if(src->set.postfieldsize && src->set.str[i]) {
    /* postfieldsize is curl_off_t, Curl_memdup() takes a size_t ... */
    dst->set.str[i] = Curl_memdup(src->set.str[i],
                                  curlx_sotouz(src->set.postfieldsize));
    if(!dst->set.str[i])
      return CURLE_OUT_OF_MEMORY;
    /* point to the new copy */
    dst->set.postfields = dst->set.str[i];
  }

  /* Duplicate mime data. */
  result = Curl_mime_duppart(&dst->set.mimepost, &src->set.mimepost);

  if(src->set.resolve)
    dst->state.resolve = dst->set.resolve;

  return result;
}

/*
 * curl_easy_duphandle() is an external interface to allow duplication of a
 * given input easy handle. The returned handle will be a new working handle
 * with all options set exactly as the input source handle.
 */
struct Curl_easy *curl_easy_duphandle(struct Curl_easy *data)
{
  struct Curl_easy *outcurl = calloc(1, sizeof(struct Curl_easy));
  if(NULL == outcurl)
    goto fail;

  /*
   * We setup a few buffers we need. We should probably make them
   * get setup on-demand in the code, as that would probably decrease
   * the likeliness of us forgetting to init a buffer here in the future.
   */
  outcurl->set.buffer_size = data->set.buffer_size;

  /* copy all userdefined values */
  if(dupset(outcurl, data))
    goto fail;

  Curl_dyn_init(&outcurl->state.headerb, CURL_MAX_HTTP_HEADER);

  /* the connection cache is setup on demand */
  outcurl->state.conn_cache = NULL;
  outcurl->state.lastconnect_id = -1;

  outcurl->progress.flags    = data->progress.flags;
  outcurl->progress.callback = data->progress.callback;

  if(data->cookies) {
    /* If cookies are enabled in the parent handle, we enable them
       in the clone as well! */
    outcurl->cookies = Curl_cookie_init(data,
                                        data->cookies->filename,
                                        outcurl->cookies,
                                        data->set.cookiesession);
    if(!outcurl->cookies)
      goto fail;
  }

  /* duplicate all values in 'change' */
  if(data->state.cookielist) {
    outcurl->state.cookielist =
      Curl_slist_duplicate(data->state.cookielist);
    if(!outcurl->state.cookielist)
      goto fail;
  }

  if(data->state.url) {
    outcurl->state.url = strdup(data->state.url);
    if(!outcurl->state.url)
      goto fail;
    outcurl->state.url_alloc = TRUE;
  }

  if(data->state.referer) {
    outcurl->state.referer = strdup(data->state.referer);
    if(!outcurl->state.referer)
      goto fail;
    outcurl->state.referer_alloc = TRUE;
  }

  /* Reinitialize an SSL engine for the new handle
   * note: the engine name has already been copied by dupset */
  if(outcurl->set.str[STRING_SSL_ENGINE]) {
    if(Curl_ssl_set_engine(outcurl, outcurl->set.str[STRING_SSL_ENGINE]))
      goto fail;
  }

#ifdef USE_ALTSVC
  if(data->asi) {
    outcurl->asi = Curl_altsvc_init();
    if(!outcurl->asi)
      goto fail;
    if(outcurl->set.str[STRING_ALTSVC])
      (void)Curl_altsvc_load(outcurl->asi, outcurl->set.str[STRING_ALTSVC]);
  }
#endif
#ifndef CURL_DISABLE_HSTS
  if(data->hsts) {
    outcurl->hsts = Curl_hsts_init();
    if(!outcurl->hsts)
      goto fail;
    if(outcurl->set.str[STRING_HSTS])
      (void)Curl_hsts_loadfile(outcurl,
                               outcurl->hsts, outcurl->set.str[STRING_HSTS]);
    (void)Curl_hsts_loadcb(outcurl, outcurl->hsts);
  }
#endif
  /* Clone the resolver handle, if present, for the new handle */
  if(Curl_resolver_duphandle(outcurl,
                             &outcurl->state.async.resolver,
                             data->state.async.resolver))
    goto fail;

#ifdef USE_ARES
  {
    CURLcode rc;

    rc = Curl_set_dns_servers(outcurl, data->set.str[STRING_DNS_SERVERS]);
    if(rc && rc != CURLE_NOT_BUILT_IN)
      goto fail;

    rc = Curl_set_dns_interface(outcurl, data->set.str[STRING_DNS_INTERFACE]);
    if(rc && rc != CURLE_NOT_BUILT_IN)
      goto fail;

    rc = Curl_set_dns_local_ip4(outcurl, data->set.str[STRING_DNS_LOCAL_IP4]);
    if(rc && rc != CURLE_NOT_BUILT_IN)
      goto fail;

    rc = Curl_set_dns_local_ip6(outcurl, data->set.str[STRING_DNS_LOCAL_IP6]);
    if(rc && rc != CURLE_NOT_BUILT_IN)
      goto fail;
  }
#endif /* USE_ARES */

  Curl_convert_setup(outcurl);

  Curl_initinfo(outcurl);

  outcurl->magic = CURLEASY_MAGIC_NUMBER;

  /* we reach this point and thus we are OK */

  return outcurl;

  fail:

  if(outcurl) {
    curl_slist_free_all(outcurl->state.cookielist);
    outcurl->state.cookielist = NULL;
    Curl_safefree(outcurl->state.buffer);
    Curl_dyn_free(&outcurl->state.headerb);
    Curl_safefree(outcurl->state.url);
    Curl_safefree(outcurl->state.referer);
    Curl_altsvc_cleanup(&outcurl->asi);
    Curl_hsts_cleanup(&outcurl->hsts);
    Curl_freeset(outcurl);
    free(outcurl);
  }

  return NULL;
}

/*
 * curl_easy_reset() is an external interface that allows an app to re-
 * initialize a session handle to the default values.
 */
void curl_easy_reset(struct Curl_easy *data)
{
  Curl_free_request_state(data);

  /* zero out UserDefined data: */
  Curl_freeset(data);
  memset(&data->set, 0, sizeof(struct UserDefined));
  (void)Curl_init_userdefined(data);

  /* zero out Progress data: */
  memset(&data->progress, 0, sizeof(struct Progress));

  /* zero out PureInfo data: */
  Curl_initinfo(data);

  data->progress.flags |= PGRS_HIDE;
  data->state.current_speed = -1; /* init to negative == impossible */
  data->state.retrycount = 0;     /* reset the retry counter */

  /* zero out authentication data: */
  memset(&data->state.authhost, 0, sizeof(struct auth));
  memset(&data->state.authproxy, 0, sizeof(struct auth));

#if !defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_CRYPTO_AUTH)
  Curl_http_auth_cleanup_digest(data);
#endif
}

/*
 * curl_easy_pause() allows an application to pause or unpause a specific
 * transfer and direction. This function sets the full new state for the
 * current connection this easy handle operates on.
 *
 * NOTE: if you have the receiving paused and you call this function to remove
 * the pausing, you may get your write callback called at this point.
 *
 * Action is a bitmask consisting of CURLPAUSE_* bits in curl/curl.h
 *
 * NOTE: This is one of few API functions that are allowed to be called from
 * within a callback.
 */
CURLcode curl_easy_pause(struct Curl_easy *data, int action)
{
  struct SingleRequest *k;
  CURLcode result = CURLE_OK;
  int oldstate;
  int newstate;

  if(!GOOD_EASY_HANDLE(data) || !data->conn)
    /* crazy input, don't continue */
    return CURLE_BAD_FUNCTION_ARGUMENT;

  k = &data->req;
  oldstate = k->keepon & (KEEP_RECV_PAUSE| KEEP_SEND_PAUSE);

  /* first switch off both pause bits then set the new pause bits */
  newstate = (k->keepon &~ (KEEP_RECV_PAUSE| KEEP_SEND_PAUSE)) |
    ((action & CURLPAUSE_RECV)?KEEP_RECV_PAUSE:0) |
    ((action & CURLPAUSE_SEND)?KEEP_SEND_PAUSE:0);

  if((newstate & (KEEP_RECV_PAUSE| KEEP_SEND_PAUSE)) == oldstate) {
    /* Not changing any pause state, return */
    DEBUGF(infof(data, "pause: no change, early return"));
    return CURLE_OK;
  }

  /* Unpause parts in active mime tree. */
  if((k->keepon & ~newstate & KEEP_SEND_PAUSE) &&
     (data->mstate == MSTATE_PERFORMING ||
      data->mstate == MSTATE_RATELIMITING) &&
     data->state.fread_func == (curl_read_callback) Curl_mime_read) {
    Curl_mime_unpause(data->state.in);
  }

  /* put it back in the keepon */
  k->keepon = newstate;

  if(!(newstate & KEEP_RECV_PAUSE)) {
    Curl_http2_stream_pause(data, FALSE);

    if(data->state.tempcount) {
      /* there are buffers for sending that can be delivered as the receive
         pausing is lifted! */
      unsigned int i;
      unsigned int count = data->state.tempcount;
      struct tempbuf writebuf[3]; /* there can only be three */

      /* copy the structs to allow for immediate re-pausing */
      for(i = 0; i < data->state.tempcount; i++) {
        writebuf[i] = data->state.tempwrite[i];
        Curl_dyn_init(&data->state.tempwrite[i].b, DYN_PAUSE_BUFFER);
      }
      data->state.tempcount = 0;

      for(i = 0; i < count; i++) {
        /* even if one function returns error, this loops through and frees
           all buffers */
        if(!result)
          result = Curl_client_write(data, writebuf[i].type,
                                     Curl_dyn_ptr(&writebuf[i].b),
                                     Curl_dyn_len(&writebuf[i].b));
        Curl_dyn_free(&writebuf[i].b);
      }

      if(result)
        return result;
    }
  }

  /* if there's no error and we're not pausing both directions, we want
     to have this handle checked soon */
  if((newstate & (KEEP_RECV_PAUSE|KEEP_SEND_PAUSE)) !=
     (KEEP_RECV_PAUSE|KEEP_SEND_PAUSE)) {
    Curl_expire(data, 0, EXPIRE_RUN_NOW); /* get this handle going again */

    /* reset the too-slow time keeper */
    data->state.keeps_speed.tv_sec = 0;

    if(!data->state.tempcount)
      /* if not pausing again, force a recv/send check of this connection as
         the data might've been read off the socket already */
      data->conn->cselect_bits = CURL_CSELECT_IN | CURL_CSELECT_OUT;
    if(data->multi) {
      if(Curl_update_timer(data->multi))
        return CURLE_ABORTED_BY_CALLBACK;
    }
  }

  if(!data->state.done)
    /* This transfer may have been moved in or out of the bundle, update the
       corresponding socket callback, if used */
    result = Curl_updatesocket(data);

  return result;
}


static CURLcode easy_connection(struct Curl_easy *data,
                                curl_socket_t *sfd,
                                struct connectdata **connp)
{
  if(!data)
    return CURLE_BAD_FUNCTION_ARGUMENT;

  /* only allow these to be called on handles with CURLOPT_CONNECT_ONLY */
  if(!data->set.connect_only) {
    failf(data, "CONNECT_ONLY is required!");
    return CURLE_UNSUPPORTED_PROTOCOL;
  }

  *sfd = Curl_getconnectinfo(data, connp);

  if(*sfd == CURL_SOCKET_BAD) {
    failf(data, "Failed to get recent socket");
    return CURLE_UNSUPPORTED_PROTOCOL;
  }

  return CURLE_OK;
}

/*
 * Receives data from the connected socket. Use after successful
 * curl_easy_perform() with CURLOPT_CONNECT_ONLY option.
 * Returns CURLE_OK on success, error code on error.
 */
CURLcode curl_easy_recv(struct Curl_easy *data, void *buffer, size_t buflen,
                        size_t *n)
{
  curl_socket_t sfd;
  CURLcode result;
  ssize_t n1;
  struct connectdata *c;

  if(Curl_is_in_callback(data))
    return CURLE_RECURSIVE_API_CALL;

  result = easy_connection(data, &sfd, &c);
  if(result)
    return result;

  if(!data->conn)
    /* on first invoke, the transfer has been detached from the connection and
       needs to be reattached */
    Curl_attach_connnection(data, c);

  *n = 0;
  result = Curl_read(data, sfd, buffer, buflen, &n1);

  if(result)
    return result;

  *n = (size_t)n1;

  return CURLE_OK;
}

/*
 * Sends data over the connected socket. Use after successful
 * curl_easy_perform() with CURLOPT_CONNECT_ONLY option.
 */
CURLcode curl_easy_send(struct Curl_easy *data, const void *buffer,
                        size_t buflen, size_t *n)
{
  curl_socket_t sfd;
  CURLcode result;
  ssize_t n1;
  struct connectdata *c = NULL;
  SIGPIPE_VARIABLE(pipe_st);

  if(Curl_is_in_callback(data))
    return CURLE_RECURSIVE_API_CALL;

  result = easy_connection(data, &sfd, &c);
  if(result)
    return result;

  if(!data->conn)
    /* on first invoke, the transfer has been detached from the connection and
       needs to be reattached */
    Curl_attach_connnection(data, c);

  *n = 0;
  sigpipe_ignore(data, &pipe_st);
  result = Curl_write(data, sfd, buffer, buflen, &n1);
  sigpipe_restore(&pipe_st);

  if(n1 == -1)
    return CURLE_SEND_ERROR;

  /* detect EAGAIN */
  if(!result && !n1)
    return CURLE_AGAIN;

  *n = (size_t)n1;

  return result;
}

/*
 * Wrapper to call functions in Curl_conncache_foreach()
 *
 * Returns always 0.
 */
static int conn_upkeep(struct Curl_easy *data,
                       struct connectdata *conn,
                       void *param)
{
  /* Param is unused. */
  (void)param;

  if(conn->handler->connection_check) {
    /* briefly attach the connection to this transfer for the purpose of
       checking it */
    Curl_attach_connnection(data, conn);

    /* Do a protocol-specific keepalive check on the connection. */
    conn->handler->connection_check(data, conn, CONNCHECK_KEEPALIVE);
    /* detach the connection again */
    Curl_detach_connnection(data);
  }

  return 0; /* continue iteration */
}

static CURLcode upkeep(struct conncache *conn_cache, void *data)
{
  /* Loop over every connection and make connection alive. */
  Curl_conncache_foreach(data,
                         conn_cache,
                         data,
                         conn_upkeep);
  return CURLE_OK;
}

/*
 * Performs connection upkeep for the given session handle.
 */
CURLcode curl_easy_upkeep(struct Curl_easy *data)
{
  /* Verify that we got an easy handle we can work with. */
  if(!GOOD_EASY_HANDLE(data))
    return CURLE_BAD_FUNCTION_ARGUMENT;

  if(data->multi_easy) {
    /* Use the common function to keep connections alive. */
    return upkeep(&data->multi_easy->conn_cache, data);
  }
  else {
    /* No connections, so just return success */
    return CURLE_OK;
  }
}
