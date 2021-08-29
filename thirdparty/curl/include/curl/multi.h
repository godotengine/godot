#ifndef CURLINC_MULTI_H
#define CURLINC_MULTI_H
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
/*
  This is an "external" header file. Don't give away any internals here!

  GOALS

  o Enable a "pull" interface. The application that uses libcurl decides where
    and when to ask libcurl to get/send data.

  o Enable multiple simultaneous transfers in the same thread without making it
    complicated for the application.

  o Enable the application to select() on its own file descriptors and curl's
    file descriptors simultaneous easily.

*/

/*
 * This header file should not really need to include "curl.h" since curl.h
 * itself includes this file and we expect user applications to do #include
 * <curl/curl.h> without the need for especially including multi.h.
 *
 * For some reason we added this include here at one point, and rather than to
 * break existing (wrongly written) libcurl applications, we leave it as-is
 * but with this warning attached.
 */
#include "curl.h"

#ifdef  __cplusplus
extern "C" {
#endif

#if defined(BUILDING_LIBCURL) || defined(CURL_STRICTER)
typedef struct Curl_multi CURLM;
#else
typedef void CURLM;
#endif

typedef enum {
  CURLM_CALL_MULTI_PERFORM = -1, /* please call curl_multi_perform() or
                                    curl_multi_socket*() soon */
  CURLM_OK,
  CURLM_BAD_HANDLE,      /* the passed-in handle is not a valid CURLM handle */
  CURLM_BAD_EASY_HANDLE, /* an easy handle was not good/valid */
  CURLM_OUT_OF_MEMORY,   /* if you ever get this, you're in deep sh*t */
  CURLM_INTERNAL_ERROR,  /* this is a libcurl bug */
  CURLM_BAD_SOCKET,      /* the passed in socket argument did not match */
  CURLM_UNKNOWN_OPTION,  /* curl_multi_setopt() with unsupported option */
  CURLM_ADDED_ALREADY,   /* an easy handle already added to a multi handle was
                            attempted to get added - again */
  CURLM_RECURSIVE_API_CALL, /* an api function was called from inside a
                               callback */
  CURLM_WAKEUP_FAILURE,  /* wakeup is unavailable or failed */
  CURLM_BAD_FUNCTION_ARGUMENT, /* function called with a bad parameter */
  CURLM_ABORTED_BY_CALLBACK,
  CURLM_LAST
} CURLMcode;

/* just to make code nicer when using curl_multi_socket() you can now check
   for CURLM_CALL_MULTI_SOCKET too in the same style it works for
   curl_multi_perform() and CURLM_CALL_MULTI_PERFORM */
#define CURLM_CALL_MULTI_SOCKET CURLM_CALL_MULTI_PERFORM

/* bitmask bits for CURLMOPT_PIPELINING */
#define CURLPIPE_NOTHING   0L
#define CURLPIPE_HTTP1     1L
#define CURLPIPE_MULTIPLEX 2L

typedef enum {
  CURLMSG_NONE, /* first, not used */
  CURLMSG_DONE, /* This easy handle has completed. 'result' contains
                   the CURLcode of the transfer */
  CURLMSG_LAST /* last, not used */
} CURLMSG;

struct CURLMsg {
  CURLMSG msg;       /* what this message means */
  CURL *easy_handle; /* the handle it concerns */
  union {
    void *whatever;    /* message-specific data */
    CURLcode result;   /* return code for transfer */
  } data;
};
typedef struct CURLMsg CURLMsg;

/* Based on poll(2) structure and values.
 * We don't use pollfd and POLL* constants explicitly
 * to cover platforms without poll(). */
#define CURL_WAIT_POLLIN    0x0001
#define CURL_WAIT_POLLPRI   0x0002
#define CURL_WAIT_POLLOUT   0x0004

struct curl_waitfd {
  curl_socket_t fd;
  short events;
  short revents; /* not supported yet */
};

/*
 * Name:    curl_multi_init()
 *
 * Desc:    inititalize multi-style curl usage
 *
 * Returns: a new CURLM handle to use in all 'curl_multi' functions.
 */
CURL_EXTERN CURLM *curl_multi_init(void);

/*
 * Name:    curl_multi_add_handle()
 *
 * Desc:    add a standard curl handle to the multi stack
 *
 * Returns: CURLMcode type, general multi error code.
 */
CURL_EXTERN CURLMcode curl_multi_add_handle(CURLM *multi_handle,
                                            CURL *curl_handle);

 /*
  * Name:    curl_multi_remove_handle()
  *
  * Desc:    removes a curl handle from the multi stack again
  *
  * Returns: CURLMcode type, general multi error code.
  */
CURL_EXTERN CURLMcode curl_multi_remove_handle(CURLM *multi_handle,
                                               CURL *curl_handle);

 /*
  * Name:    curl_multi_fdset()
  *
  * Desc:    Ask curl for its fd_set sets. The app can use these to select() or
  *          poll() on. We want curl_multi_perform() called as soon as one of
  *          them are ready.
  *
  * Returns: CURLMcode type, general multi error code.
  */
CURL_EXTERN CURLMcode curl_multi_fdset(CURLM *multi_handle,
                                       fd_set *read_fd_set,
                                       fd_set *write_fd_set,
                                       fd_set *exc_fd_set,
                                       int *max_fd);

/*
 * Name:     curl_multi_wait()
 *
 * Desc:     Poll on all fds within a CURLM set as well as any
 *           additional fds passed to the function.
 *
 * Returns:  CURLMcode type, general multi error code.
 */
CURL_EXTERN CURLMcode curl_multi_wait(CURLM *multi_handle,
                                      struct curl_waitfd extra_fds[],
                                      unsigned int extra_nfds,
                                      int timeout_ms,
                                      int *ret);

/*
 * Name:     curl_multi_poll()
 *
 * Desc:     Poll on all fds within a CURLM set as well as any
 *           additional fds passed to the function.
 *
 * Returns:  CURLMcode type, general multi error code.
 */
CURL_EXTERN CURLMcode curl_multi_poll(CURLM *multi_handle,
                                      struct curl_waitfd extra_fds[],
                                      unsigned int extra_nfds,
                                      int timeout_ms,
                                      int *ret);

/*
 * Name:     curl_multi_wakeup()
 *
 * Desc:     wakes up a sleeping curl_multi_poll call.
 *
 * Returns:  CURLMcode type, general multi error code.
 */
CURL_EXTERN CURLMcode curl_multi_wakeup(CURLM *multi_handle);

 /*
  * Name:    curl_multi_perform()
  *
  * Desc:    When the app thinks there's data available for curl it calls this
  *          function to read/write whatever there is right now. This returns
  *          as soon as the reads and writes are done. This function does not
  *          require that there actually is data available for reading or that
  *          data can be written, it can be called just in case. It returns
  *          the number of handles that still transfer data in the second
  *          argument's integer-pointer.
  *
  * Returns: CURLMcode type, general multi error code. *NOTE* that this only
  *          returns errors etc regarding the whole multi stack. There might
  *          still have occurred problems on individual transfers even when
  *          this returns OK.
  */
CURL_EXTERN CURLMcode curl_multi_perform(CURLM *multi_handle,
                                         int *running_handles);

 /*
  * Name:    curl_multi_cleanup()
  *
  * Desc:    Cleans up and removes a whole multi stack. It does not free or
  *          touch any individual easy handles in any way. We need to define
  *          in what state those handles will be if this function is called
  *          in the middle of a transfer.
  *
  * Returns: CURLMcode type, general multi error code.
  */
CURL_EXTERN CURLMcode curl_multi_cleanup(CURLM *multi_handle);

/*
 * Name:    curl_multi_info_read()
 *
 * Desc:    Ask the multi handle if there's any messages/informationals from
 *          the individual transfers. Messages include informationals such as
 *          error code from the transfer or just the fact that a transfer is
 *          completed. More details on these should be written down as well.
 *
 *          Repeated calls to this function will return a new struct each
 *          time, until a special "end of msgs" struct is returned as a signal
 *          that there is no more to get at this point.
 *
 *          The data the returned pointer points to will not survive calling
 *          curl_multi_cleanup().
 *
 *          The 'CURLMsg' struct is meant to be very simple and only contain
 *          very basic information. If more involved information is wanted,
 *          we will provide the particular "transfer handle" in that struct
 *          and that should/could/would be used in subsequent
 *          curl_easy_getinfo() calls (or similar). The point being that we
 *          must never expose complex structs to applications, as then we'll
 *          undoubtably get backwards compatibility problems in the future.
 *
 * Returns: A pointer to a filled-in struct, or NULL if it failed or ran out
 *          of structs. It also writes the number of messages left in the
 *          queue (after this read) in the integer the second argument points
 *          to.
 */
CURL_EXTERN CURLMsg *curl_multi_info_read(CURLM *multi_handle,
                                          int *msgs_in_queue);

/*
 * Name:    curl_multi_strerror()
 *
 * Desc:    The curl_multi_strerror function may be used to turn a CURLMcode
 *          value into the equivalent human readable error string.  This is
 *          useful for printing meaningful error messages.
 *
 * Returns: A pointer to a null-terminated error message.
 */
CURL_EXTERN const char *curl_multi_strerror(CURLMcode);

/*
 * Name:    curl_multi_socket() and
 *          curl_multi_socket_all()
 *
 * Desc:    An alternative version of curl_multi_perform() that allows the
 *          application to pass in one of the file descriptors that have been
 *          detected to have "action" on them and let libcurl perform.
 *          See man page for details.
 */
#define CURL_POLL_NONE   0
#define CURL_POLL_IN     1
#define CURL_POLL_OUT    2
#define CURL_POLL_INOUT  3
#define CURL_POLL_REMOVE 4

#define CURL_SOCKET_TIMEOUT CURL_SOCKET_BAD

#define CURL_CSELECT_IN   0x01
#define CURL_CSELECT_OUT  0x02
#define CURL_CSELECT_ERR  0x04

typedef int (*curl_socket_callback)(CURL *easy,      /* easy handle */
                                    curl_socket_t s, /* socket */
                                    int what,        /* see above */
                                    void *userp,     /* private callback
                                                        pointer */
                                    void *socketp);  /* private socket
                                                        pointer */
/*
 * Name:    curl_multi_timer_callback
 *
 * Desc:    Called by libcurl whenever the library detects a change in the
 *          maximum number of milliseconds the app is allowed to wait before
 *          curl_multi_socket() or curl_multi_perform() must be called
 *          (to allow libcurl's timed events to take place).
 *
 * Returns: The callback should return zero.
 */
typedef int (*curl_multi_timer_callback)(CURLM *multi,    /* multi handle */
                                         long timeout_ms, /* see above */
                                         void *userp);    /* private callback
                                                             pointer */

CURL_EXTERN CURLMcode curl_multi_socket(CURLM *multi_handle, curl_socket_t s,
                                        int *running_handles);

CURL_EXTERN CURLMcode curl_multi_socket_action(CURLM *multi_handle,
                                               curl_socket_t s,
                                               int ev_bitmask,
                                               int *running_handles);

CURL_EXTERN CURLMcode curl_multi_socket_all(CURLM *multi_handle,
                                            int *running_handles);

#ifndef CURL_ALLOW_OLD_MULTI_SOCKET
/* This macro below was added in 7.16.3 to push users who recompile to use
   the new curl_multi_socket_action() instead of the old curl_multi_socket()
*/
#define curl_multi_socket(x,y,z) curl_multi_socket_action(x,y,0,z)
#endif

/*
 * Name:    curl_multi_timeout()
 *
 * Desc:    Returns the maximum number of milliseconds the app is allowed to
 *          wait before curl_multi_socket() or curl_multi_perform() must be
 *          called (to allow libcurl's timed events to take place).
 *
 * Returns: CURLM error code.
 */
CURL_EXTERN CURLMcode curl_multi_timeout(CURLM *multi_handle,
                                         long *milliseconds);

typedef enum {
  /* This is the socket callback function pointer */
  CURLOPT(CURLMOPT_SOCKETFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 1),

  /* This is the argument passed to the socket callback */
  CURLOPT(CURLMOPT_SOCKETDATA, CURLOPTTYPE_OBJECTPOINT, 2),

    /* set to 1 to enable pipelining for this multi handle */
  CURLOPT(CURLMOPT_PIPELINING, CURLOPTTYPE_LONG, 3),

   /* This is the timer callback function pointer */
  CURLOPT(CURLMOPT_TIMERFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 4),

  /* This is the argument passed to the timer callback */
  CURLOPT(CURLMOPT_TIMERDATA, CURLOPTTYPE_OBJECTPOINT, 5),

  /* maximum number of entries in the connection cache */
  CURLOPT(CURLMOPT_MAXCONNECTS, CURLOPTTYPE_LONG, 6),

  /* maximum number of (pipelining) connections to one host */
  CURLOPT(CURLMOPT_MAX_HOST_CONNECTIONS, CURLOPTTYPE_LONG, 7),

  /* maximum number of requests in a pipeline */
  CURLOPT(CURLMOPT_MAX_PIPELINE_LENGTH, CURLOPTTYPE_LONG, 8),

  /* a connection with a content-length longer than this
     will not be considered for pipelining */
  CURLOPT(CURLMOPT_CONTENT_LENGTH_PENALTY_SIZE, CURLOPTTYPE_OFF_T, 9),

  /* a connection with a chunk length longer than this
     will not be considered for pipelining */
  CURLOPT(CURLMOPT_CHUNK_LENGTH_PENALTY_SIZE, CURLOPTTYPE_OFF_T, 10),

  /* a list of site names(+port) that are blocked from pipelining */
  CURLOPT(CURLMOPT_PIPELINING_SITE_BL, CURLOPTTYPE_OBJECTPOINT, 11),

  /* a list of server types that are blocked from pipelining */
  CURLOPT(CURLMOPT_PIPELINING_SERVER_BL, CURLOPTTYPE_OBJECTPOINT, 12),

  /* maximum number of open connections in total */
  CURLOPT(CURLMOPT_MAX_TOTAL_CONNECTIONS, CURLOPTTYPE_LONG, 13),

   /* This is the server push callback function pointer */
  CURLOPT(CURLMOPT_PUSHFUNCTION, CURLOPTTYPE_FUNCTIONPOINT, 14),

  /* This is the argument passed to the server push callback */
  CURLOPT(CURLMOPT_PUSHDATA, CURLOPTTYPE_OBJECTPOINT, 15),

  /* maximum number of concurrent streams to support on a connection */
  CURLOPT(CURLMOPT_MAX_CONCURRENT_STREAMS, CURLOPTTYPE_LONG, 16),

  CURLMOPT_LASTENTRY /* the last unused */
} CURLMoption;


/*
 * Name:    curl_multi_setopt()
 *
 * Desc:    Sets options for the multi handle.
 *
 * Returns: CURLM error code.
 */
CURL_EXTERN CURLMcode curl_multi_setopt(CURLM *multi_handle,
                                        CURLMoption option, ...);


/*
 * Name:    curl_multi_assign()
 *
 * Desc:    This function sets an association in the multi handle between the
 *          given socket and a private pointer of the application. This is
 *          (only) useful for curl_multi_socket uses.
 *
 * Returns: CURLM error code.
 */
CURL_EXTERN CURLMcode curl_multi_assign(CURLM *multi_handle,
                                        curl_socket_t sockfd, void *sockp);


/*
 * Name: curl_push_callback
 *
 * Desc: This callback gets called when a new stream is being pushed by the
 *       server. It approves or denies the new stream. It can also decide
 *       to completely fail the connection.
 *
 * Returns: CURL_PUSH_OK, CURL_PUSH_DENY or CURL_PUSH_ERROROUT
 */
#define CURL_PUSH_OK       0
#define CURL_PUSH_DENY     1
#define CURL_PUSH_ERROROUT 2 /* added in 7.72.0 */

struct curl_pushheaders;  /* forward declaration only */

CURL_EXTERN char *curl_pushheader_bynum(struct curl_pushheaders *h,
                                        size_t num);
CURL_EXTERN char *curl_pushheader_byname(struct curl_pushheaders *h,
                                         const char *name);

typedef int (*curl_push_callback)(CURL *parent,
                                  CURL *easy,
                                  size_t num_headers,
                                  struct curl_pushheaders *headers,
                                  void *userp);

#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif
