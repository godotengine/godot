#ifndef HEADER_CURL_MULTIIF_H
#define HEADER_CURL_MULTIIF_H
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
 * Prototypes for library-wide functions provided by multi.c
 */

CURLcode Curl_updatesocket(struct Curl_easy *data);
void Curl_expire(struct Curl_easy *data, timediff_t milli, expire_id);
void Curl_expire_clear(struct Curl_easy *data);
void Curl_expire_done(struct Curl_easy *data, expire_id id);
CURLMcode Curl_update_timer(struct Curl_multi *multi) WARN_UNUSED_RESULT;
void Curl_attach_connnection(struct Curl_easy *data,
                             struct connectdata *conn);
void Curl_detach_connnection(struct Curl_easy *data);
bool Curl_multiplex_wanted(const struct Curl_multi *multi);
void Curl_set_in_callback(struct Curl_easy *data, bool value);
bool Curl_is_in_callback(struct Curl_easy *easy);
CURLcode Curl_preconnect(struct Curl_easy *data);

/* Internal version of curl_multi_init() accepts size parameters for the
   socket and connection hashes */
struct Curl_multi *Curl_multi_handle(int hashsize, int chashsize);

/* the write bits start at bit 16 for the *getsock() bitmap */
#define GETSOCK_WRITEBITSTART 16

#define GETSOCK_BLANK 0 /* no bits set */

/* set the bit for the given sock number to make the bitmap for writable */
#define GETSOCK_WRITESOCK(x) (1 << (GETSOCK_WRITEBITSTART + (x)))

/* set the bit for the given sock number to make the bitmap for readable */
#define GETSOCK_READSOCK(x) (1 << (x))

#ifdef DEBUGBUILD
 /*
  * Curl_multi_dump is not a stable public function, this is only meant to
  * allow easier tracking of the internal handle's state and what sockets
  * they use. Only for research and development DEBUGBUILD enabled builds.
  */
void Curl_multi_dump(struct Curl_multi *multi);
#endif

/* Return the value of the CURLMOPT_MAX_HOST_CONNECTIONS option */
size_t Curl_multi_max_host_connections(struct Curl_multi *multi);

/* Return the value of the CURLMOPT_MAX_TOTAL_CONNECTIONS option */
size_t Curl_multi_max_total_connections(struct Curl_multi *multi);

void Curl_multiuse_state(struct Curl_easy *data,
                         int bundlestate); /* use BUNDLE_* defines */

/*
 * Curl_multi_closed()
 *
 * Used by the connect code to tell the multi_socket code that one of the
 * sockets we were using is about to be closed.  This function will then
 * remove it from the sockethash for this handle to make the multi_socket API
 * behave properly, especially for the case when libcurl will create another
 * socket again and it gets the same file descriptor number.
 */

void Curl_multi_closed(struct Curl_easy *data, curl_socket_t s);

/*
 * Add a handle and move it into PERFORM state at once. For pushed streams.
 */
CURLMcode Curl_multi_add_perform(struct Curl_multi *multi,
                                 struct Curl_easy *data,
                                 struct connectdata *conn);


/* Return the value of the CURLMOPT_MAX_CONCURRENT_STREAMS option */
unsigned int Curl_multi_max_concurrent_streams(struct Curl_multi *multi);

#endif /* HEADER_CURL_MULTIIF_H */
