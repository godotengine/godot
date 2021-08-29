#ifndef HEADER_CURL_CONNCACHE_H
#define HEADER_CURL_CONNCACHE_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2015 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
 * Copyright (C) 2012 - 2014, Linus Nielsen Feltzing, <linus@haxx.se>
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
 * All accesses to struct fields and changing of data in the connection cache
 * and connectbundles must be done with the conncache LOCKED. The cache might
 * be shared.
 */

#include "timeval.h"

struct connectdata;

struct conncache {
  struct Curl_hash hash;
  size_t num_conn;
  long next_connection_id;
  struct curltime last_cleanup;
  /* handle used for closing cached connections */
  struct Curl_easy *closure_handle;
};

#define BUNDLE_NO_MULTIUSE -1
#define BUNDLE_UNKNOWN     0  /* initial value */
#define BUNDLE_MULTIPLEX   2

#ifdef CURLDEBUG
/* the debug versions of these macros make extra certain that the lock is
   never doubly locked or unlocked */
#define CONNCACHE_LOCK(x)                                               \
  do {                                                                  \
    if((x)->share) {                                                    \
      Curl_share_lock((x), CURL_LOCK_DATA_CONNECT,                      \
                      CURL_LOCK_ACCESS_SINGLE);                         \
      DEBUGASSERT(!(x)->state.conncache_lock);                          \
      (x)->state.conncache_lock = TRUE;                                 \
    }                                                                   \
  } while(0)

#define CONNCACHE_UNLOCK(x)                                             \
  do {                                                                  \
    if((x)->share) {                                                    \
      DEBUGASSERT((x)->state.conncache_lock);                           \
      (x)->state.conncache_lock = FALSE;                                \
      Curl_share_unlock((x), CURL_LOCK_DATA_CONNECT);                   \
    }                                                                   \
  } while(0)
#else
#define CONNCACHE_LOCK(x) if((x)->share)                                \
    Curl_share_lock((x), CURL_LOCK_DATA_CONNECT, CURL_LOCK_ACCESS_SINGLE)
#define CONNCACHE_UNLOCK(x) if((x)->share)              \
    Curl_share_unlock((x), CURL_LOCK_DATA_CONNECT)
#endif

struct connectbundle {
  int multiuse;                 /* supports multi-use */
  size_t num_connections;       /* Number of connections in the bundle */
  struct Curl_llist conn_list;  /* The connectdata members of the bundle */
};

/* returns 1 on error, 0 is fine */
int Curl_conncache_init(struct conncache *, int size);
void Curl_conncache_destroy(struct conncache *connc);

/* return the correct bundle, to a host or a proxy */
struct connectbundle *Curl_conncache_find_bundle(struct Curl_easy *data,
                                                 struct connectdata *conn,
                                                 struct conncache *connc,
                                                 const char **hostp);
/* returns number of connections currently held in the connection cache */
size_t Curl_conncache_size(struct Curl_easy *data);

bool Curl_conncache_return_conn(struct Curl_easy *data,
                                struct connectdata *conn);
CURLcode Curl_conncache_add_conn(struct Curl_easy *data) WARN_UNUSED_RESULT;
void Curl_conncache_remove_conn(struct Curl_easy *data,
                                struct connectdata *conn,
                                bool lock);
bool Curl_conncache_foreach(struct Curl_easy *data,
                            struct conncache *connc,
                            void *param,
                            int (*func)(struct Curl_easy *data,
                                        struct connectdata *conn,
                                        void *param));

struct connectdata *
Curl_conncache_find_first_connection(struct conncache *connc);

struct connectdata *
Curl_conncache_extract_bundle(struct Curl_easy *data,
                              struct connectbundle *bundle);
struct connectdata *
Curl_conncache_extract_oldest(struct Curl_easy *data);
void Curl_conncache_close_all_connections(struct conncache *connc);
void Curl_conncache_print(struct conncache *connc);

#endif /* HEADER_CURL_CONNCACHE_H */
