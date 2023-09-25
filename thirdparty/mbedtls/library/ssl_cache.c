/*
 *  SSL session cache implementation
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may
 *  not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
/*
 * These session callbacks use a simple chained list
 * to store and retrieve the session information.
 */

#include "common.h"

#if defined(MBEDTLS_SSL_CACHE_C)

#include "mbedtls/platform.h"

#include "mbedtls/ssl_cache.h"
#include "ssl_misc.h"

#include <string.h>

void mbedtls_ssl_cache_init(mbedtls_ssl_cache_context *cache)
{
    memset(cache, 0, sizeof(mbedtls_ssl_cache_context));

    cache->timeout = MBEDTLS_SSL_CACHE_DEFAULT_TIMEOUT;
    cache->max_entries = MBEDTLS_SSL_CACHE_DEFAULT_MAX_ENTRIES;

#if defined(MBEDTLS_THREADING_C)
    mbedtls_mutex_init(&cache->mutex);
#endif
}

MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_cache_find_entry(mbedtls_ssl_cache_context *cache,
                                unsigned char const *session_id,
                                size_t session_id_len,
                                mbedtls_ssl_cache_entry **dst)
{
    int ret = 1;
#if defined(MBEDTLS_HAVE_TIME)
    mbedtls_time_t t = mbedtls_time(NULL);
#endif
    mbedtls_ssl_cache_entry *cur;

    for (cur = cache->chain; cur != NULL; cur = cur->next) {
#if defined(MBEDTLS_HAVE_TIME)
        if (cache->timeout != 0 &&
            (int) (t - cur->timestamp) > cache->timeout) {
            continue;
        }
#endif

        if (session_id_len != cur->session_id_len ||
            memcmp(session_id, cur->session_id,
                   cur->session_id_len) != 0) {
            continue;
        }

        break;
    }

    if (cur != NULL) {
        *dst = cur;
        ret = 0;
    }

    return ret;
}


int mbedtls_ssl_cache_get(void *data,
                          unsigned char const *session_id,
                          size_t session_id_len,
                          mbedtls_ssl_session *session)
{
    int ret = 1;
    mbedtls_ssl_cache_context *cache = (mbedtls_ssl_cache_context *) data;
    mbedtls_ssl_cache_entry *entry;

#if defined(MBEDTLS_THREADING_C)
    if ((ret = mbedtls_mutex_lock(&cache->mutex)) != 0) {
        return ret;
    }
#endif

    ret = ssl_cache_find_entry(cache, session_id, session_id_len, &entry);
    if (ret != 0) {
        goto exit;
    }

    ret = mbedtls_ssl_session_load(session,
                                   entry->session,
                                   entry->session_len);
    if (ret != 0) {
        goto exit;
    }

    ret = 0;

exit:
#if defined(MBEDTLS_THREADING_C)
    if (mbedtls_mutex_unlock(&cache->mutex) != 0) {
        ret = MBEDTLS_ERR_THREADING_MUTEX_ERROR;
    }
#endif

    return ret;
}

/* zeroize a cache entry */
static void ssl_cache_entry_zeroize(mbedtls_ssl_cache_entry *entry)
{
    if (entry == NULL) {
        return;
    }

    /* zeroize and free session structure */
    if (entry->session != NULL) {
        mbedtls_platform_zeroize(entry->session, entry->session_len);
        mbedtls_free(entry->session);
    }

    /* zeroize the whole entry structure */
    mbedtls_platform_zeroize(entry, sizeof(mbedtls_ssl_cache_entry));
}

MBEDTLS_CHECK_RETURN_CRITICAL
static int ssl_cache_pick_writing_slot(mbedtls_ssl_cache_context *cache,
                                       unsigned char const *session_id,
                                       size_t session_id_len,
                                       mbedtls_ssl_cache_entry **dst)
{
#if defined(MBEDTLS_HAVE_TIME)
    mbedtls_time_t t = mbedtls_time(NULL), oldest = 0;
#endif /* MBEDTLS_HAVE_TIME */

    mbedtls_ssl_cache_entry *old = NULL;
    int count = 0;
    mbedtls_ssl_cache_entry *cur, *last;

    /* Check 1: Is there already an entry with the given session ID?
     *
     * If yes, overwrite it.
     *
     * If not, `count` will hold the size of the session cache
     * at the end of this loop, and `last` will point to the last
     * entry, both of which will be used later. */

    last = NULL;
    for (cur = cache->chain; cur != NULL; cur = cur->next) {
        count++;
        if (session_id_len == cur->session_id_len &&
            memcmp(session_id, cur->session_id, cur->session_id_len) == 0) {
            goto found;
        }
        last = cur;
    }

    /* Check 2: Is there an outdated entry in the cache?
     *
     * If so, overwrite it.
     *
     * If not, remember the oldest entry in `old` for later.
     */

#if defined(MBEDTLS_HAVE_TIME)
    for (cur = cache->chain; cur != NULL; cur = cur->next) {
        if (cache->timeout != 0 &&
            (int) (t - cur->timestamp) > cache->timeout) {
            goto found;
        }

        if (oldest == 0 || cur->timestamp < oldest) {
            oldest = cur->timestamp;
            old = cur;
        }
    }
#endif /* MBEDTLS_HAVE_TIME */

    /* Check 3: Is there free space in the cache? */

    if (count < cache->max_entries) {
        /* Create new entry */
        cur = mbedtls_calloc(1, sizeof(mbedtls_ssl_cache_entry));
        if (cur == NULL) {
            return 1;
        }

        /* Append to the end of the linked list. */
        if (last == NULL) {
            cache->chain = cur;
        } else {
            last->next = cur;
        }

        goto found;
    }

    /* Last resort: The cache is full and doesn't contain any outdated
     * elements. In this case, we evict the oldest one, judged by timestamp
     * (if present) or cache-order. */

#if defined(MBEDTLS_HAVE_TIME)
    if (old == NULL) {
        /* This should only happen on an ill-configured cache
         * with max_entries == 0. */
        return 1;
    }
#else /* MBEDTLS_HAVE_TIME */
    /* Reuse first entry in chain, but move to last place. */
    if (cache->chain == NULL) {
        return 1;
    }

    old = cache->chain;
    cache->chain = old->next;
    old->next = NULL;
    last->next = old;
#endif /* MBEDTLS_HAVE_TIME */

    /* Now `old` points to the oldest entry to be overwritten. */
    cur = old;

found:

    /* If we're reusing an entry, free it first. */
    if (cur->session != NULL) {
        /* `ssl_cache_entry_zeroize` would break the chain,
         * so we reuse `old` to record `next` temporarily. */
        old = cur->next;
        ssl_cache_entry_zeroize(cur);
        cur->next = old;
    }

#if defined(MBEDTLS_HAVE_TIME)
    cur->timestamp = t;
#endif

    *dst = cur;
    return 0;
}

int mbedtls_ssl_cache_set(void *data,
                          unsigned char const *session_id,
                          size_t session_id_len,
                          const mbedtls_ssl_session *session)
{
    int ret = 1;
    mbedtls_ssl_cache_context *cache = (mbedtls_ssl_cache_context *) data;
    mbedtls_ssl_cache_entry *cur;

    size_t session_serialized_len;
    unsigned char *session_serialized = NULL;

#if defined(MBEDTLS_THREADING_C)
    if ((ret = mbedtls_mutex_lock(&cache->mutex)) != 0) {
        return ret;
    }
#endif

    ret = ssl_cache_pick_writing_slot(cache,
                                      session_id, session_id_len,
                                      &cur);
    if (ret != 0) {
        goto exit;
    }

    /* Check how much space we need to serialize the session
     * and allocate a sufficiently large buffer. */
    ret = mbedtls_ssl_session_save(session, NULL, 0, &session_serialized_len);
    if (ret != MBEDTLS_ERR_SSL_BUFFER_TOO_SMALL) {
        ret = 1;
        goto exit;
    }

    session_serialized = mbedtls_calloc(1, session_serialized_len);
    if (session_serialized == NULL) {
        ret = MBEDTLS_ERR_SSL_ALLOC_FAILED;
        goto exit;
    }

    /* Now serialize the session into the allocated buffer. */
    ret = mbedtls_ssl_session_save(session,
                                   session_serialized,
                                   session_serialized_len,
                                   &session_serialized_len);
    if (ret != 0) {
        goto exit;
    }

    if (session_id_len > sizeof(cur->session_id)) {
        ret = 1;
        goto exit;
    }
    cur->session_id_len = session_id_len;
    memcpy(cur->session_id, session_id, session_id_len);

    cur->session = session_serialized;
    cur->session_len = session_serialized_len;
    session_serialized = NULL;

    ret = 0;

exit:
#if defined(MBEDTLS_THREADING_C)
    if (mbedtls_mutex_unlock(&cache->mutex) != 0) {
        ret = MBEDTLS_ERR_THREADING_MUTEX_ERROR;
    }
#endif

    if (session_serialized != NULL) {
        mbedtls_platform_zeroize(session_serialized, session_serialized_len);
        mbedtls_free(session_serialized);
        session_serialized = NULL;
    }

    return ret;
}

int mbedtls_ssl_cache_remove(void *data,
                             unsigned char const *session_id,
                             size_t session_id_len)
{
    int ret = 1;
    mbedtls_ssl_cache_context *cache = (mbedtls_ssl_cache_context *) data;
    mbedtls_ssl_cache_entry *entry;
    mbedtls_ssl_cache_entry *prev;

#if defined(MBEDTLS_THREADING_C)
    if ((ret = mbedtls_mutex_lock(&cache->mutex)) != 0) {
        return ret;
    }
#endif

    ret = ssl_cache_find_entry(cache, session_id, session_id_len, &entry);
    /* No valid entry found, exit with success */
    if (ret != 0) {
        ret = 0;
        goto exit;
    }

    /* Now we remove the entry from the chain */
    if (entry == cache->chain) {
        cache->chain = entry->next;
        goto free;
    }
    for (prev = cache->chain; prev->next != NULL; prev = prev->next) {
        if (prev->next == entry) {
            prev->next = entry->next;
            break;
        }
    }

free:
    ssl_cache_entry_zeroize(entry);
    mbedtls_free(entry);
    ret = 0;

exit:
#if defined(MBEDTLS_THREADING_C)
    if (mbedtls_mutex_unlock(&cache->mutex) != 0) {
        ret = MBEDTLS_ERR_THREADING_MUTEX_ERROR;
    }
#endif

    return ret;
}

#if defined(MBEDTLS_HAVE_TIME)
void mbedtls_ssl_cache_set_timeout(mbedtls_ssl_cache_context *cache, int timeout)
{
    if (timeout < 0) {
        timeout = 0;
    }

    cache->timeout = timeout;
}
#endif /* MBEDTLS_HAVE_TIME */

void mbedtls_ssl_cache_set_max_entries(mbedtls_ssl_cache_context *cache, int max)
{
    if (max < 0) {
        max = 0;
    }

    cache->max_entries = max;
}

void mbedtls_ssl_cache_free(mbedtls_ssl_cache_context *cache)
{
    mbedtls_ssl_cache_entry *cur, *prv;

    cur = cache->chain;

    while (cur != NULL) {
        prv = cur;
        cur = cur->next;

        ssl_cache_entry_zeroize(prv);
        mbedtls_free(prv);
    }

#if defined(MBEDTLS_THREADING_C)
    mbedtls_mutex_free(&cache->mutex);
#endif
    cache->chain = NULL;
}

#endif /* MBEDTLS_SSL_CACHE_C */
