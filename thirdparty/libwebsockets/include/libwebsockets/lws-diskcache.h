/*
 * libwebsockets - disk cache helpers
 *
 * Copyright (C) 2010-2018 Andy Green <andy@warmcat.com>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation:
 *  version 2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 *
 * included from libwebsockets.h
 */

/*! \defgroup diskcache LWS disk cache
 * ## Disk cache API
 *
 * Lws provides helper apis useful if you need a disk cache containing hashed
 * files and need to delete files from it on an LRU basis to keep it below some
 * size limit.
 *
 * The API `lws_diskcache_prepare()` deals with creating the cache dir and
 * 256 subdirs, which are used according to the first two chars of the hex
 * hash of the cache file.
 *
 * `lws_diskcache_create()` and `lws_diskcache_destroy()` allocate and free
 * an opaque struct that represents the disk cache.
 *
 * `lws_diskcache_trim()` should be called at eg, 1s intervals to perform the
 * cache dir monitoring and LRU autodelete in the background lazily.  It can
 * be done in its own thread or on a timer... it monitors the directories in a
 * stateful way that stats one or more file in the cache per call, and keeps
 * a list of the oldest files as it goes.  When it completes a scan, if the
 * aggregate size is over the limit, it will delete oldest files first to try
 * to keep it under the limit.
 *
 * The cache size monitoring is extremely efficient in time and memory even when
 * the cache directory becomes huge.
 *
 * `lws_diskcache_query()` is used to determine if the file already exists in
 * the cache, or if it must be created.  If it must be created, then the file
 * is opened using a temp name that must be converted to a findable name with
 * `lws_diskcache_finalize_name()` when the generation of the file contents are
 * complete.  Aborted cached files that did not complete generation will be
 * flushed by the LRU eventually.  If the file already exists, it is 'touched'
 * to make it new again and the fd returned.
 *
 */
///@{

struct lws_diskcache_scan;

/**
 * lws_diskcache_create() - creates an opaque struct representing the disk cache
 *
 * \param cache_dir_base: The cache dir path, eg `/var/cache/mycache`
 * \param cache_size_limit: maximum size on disk the cache is allowed to use
 *
 * This returns an opaque `struct lws_diskcache_scan *` which represents the
 * disk cache, the trim scanning state and so on.  You should use
 * `lws_diskcache_destroy()` to free it to destroy it.
 */
LWS_VISIBLE LWS_EXTERN struct lws_diskcache_scan *
lws_diskcache_create(const char *cache_dir_base, uint64_t cache_size_limit);

/**
 * lws_diskcache_destroy() - destroys the pointer returned by ...create()
 *
 * \param lds: pointer to the pointer returned by lws_diskcache_create()
 *
 * Frees *lds and any allocations it did, and then sets *lds to NULL and
 * returns.
 */
LWS_VISIBLE LWS_EXTERN void
lws_diskcache_destroy(struct lws_diskcache_scan **lds);

/**
 * lws_diskcache_prepare() - ensures the cache dir structure exists on disk
 *
 * \param cache_base_dir: The cache dir path, eg `/var/cache/mycache`
 * \param mode: octal dir mode to enforce, like 0700
 * \param uid: uid the cache dir should belong to
 *
 * This should be called while your app is still privileged.  It will create
 * the cache directory structure on disk as necessary, enforce the given access
 * mode on it and set the given uid as the owner.  It won't make any trouble
 * if the cache already exists.
 *
 * Typically the mode is 0700 and the owner is the user that your application
 * will transition to use when it drops root privileges.
 */
LWS_VISIBLE LWS_EXTERN int
lws_diskcache_prepare(const char *cache_base_dir, int mode, int uid);

#define LWS_DISKCACHE_QUERY_NO_CACHE	0
#define LWS_DISKCACHE_QUERY_EXISTS	1
#define LWS_DISKCACHE_QUERY_CREATING	2
#define LWS_DISKCACHE_QUERY_ONGOING	3 /* something else is creating it */

/**
 * lws_diskcache_query() - ensures the cache dir structure exists on disk
 *
 * \param lds: The opaque struct representing the disk cache
 * \param is_bot: nonzero means the request is from a bot.  Don't create new cache contents if so.
 * \param hash_hex: hex string representation of the cache object hash
 * \param _fd: pointer to the fd to be set
 * \param cache: destination string to take the cache filepath
 * \param cache_len: length of the buffer at `cache`
 * \param extant_cache_len: pointer to a size_t to take any extant cached file size
 *
 * This function is called when you want to find if the hashed name already
 * exists in the cache.  The possibilities for the return value are
 *
 *  - LWS_DISKCACHE_QUERY_NO_CACHE: It's not in the cache and you can't create
 *    it in the cache for whatever reason.
 *  - LWS_DISKCACHE_QUERY_EXISTS: It exists in the cache.  It's open RDONLY and
 *    *_fd has been set to the file descriptor.  *extant_cache_len has been set
 *    to the size of the cached file in bytes.  cache has been set to the
 *    full filepath of the cached file.  Closing _fd is your responsibility.
 *  - LWS_DISKCACHE_QUERY_CREATING: It didn't exist, but a temp file has been
 *    created in the cache and *_fd set to a file descriptor opened on it RDWR.
 *    You should create the contents, and call `lws_diskcache_finalize_name()`
 *    when it is done.  Closing _fd is your responsibility.
 *  - LWS_DISKCACHE_QUERY_ONGOING: not returned by this api, but you may find it
 *    desirable to make a wrapper function which can handle another asynchronous
 *    process that is already creating the cached file.  This can be used to
 *    indicate that situation externally... how to determine the same thing is
 *    already being generated is out of scope of this api.
 */
LWS_VISIBLE LWS_EXTERN int
lws_diskcache_query(struct lws_diskcache_scan *lds, int is_bot,
		    const char *hash_hex, int *_fd, char *cache, int cache_len,
		    size_t *extant_cache_len);

/**
 * lws_diskcache_query() - ensures the cache dir structure exists on disk
 *
 * \param cache: The cache file temp name returned with LWS_DISKCACHE_QUERY_CREATING
 *
 * This renames the cache file you are creating to its final name.  It should
 * be called on the temp name returned by `lws_diskcache_query()` if it gave a
 * LWS_DISKCACHE_QUERY_CREATING return, after you have filled the cache file and
 * closed it.
 */
LWS_VISIBLE LWS_EXTERN int
lws_diskcache_finalize_name(char *cache);

/**
 * lws_diskcache_trim() - performs one or more file checks in the cache for size management
 *
 * \param lds: The opaque object representing the cache
 *
 * This should be called periodically to statefully walk the cache on disk
 * collecting the oldest files.  When it has visited every file, if the cache
 * is oversize it will delete the oldest files until it's back under size again.
 *
 * Each time it's called, it will look at one or more dir in the cache.  If
 * called when the cache is oversize, it increases the amount of work done each
 * call until it is reduced again.  Typically it will take 256 calls before it
 * deletes anything, so if called once per second, it will delete files once
 * every 4 minutes.  Each call is very inexpensive both in memory and time.
 */
LWS_VISIBLE LWS_EXTERN int
lws_diskcache_trim(struct lws_diskcache_scan *lds);


/**
 * lws_diskcache_secs_to_idle() - see how long to idle before calling trim
 *
 * \param lds: The opaque object representing the cache
 *
 * If the cache is undersize, there's no need to monitor it immediately.  This
 * suggests how long to "sleep" before calling `lws_diskcache_trim()` again.
 */
LWS_VISIBLE LWS_EXTERN int
lws_diskcache_secs_to_idle(struct lws_diskcache_scan *lds);
