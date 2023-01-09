/*
 *  sync abstraction
 *  Copyright 2015-2016 Collabora Ltd.
 *
 *  Based on the implementation from the Android Open Source Project,
 *
 *  Copyright 2012 Google, Inc
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without limitation
 *  the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *  and/or sell copies of the Software, and to permit persons to whom the
 *  Software is furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 *  OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 *  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 *  OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef _LIBSYNC_H
#define _LIBSYNC_H

#include <assert.h>
#include <errno.h>
#include <poll.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <time.h>

#if defined(__cplusplus)
extern "C" {
#endif

#ifdef ANDROID
/* On Android, rely on the system's libsync instead of rolling our own
 * sync_wait() and sync_merge().  This gives us compatibility with pre-4.7
 * Android kernels.
 */
#include <android/sync.h>

/**
 * Check if the fd represents a valid fence-fd.
 *
 * The android variant of this debug helper is implemented on top of the
 * system's libsync for compatibility with pre-4.7 android kernels.
 */
static inline bool
sync_valid_fd(int fd)
{
	/* sync_file_info() only available in SDK 26. */
#if ANDROID_API_LEVEL >= 26
	struct sync_file_info *info = sync_file_info(fd);
	if (!info)
		return false;
	sync_file_info_free(info);
#endif
	return true;
}
#else

#ifndef SYNC_IOC_MERGE
/* duplicated from linux/sync_file.h to avoid build-time dependency
 * on new (v4.7) kernel headers.  Once distro's are mostly using
 * something newer than v4.7 drop this and #include <linux/sync_file.h>
 * instead.
 */
struct sync_merge_data {
	char	name[32];
	int32_t	fd2;
	int32_t	fence;
	uint32_t	flags;
	uint32_t	pad;
};

struct sync_file_info {
	char	name[32];
	int32_t	status;
	uint32_t	flags;
	uint32_t	num_fences;
	uint32_t	pad;

	uint64_t	sync_fence_info;
};

#define SYNC_IOC_MAGIC		'>'
#define SYNC_IOC_MERGE		_IOWR(SYNC_IOC_MAGIC, 3, struct sync_merge_data)
#define SYNC_IOC_FILE_INFO	_IOWR(SYNC_IOC_MAGIC, 4, struct sync_file_info)
#endif


static inline int sync_wait(int fd, int timeout)
{
	struct pollfd fds = {0};
	int ret;
	struct timespec poll_start, poll_end;

	fds.fd = fd;
	fds.events = POLLIN;

	do {
		clock_gettime(CLOCK_MONOTONIC, &poll_start);
		ret = poll(&fds, 1, timeout);
		clock_gettime(CLOCK_MONOTONIC, &poll_end);
		if (ret > 0) {
			if (fds.revents & (POLLERR | POLLNVAL)) {
				errno = EINVAL;
				return -1;
			}
			return 0;
		} else if (ret == 0) {
			errno = ETIME;
			return -1;
		}
		timeout -= (poll_end.tv_sec - poll_start.tv_sec) * 1000 +
			(poll_end.tv_nsec - poll_end.tv_nsec) / 1000000;
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	return ret;
}

static inline int sync_merge(const char *name, int fd1, int fd2)
{
	struct sync_merge_data data = {{0}};
	int ret;

	data.fd2 = fd2;
	strncpy(data.name, name, sizeof(data.name));

	do {
		ret = ioctl(fd1, SYNC_IOC_MERGE, &data);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	if (ret < 0)
		return ret;

	return data.fence;
}

/**
 * Check if the fd represents a valid fence-fd.
 */
static inline bool
sync_valid_fd(int fd)
{
	struct sync_file_info info = {{0}};
	return ioctl(fd, SYNC_IOC_FILE_INFO, &info) >= 0;
}

#endif /* !ANDROID */

/* accumulate fd2 into fd1.  If *fd1 is not a valid fd then dup fd2,
 * otherwise sync_merge() and close the old *fd1.  This can be used
 * to implement the pattern:
 *
 *    init()
 *    {
 *       batch.fence_fd = -1;
 *    }
 *
 *    // does *NOT* take ownership of fd
 *    server_sync(int fd)
 *    {
 *       if (sync_accumulate("foo", &batch.fence_fd, fd)) {
 *          ... error ...
 *       }
 *    }
 */
static inline int sync_accumulate(const char *name, int *fd1, int fd2)
{
	int ret;

	assert(fd2 >= 0);

	if (*fd1 < 0) {
		*fd1 = dup(fd2);
		return 0;
	}

	ret = sync_merge(name, *fd1, fd2);
	if (ret < 0) {
		/* leave *fd1 as it is */
		return ret;
	}

	close(*fd1);
	*fd1 = ret;

	return 0;
}

/* Helper macro to complain if fd is non-negative and not a valid fence fd.
 * Sprinkle this around to help catch fd lifetime issues.
 */
#ifdef DEBUG
#  include "util/log.h"
#  define validate_fence_fd(fd) do {                                         \
      if (((fd) >= 0) && !sync_valid_fd(fd))                                 \
         mesa_loge("%s:%d: invalid fence fd: %d", __func__, __LINE__, (fd)); \
   } while (0)
#else
#  define validate_fence_fd(fd) do {} while (0)
#endif

#if defined(__cplusplus)
}
#endif

#endif
