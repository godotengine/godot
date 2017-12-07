/*
 * libusb synchronization using POSIX Threads
 *
 * Copyright © 2011 Vitali Lovich <vlovich@aliph.com>
 * Copyright © 2011 Peter Stuge <peter@stuge.se>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <config.h>

#include <time.h>
#if defined(__linux__) || defined(__OpenBSD__)
# if defined(__OpenBSD__)
#  define _BSD_SOURCE
# endif
# include <unistd.h>
# include <sys/syscall.h>
#elif defined(__APPLE__)
# include <mach/mach.h>
#elif defined(__CYGWIN__)
# include <windows.h>
#endif

#include "threads_posix.h"
#include "libusbi.h"

int usbi_cond_timedwait(pthread_cond_t *cond,
	pthread_mutex_t *mutex, const struct timeval *tv)
{
	struct timespec timeout;
	int r;

	r = usbi_backend.clock_gettime(USBI_CLOCK_REALTIME, &timeout);
	if (r < 0)
		return r;

	timeout.tv_sec += tv->tv_sec;
	timeout.tv_nsec += tv->tv_usec * 1000;
	while (timeout.tv_nsec >= 1000000000L) {
		timeout.tv_nsec -= 1000000000L;
		timeout.tv_sec++;
	}

	return pthread_cond_timedwait(cond, mutex, &timeout);
}

int usbi_get_tid(void)
{
	int ret = -1;
#if defined(__ANDROID__)
	ret = gettid();
#elif defined(__linux__)
	ret = syscall(SYS_gettid);
#elif defined(__OpenBSD__)
	/* The following only works with OpenBSD > 5.1 as it requires
	   real thread support. For 5.1 and earlier, -1 is returned. */
	ret = syscall(SYS_getthrid);
#elif defined(__APPLE__)
	ret = mach_thread_self();
	mach_port_deallocate(mach_task_self(), ret);
#elif defined(__CYGWIN__)
	ret = GetCurrentThreadId();
#endif
/* TODO: NetBSD thread ID support */
	return ret;
}
