/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "timestamp.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>

// clock_gettime() is not implemented on older versions of OS X (< 10.12)
#if defined(__APPLE__) && !defined(CLOCK_MONOTONIC)
#include <sys/time.h>
#define CLOCK_MONOTONIC 0
int clock_gettime(int clk_id, struct timespec *t) {
	(void)clk_id;

	// gettimeofday() does not return monotonic time but it should be good enough.
	struct timeval now;
	if (gettimeofday(&now, NULL))
		return -1;

	t->tv_sec = now.tv_sec;
	t->tv_nsec = now.tv_usec * 1000;
	return 0;
}
#endif // defined(__APPLE__) && !defined(CLOCK_MONOTONIC)

#endif

timestamp_t current_timestamp() {
#ifdef _WIN32
	return (timestamp_t)GetTickCount();
#else // POSIX
	struct timespec ts;
	if (clock_gettime(CLOCK_MONOTONIC, &ts))
		return 0;
	return (timestamp_t)ts.tv_sec * 1000 + (timestamp_t)ts.tv_nsec / 1000000;
#endif
}
