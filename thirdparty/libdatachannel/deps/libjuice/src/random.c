/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "random.h"
#include "log.h"
#include "thread.h" // for mutexes

#include <math.h>
#include <stdbool.h>
#include <time.h>

// getrandom() is not available in Android NDK API < 28 and needs glibc >= 2.25
#if defined(__linux__) && !defined(__ANDROID__) && (!defined(__GLIBC__) || __GLIBC__ > 2 || __GLIBC_MINOR__ >= 25)

#include <errno.h>
#include <sys/random.h>

static int random_bytes(void *buf, size_t size) {
	ssize_t ret = getrandom(buf, size, 0);
	if (ret < 0) {
		JLOG_WARN("getrandom failed, errno=%d", errno);
		return -1;
	}
	if ((size_t)ret < size) {
		JLOG_WARN("getrandom returned too few bytes, size=%zu, returned=%zu", size, (size_t)ret);
		return -1;
	}
	return 0;
}

#elif defined(_WIN32)

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601 // Windows 7
#endif

#include <windows.h>
//
#include <bcrypt.h>

static int random_bytes(void *buf, size_t size) {
	// Requires Windows 7 or later
	NTSTATUS status = BCryptGenRandom(NULL, (PUCHAR)buf, (ULONG)size, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
	return !status ? 0 : -1;
}

#else
static int random_bytes(void *buf, size_t size) {
	(void)buf;
	(void)size;
	return -1;
}
#endif

static unsigned int generate_seed() {
#ifdef _WIN32
	return (unsigned int)GetTickCount();
#else
	struct timespec ts;
	if (clock_gettime(CLOCK_REALTIME, &ts) == 0)
		return (unsigned int)(ts.tv_sec ^ ts.tv_nsec);
	else
		return (unsigned int)time(NULL);
#endif
}

void juice_random(void *buf, size_t size) {
	if (random_bytes(buf, size) == 0)
		return;

	// rand() is not thread-safe
	static mutex_t rand_mutex = MUTEX_INITIALIZER;
	mutex_lock(&rand_mutex);

	static bool srandom_called = false;
#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#define random_func random
#define srandom_func srandom
	if (!srandom_called)
		JLOG_DEBUG("Using random() for random bytes");
#else
#define random_func rand
#define srandom_func srand
	if (!srandom_called)
		JLOG_WARN("Falling back on rand() for random bytes");
#endif
	if (!srandom_called) {
		srandom_func(generate_seed());
		srandom_called = true;
	}
	// RAND_MAX is guaranteed to be at least 2^15 - 1
	uint8_t *bytes = buf;
	for (size_t i = 0; i < size; ++i)
		bytes[i] = (uint8_t)((random_func() & 0x7f80) >> 7);

	mutex_unlock(&rand_mutex);
}

void juice_random_str64(char *buf, size_t size) {
	static const char chars64[] =
	    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	size_t i = 0;
	for (i = 0; i + 1 < size; ++i) {
		uint8_t byte = 0;
		juice_random(&byte, 1);
		buf[i] = chars64[byte & 0x3F];
	}
	buf[i] = '\0';
}

uint32_t juice_rand32(void) {
	uint32_t r = 0;
	juice_random(&r, sizeof(r));
	return r;
}

uint64_t juice_rand64(void) {
	uint64_t r = 0;
	juice_random(&r, sizeof(r));
	return r;
}
