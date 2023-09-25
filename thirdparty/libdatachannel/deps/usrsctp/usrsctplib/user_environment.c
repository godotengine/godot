/*-
 * Copyright (c) 2009-2010 Brad Penoff
 * Copyright (c) 2009-2010 Humaira Kamal
 * Copyright (c) 2011-2012 Irene Ruengeler
 * Copyright (c) 2011-2012 Michael Tuexen
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/* __Userspace__ */

#if defined(_WIN32)
#if !defined(_CRT_RAND_S) && !defined(FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION)
#define _CRT_RAND_S
#endif
#else
#include <stdint.h>
#include <netinet/sctp_os_userspace.h>
#endif
#ifdef INVARIANTS
#include <netinet/sctp_pcb.h>
#endif
#include <user_environment.h>
#include <sys/types.h>
/* #include <sys/param.h> defines MIN */
#if !defined(MIN)
#define MIN(arg1,arg2) ((arg1) < (arg2) ? (arg1) : (arg2))
#endif

#define uHZ 1000

/* See user_include/user_environment.h for comments about these variables */
int maxsockets = 25600;
int hz = uHZ;
int ip_defttl = 64;
int ipport_firstauto = 49152, ipport_lastauto = 65535;
int nmbclusters = 65536;

/* Source ip_output.c. extern'd in ip_var.h */
u_short ip_id = 0; /*__Userspace__ TODO Should it be initialized to zero? */

/* used in user_include/user_atomic.h in order to make the operations
 * defined there truly atomic
 */
userland_mutex_t atomic_mtx;

/* If the entropy device is not loaded, make a token effort to
 * provide _some_ kind of randomness. This should only be used
 * inside other RNG's, like arc4random(9).
 */
#if defined(FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION)
#include <string.h>

void
init_random(void)
{
	return;
}

void
read_random(void *buf, size_t size)
{
	memset(buf, 'A', size);
	return;
}

void
finish_random(void)
{
	return;
}
/* This define can be used to optionally use OpenSSL's random number utility,
 * which is capable of bypassing the chromium sandbox which normally would
 * prevent opening files, including /dev/urandom.
 */
#elif defined(SCTP_USE_OPENSSL_RAND)
#include <openssl/rand.h>

/* Requiring BoringSSL because it guarantees that RAND_bytes will succeed. */
#ifndef OPENSSL_IS_BORINGSSL
#error Only BoringSSL is supported with SCTP_USE_OPENSSL_RAND.
#endif

void
init_random(void)
{
	return;
}

void
read_random(void *buf, size_t size)
{
	RAND_bytes((uint8_t *)buf, size);
	return;
}

void
finish_random(void)
{
	return;
}
#elif defined(__FreeBSD__) || defined(__DragonFly__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__APPLE__) || defined(__Bitrig__)
#include <stdlib.h>

void
init_random(void)
{
	return;
}

void
read_random(void *buf, size_t size)
{
	arc4random_buf(buf, size);
	return;
}

void
finish_random(void)
{
	return;
}
#elif defined(_WIN32)
#include <stdlib.h>

void
init_random(void)
{
	return;
}

void
read_random(void *buf, size_t size)
{
	unsigned int randval;
	size_t position, remaining;

	position = 0;
	while (position < size) {
		if (rand_s(&randval) == 0) {
			remaining = MIN(size - position, sizeof(unsigned int));
			memcpy((char *)buf + position, &randval, remaining);
			position += sizeof(unsigned int);
		}
	}
	return;
}

void
finish_random(void)
{
	return;
}
#elif (defined(__ANDROID__) && (__ANDROID_API__ < 28)) || defined(__QNX__) || defined(__EMSCRIPTEN__)
#include <fcntl.h>

static int fd = -1;

void
init_random(void)
{
	fd = open("/dev/urandom", O_RDONLY);
	return;
}

void
read_random(void *buf, size_t size)
{
	size_t position;
	ssize_t n;

	position = 0;
	while (position < size) {
		n = read(fd, (char *)buf + position, size - position);
		if (n > 0) {
			position += n;
		}
	}
	return;
}

void
finish_random(void)
{
	close(fd);
	return;
}
#elif defined(__ANDROID__) && (__ANDROID_API__ >= 28)
#include <sys/random.h>

void
init_random(void)
{
	return;
}

void
read_random(void *buf, size_t size)
{
	size_t position;
	ssize_t n;

	position = 0;
	while (position < size) {
		n = getrandom((char *)buf + position, size - position, 0);
		if (n > 0) {
			position += n;
		}
	}
	return;
}

void
finish_random(void)
{
	return;
}
#elif defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
void __msan_unpoison(void *, size_t);
#endif
#endif

#ifdef __NR_getrandom
#if !defined(GRND_NONBLOCK)
#define GRND_NONBLOCK 1
#endif
static int getrandom_available = 0;
#endif
static int fd = -1;

void
init_random(void)
{
#ifdef __NR_getrandom
	char dummy;
	ssize_t n = syscall(__NR_getrandom, &dummy, sizeof(dummy), GRND_NONBLOCK);
	if (n > 0 || errno == EINTR || errno == EAGAIN) {
		/* Either getrandom succeeded, was interrupted or is waiting for entropy;
		 * all of which mean the syscall is available.
		 */
		getrandom_available = 1;
	} else {
#ifdef INVARIANTS
		if (errno != ENOSYS) {
			panic("getrandom syscall returned unexpected error: %d", errno);
		}
#endif
		/* If the syscall isn't available, fall back to /dev/urandom. */
#endif
		fd = open("/dev/urandom", O_RDONLY);
#ifdef __NR_getrandom
	}
#endif
	return;
}

void
read_random(void *buf, size_t size)
{
	size_t position;
	ssize_t n;

	position = 0;
	while (position < size) {
#ifdef __NR_getrandom
		if (getrandom_available) {
			/* Using syscall directly because getrandom isn't present in glibc < 2.25.
			 */
			n = syscall(__NR_getrandom, (char *)buf + position, size - position, 0);
			if (n > 0) {
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
				/* Need to do this because MSan doesn't realize that syscall has
				 * initialized the output buffer.
				 */
				__msan_unpoison(buf + position, n);
#endif
#endif
				position += n;
			} else if (errno != EINTR && errno != EAGAIN) {
#ifdef INVARIANTS
				panic("getrandom syscall returned unexpected error: %d", errno);
#endif
			}
		} else
#endif /* __NR_getrandom */
		{
			n = read(fd, (char *)buf + position, size - position);
			if (n > 0) {
				position += n;
			}
		}
	}
	return;
}

void
finish_random(void)
{
	if (fd != -1) {
		close(fd);
	}
	return;
}
#elif defined(__Fuchsia__)
#include <zircon/syscalls.h>

void
init_random(void)
{
	return;
}

void
read_random(void *buf, size_t size)
{
	zx_cprng_draw(buf, size);
	return;
}

void
finish_random(void)
{
	return;
}
#elif defined(__native_client__)
#include <nacl/nacl_random.h>

void
init_random(void)
{
	return;
}

void
read_random(void *buf, size_t size)
{
	size_t position;
	size_t n;

	position = 0;
	while (position < size) {
		if (nacl_secure_random((char *)buf + position, size - position, &n) == 0)
			position += n;
		}
	}
	return;
}

void
finish_random(void)
{
	return;
}
#else
#error "Unknown platform. Please provide platform specific RNG."
#endif
