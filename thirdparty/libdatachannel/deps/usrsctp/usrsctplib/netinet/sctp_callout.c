/*-
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2001-2007, by Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2008-2012, by Randall Stewart. All rights reserved.
 * Copyright (c) 2008-2012, by Michael Tuexen. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * a) Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * b) Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the distribution.
 *
 * c) Neither the name of Cisco Systems, Inc. nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(__Userspace__)
#include <sys/types.h>
#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#include <pthread.h>
#endif
#if defined(__native_client__)
#include <sys/select.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <user_atomic.h>
#include <netinet/sctp_sysctl.h>
#include <netinet/sctp_pcb.h>
#else
#include <netinet/sctp_os.h>
#include <netinet/sctp_callout.h>
#include <netinet/sctp_pcb.h>
#endif
#include <netinet/sctputil.h>

/*
 * Callout/Timer routines for OS that doesn't have them
 */
#if defined(__APPLE__) || defined(__Userspace__)
static uint32_t ticks = 0;
#else
extern int ticks;
#endif

uint32_t sctp_get_tick_count(void) {
	uint32_t ret;

	SCTP_TIMERQ_LOCK();
	ret = ticks;
	SCTP_TIMERQ_UNLOCK();
	return ret;
}

/*
 * SCTP_TIMERQ_LOCK protects:
 * - SCTP_BASE_INFO(callqueue)
 * - sctp_os_timer_next: next timer to check
 */
static sctp_os_timer_t *sctp_os_timer_next = NULL;

void
sctp_os_timer_init(sctp_os_timer_t *c)
{
	memset(c, 0, sizeof(*c));
}

int
sctp_os_timer_start(sctp_os_timer_t *c, uint32_t to_ticks, void (*ftn) (void *),
                    void *arg)
{
	int ret = 0;

	/* paranoia */
	if ((c == NULL) || (ftn == NULL))
		return (ret);

	SCTP_TIMERQ_LOCK();
	/* check to see if we're rescheduling a timer */
	if (c->c_flags & SCTP_CALLOUT_PENDING) {
		ret = 1;
		if (c == sctp_os_timer_next) {
			sctp_os_timer_next = TAILQ_NEXT(c, tqe);
		}
		TAILQ_REMOVE(&SCTP_BASE_INFO(callqueue), c, tqe);
		/*
		 * part of the normal "stop a pending callout" process
		 * is to clear the CALLOUT_ACTIVE and CALLOUT_PENDING
		 * flags.  We don't bother since we are setting these
		 * below and we still hold the lock.
		 */
	}

	/*
	 * We could unlock/splx here and lock/spl at the TAILQ_INSERT_TAIL,
	 * but there's no point since doing this setup doesn't take much time.
	 */
	if (to_ticks == 0)
		to_ticks = 1;

	c->c_arg = arg;
	c->c_flags = (SCTP_CALLOUT_ACTIVE | SCTP_CALLOUT_PENDING);
	c->c_func = ftn;
	c->c_time = ticks + to_ticks;
	TAILQ_INSERT_TAIL(&SCTP_BASE_INFO(callqueue), c, tqe);
	SCTP_TIMERQ_UNLOCK();
	return (ret);
}

int
sctp_os_timer_stop(sctp_os_timer_t *c)
{
	SCTP_TIMERQ_LOCK();
	/*
	 * Don't attempt to delete a callout that's not on the queue.
	 */
	if ((c->c_flags & SCTP_CALLOUT_PENDING) == 0) {
		c->c_flags &= ~SCTP_CALLOUT_ACTIVE;
		SCTP_TIMERQ_UNLOCK();
		return (0);
	}
	c->c_flags &= ~(SCTP_CALLOUT_ACTIVE | SCTP_CALLOUT_PENDING);
	if (c == sctp_os_timer_next) {
		sctp_os_timer_next = TAILQ_NEXT(c, tqe);
	}
	TAILQ_REMOVE(&SCTP_BASE_INFO(callqueue), c, tqe);
	SCTP_TIMERQ_UNLOCK();
	return (1);
}

void
sctp_handle_tick(uint32_t elapsed_ticks)
{
	sctp_os_timer_t *c;
	void (*c_func)(void *);
	void *c_arg;

	SCTP_TIMERQ_LOCK();
	/* update our tick count */
	ticks += elapsed_ticks;
	c = TAILQ_FIRST(&SCTP_BASE_INFO(callqueue));
	while (c) {
		if (SCTP_UINT32_GE(ticks, c->c_time)) {
			sctp_os_timer_next = TAILQ_NEXT(c, tqe);
			TAILQ_REMOVE(&SCTP_BASE_INFO(callqueue), c, tqe);
			c_func = c->c_func;
			c_arg = c->c_arg;
			c->c_flags &= ~SCTP_CALLOUT_PENDING;
			SCTP_TIMERQ_UNLOCK();
			c_func(c_arg);
			SCTP_TIMERQ_LOCK();
			c = sctp_os_timer_next;
		} else {
			c = TAILQ_NEXT(c, tqe);
		}
	}
	sctp_os_timer_next = NULL;
	SCTP_TIMERQ_UNLOCK();
}

#if defined(__APPLE__) && !defined(__Userspace__)
void
sctp_timeout(void *arg SCTP_UNUSED)
{
	sctp_handle_tick(SCTP_BASE_VAR(sctp_main_timer_ticks));
	sctp_start_main_timer();
}
#endif

#if defined(__Userspace__)
#define TIMEOUT_INTERVAL 10

void *
user_sctp_timer_iterate(void *arg)
{
	sctp_userspace_set_threadname("SCTP timer");
	for (;;) {
#if defined(_WIN32)
		Sleep(TIMEOUT_INTERVAL);
#else
		struct timespec amount, remaining;

		remaining.tv_sec = 0;
		remaining.tv_nsec = TIMEOUT_INTERVAL * 1000 * 1000;
		do {
			amount = remaining;
		} while (nanosleep(&amount, &remaining) == -1);
#endif
		if (atomic_cmpset_int(&SCTP_BASE_VAR(timer_thread_should_exit), 1, 1)) {
			break;
		}
		sctp_handle_tick(sctp_msecs_to_ticks(TIMEOUT_INTERVAL));
	}
	return (NULL);
}

void
sctp_start_timer_thread(void)
{
	/*
	 * No need to do SCTP_TIMERQ_LOCK_INIT();
	 * here, it is being done in sctp_pcb_init()
	 */
	int rc;

	rc = sctp_userspace_thread_create(&SCTP_BASE_VAR(timer_thread), user_sctp_timer_iterate);
	if (rc) {
		SCTP_PRINTF("ERROR; return code from sctp_thread_create() is %d\n", rc);
	} else {
		SCTP_BASE_VAR(timer_thread_started) = 1;
	}
}

void
sctp_stop_timer_thread(void)
{
	atomic_cmpset_int(&SCTP_BASE_VAR(timer_thread_should_exit), 0, 1);
	if (SCTP_BASE_VAR(timer_thread_started)) {
#if defined(_WIN32)
		WaitForSingleObject(SCTP_BASE_VAR(timer_thread), INFINITE);
		CloseHandle(SCTP_BASE_VAR(timer_thread));
#else
		pthread_join(SCTP_BASE_VAR(timer_thread), NULL);
#endif
	}
}
#endif
