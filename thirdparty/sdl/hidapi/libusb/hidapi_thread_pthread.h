/*******************************************************
 HIDAPI - Multi-Platform library for
 communication with HID devices.

 Alan Ott
 Signal 11 Software

 libusb/hidapi Team

 Sam Lantinga

 Copyright 2023, All Rights Reserved.

 At the discretion of the user of this library,
 this software may be licensed under the terms of the
 GNU General Public License v3, a BSD-Style license, or the
 original HIDAPI license as outlined in the LICENSE.txt,
 LICENSE-gpl3.txt, LICENSE-bsd.txt, and LICENSE-orig.txt
 files located at the root of the source distribution.
 These files may also be found in the public source
 code repository located at:
        https://github.com/libusb/hidapi .
********************************************************/

#include <pthread.h>

#if defined(__ANDROID__) && __ANDROID_API__ < __ANDROID_API_N__

/* Barrier implementation because Android/Bionic don't have pthread_barrier.
   This implementation came from Brent Priddy and was posted on
   StackOverflow. It is used with his permission. */
typedef int pthread_barrierattr_t;
typedef struct pthread_barrier {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int trip_count;
} pthread_barrier_t;

static int pthread_barrier_init(pthread_barrier_t *barrier, const pthread_barrierattr_t *attr, unsigned int count)
{
	if(count == 0) {
		errno = EINVAL;
		return -1;
	}

	if(pthread_mutex_init(&barrier->mutex, 0) < 0) {
		return -1;
	}
	if(pthread_cond_init(&barrier->cond, 0) < 0) {
		pthread_mutex_destroy(&barrier->mutex);
		return -1;
	}
	barrier->trip_count = count;
	barrier->count = 0;

	return 0;
}

static int pthread_barrier_destroy(pthread_barrier_t *barrier)
{
	pthread_cond_destroy(&barrier->cond);
	pthread_mutex_destroy(&barrier->mutex);
	return 0;
}

static int pthread_barrier_wait(pthread_barrier_t *barrier)
{
	pthread_mutex_lock(&barrier->mutex);
	++(barrier->count);
	if(barrier->count >= barrier->trip_count) {
		barrier->count = 0;
		pthread_cond_broadcast(&barrier->cond);
		pthread_mutex_unlock(&barrier->mutex);
		return 1;
	}
	else {
		pthread_cond_wait(&barrier->cond, &(barrier->mutex));
		pthread_mutex_unlock(&barrier->mutex);
		return 0;
	}
}

#endif

#define HIDAPI_THREAD_TIMED_OUT	ETIMEDOUT

typedef struct timespec hidapi_timespec;

typedef struct
{
	pthread_t thread;
	pthread_mutex_t mutex; /* Protects input_reports */
	pthread_cond_t condition;
	pthread_barrier_t barrier; /* Ensures correct startup sequence */

} hidapi_thread_state;

static void hidapi_thread_state_init(hidapi_thread_state *state)
{
	pthread_mutex_init(&state->mutex, NULL);
	pthread_cond_init(&state->condition, NULL);
	pthread_barrier_init(&state->barrier, NULL, 2);
}

static void hidapi_thread_state_destroy(hidapi_thread_state *state)
{
	pthread_barrier_destroy(&state->barrier);
	pthread_cond_destroy(&state->condition);
	pthread_mutex_destroy(&state->mutex);
}

#define hidapi_thread_cleanup_push	pthread_cleanup_push
#define hidapi_thread_cleanup_pop	pthread_cleanup_pop

static void hidapi_thread_mutex_lock(hidapi_thread_state *state)
{
	pthread_mutex_lock(&state->mutex);
}

static void hidapi_thread_mutex_unlock(hidapi_thread_state *state)
{
	pthread_mutex_unlock(&state->mutex);
}

static void hidapi_thread_cond_wait(hidapi_thread_state *state)
{
	pthread_cond_wait(&state->condition, &state->mutex);
}

static int hidapi_thread_cond_timedwait(hidapi_thread_state *state, hidapi_timespec *ts)
{
	return pthread_cond_timedwait(&state->condition, &state->mutex, ts);
}

static void hidapi_thread_cond_signal(hidapi_thread_state *state)
{
	pthread_cond_signal(&state->condition);
}

static void hidapi_thread_cond_broadcast(hidapi_thread_state *state)
{
	pthread_cond_broadcast(&state->condition);
}

static void hidapi_thread_barrier_wait(hidapi_thread_state *state)
{
	pthread_barrier_wait(&state->barrier);
}

static void hidapi_thread_create(hidapi_thread_state *state, void *(*func)(void*), void *func_arg)
{
	pthread_create(&state->thread, NULL, func, func_arg);
}

static void hidapi_thread_join(hidapi_thread_state *state)
{
	pthread_join(state->thread, NULL);
}

static void hidapi_thread_gettime(hidapi_timespec *ts)
{
	clock_gettime(CLOCK_REALTIME, ts);
}

static void hidapi_thread_addtime(hidapi_timespec *ts, int milliseconds)
{
    ts->tv_sec += milliseconds / 1000;
    ts->tv_nsec += (milliseconds % 1000) * 1000000;
    if (ts->tv_nsec >= 1000000000L) {
        ts->tv_sec++;
        ts->tv_nsec -= 1000000000L;
    }
}
