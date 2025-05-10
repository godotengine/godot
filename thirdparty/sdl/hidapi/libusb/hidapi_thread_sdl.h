/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

/* Barrier implementation because Android/Bionic don't have pthread_barrier.
   This implementation came from Brent Priddy and was posted on
   StackOverflow. It is used with his permission. */

typedef struct _SDL_ThreadBarrier
{
    SDL_Mutex *mutex;
    SDL_Condition *cond;
    Uint32 count;
    Uint32 trip_count;
} SDL_ThreadBarrier;

static int SDL_CreateThreadBarrier(SDL_ThreadBarrier *barrier, Uint32 count)
{
    SDL_assert(barrier != NULL);
    SDL_assert(count != 0);

    barrier->mutex = SDL_CreateMutex();
    if (barrier->mutex == NULL) {
        return -1; /* Error set by CreateMutex */
    }
    barrier->cond = SDL_CreateCondition();
    if (barrier->cond == NULL) {
        return -1; /* Error set by CreateCond */
    }

    barrier->trip_count = count;
    barrier->count = 0;

    return 0;
}

static void SDL_DestroyThreadBarrier(SDL_ThreadBarrier *barrier)
{
    SDL_DestroyCondition(barrier->cond);
    SDL_DestroyMutex(barrier->mutex);
}

static int SDL_WaitThreadBarrier(SDL_ThreadBarrier *barrier)
{
    SDL_LockMutex(barrier->mutex);
    barrier->count += 1;
    if (barrier->count >= barrier->trip_count) {
        barrier->count = 0;
        SDL_BroadcastCondition(barrier->cond);
        SDL_UnlockMutex(barrier->mutex);
        return 1;
    }
    SDL_WaitCondition(barrier->cond, barrier->mutex);
    SDL_UnlockMutex(barrier->mutex);
    return 0;
}

#include "../../thread/SDL_systhread.h"

#define HIDAPI_THREAD_TIMED_OUT 1

typedef Uint64 hidapi_timespec;

typedef struct
{
    SDL_Thread *thread;
    SDL_Mutex *mutex; /* Protects input_reports */
    SDL_Condition *condition;
    SDL_ThreadBarrier barrier; /* Ensures correct startup sequence */

} hidapi_thread_state;

static void hidapi_thread_state_init(hidapi_thread_state *state)
{
    state->mutex = SDL_CreateMutex();
    state->condition = SDL_CreateCondition();
    SDL_CreateThreadBarrier(&state->barrier, 2);
}

static void hidapi_thread_state_destroy(hidapi_thread_state *state)
{
    SDL_DestroyThreadBarrier(&state->barrier);
    SDL_DestroyCondition(state->condition);
    SDL_DestroyMutex(state->mutex);
}

static void hidapi_thread_cleanup_push(void (*routine)(void *), void *arg)
{
    /* There isn't an equivalent in SDL, and it's only useful for threads calling hid_read_timeout() */
}

static void hidapi_thread_cleanup_pop(int execute)
{
}

static void hidapi_thread_mutex_lock(hidapi_thread_state *state)
{
    SDL_LockMutex(state->mutex);
}

static void hidapi_thread_mutex_unlock(hidapi_thread_state *state)
{
    SDL_UnlockMutex(state->mutex);
}

static void hidapi_thread_cond_wait(hidapi_thread_state *state)
{
    SDL_WaitCondition(state->condition, state->mutex);
}

static int hidapi_thread_cond_timedwait(hidapi_thread_state *state, hidapi_timespec *ts)
{
    Sint64 timeout_ns;
    Sint32 timeout_ms;

    timeout_ns = (Sint64)(*ts - SDL_GetTicksNS());
    if (timeout_ns <= 0) {
        timeout_ms = 0;
    } else {
        timeout_ms = (Sint32)SDL_NS_TO_MS(timeout_ns);
    }
    if (SDL_WaitConditionTimeout(state->condition, state->mutex, timeout_ms)) {
        return 0;
    } else {
        return HIDAPI_THREAD_TIMED_OUT;
    }
}

static void hidapi_thread_cond_signal(hidapi_thread_state *state)
{
    SDL_SignalCondition(state->condition);
}

static void hidapi_thread_cond_broadcast(hidapi_thread_state *state)
{
    SDL_BroadcastCondition(state->condition);
}

static void hidapi_thread_barrier_wait(hidapi_thread_state *state)
{
    SDL_WaitThreadBarrier(&state->barrier);
}

typedef struct
{
    void *(*func)(void*);
    void *func_arg;

} RunInputThreadParam;

static int RunInputThread(void *param)
{
    RunInputThreadParam *data = (RunInputThreadParam *)param;
    void *(*func)(void*) = data->func;
    void *func_arg = data->func_arg;
    SDL_free(data);
    func(func_arg);
    return 0;
}

static void hidapi_thread_create(hidapi_thread_state *state, void *(*func)(void*), void *func_arg)
{
    RunInputThreadParam *param = (RunInputThreadParam *)malloc(sizeof(*param));
    /* Note that the hidapi code didn't check for thread creation failure.
     * We'll crash if malloc() fails
     */
    param->func = func;
    param->func_arg = func_arg;
    state->thread = SDL_CreateThread(RunInputThread, "libusb", param);
}

static void hidapi_thread_join(hidapi_thread_state *state)
{
    SDL_WaitThread(state->thread, NULL);
}

static void hidapi_thread_gettime(hidapi_timespec *ts)
{
    *ts = SDL_GetTicksNS();
}

static void hidapi_thread_addtime(hidapi_timespec *ts, int milliseconds)
{
    *ts += SDL_MS_TO_NS(milliseconds);
}
