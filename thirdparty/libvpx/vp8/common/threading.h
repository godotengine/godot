/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#ifndef VP8_COMMON_THREADING_H_
#define VP8_COMMON_THREADING_H_

#include "./vpx_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#if CONFIG_OS_SUPPORT && CONFIG_MULTITHREAD

/* Thread management macros */
#if defined(_WIN32) && !HAVE_PTHREAD_H
/* Win32 */
#include <process.h>
#include <windows.h>
#define THREAD_FUNCTION unsigned int __stdcall
#define THREAD_FUNCTION_RETURN DWORD
#define THREAD_SPECIFIC_INDEX DWORD
#define pthread_t HANDLE
#define pthread_attr_t DWORD
#define pthread_detach(thread) if(thread!=NULL)CloseHandle(thread)
#define thread_sleep(nms) Sleep(nms)
#define pthread_cancel(thread) terminate_thread(thread,0)
#define ts_key_create(ts_key, destructor) {ts_key = TlsAlloc();};
#define pthread_getspecific(ts_key) TlsGetValue(ts_key)
#define pthread_setspecific(ts_key, value) TlsSetValue(ts_key, (void *)value)
#define pthread_self() GetCurrentThreadId()

#elif defined(__OS2__)
/* OS/2 */
#define INCL_DOS
#include <os2.h>

#include <stdlib.h>
#define THREAD_FUNCTION void *
#define THREAD_FUNCTION_RETURN void *
#define THREAD_SPECIFIC_INDEX PULONG
#define pthread_t TID
#define pthread_attr_t ULONG
#define pthread_detach(thread) 0
#define thread_sleep(nms) DosSleep(nms)
#define pthread_cancel(thread) DosKillThread(thread)
#define ts_key_create(ts_key, destructor) \
    DosAllocThreadLocalMemory(1, &(ts_key));
#define pthread_getspecific(ts_key) ((void *)(*(ts_key)))
#define pthread_setspecific(ts_key, value) (*(ts_key)=(ULONG)(value))
#define pthread_self() _gettid()
#else
#ifdef __APPLE__
#include <mach/mach_init.h>
#include <mach/semaphore.h>
#include <mach/task.h>
#include <time.h>
#include <unistd.h>

#else
#include <semaphore.h>
#endif

#include <pthread.h>
/* pthreads */
/* Nearly everything is already defined */
#define THREAD_FUNCTION void *
#define THREAD_FUNCTION_RETURN void *
#define THREAD_SPECIFIC_INDEX pthread_key_t
#define ts_key_create(ts_key, destructor) pthread_key_create (&(ts_key), destructor);
#endif

/* Synchronization macros: Win32 and Pthreads */
#if defined(_WIN32) && !HAVE_PTHREAD_H
#define sem_t HANDLE
#define pause(voidpara) __asm PAUSE
#define sem_init(sem, sem_attr1, sem_init_value) (int)((*sem = CreateSemaphore(NULL,0,32768,NULL))==NULL)
#define sem_wait(sem) (int)(WAIT_OBJECT_0 != WaitForSingleObject(*sem,INFINITE))
#define sem_post(sem) ReleaseSemaphore(*sem,1,NULL)
#define sem_destroy(sem) if(*sem)((int)(CloseHandle(*sem))==TRUE)
#define thread_sleep(nms) Sleep(nms)

#elif defined(__OS2__)
typedef struct
{
    HEV  event;
    HMTX wait_mutex;
    HMTX count_mutex;
    int  count;
} sem_t;

static inline int sem_init(sem_t *sem, int pshared, unsigned int value)
{
    DosCreateEventSem(NULL, &sem->event, pshared ? DC_SEM_SHARED : 0,
                      value > 0 ? TRUE : FALSE);
    DosCreateMutexSem(NULL, &sem->wait_mutex, 0, FALSE);
    DosCreateMutexSem(NULL, &sem->count_mutex, 0, FALSE);

    sem->count = value;

    return 0;
}

static inline int sem_wait(sem_t * sem)
{
    DosRequestMutexSem(sem->wait_mutex, -1);

    DosWaitEventSem(sem->event, -1);

    DosRequestMutexSem(sem->count_mutex, -1);

    sem->count--;
    if (sem->count == 0)
    {
        ULONG post_count;

        DosResetEventSem(sem->event, &post_count);
    }

    DosReleaseMutexSem(sem->count_mutex);

    DosReleaseMutexSem(sem->wait_mutex);

    return 0;
}

static inline int sem_post(sem_t * sem)
{
    DosRequestMutexSem(sem->count_mutex, -1);

    if (sem->count < 32768)
    {
        sem->count++;
        DosPostEventSem(sem->event);
    }

    DosReleaseMutexSem(sem->count_mutex);

    return 0;
}

static inline int sem_destroy(sem_t * sem)
{
    DosCloseEventSem(sem->event);
    DosCloseMutexSem(sem->wait_mutex);
    DosCloseMutexSem(sem->count_mutex);

    return 0;
}

#define thread_sleep(nms) DosSleep(nms)

#else

#ifdef __APPLE__
#define sem_t semaphore_t
#define sem_init(X,Y,Z) semaphore_create(mach_task_self(), X, SYNC_POLICY_FIFO, Z)
#define sem_wait(sem) (semaphore_wait(*sem) )
#define sem_post(sem) semaphore_signal(*sem)
#define sem_destroy(sem) semaphore_destroy(mach_task_self(),*sem)
#define thread_sleep(nms) /* { struct timespec ts;ts.tv_sec=0; ts.tv_nsec = 1000*nms;nanosleep(&ts, NULL);} */
#else
#include <unistd.h>
#include <sched.h>
#define thread_sleep(nms) sched_yield();/* {struct timespec ts;ts.tv_sec=0; ts.tv_nsec = 1000*nms;nanosleep(&ts, NULL);} */
#endif
/* Not Windows. Assume pthreads */

#endif

#if ARCH_X86 || ARCH_X86_64
#include "vpx_ports/x86.h"
#else
#define x86_pause_hint()
#endif

#include "vpx_util/vpx_thread.h"

static INLINE void mutex_lock(pthread_mutex_t *const mutex) {
    const int kMaxTryLocks = 4000;
    int locked = 0;
    int i;

    for (i = 0; i < kMaxTryLocks; ++i) {
        if (!pthread_mutex_trylock(mutex)) {
            locked = 1;
            break;
        }
    }

    if (!locked)
        pthread_mutex_lock(mutex);
}

static INLINE int protected_read(pthread_mutex_t *const mutex, const int *p) {
    int ret;
    mutex_lock(mutex);
    ret = *p;
    pthread_mutex_unlock(mutex);
    return ret;
}

static INLINE void sync_read(pthread_mutex_t *const mutex, int mb_col,
                             const int *last_row_current_mb_col,
                             const int nsync) {
    while (mb_col > (protected_read(mutex, last_row_current_mb_col) - nsync)) {
        x86_pause_hint();
        thread_sleep(0);
    }
}

static INLINE void protected_write(pthread_mutex_t *mutex, int *p, int v) {
    mutex_lock(mutex);
    *p = v;
    pthread_mutex_unlock(mutex);
}

#endif /* CONFIG_OS_SUPPORT && CONFIG_MULTITHREAD */

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP8_COMMON_THREADING_H_
