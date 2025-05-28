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
#include "SDL_internal.h"

#include <pthread.h>

#ifdef HAVE_PTHREAD_NP_H
#include <pthread_np.h>
#endif

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif
#include <errno.h>

#ifdef SDL_PLATFORM_LINUX
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "../../core/linux/SDL_dbus.h"
#endif // SDL_PLATFORM_LINUX

#if (defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_ANDROID) || defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_IOS)) && defined(HAVE_DLOPEN)
#include <dlfcn.h>
#ifndef RTLD_DEFAULT
#define RTLD_DEFAULT NULL
#endif
#endif

#include "../SDL_thread_c.h"
#include "../SDL_systhread.h"
#ifdef SDL_PLATFORM_ANDROID
#include "../../core/android/SDL_android.h"
#endif

#ifdef SDL_PLATFORM_HAIKU
#include <kernel/OS.h>
#endif

#ifdef HAVE_SIGNAL_H
// List of signals to mask in the subthreads
static const int sig_list[] = {
    SIGHUP, SIGINT, SIGQUIT, SIGPIPE, SIGALRM, SIGTERM, SIGCHLD, SIGWINCH,
    SIGVTALRM, SIGPROF, 0
};
#endif

static void *RunThread(void *data)
{
#ifdef SDL_PLATFORM_ANDROID
    Android_JNI_SetupThread();
#endif
    SDL_RunThread((SDL_Thread *)data);
    return NULL;
}

#if (defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_IOS)) && defined(HAVE_DLOPEN)
static bool checked_setname = false;
static int (*ppthread_setname_np)(const char *) = NULL;
#elif (defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_ANDROID)) && defined(HAVE_DLOPEN)
static bool checked_setname = false;
static int (*ppthread_setname_np)(pthread_t, const char *) = NULL;
#endif
bool SDL_SYS_CreateThread(SDL_Thread *thread,
                          SDL_FunctionPointer pfnBeginThread,
                          SDL_FunctionPointer pfnEndThread)
{
    pthread_attr_t type;

// do this here before any threads exist, so there's no race condition.
#if (defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_IOS) || defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_ANDROID)) && defined(HAVE_DLOPEN)
    if (!checked_setname) {
        void *fn = dlsym(RTLD_DEFAULT, "pthread_setname_np");
#if defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_IOS)
        ppthread_setname_np = (int (*)(const char *))fn;
#elif defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_ANDROID)
        ppthread_setname_np = (int (*)(pthread_t, const char *))fn;
#endif
        checked_setname = true;
    }
    #endif

    // Set the thread attributes
    if (pthread_attr_init(&type) != 0) {
        return SDL_SetError("Couldn't initialize pthread attributes");
    }
    pthread_attr_setdetachstate(&type, PTHREAD_CREATE_JOINABLE);

    // Set caller-requested stack size. Otherwise: use the system default.
    if (thread->stacksize) {
        pthread_attr_setstacksize(&type, thread->stacksize);
    }

    // Create the thread and go!
    if (pthread_create(&thread->handle, &type, RunThread, thread) != 0) {
        return SDL_SetError("Not enough resources to create thread");
    }

    return true;
}

void SDL_SYS_SetupThread(const char *name)
{
#ifdef HAVE_SIGNAL_H
    int i;
    sigset_t mask;
#endif

    if (name) {
#if (defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_IOS) || defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_ANDROID)) && defined(HAVE_DLOPEN)
        SDL_assert(checked_setname);
        if (ppthread_setname_np) {
#if defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_IOS)
            ppthread_setname_np(name);
#elif defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_ANDROID)
            if (ppthread_setname_np(pthread_self(), name) == ERANGE) {
                char namebuf[16]; // Limited to 16 char
                SDL_strlcpy(namebuf, name, sizeof(namebuf));
                ppthread_setname_np(pthread_self(), namebuf);
            }
#endif
        }
#elif defined(HAVE_PTHREAD_SETNAME_NP)
#ifdef SDL_PLATFORM_NETBSD
        pthread_setname_np(pthread_self(), "%s", name);
#else
        if (pthread_setname_np(pthread_self(), name) == ERANGE) {
            char namebuf[16]; // Limited to 16 char
            SDL_strlcpy(namebuf, name, sizeof(namebuf));
            pthread_setname_np(pthread_self(), namebuf);
        }
#endif
#elif defined(HAVE_PTHREAD_SET_NAME_NP)
        pthread_set_name_np(pthread_self(), name);
#elif defined(SDL_PLATFORM_HAIKU)
        // The docs say the thread name can't be longer than B_OS_NAME_LENGTH.
        char namebuf[B_OS_NAME_LENGTH];
        SDL_strlcpy(namebuf, name, sizeof(namebuf));
        rename_thread(find_thread(NULL), namebuf);
#endif
    }

#ifdef HAVE_SIGNAL_H
    // Mask asynchronous signals for this thread
    sigemptyset(&mask);
    for (i = 0; sig_list[i]; ++i) {
        sigaddset(&mask, sig_list[i]);
    }
    pthread_sigmask(SIG_BLOCK, &mask, 0);
#endif

#ifdef PTHREAD_CANCEL_ASYNCHRONOUS
    // Allow ourselves to be asynchronously cancelled
    {
        int oldstate;
        pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &oldstate);
    }
#endif
}

SDL_ThreadID SDL_GetCurrentThreadID(void)
{
    return (SDL_ThreadID)pthread_self();
}

bool SDL_SYS_SetThreadPriority(SDL_ThreadPriority priority)
{
#ifdef SDL_PLATFORM_RISCOS
    // FIXME: Setting thread priority does not seem to be supported
    return true;
#else
    struct sched_param sched;
    int policy;
    int pri_policy;
    pthread_t thread = pthread_self();
    const char *policyhint = SDL_GetHint(SDL_HINT_THREAD_PRIORITY_POLICY);
    const bool timecritical_realtime_hint = SDL_GetHintBoolean(SDL_HINT_THREAD_FORCE_REALTIME_TIME_CRITICAL, false);

    if (pthread_getschedparam(thread, &policy, &sched) != 0) {
        return SDL_SetError("pthread_getschedparam() failed");
    }

    /* Higher priority levels may require changing the pthread scheduler policy
     * for the thread.  SDL will make such changes by default but there is
     * also a hint allowing that behavior to be overridden. */
    switch (priority) {
    case SDL_THREAD_PRIORITY_LOW:
    case SDL_THREAD_PRIORITY_NORMAL:
        pri_policy = SCHED_OTHER;
        break;
    case SDL_THREAD_PRIORITY_HIGH:
    case SDL_THREAD_PRIORITY_TIME_CRITICAL:
#if defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_IOS) || defined(SDL_PLATFORM_TVOS)
        // Apple requires SCHED_RR for high priority threads
        pri_policy = SCHED_RR;
        break;
#else
        pri_policy = SCHED_OTHER;
        break;
#endif
    default:
        pri_policy = policy;
        break;
    }

    if (timecritical_realtime_hint && priority == SDL_THREAD_PRIORITY_TIME_CRITICAL) {
        pri_policy = SCHED_RR;
    }

    if (policyhint) {
        if (SDL_strcmp(policyhint, "current") == 0) {
            // Leave current thread scheduler policy unchanged
        } else if (SDL_strcmp(policyhint, "other") == 0) {
            policy = SCHED_OTHER;
        } else if (SDL_strcmp(policyhint, "rr") == 0) {
            policy = SCHED_RR;
        } else if (SDL_strcmp(policyhint, "fifo") == 0) {
            policy = SCHED_FIFO;
        } else {
            policy = pri_policy;
        }
    } else {
        policy = pri_policy;
    }

#ifdef SDL_PLATFORM_LINUX
    {
        pid_t linuxTid = syscall(SYS_gettid);
        return SDL_SetLinuxThreadPriorityAndPolicy(linuxTid, priority, policy);
    }
#else
    if (priority == SDL_THREAD_PRIORITY_LOW) {
        sched.sched_priority = sched_get_priority_min(policy);
    } else if (priority == SDL_THREAD_PRIORITY_TIME_CRITICAL) {
        sched.sched_priority = sched_get_priority_max(policy);
    } else {
        int min_priority = sched_get_priority_min(policy);
        int max_priority = sched_get_priority_max(policy);

#if defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_IOS) || defined(SDL_PLATFORM_TVOS)
        if (min_priority == 15 && max_priority == 47) {
            // Apple has a specific set of thread priorities
            if (priority == SDL_THREAD_PRIORITY_HIGH) {
                sched.sched_priority = 45;
            } else {
                sched.sched_priority = 37;
            }
        } else
#endif // SDL_PLATFORM_MACOS || SDL_PLATFORM_IOS || SDL_PLATFORM_TVOS
        {
            sched.sched_priority = (min_priority + (max_priority - min_priority) / 2);
            if (priority == SDL_THREAD_PRIORITY_HIGH) {
                sched.sched_priority += ((max_priority - min_priority) / 4);
            }
        }
    }
    if (pthread_setschedparam(thread, policy, &sched) != 0) {
        return SDL_SetError("pthread_setschedparam() failed");
    }
    return true;
#endif // linux
#endif // #if SDL_PLATFORM_RISCOS
}

void SDL_SYS_WaitThread(SDL_Thread *thread)
{
    pthread_join(thread->handle, 0);
}

void SDL_SYS_DetachThread(SDL_Thread *thread)
{
    pthread_detach(thread->handle);
}
