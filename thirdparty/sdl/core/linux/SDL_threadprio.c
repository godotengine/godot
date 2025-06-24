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

#ifdef SDL_PLATFORM_LINUX

#ifndef SDL_THREADS_DISABLED
#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

// RLIMIT_RTTIME requires kernel >= 2.6.25 and is in glibc >= 2.14
#ifndef RLIMIT_RTTIME
#define RLIMIT_RTTIME 15
#endif
// SCHED_RESET_ON_FORK is in kernel >= 2.6.32.
#ifndef SCHED_RESET_ON_FORK
#define SCHED_RESET_ON_FORK 0x40000000
#endif

#include "SDL_dbus.h"

#ifdef SDL_USE_LIBDBUS

// d-bus queries to org.freedesktop.RealtimeKit1.
#define RTKIT_DBUS_NODE      "org.freedesktop.RealtimeKit1"
#define RTKIT_DBUS_PATH      "/org/freedesktop/RealtimeKit1"
#define RTKIT_DBUS_INTERFACE "org.freedesktop.RealtimeKit1"

// d-bus queries to the XDG portal interface to RealtimeKit1
#define XDG_PORTAL_DBUS_NODE      "org.freedesktop.portal.Desktop"
#define XDG_PORTAL_DBUS_PATH      "/org/freedesktop/portal/desktop"
#define XDG_PORTAL_DBUS_INTERFACE "org.freedesktop.portal.Realtime"

static bool rtkit_use_session_conn;
static const char *rtkit_dbus_node;
static const char *rtkit_dbus_path;
static const char *rtkit_dbus_interface;

static pthread_once_t rtkit_initialize_once = PTHREAD_ONCE_INIT;
static Sint32 rtkit_min_nice_level = -20;
static Sint32 rtkit_max_realtime_priority = 99;
static Sint64 rtkit_max_rttime_usec = 200000;

/*
 * Checking that the RTTimeUSecMax property exists and is an int64 confirms that:
 *  - The desktop portal exists and supports the realtime interface.
 *  - The realtime interface is new enough to have the required bug fixes applied.
 */
static bool realtime_portal_supported(DBusConnection *conn)
{
    Sint64 res;
    return SDL_DBus_QueryPropertyOnConnection(conn, XDG_PORTAL_DBUS_NODE, XDG_PORTAL_DBUS_PATH, XDG_PORTAL_DBUS_INTERFACE,
                                              "RTTimeUSecMax", DBUS_TYPE_INT64, &res);
}

static void set_rtkit_interface(void)
{
    SDL_DBusContext *dbus = SDL_DBus_GetContext();

    // xdg-desktop-portal works in all instances, so check for it first.
    if (dbus && realtime_portal_supported(dbus->session_conn)) {
        rtkit_use_session_conn = true;
        rtkit_dbus_node = XDG_PORTAL_DBUS_NODE;
        rtkit_dbus_path = XDG_PORTAL_DBUS_PATH;
        rtkit_dbus_interface = XDG_PORTAL_DBUS_INTERFACE;
    } else { // Fall back to the standard rtkit interface in all other cases.
        rtkit_use_session_conn = false;
        rtkit_dbus_node = RTKIT_DBUS_NODE;
        rtkit_dbus_path = RTKIT_DBUS_PATH;
        rtkit_dbus_interface = RTKIT_DBUS_INTERFACE;
    }
}

static DBusConnection *get_rtkit_dbus_connection(void)
{
    SDL_DBusContext *dbus = SDL_DBus_GetContext();

    if (dbus) {
        return rtkit_use_session_conn ? dbus->session_conn : dbus->system_conn;
    }

    return NULL;
}

static void rtkit_initialize(void)
{
    DBusConnection *dbus_conn;

    set_rtkit_interface();
    dbus_conn = get_rtkit_dbus_connection();

    // Try getting minimum nice level: this is often greater than PRIO_MIN (-20).
    if (!dbus_conn || !SDL_DBus_QueryPropertyOnConnection(dbus_conn, rtkit_dbus_node, rtkit_dbus_path, rtkit_dbus_interface, "MinNiceLevel",
                                                                 DBUS_TYPE_INT32, &rtkit_min_nice_level)) {
        rtkit_min_nice_level = -20;
    }

    // Try getting maximum realtime priority: this can be less than the POSIX default (99).
    if (!dbus_conn || !SDL_DBus_QueryPropertyOnConnection(dbus_conn, rtkit_dbus_node, rtkit_dbus_path, rtkit_dbus_interface, "MaxRealtimePriority",
                                                                 DBUS_TYPE_INT32, &rtkit_max_realtime_priority)) {
        rtkit_max_realtime_priority = 99;
    }

    // Try getting maximum rttime allowed by rtkit: exceeding this value will result in SIGKILL
    if (!dbus_conn || !SDL_DBus_QueryPropertyOnConnection(dbus_conn, rtkit_dbus_node, rtkit_dbus_path, rtkit_dbus_interface, "RTTimeUSecMax",
                                                                 DBUS_TYPE_INT64, &rtkit_max_rttime_usec)) {
        rtkit_max_rttime_usec = 200000;
    }
}

static bool rtkit_initialize_realtime_thread(void)
{
    // Following is an excerpt from rtkit README that outlines the requirements
    // a thread must meet before making rtkit requests:
    //
    //   * Only clients with RLIMIT_RTTIME set will get RT scheduling
    //
    //   * RT scheduling will only be handed out to processes with
    //     SCHED_RESET_ON_FORK set to guarantee that the scheduling
    //     settings cannot 'leak' to child processes, thus making sure
    //     that 'RT fork bombs' cannot be used to bypass RLIMIT_RTTIME
    //     and take the system down.
    //
    //   * Limits are enforced on all user controllable resources, only
    //     a maximum number of users, processes, threads can request RT
    //     scheduling at the same time.
    //
    //   * Only a limited number of threads may be made RT in a
    //     specific time frame.
    //
    //   * Client authorization is verified with PolicyKit

    int err;
    struct rlimit rlimit;
    int nLimit = RLIMIT_RTTIME;
    pid_t nPid = 0; // self
    int nSchedPolicy = sched_getscheduler(nPid) | SCHED_RESET_ON_FORK;
    struct sched_param schedParam;

    SDL_zero(schedParam);

    // Requirement #1: Set RLIMIT_RTTIME
    err = getrlimit(nLimit, &rlimit);
    if (err) {
        return false;
    }

    // Current rtkit allows a max of 200ms right now
    rlimit.rlim_max = rtkit_max_rttime_usec;
    rlimit.rlim_cur = rlimit.rlim_max / 2;
    err = setrlimit(nLimit, &rlimit);
    if (err) {
        return false;
    }

    // Requirement #2: Add SCHED_RESET_ON_FORK to the scheduler policy
    err = sched_getparam(nPid, &schedParam);
    if (err) {
        return false;
    }

    err = sched_setscheduler(nPid, nSchedPolicy, &schedParam);
    if (err) {
        return false;
    }

    return true;
}

static bool rtkit_setpriority_nice(pid_t thread, int nice_level)
{
    DBusConnection *dbus_conn;
    Uint64 pid = (Uint64)getpid();
    Uint64 tid = (Uint64)thread;
    Sint32 nice = (Sint32)nice_level;

    pthread_once(&rtkit_initialize_once, rtkit_initialize);
    dbus_conn = get_rtkit_dbus_connection();

    if (nice < rtkit_min_nice_level) {
        nice = rtkit_min_nice_level;
    }

    if (!dbus_conn || !SDL_DBus_CallMethodOnConnection(dbus_conn,
                                                              rtkit_dbus_node, rtkit_dbus_path, rtkit_dbus_interface, "MakeThreadHighPriorityWithPID",
                                                              DBUS_TYPE_UINT64, &pid, DBUS_TYPE_UINT64, &tid, DBUS_TYPE_INT32, &nice, DBUS_TYPE_INVALID,
                                                              DBUS_TYPE_INVALID)) {
        return false;
    }
    return true;
}

static bool rtkit_setpriority_realtime(pid_t thread, int rt_priority)
{
    DBusConnection *dbus_conn;
    Uint64 pid = (Uint64)getpid();
    Uint64 tid = (Uint64)thread;
    Uint32 priority = (Uint32)rt_priority;

    pthread_once(&rtkit_initialize_once, rtkit_initialize);
    dbus_conn = get_rtkit_dbus_connection();

    if (priority > rtkit_max_realtime_priority) {
        priority = rtkit_max_realtime_priority;
    }

    // We always perform the thread state changes necessary for rtkit.
    // This wastes some system calls if the state is already set but
    // typically code sets a thread priority and leaves it so it's
    // not expected that this wasted effort will be an issue.
    // We also do not quit if this fails, we let the rtkit request
    // go through to determine whether it really needs to fail or not.
    rtkit_initialize_realtime_thread();

    if (!dbus_conn || !SDL_DBus_CallMethodOnConnection(dbus_conn,
                                                              rtkit_dbus_node, rtkit_dbus_path, rtkit_dbus_interface, "MakeThreadRealtimeWithPID",
                                                              DBUS_TYPE_UINT64, &pid, DBUS_TYPE_UINT64, &tid, DBUS_TYPE_UINT32, &priority, DBUS_TYPE_INVALID,
                                                              DBUS_TYPE_INVALID)) {
        return false;
    }
    return true;
}
#else

#define rtkit_max_realtime_priority 99

#endif // dbus
#endif // threads

// this is a public symbol, so it has to exist even if threads are disabled.
bool SDL_SetLinuxThreadPriority(Sint64 threadID, int priority)
{
#ifdef SDL_THREADS_DISABLED
    return SDL_Unsupported();
#else
    if (setpriority(PRIO_PROCESS, (id_t)threadID, priority) == 0) {
        return true;
    }

#ifdef SDL_USE_LIBDBUS
    /* Note that this fails you most likely:
         * Have your process's scheduler incorrectly configured.
           See the requirements at:
           http://git.0pointer.net/rtkit.git/tree/README#n16
         * Encountered dbus/polkit security restrictions. Note
           that the RealtimeKit1 dbus endpoint is inaccessible
           over ssh connections for most common distro configs.
           You might want to check your local config for details:
           /usr/share/polkit-1/actions/org.freedesktop.RealtimeKit1.policy

       README and sample code at: http://git.0pointer.net/rtkit.git
    */
    if (rtkit_setpriority_nice((pid_t)threadID, priority)) {
        return true;
    }
#endif

    return SDL_SetError("setpriority() failed");
#endif
}

// this is a public symbol, so it has to exist even if threads are disabled.
bool SDL_SetLinuxThreadPriorityAndPolicy(Sint64 threadID, int sdlPriority, int schedPolicy)
{
#ifdef SDL_THREADS_DISABLED
    return SDL_Unsupported();
#else
    int osPriority;

    if (schedPolicy == SCHED_RR || schedPolicy == SCHED_FIFO) {
        if (sdlPriority == SDL_THREAD_PRIORITY_LOW) {
            osPriority = 1;
        } else if (sdlPriority == SDL_THREAD_PRIORITY_HIGH) {
            osPriority = rtkit_max_realtime_priority * 3 / 4;
        } else if (sdlPriority == SDL_THREAD_PRIORITY_TIME_CRITICAL) {
            osPriority = rtkit_max_realtime_priority;
        } else {
            osPriority = rtkit_max_realtime_priority / 2;
        }
    } else {
        if (sdlPriority == SDL_THREAD_PRIORITY_LOW) {
            osPriority = 19;
        } else if (sdlPriority == SDL_THREAD_PRIORITY_HIGH) {
            osPriority = -10;
        } else if (sdlPriority == SDL_THREAD_PRIORITY_TIME_CRITICAL) {
            osPriority = -20;
        } else {
            osPriority = 0;
        }

        if (setpriority(PRIO_PROCESS, (id_t)threadID, osPriority) == 0) {
            return true;
        }
    }

#ifdef SDL_USE_LIBDBUS
    /* Note that this fails you most likely:
     * Have your process's scheduler incorrectly configured.
       See the requirements at:
       http://git.0pointer.net/rtkit.git/tree/README#n16
     * Encountered dbus/polkit security restrictions. Note
       that the RealtimeKit1 dbus endpoint is inaccessible
       over ssh connections for most common distro configs.
       You might want to check your local config for details:
       /usr/share/polkit-1/actions/org.freedesktop.RealtimeKit1.policy

       README and sample code at: http://git.0pointer.net/rtkit.git
    */
    if (schedPolicy == SCHED_RR || schedPolicy == SCHED_FIFO) {
        if (rtkit_setpriority_realtime((pid_t)threadID, osPriority)) {
            return true;
        }
    } else {
        if (rtkit_setpriority_nice((pid_t)threadID, osPriority)) {
            return true;
        }
    }
#endif

    return SDL_SetError("setpriority() failed");
#endif
}

#endif // SDL_PLATFORM_LINUX
