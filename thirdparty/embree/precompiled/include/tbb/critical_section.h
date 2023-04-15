/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "internal/_deprecated_header_message_guard.h"

#if !defined(__TBB_show_deprecation_message_critical_section_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_critical_section_H
#pragma message("TBB Warning: tbb/critical_section.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef _TBB_CRITICAL_SECTION_H_
#define _TBB_CRITICAL_SECTION_H_

#define __TBB_critical_section_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#if _WIN32||_WIN64
#include "machine/windows_api.h"
#else
#include <pthread.h>
#include <errno.h>
#endif  // _WIN32||WIN64

#include "tbb_stddef.h"
#include "tbb_thread.h"
#include "tbb_exception.h"

#include "tbb_profiling.h"

namespace tbb {

    namespace internal {
class critical_section_v4 : internal::no_copy {
#if _WIN32||_WIN64
    CRITICAL_SECTION my_impl;
#else
    pthread_mutex_t my_impl;
#endif
    tbb_thread::id my_tid;
public:

    void __TBB_EXPORTED_METHOD internal_construct();

    critical_section_v4() {
#if _WIN32||_WIN64
        InitializeCriticalSectionEx( &my_impl, 4000, 0 );
#else
        pthread_mutex_init(&my_impl, NULL);
#endif
        internal_construct();
    }

    ~critical_section_v4() {
        __TBB_ASSERT(my_tid == tbb_thread::id(), "Destroying a still-held critical section");
#if _WIN32||_WIN64
        DeleteCriticalSection(&my_impl);
#else
        pthread_mutex_destroy(&my_impl);
#endif
    }

    class scoped_lock : internal::no_copy {
    private:
        critical_section_v4 &my_crit;
    public:
        scoped_lock( critical_section_v4& lock_me) :my_crit(lock_me) {
            my_crit.lock();
        }

        ~scoped_lock() {
            my_crit.unlock();
        }
    };

    void lock() {
        tbb_thread::id local_tid = this_tbb_thread::get_id();
        if(local_tid == my_tid) throw_exception( eid_improper_lock );
#if _WIN32||_WIN64
        EnterCriticalSection( &my_impl );
#else
        int rval = pthread_mutex_lock(&my_impl);
        __TBB_ASSERT_EX(!rval, "critical_section::lock: pthread_mutex_lock failed");
#endif
        __TBB_ASSERT(my_tid == tbb_thread::id(), NULL);
        my_tid = local_tid;
    }

    bool try_lock() {
        bool gotlock;
        tbb_thread::id local_tid = this_tbb_thread::get_id();
        if(local_tid == my_tid) return false;
#if _WIN32||_WIN64
        gotlock = TryEnterCriticalSection( &my_impl ) != 0;
#else
        int rval = pthread_mutex_trylock(&my_impl);
        // valid returns are 0 (locked) and [EBUSY]
        __TBB_ASSERT(rval == 0 || rval == EBUSY, "critical_section::trylock: pthread_mutex_trylock failed");
        gotlock = rval == 0;
#endif
        if(gotlock)  {
            my_tid = local_tid;
        }
        return gotlock;
    }

    void unlock() {
        __TBB_ASSERT(this_tbb_thread::get_id() == my_tid, "thread unlocking critical_section is not thread that locked it");
        my_tid = tbb_thread::id();
#if _WIN32||_WIN64
        LeaveCriticalSection( &my_impl );
#else
        int rval = pthread_mutex_unlock(&my_impl);
        __TBB_ASSERT_EX(!rval, "critical_section::unlock: pthread_mutex_unlock failed");
#endif
    }

    static const bool is_rw_mutex = false;
    static const bool is_recursive_mutex = false;
    static const bool is_fair_mutex = true;
}; // critical_section_v4
} // namespace internal
__TBB_DEPRECATED_IN_VERBOSE_MODE_MSG("tbb::critical_section is deprecated, use std::mutex") typedef internal::critical_section_v4 critical_section;

__TBB_DEFINE_PROFILING_SET_NAME(critical_section)
} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_critical_section_H_include_area

#endif  // _TBB_CRITICAL_SECTION_H_
