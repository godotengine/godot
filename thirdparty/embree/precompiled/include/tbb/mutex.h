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

#if !defined(__TBB_show_deprecation_message_mutex_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_mutex_H
#pragma message("TBB Warning: tbb/mutex.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_mutex_H
#define __TBB_mutex_H

#define __TBB_mutex_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#if _WIN32||_WIN64
#include "machine/windows_api.h"
#else
#include <pthread.h>
#endif /* _WIN32||_WIN64 */

#include <new>
#include "aligned_space.h"
#include "tbb_stddef.h"
#include "tbb_profiling.h"

namespace tbb {

//! Wrapper around the platform's native lock.
/** @ingroup synchronization */
class __TBB_DEPRECATED_IN_VERBOSE_MODE_MSG("tbb::critical_section is deprecated, use std::mutex") mutex : internal::mutex_copy_deprecated_and_disabled {
public:
    //! Construct unacquired mutex.
    mutex() {
#if TBB_USE_ASSERT || TBB_USE_THREADING_TOOLS
    internal_construct();
#else
  #if _WIN32||_WIN64
        InitializeCriticalSectionEx(&impl, 4000, 0);
  #else
        int error_code = pthread_mutex_init(&impl,NULL);
        if( error_code )
            tbb::internal::handle_perror(error_code,"mutex: pthread_mutex_init failed");
  #endif /* _WIN32||_WIN64*/
#endif /* TBB_USE_ASSERT */
    };

    ~mutex() {
#if TBB_USE_ASSERT
        internal_destroy();
#else
  #if _WIN32||_WIN64
        DeleteCriticalSection(&impl);
  #else
        pthread_mutex_destroy(&impl);

  #endif /* _WIN32||_WIN64 */
#endif /* TBB_USE_ASSERT */
    };

    class scoped_lock;
    friend class scoped_lock;

    //! The scoped locking pattern
    /** It helps to avoid the common problem of forgetting to release lock.
        It also nicely provides the "node" for queuing locks. */
    class scoped_lock : internal::no_copy {
    public:
        //! Construct lock that has not acquired a mutex.
        scoped_lock() : my_mutex(NULL) {};

        //! Acquire lock on given mutex.
        scoped_lock( mutex& mutex ) {
            acquire( mutex );
        }

        //! Release lock (if lock is held).
        ~scoped_lock() {
            if( my_mutex )
                release();
        }

        //! Acquire lock on given mutex.
        void acquire( mutex& mutex ) {
#if TBB_USE_ASSERT
            internal_acquire(mutex);
#else
            mutex.lock();
            my_mutex = &mutex;
#endif /* TBB_USE_ASSERT */
        }

        //! Try acquire lock on given mutex.
        bool try_acquire( mutex& mutex ) {
#if TBB_USE_ASSERT
            return internal_try_acquire (mutex);
#else
            bool result = mutex.try_lock();
            if( result )
                my_mutex = &mutex;
            return result;
#endif /* TBB_USE_ASSERT */
        }

        //! Release lock
        void release() {
#if TBB_USE_ASSERT
            internal_release ();
#else
            my_mutex->unlock();
            my_mutex = NULL;
#endif /* TBB_USE_ASSERT */
        }

    private:
        //! The pointer to the current mutex to work
        mutex* my_mutex;

        //! All checks from acquire using mutex.state were moved here
        void __TBB_EXPORTED_METHOD internal_acquire( mutex& m );

        //! All checks from try_acquire using mutex.state were moved here
        bool __TBB_EXPORTED_METHOD internal_try_acquire( mutex& m );

        //! All checks from release using mutex.state were moved here
        void __TBB_EXPORTED_METHOD internal_release();

        friend class mutex;
    };

    // Mutex traits
    static const bool is_rw_mutex = false;
    static const bool is_recursive_mutex = false;
    static const bool is_fair_mutex = false;

    // ISO C++0x compatibility methods

    //! Acquire lock
    void lock() {
#if TBB_USE_ASSERT
        aligned_space<scoped_lock> tmp;
        new(tmp.begin()) scoped_lock(*this);
#else
  #if _WIN32||_WIN64
        EnterCriticalSection(&impl);
  #else
        int error_code = pthread_mutex_lock(&impl);
        if( error_code )
            tbb::internal::handle_perror(error_code,"mutex: pthread_mutex_lock failed");
  #endif /* _WIN32||_WIN64 */
#endif /* TBB_USE_ASSERT */
    }

    //! Try acquiring lock (non-blocking)
    /** Return true if lock acquired; false otherwise. */
    bool try_lock() {
#if TBB_USE_ASSERT
        aligned_space<scoped_lock> tmp;
        scoped_lock& s = *tmp.begin();
        s.my_mutex = NULL;
        return s.internal_try_acquire(*this);
#else
  #if _WIN32||_WIN64
        return TryEnterCriticalSection(&impl)!=0;
  #else
        return pthread_mutex_trylock(&impl)==0;
  #endif /* _WIN32||_WIN64 */
#endif /* TBB_USE_ASSERT */
    }

    //! Release lock
    void unlock() {
#if TBB_USE_ASSERT
        aligned_space<scoped_lock> tmp;
        scoped_lock& s = *tmp.begin();
        s.my_mutex = this;
        s.internal_release();
#else
  #if _WIN32||_WIN64
        LeaveCriticalSection(&impl);
  #else
        pthread_mutex_unlock(&impl);
  #endif /* _WIN32||_WIN64 */
#endif /* TBB_USE_ASSERT */
    }

    //! Return native_handle
  #if _WIN32||_WIN64
    typedef LPCRITICAL_SECTION native_handle_type;
  #else
    typedef pthread_mutex_t* native_handle_type;
  #endif
    native_handle_type native_handle() { return (native_handle_type) &impl; }

    enum state_t {
        INITIALIZED=0x1234,
        DESTROYED=0x789A,
        HELD=0x56CD
    };
private:
#if _WIN32||_WIN64
    CRITICAL_SECTION impl;
    enum state_t state;
#else
    pthread_mutex_t impl;
#endif /* _WIN32||_WIN64 */

    //! All checks from mutex constructor using mutex.state were moved here
    void __TBB_EXPORTED_METHOD internal_construct();

    //! All checks from mutex destructor using mutex.state were moved here
    void __TBB_EXPORTED_METHOD internal_destroy();

#if _WIN32||_WIN64
public:
    //!  Set the internal state
    void set_state( state_t to ) { state = to; }
#endif
};

__TBB_DEFINE_PROFILING_SET_NAME(mutex)

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_mutex_H_include_area

#endif /* __TBB_mutex_H */
