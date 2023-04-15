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

#ifndef __TBB_spin_mutex_H
#define __TBB_spin_mutex_H

#define __TBB_spin_mutex_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include <cstddef>
#include <new>
#include "aligned_space.h"
#include "tbb_stddef.h"
#include "tbb_machine.h"
#include "tbb_profiling.h"
#include "internal/_mutex_padding.h"

namespace tbb {

//! A lock that occupies a single byte.
/** A spin_mutex is a spin mutex that fits in a single byte.
    It should be used only for locking short critical sections
    (typically less than 20 instructions) when fairness is not an issue.
    If zero-initialized, the mutex is considered unheld.
    @ingroup synchronization */
class spin_mutex : internal::mutex_copy_deprecated_and_disabled {
    //! 0 if lock is released, 1 if lock is acquired.
    __TBB_atomic_flag flag;

public:
    //! Construct unacquired lock.
    /** Equivalent to zero-initialization of *this. */
    spin_mutex() : flag(0) {
#if TBB_USE_THREADING_TOOLS
        internal_construct();
#endif
    }

    //! Represents acquisition of a mutex.
    class scoped_lock : internal::no_copy {
    private:
        //! Points to currently held mutex, or NULL if no lock is held.
        spin_mutex* my_mutex;

        //! Value to store into spin_mutex::flag to unlock the mutex.
        /** This variable is no longer used. Instead, 0 and 1 are used to
            represent that the lock is free and acquired, respectively.
            We keep the member variable here to ensure backward compatibility */
        __TBB_Flag my_unlock_value;

        //! Like acquire, but with ITT instrumentation.
        void __TBB_EXPORTED_METHOD internal_acquire( spin_mutex& m );

        //! Like try_acquire, but with ITT instrumentation.
        bool __TBB_EXPORTED_METHOD internal_try_acquire( spin_mutex& m );

        //! Like release, but with ITT instrumentation.
        void __TBB_EXPORTED_METHOD internal_release();

        friend class spin_mutex;

    public:
        //! Construct without acquiring a mutex.
        scoped_lock() : my_mutex(NULL), my_unlock_value(0) {}

        //! Construct and acquire lock on a mutex.
        scoped_lock( spin_mutex& m ) : my_unlock_value(0) {
            internal::suppress_unused_warning(my_unlock_value);
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
            my_mutex=NULL;
            internal_acquire(m);
#else
            my_mutex=&m;
            __TBB_LockByte(m.flag);
#endif /* TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT*/
        }

        //! Acquire lock.
        void acquire( spin_mutex& m ) {
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
            internal_acquire(m);
#else
            my_mutex = &m;
            __TBB_LockByte(m.flag);
#endif /* TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT*/
        }

        //! Try acquiring lock (non-blocking)
        /** Return true if lock acquired; false otherwise. */
        bool try_acquire( spin_mutex& m ) {
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
            return internal_try_acquire(m);
#else
            bool result = __TBB_TryLockByte(m.flag);
            if( result )
                my_mutex = &m;
            return result;
#endif /* TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT*/
        }

        //! Release lock
        void release() {
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
            internal_release();
#else
            __TBB_UnlockByte(my_mutex->flag);
            my_mutex = NULL;
#endif /* TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT */
        }

        //! Destroy lock.  If holding a lock, releases the lock first.
        ~scoped_lock() {
            if( my_mutex ) {
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
                internal_release();
#else
                __TBB_UnlockByte(my_mutex->flag);
#endif /* TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT */
            }
        }
    };

    //! Internal constructor with ITT instrumentation.
    void __TBB_EXPORTED_METHOD internal_construct();

    // Mutex traits
    static const bool is_rw_mutex = false;
    static const bool is_recursive_mutex = false;
    static const bool is_fair_mutex = false;

    // ISO C++0x compatibility methods

    //! Acquire lock
    void lock() {
#if TBB_USE_THREADING_TOOLS
        aligned_space<scoped_lock> tmp;
        new(tmp.begin()) scoped_lock(*this);
#else
        __TBB_LockByte(flag);
#endif /* TBB_USE_THREADING_TOOLS*/
    }

    //! Try acquiring lock (non-blocking)
    /** Return true if lock acquired; false otherwise. */
    bool try_lock() {
#if TBB_USE_THREADING_TOOLS
        aligned_space<scoped_lock> tmp;
        return (new(tmp.begin()) scoped_lock)->internal_try_acquire(*this);
#else
        return __TBB_TryLockByte(flag);
#endif /* TBB_USE_THREADING_TOOLS*/
    }

    //! Release lock
    void unlock() {
#if TBB_USE_THREADING_TOOLS
        aligned_space<scoped_lock> tmp;
        scoped_lock& s = *tmp.begin();
        s.my_mutex = this;
        s.internal_release();
#else
        __TBB_UnlockByte(flag);
#endif /* TBB_USE_THREADING_TOOLS */
    }

    friend class scoped_lock;
}; // end of spin_mutex

__TBB_DEFINE_PROFILING_SET_NAME(spin_mutex)

} // namespace tbb

#if ( __TBB_x86_32 || __TBB_x86_64 )
#include "internal/_x86_eliding_mutex_impl.h"
#endif

namespace tbb {
//! A cross-platform spin mutex with speculative lock acquisition.
/** On platforms with proper HW support, this lock may speculatively execute
    its critical sections, using HW mechanisms to detect real data races and
    ensure atomicity of the critical sections. In particular, it uses
    Intel(R) Transactional Synchronization Extensions (Intel(R) TSX).
    Without such HW support, it behaves like a spin_mutex.
    It should be used for locking short critical sections where the lock is
    contended but the data it protects are not.  If zero-initialized, the
    mutex is considered unheld.
    @ingroup synchronization */

#if ( __TBB_x86_32 || __TBB_x86_64 )
typedef interface7::internal::padded_mutex<interface7::internal::x86_eliding_mutex,false> speculative_spin_mutex;
#else
typedef interface7::internal::padded_mutex<spin_mutex,false> speculative_spin_mutex;
#endif
__TBB_DEFINE_PROFILING_SET_NAME(speculative_spin_mutex)

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_spin_mutex_H_include_area

#endif /* __TBB_spin_mutex_H */
