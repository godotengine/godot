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

#ifndef __TBB_spin_rw_mutex_H
#define __TBB_spin_rw_mutex_H

#include "tbb_stddef.h"
#include "tbb_machine.h"
#include "tbb_profiling.h"
#include "internal/_mutex_padding.h"

namespace tbb {

#if __TBB_TSX_AVAILABLE
namespace interface8 { namespace internal {
    class x86_rtm_rw_mutex;
}}
#endif

class spin_rw_mutex_v3;
typedef spin_rw_mutex_v3 spin_rw_mutex;

//! Fast, unfair, spinning reader-writer lock with backoff and writer-preference
/** @ingroup synchronization */
class spin_rw_mutex_v3 : internal::mutex_copy_deprecated_and_disabled {
    //! @cond INTERNAL

    //! Internal acquire write lock.
    bool __TBB_EXPORTED_METHOD internal_acquire_writer();

    //! Out of line code for releasing a write lock.
    /** This code has debug checking and instrumentation for Intel(R) Thread Checker and Intel(R) Thread Profiler. */
    void __TBB_EXPORTED_METHOD internal_release_writer();

    //! Internal acquire read lock.
    void __TBB_EXPORTED_METHOD internal_acquire_reader();

    //! Internal upgrade reader to become a writer.
    bool __TBB_EXPORTED_METHOD internal_upgrade();

    //! Out of line code for downgrading a writer to a reader.
    /** This code has debug checking and instrumentation for Intel(R) Thread Checker and Intel(R) Thread Profiler. */
    void __TBB_EXPORTED_METHOD internal_downgrade();

    //! Internal release read lock.
    void __TBB_EXPORTED_METHOD internal_release_reader();

    //! Internal try_acquire write lock.
    bool __TBB_EXPORTED_METHOD internal_try_acquire_writer();

    //! Internal try_acquire read lock.
    bool __TBB_EXPORTED_METHOD internal_try_acquire_reader();

    //! @endcond
public:
    //! Construct unacquired mutex.
    spin_rw_mutex_v3() : state(0) {
#if TBB_USE_THREADING_TOOLS
        internal_construct();
#endif
    }

#if TBB_USE_ASSERT
    //! Destructor asserts if the mutex is acquired, i.e. state is zero.
    ~spin_rw_mutex_v3() {
        __TBB_ASSERT( !state, "destruction of an acquired mutex");
    };
#endif /* TBB_USE_ASSERT */

    //! The scoped locking pattern
    /** It helps to avoid the common problem of forgetting to release lock.
        It also nicely provides the "node" for queuing locks. */
    class scoped_lock : internal::no_copy {
#if __TBB_TSX_AVAILABLE
        friend class tbb::interface8::internal::x86_rtm_rw_mutex;
#endif
    public:
        //! Construct lock that has not acquired a mutex.
        /** Equivalent to zero-initialization of *this. */
        scoped_lock() : mutex(NULL), is_writer(false) {}

        //! Acquire lock on given mutex.
        scoped_lock( spin_rw_mutex& m, bool write = true ) : mutex(NULL) {
            acquire(m, write);
        }

        //! Release lock (if lock is held).
        ~scoped_lock() {
            if( mutex ) release();
        }

        //! Acquire lock on given mutex.
        void acquire( spin_rw_mutex& m, bool write = true ) {
            __TBB_ASSERT( !mutex, "holding mutex already" );
            is_writer = write;
            mutex = &m;
            if( write ) mutex->internal_acquire_writer();
            else        mutex->internal_acquire_reader();
        }

        //! Upgrade reader to become a writer.
        /** Returns whether the upgrade happened without releasing and re-acquiring the lock */
        bool upgrade_to_writer() {
            __TBB_ASSERT( mutex, "mutex is not acquired" );
            if (is_writer) return true; // Already a writer
            is_writer = true;
            return mutex->internal_upgrade();
        }

        //! Release lock.
        void release() {
            __TBB_ASSERT( mutex, "mutex is not acquired" );
            spin_rw_mutex *m = mutex;
            mutex = NULL;
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
            if( is_writer ) m->internal_release_writer();
            else            m->internal_release_reader();
#else
            if( is_writer ) __TBB_AtomicAND( &m->state, READERS );
            else            __TBB_FetchAndAddWrelease( &m->state, -(intptr_t)ONE_READER);
#endif /* TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT */
        }

        //! Downgrade writer to become a reader.
        bool downgrade_to_reader() {
            __TBB_ASSERT( mutex, "mutex is not acquired" );
            if (!is_writer) return true; // Already a reader
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
            mutex->internal_downgrade();
#else
            __TBB_FetchAndAddW( &mutex->state, ((intptr_t)ONE_READER-WRITER));
#endif /* TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT */
            is_writer = false;
            return true;
        }

        //! Try acquire lock on given mutex.
        bool try_acquire( spin_rw_mutex& m, bool write = true ) {
            __TBB_ASSERT( !mutex, "holding mutex already" );
            bool result;
            is_writer = write;
            result = write? m.internal_try_acquire_writer()
                          : m.internal_try_acquire_reader();
            if( result )
                mutex = &m;
            return result;
        }

    protected:

        //! The pointer to the current mutex that is held, or NULL if no mutex is held.
        spin_rw_mutex* mutex;

        //! If mutex!=NULL, then is_writer is true if holding a writer lock, false if holding a reader lock.
        /** Not defined if not holding a lock. */
        bool is_writer;
    };

    // Mutex traits
    static const bool is_rw_mutex = true;
    static const bool is_recursive_mutex = false;
    static const bool is_fair_mutex = false;

    // ISO C++0x compatibility methods

    //! Acquire writer lock
    void lock() {internal_acquire_writer();}

    //! Try acquiring writer lock (non-blocking)
    /** Return true if lock acquired; false otherwise. */
    bool try_lock() {return internal_try_acquire_writer();}

    //! Release lock
    void unlock() {
#if TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT
        if( state&WRITER ) internal_release_writer();
        else               internal_release_reader();
#else
        if( state&WRITER ) __TBB_AtomicAND( &state, READERS );
        else               __TBB_FetchAndAddWrelease( &state, -(intptr_t)ONE_READER);
#endif /* TBB_USE_THREADING_TOOLS||TBB_USE_ASSERT */
    }

    // Methods for reader locks that resemble ISO C++0x compatibility methods.

    //! Acquire reader lock
    void lock_read() {internal_acquire_reader();}

    //! Try acquiring reader lock (non-blocking)
    /** Return true if reader lock acquired; false otherwise. */
    bool try_lock_read() {return internal_try_acquire_reader();}

protected:
    typedef intptr_t state_t;
    static const state_t WRITER = 1;
    static const state_t WRITER_PENDING = 2;
    static const state_t READERS = ~(WRITER | WRITER_PENDING);
    static const state_t ONE_READER = 4;
    static const state_t BUSY = WRITER | READERS;
    //! State of lock
    /** Bit 0 = writer is holding lock
        Bit 1 = request by a writer to acquire lock (hint to readers to wait)
        Bit 2..N = number of readers holding lock */
    state_t state;

private:
    void __TBB_EXPORTED_METHOD internal_construct();
};

__TBB_DEFINE_PROFILING_SET_NAME(spin_rw_mutex)

} // namespace tbb

#if __TBB_TSX_AVAILABLE
#include "internal/_x86_rtm_rw_mutex_impl.h"
#endif

namespace tbb {
namespace interface8 {
//! A cross-platform spin reader/writer mutex with speculative lock acquisition.
/** On platforms with proper HW support, this lock may speculatively execute
    its critical sections, using HW mechanisms to detect real data races and
    ensure atomicity of the critical sections. In particular, it uses
    Intel(R) Transactional Synchronization Extensions (Intel(R) TSX).
    Without such HW support, it behaves like a spin_rw_mutex.
    It should be used for locking short critical sections where the lock is
    contended but the data it protects are not.
    @ingroup synchronization */
#if __TBB_TSX_AVAILABLE
typedef interface7::internal::padded_mutex<tbb::interface8::internal::x86_rtm_rw_mutex,true> speculative_spin_rw_mutex;
#else
typedef interface7::internal::padded_mutex<tbb::spin_rw_mutex,true> speculative_spin_rw_mutex;
#endif
}  // namespace interface8

using interface8::speculative_spin_rw_mutex;
__TBB_DEFINE_PROFILING_SET_NAME(speculative_spin_rw_mutex)
} // namespace tbb
#endif /* __TBB_spin_rw_mutex_H */
