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

#ifndef __TBB__x86_rtm_rw_mutex_impl_H
#define __TBB__x86_rtm_rw_mutex_impl_H

#ifndef __TBB_spin_rw_mutex_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#if __TBB_TSX_AVAILABLE

#include "../tbb_stddef.h"
#include "../tbb_machine.h"
#include "../tbb_profiling.h"
#include "../spin_rw_mutex.h"

namespace tbb {
namespace interface8 {
namespace internal {

enum RTM_type {
    RTM_not_in_mutex,
    RTM_transacting_reader,
    RTM_transacting_writer,
    RTM_real_reader,
    RTM_real_writer
};

static const unsigned long speculation_granularity = 64;

//! Fast, unfair, spinning speculation-enabled reader-writer lock with backoff and
//  writer-preference
/** @ingroup synchronization */
class x86_rtm_rw_mutex: private spin_rw_mutex {
#if __TBB_USE_X86_RTM_RW_MUTEX || __TBB_GCC_VERSION < 40000
// bug in gcc 3.x.x causes syntax error in spite of the friend declaration below.
// Make the scoped_lock public in that case.
public:
#else
private:
#endif
    friend class interface7::internal::padded_mutex<x86_rtm_rw_mutex,true>;
    class scoped_lock;   // should be private
    friend class scoped_lock;
private:
    //! @cond INTERNAL

    //! Internal construct unacquired mutex.
    void __TBB_EXPORTED_METHOD internal_construct();

    //! Internal acquire write lock.
    // only_speculate == true if we're doing a try_lock, else false.
    void __TBB_EXPORTED_METHOD internal_acquire_writer(x86_rtm_rw_mutex::scoped_lock&, bool only_speculate=false);

    //! Internal acquire read lock.
    // only_speculate == true if we're doing a try_lock, else false.
    void __TBB_EXPORTED_METHOD internal_acquire_reader(x86_rtm_rw_mutex::scoped_lock&, bool only_speculate=false);

    //! Internal upgrade reader to become a writer.
    bool __TBB_EXPORTED_METHOD internal_upgrade( x86_rtm_rw_mutex::scoped_lock& );

    //! Out of line code for downgrading a writer to a reader.
    bool __TBB_EXPORTED_METHOD internal_downgrade( x86_rtm_rw_mutex::scoped_lock& );

    //! Internal try_acquire write lock.
    bool __TBB_EXPORTED_METHOD internal_try_acquire_writer( x86_rtm_rw_mutex::scoped_lock& );

    //! Internal release lock.
    void __TBB_EXPORTED_METHOD internal_release( x86_rtm_rw_mutex::scoped_lock& );

    static x86_rtm_rw_mutex* internal_get_mutex( const spin_rw_mutex::scoped_lock& lock )
    {
        return static_cast<x86_rtm_rw_mutex*>( lock.mutex );
    }
    static void internal_set_mutex( spin_rw_mutex::scoped_lock& lock, spin_rw_mutex* mtx )
    {
        lock.mutex = mtx;
    }
    //! @endcond
public:
    //! Construct unacquired mutex.
    x86_rtm_rw_mutex() {
        w_flag = false;
#if TBB_USE_THREADING_TOOLS
        internal_construct();
#endif
    }

#if TBB_USE_ASSERT
    //! Empty destructor.
    ~x86_rtm_rw_mutex() {}
#endif /* TBB_USE_ASSERT */

    // Mutex traits
    static const bool is_rw_mutex = true;
    static const bool is_recursive_mutex = false;
    static const bool is_fair_mutex = false;

#if __TBB_USE_X86_RTM_RW_MUTEX || __TBB_GCC_VERSION < 40000
#else
    // by default we will not provide the scoped_lock interface.  The user
    // should use the padded version of the mutex.  scoped_lock is used in
    // padded_mutex template.
private:
#endif
    //! The scoped locking pattern
    /** It helps to avoid the common problem of forgetting to release lock.
        It also nicely provides the "node" for queuing locks. */
    // Speculation-enabled scoped lock for spin_rw_mutex
    // The idea is to be able to reuse the acquire/release methods of spin_rw_mutex
    // and its scoped lock wherever possible.  The only way to use a speculative lock is to use
    // a scoped_lock. (because transaction_state must be local)

    class scoped_lock : tbb::internal::no_copy {
        friend class x86_rtm_rw_mutex;
        spin_rw_mutex::scoped_lock my_scoped_lock;

        RTM_type transaction_state;

    public:
        //! Construct lock that has not acquired a mutex.
        /** Equivalent to zero-initialization of *this. */
        scoped_lock() : my_scoped_lock(), transaction_state(RTM_not_in_mutex) {
        }

        //! Acquire lock on given mutex.
        scoped_lock( x86_rtm_rw_mutex& m, bool write = true ) : my_scoped_lock(),
            transaction_state(RTM_not_in_mutex) {
            acquire(m, write);
        }

        //! Release lock (if lock is held).
        ~scoped_lock() {
            if(transaction_state != RTM_not_in_mutex) release();
        }

        //! Acquire lock on given mutex.
        void acquire( x86_rtm_rw_mutex& m, bool write = true ) {
            if( write ) m.internal_acquire_writer(*this);
            else        m.internal_acquire_reader(*this);
        }

        //! Release lock
        void release() {
            x86_rtm_rw_mutex* mutex = x86_rtm_rw_mutex::internal_get_mutex(my_scoped_lock);
            __TBB_ASSERT( mutex, "lock is not acquired" );
            __TBB_ASSERT( transaction_state!=RTM_not_in_mutex, "lock is not acquired" );
            return mutex->internal_release(*this);
        }

        //! Upgrade reader to become a writer.
        /** Returns whether the upgrade happened without releasing and re-acquiring the lock */
        bool upgrade_to_writer() {
            x86_rtm_rw_mutex* mutex = x86_rtm_rw_mutex::internal_get_mutex(my_scoped_lock);
            __TBB_ASSERT( mutex, "lock is not acquired" );
            if (transaction_state == RTM_transacting_writer || transaction_state == RTM_real_writer)
                return true; // Already a writer
            return mutex->internal_upgrade(*this);
        }

        //! Downgrade writer to become a reader.
        /** Returns whether the downgrade happened without releasing and re-acquiring the lock */
        bool downgrade_to_reader() {
            x86_rtm_rw_mutex* mutex = x86_rtm_rw_mutex::internal_get_mutex(my_scoped_lock);
            __TBB_ASSERT( mutex, "lock is not acquired" );
            if (transaction_state == RTM_transacting_reader || transaction_state == RTM_real_reader)
                return true; // Already a reader
            return mutex->internal_downgrade(*this);
        }

        //! Attempt to acquire mutex.
        /** returns true if successful.  */
        bool try_acquire( x86_rtm_rw_mutex& m, bool write = true ) {
#if TBB_USE_ASSERT
            x86_rtm_rw_mutex* mutex = x86_rtm_rw_mutex::internal_get_mutex(my_scoped_lock);
            __TBB_ASSERT( !mutex, "lock is already acquired" );
#endif
            // have to assign m to our mutex.
            // cannot set the mutex, because try_acquire in spin_rw_mutex depends on it being NULL.
            if(write) return m.internal_try_acquire_writer(*this);
            // speculatively acquire the lock.  If this fails, do try_acquire on the spin_rw_mutex.
            m.internal_acquire_reader(*this, /*only_speculate=*/true);
            if(transaction_state == RTM_transacting_reader) return true;
            if( my_scoped_lock.try_acquire(m, false)) {
                transaction_state = RTM_real_reader;
                return true;
            }
            return false;
        }

        };  // class x86_rtm_rw_mutex::scoped_lock

    // ISO C++0x compatibility methods not provided because we cannot maintain
    // state about whether a thread is in a transaction.

private:
    char pad[speculation_granularity-sizeof(spin_rw_mutex)]; // padding

    // If true, writer holds the spin_rw_mutex.
    tbb::atomic<bool> w_flag;  // want this on a separate cache line

};  // x86_rtm_rw_mutex

}  // namespace internal
}  // namespace interface8
}  // namespace tbb

#endif  /* __TBB_TSX_AVAILABLE */
#endif /* __TBB__x86_rtm_rw_mutex_impl_H */
