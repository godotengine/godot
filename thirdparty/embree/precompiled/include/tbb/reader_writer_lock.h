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

#if !defined(__TBB_show_deprecation_message_reader_writer_lock_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_reader_writer_lock_H
#pragma message("TBB Warning: tbb/reader_writer_lock.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_reader_writer_lock_H
#define __TBB_reader_writer_lock_H

#define __TBB_reader_writer_lock_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "tbb_thread.h"
#include "tbb_allocator.h"
#include "atomic.h"

namespace tbb {
namespace interface5 {
//! Writer-preference reader-writer lock with local-only spinning on readers.
/** Loosely adapted from Mellor-Crummey and Scott pseudocode at
    http://www.cs.rochester.edu/research/synchronization/pseudocode/rw.html#s_wp
    @ingroup synchronization */
    class __TBB_DEPRECATED_IN_VERBOSE_MODE_MSG("tbb::reader_writer_lock is deprecated, use std::shared_mutex")
    reader_writer_lock : tbb::internal::no_copy {
 public:
    friend class scoped_lock;
    friend class scoped_lock_read;
    //! Status type for nodes associated with lock instances
    /** waiting_nonblocking: the wait state for nonblocking lock
          instances; for writes, these transition straight to active
          states; for reads, these are unused.

        waiting: the start and spin state for all lock instances; these will
          transition to active state when appropriate.  Non-blocking write locks
          transition from this state to waiting_nonblocking immediately.

        active: the active state means that the lock instance holds
          the lock; it will transition to invalid state during node deletion

        invalid: the end state for all nodes; this is set in the
          destructor so if we encounter this state, we are looking at
          memory that has already been freed

        The state diagrams below describe the status transitions.
        Single arrows indicate that the thread that owns the node is
        responsible for the transition; double arrows indicate that
        any thread could make the transition.

        State diagram for scoped_lock status:

        waiting ----------> waiting_nonblocking
          |     _____________/       |
          V    V                     V
        active -----------------> invalid

        State diagram for scoped_lock_read status:

        waiting
          |
          V
        active ----------------->invalid

    */
    enum status_t { waiting_nonblocking, waiting, active, invalid };

    //! Constructs a new reader_writer_lock
    reader_writer_lock() {
        internal_construct();
    }

    //! Destructs a reader_writer_lock object
    ~reader_writer_lock() {
        internal_destroy();
    }

    //! The scoped lock pattern for write locks
    /** Scoped locks help avoid the common problem of forgetting to release the lock.
        This type also serves as the node for queuing locks. */
    class scoped_lock : tbb::internal::no_copy {
    public:
        friend class reader_writer_lock;

        //! Construct with blocking attempt to acquire write lock on the passed-in lock
        scoped_lock(reader_writer_lock& lock) {
            internal_construct(lock);
        }

        //! Destructor, releases the write lock
        ~scoped_lock() {
            internal_destroy();
        }

        void* operator new(size_t s) {
            return tbb::internal::allocate_via_handler_v3(s);
        }
        void operator delete(void* p) {
            tbb::internal::deallocate_via_handler_v3(p);
        }

    private:
        //! The pointer to the mutex to lock
        reader_writer_lock *mutex;
        //! The next queued competitor for the mutex
        scoped_lock* next;
        //! Status flag of the thread associated with this node
        atomic<status_t> status;

        //! Construct scoped_lock that is not holding lock
        scoped_lock();

        void __TBB_EXPORTED_METHOD internal_construct(reader_writer_lock&);
        void __TBB_EXPORTED_METHOD internal_destroy();
   };

    //! The scoped lock pattern for read locks
    class scoped_lock_read : tbb::internal::no_copy {
    public:
        friend class reader_writer_lock;

        //! Construct with blocking attempt to acquire read lock on the passed-in lock
        scoped_lock_read(reader_writer_lock& lock) {
            internal_construct(lock);
        }

        //! Destructor, releases the read lock
        ~scoped_lock_read() {
            internal_destroy();
        }

        void* operator new(size_t s) {
            return tbb::internal::allocate_via_handler_v3(s);
        }
        void operator delete(void* p) {
            tbb::internal::deallocate_via_handler_v3(p);
        }

    private:
        //! The pointer to the mutex to lock
        reader_writer_lock *mutex;
        //! The next queued competitor for the mutex
        scoped_lock_read *next;
        //! Status flag of the thread associated with this node
        atomic<status_t> status;

        //! Construct scoped_lock_read that is not holding lock
        scoped_lock_read();

        void __TBB_EXPORTED_METHOD internal_construct(reader_writer_lock&);
        void __TBB_EXPORTED_METHOD internal_destroy();
    };

    //! Acquires the reader_writer_lock for write.
    /** If the lock is currently held in write mode by another
        context, the writer will block by spinning on a local
        variable.  Exceptions thrown: improper_lock The context tries
        to acquire a reader_writer_lock that it already has write
        ownership of.*/
    void __TBB_EXPORTED_METHOD lock();

    //! Tries to acquire the reader_writer_lock for write.
    /** This function does not block.  Return Value: True or false,
        depending on whether the lock is acquired or not.  If the lock
        is already held by this acquiring context, try_lock() returns
        false. */
    bool __TBB_EXPORTED_METHOD try_lock();

    //! Acquires the reader_writer_lock for read.
    /** If the lock is currently held by a writer, this reader will
        block and wait until the writers are done.  Exceptions thrown:
        improper_lock The context tries to acquire a
        reader_writer_lock that it already has write ownership of. */
    void __TBB_EXPORTED_METHOD lock_read();

    //! Tries to acquire the reader_writer_lock for read.
    /** This function does not block.  Return Value: True or false,
        depending on whether the lock is acquired or not.  */
    bool __TBB_EXPORTED_METHOD try_lock_read();

    //! Releases the reader_writer_lock
    void __TBB_EXPORTED_METHOD unlock();

 private:
    void __TBB_EXPORTED_METHOD internal_construct();
    void __TBB_EXPORTED_METHOD internal_destroy();

    //! Attempts to acquire write lock
    /** If unavailable, spins in blocking case, returns false in non-blocking case. */
    bool start_write(scoped_lock *);
    //! Sets writer_head to w and attempts to unblock
    void set_next_writer(scoped_lock *w);
    //! Relinquishes write lock to next waiting writer or group of readers
    void end_write(scoped_lock *);
    //! Checks if current thread holds write lock
    bool is_current_writer();

    //! Attempts to acquire read lock
    /** If unavailable, spins in blocking case, returns false in non-blocking case. */
    void start_read(scoped_lock_read *);
    //! Unblocks pending readers
    void unblock_readers();
    //! Relinquishes read lock by decrementing counter; last reader wakes pending writer
    void end_read();

    //! The list of pending readers
    atomic<scoped_lock_read*> reader_head;
    //! The list of pending writers
    atomic<scoped_lock*> writer_head;
    //! The last node in the list of pending writers
    atomic<scoped_lock*> writer_tail;
    //! Writer that owns the mutex; tbb_thread::id() otherwise.
    tbb_thread::id my_current_writer;
    //! Status of mutex
    atomic<uintptr_t> rdr_count_and_flags; // used with __TBB_AtomicOR, which assumes uintptr_t
};

} // namespace interface5

using interface5::reader_writer_lock;

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_reader_writer_lock_H_include_area

#endif /* __TBB_reader_writer_lock_H */
