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

#ifndef __TBB__aggregator_impl_H
#define __TBB__aggregator_impl_H

#include "../atomic.h"
#if !__TBBMALLOC_BUILD
#include "../tbb_profiling.h"
#endif

namespace tbb {
namespace interface6 {
namespace internal {

using namespace tbb::internal;

//! aggregated_operation base class
template <typename Derived>
class aggregated_operation {
 public:
    //! Zero value means "wait" status, all other values are "user" specified values and are defined into the scope of a class which uses "status".
    uintptr_t status;

    Derived *next;
    aggregated_operation() : status(0), next(NULL) {}
};

//! Aggregator base class
/** An aggregator for collecting operations coming from multiple sources and executing
    them serially on a single thread.  operation_type must be derived from
    aggregated_operation. The parameter handler_type is a functor that will be passed the
    list of operations and is expected to handle each operation appropriately, setting the
    status of each operation to non-zero.*/
template < typename operation_type >
class aggregator_generic {
public:
    aggregator_generic() : handler_busy(false) { pending_operations = NULL; }

    //! Execute an operation
    /** Places an operation into the waitlist (pending_operations), and either handles the list,
        or waits for the operation to complete, or returns.
        The long_life_time parameter specifies the life time of the given operation object.
        Operations with long_life_time == true may be accessed after execution.
        A "short" life time operation (long_life_time == false) can be destroyed
        during execution, and so any access to it after it was put into the waitlist,
        including status check, is invalid. As a consequence, waiting for completion
        of such operation causes undefined behavior.
    */
    template < typename handler_type >
    void execute(operation_type *op, handler_type &handle_operations, bool long_life_time = true) {
        operation_type *res;
        // op->status should be read before inserting the operation into the
        // aggregator waitlist since it can become invalid after executing a
        // handler (if the operation has 'short' life time.)
        const uintptr_t status = op->status;

        // ITT note: &(op->status) tag is used to cover accesses to this op node. This
        // thread has created the operation, and now releases it so that the handler
        // thread may handle the associated operation w/o triggering a race condition;
        // thus this tag will be acquired just before the operation is handled in the
        // handle_operations functor.
        call_itt_notify(releasing, &(op->status));
        // insert the operation in the queue.
        do {
            // Tools may flag the following line as a race; it is a false positive:
            // This is an atomic read; we don't provide itt_hide_load_word for atomics
            op->next = res = pending_operations; // NOT A RACE
        } while (pending_operations.compare_and_swap(op, res) != res);
        if (!res) { // first in the list; handle the operations.
            // ITT note: &pending_operations tag covers access to the handler_busy flag,
            // which this waiting handler thread will try to set before entering
            // handle_operations.
            call_itt_notify(acquired, &pending_operations);
            start_handle_operations(handle_operations);
            // The operation with 'short' life time can already be destroyed.
            if (long_life_time)
                __TBB_ASSERT(op->status, NULL);
        }
        // not first; wait for op to be ready.
        else if (!status) { // operation is blocking here.
            __TBB_ASSERT(long_life_time, "Waiting for an operation object that might be destroyed during processing.");
            call_itt_notify(prepare, &(op->status));
            spin_wait_while_eq(op->status, uintptr_t(0));
            itt_load_word_with_acquire(op->status);
        }
    }

 private:
    //! An atomically updated list (aka mailbox) of pending operations
    atomic<operation_type *> pending_operations;
    //! Controls thread access to handle_operations
    uintptr_t handler_busy;

    //! Trigger the handling of operations when the handler is free
    template < typename handler_type >
    void start_handle_operations( handler_type &handle_operations ) {
        operation_type *op_list;

        // ITT note: &handler_busy tag covers access to pending_operations as it is passed
        // between active and waiting handlers.  Below, the waiting handler waits until
        // the active handler releases, and the waiting handler acquires &handler_busy as
        // it becomes the active_handler. The release point is at the end of this
        // function, when all operations in pending_operations have been handled by the
        // owner of this aggregator.
        call_itt_notify(prepare, &handler_busy);
        // get the handler_busy:
        // only one thread can possibly spin here at a time
        spin_wait_until_eq(handler_busy, uintptr_t(0));
        call_itt_notify(acquired, &handler_busy);
        // acquire fence not necessary here due to causality rule and surrounding atomics
        __TBB_store_with_release(handler_busy, uintptr_t(1));

        // ITT note: &pending_operations tag covers access to the handler_busy flag
        // itself. Capturing the state of the pending_operations signifies that
        // handler_busy has been set and a new active handler will now process that list's
        // operations.
        call_itt_notify(releasing, &pending_operations);
        // grab pending_operations
        op_list = pending_operations.fetch_and_store(NULL);

        // handle all the operations
        handle_operations(op_list);

        // release the handler
        itt_store_word_with_release(handler_busy, uintptr_t(0));
    }
};

template < typename handler_type, typename operation_type >
class aggregator : public aggregator_generic<operation_type> {
    handler_type handle_operations;
public:
    aggregator() {}
    explicit aggregator(handler_type h) : handle_operations(h) {}

    void initialize_handler(handler_type h) { handle_operations = h; }

    void execute(operation_type *op) {
        aggregator_generic<operation_type>::execute(op, handle_operations);
    }
};

// the most-compatible friend declaration (vs, gcc, icc) is
//    template<class U, class V> friend class aggregating_functor;
template<typename aggregating_class, typename operation_list>
class aggregating_functor {
    aggregating_class *fi;
public:
    aggregating_functor() : fi() {}
    aggregating_functor(aggregating_class *fi_) : fi(fi_) {}
    void operator()(operation_list* op_list) { fi->handle_operations(op_list); }
};

} // namespace internal
} // namespace interface6

namespace internal {
    using interface6::internal::aggregated_operation;
    using interface6::internal::aggregator_generic;
    using interface6::internal::aggregator;
    using interface6::internal::aggregating_functor;
} // namespace internal

} // namespace tbb

#endif  // __TBB__aggregator_impl_H
