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

#ifndef __TBB__aggregator_H
#define __TBB__aggregator_H

#define __TBB_aggregator_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#if !TBB_PREVIEW_AGGREGATOR
#error Set TBB_PREVIEW_AGGREGATOR before including aggregator.h
#endif

#include "atomic.h"
#include "tbb_profiling.h"

namespace tbb {
namespace interface6 {

using namespace tbb::internal;

class aggregator_operation {
    template<typename handler_type> friend class aggregator_ext;
    uintptr_t status;
    aggregator_operation* my_next;
public:
    enum aggregator_operation_status { agg_waiting=0, agg_finished };
    aggregator_operation() : status(agg_waiting), my_next(NULL) {}
    /// Call start before handling this operation
    void start() { call_itt_notify(acquired, &status); }
    /// Call finish when done handling this operation
    /** The operation will be released to its originating thread, and possibly deleted. */
    void finish() { itt_store_word_with_release(status, uintptr_t(agg_finished)); }
    aggregator_operation* next() { return itt_hide_load_word(my_next);}
    void set_next(aggregator_operation* n) { itt_hide_store_word(my_next, n); }
};

namespace internal {

class basic_operation_base : public aggregator_operation {
    friend class basic_handler;
    virtual void apply_body() = 0;
public:
    basic_operation_base() : aggregator_operation() {}
    virtual ~basic_operation_base() {}
};

template<typename Body>
class basic_operation : public basic_operation_base, no_assign {
    const Body& my_body;
    void apply_body() __TBB_override { my_body(); }
public:
    basic_operation(const Body& b) : basic_operation_base(), my_body(b) {}
};

class basic_handler {
public:
    basic_handler() {}
    void operator()(aggregator_operation* op_list) const {
        while (op_list) {
            // ITT note: &(op_list->status) tag is used to cover accesses to the operation data.
            // The executing thread "acquires" the tag (see start()) and then performs
            // the associated operation w/o triggering a race condition diagnostics.
            // A thread that created the operation is waiting for its status (see execute_impl()),
            // so when this thread is done with the operation, it will "release" the tag
            // and update the status (see finish()) to give control back to the waiting thread.
            basic_operation_base& request = static_cast<basic_operation_base&>(*op_list);
            // IMPORTANT: need to advance op_list to op_list->next() before calling request.finish()
            op_list = op_list->next();
            request.start();
            request.apply_body();
            request.finish();
        }
    }
};

} // namespace internal

//! Aggregator base class and expert interface
/** An aggregator for collecting operations coming from multiple sources and executing
    them serially on a single thread. */
template <typename handler_type>
class aggregator_ext : tbb::internal::no_copy {
public:
    aggregator_ext(const handler_type& h) : handler_busy(0), handle_operations(h) { mailbox = NULL; }

    //! EXPERT INTERFACE: Enter a user-made operation into the aggregator's mailbox.
    /** Details of user-made operations must be handled by user-provided handler */
    void process(aggregator_operation *op) { execute_impl(*op); }

protected:
    /** Place operation in mailbox, then either handle mailbox or wait for the operation
        to be completed by a different thread. */
    void execute_impl(aggregator_operation& op) {
        aggregator_operation* res;

        // ITT note: &(op.status) tag is used to cover accesses to this operation. This
        // thread has created the operation, and now releases it so that the handler
        // thread may handle the associated operation w/o triggering a race condition;
        // thus this tag will be acquired just before the operation is handled in the
        // handle_operations functor.
        call_itt_notify(releasing, &(op.status));
        // insert the operation into the list
        do {
            // ITT may flag the following line as a race; it is a false positive:
            // This is an atomic read; we don't provide itt_hide_load_word for atomics
            op.my_next = res = mailbox; // NOT A RACE
        } while (mailbox.compare_and_swap(&op, res) != res);
        if (!res) { // first in the list; handle the operations
            // ITT note: &mailbox tag covers access to the handler_busy flag, which this
            // waiting handler thread will try to set before entering handle_operations.
            call_itt_notify(acquired, &mailbox);
            start_handle_operations();
            __TBB_ASSERT(op.status, NULL);
        }
        else { // not first; wait for op to be ready
            call_itt_notify(prepare, &(op.status));
            spin_wait_while_eq(op.status, uintptr_t(aggregator_operation::agg_waiting));
            itt_load_word_with_acquire(op.status);
        }
    }


private:
    //! An atomically updated list (aka mailbox) of aggregator_operations
    atomic<aggregator_operation *> mailbox;

    //! Controls thread access to handle_operations
    /** Behaves as boolean flag where 0=false, 1=true */
    uintptr_t handler_busy;

    handler_type handle_operations;

    //! Trigger the handling of operations when the handler is free
    void start_handle_operations() {
        aggregator_operation *pending_operations;

        // ITT note: &handler_busy tag covers access to mailbox as it is passed
        // between active and waiting handlers.  Below, the waiting handler waits until
        // the active handler releases, and the waiting handler acquires &handler_busy as
        // it becomes the active_handler. The release point is at the end of this
        // function, when all operations in mailbox have been handled by the
        // owner of this aggregator.
        call_itt_notify(prepare, &handler_busy);
        // get handler_busy: only one thread can possibly spin here at a time
        spin_wait_until_eq(handler_busy, uintptr_t(0));
        call_itt_notify(acquired, &handler_busy);
        // acquire fence not necessary here due to causality rule and surrounding atomics
        __TBB_store_with_release(handler_busy, uintptr_t(1));

        // ITT note: &mailbox tag covers access to the handler_busy flag itself.
        // Capturing the state of the mailbox signifies that handler_busy has been
        // set and a new active handler will now process that list's operations.
        call_itt_notify(releasing, &mailbox);
        // grab pending_operations
        pending_operations = mailbox.fetch_and_store(NULL);

        // handle all the operations
        handle_operations(pending_operations);

        // release the handler
        itt_store_word_with_release(handler_busy, uintptr_t(0));
    }
};

//! Basic aggregator interface
class aggregator : private aggregator_ext<internal::basic_handler> {
public:
    aggregator() : aggregator_ext<internal::basic_handler>(internal::basic_handler()) {}
    //! BASIC INTERFACE: Enter a function for exclusive execution by the aggregator.
    /** The calling thread stores the function object in a basic_operation and
        places the operation in the aggregator's mailbox */
    template<typename Body>
    void execute(const Body& b) {
        internal::basic_operation<Body> op(b);
        this->execute_impl(op);
    }
};

} // namespace interface6

using interface6::aggregator;
using interface6::aggregator_ext;
using interface6::aggregator_operation;

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_aggregator_H_include_area

#endif  // __TBB__aggregator_H
