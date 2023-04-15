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

#ifndef __TBB_task_group_H
#define __TBB_task_group_H

#define __TBB_task_group_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "task.h"
#include "tbb_exception.h"
#include "internal/_template_helpers.h"
#if TBB_PREVIEW_ISOLATED_TASK_GROUP && __TBB_TASK_ISOLATION
#include "task_arena.h"
#endif

#if __TBB_TASK_GROUP_CONTEXT

namespace tbb {

namespace internal {
    template<typename F> class task_handle_task;
}

class task_group;
class structured_task_group;
#if TBB_PREVIEW_ISOLATED_TASK_GROUP && __TBB_TASK_ISOLATION
class isolated_task_group;
#endif

template<typename F>
class task_handle : internal::no_assign {
    template<typename _F> friend class internal::task_handle_task;
    friend class task_group;
    friend class structured_task_group;
#if TBB_PREVIEW_ISOLATED_TASK_GROUP && __TBB_TASK_ISOLATION
    friend class isolated_task_group;
#endif

    static const intptr_t scheduled = 0x1;

    F my_func;
    intptr_t my_state;

    void mark_scheduled () {
        // The check here is intentionally lax to avoid the impact of interlocked operation
        if ( my_state & scheduled )
            internal::throw_exception( internal::eid_invalid_multiple_scheduling );
        my_state |= scheduled;
    }
public:
    task_handle( const F& f ) : my_func(f), my_state(0) {}
#if __TBB_CPP11_RVALUE_REF_PRESENT
    task_handle( F&& f ) : my_func( std::move(f)), my_state(0) {}
#endif

    void operator() () const { my_func(); }
};

enum task_group_status {
    not_complete,
    complete,
    canceled
};

namespace internal {

template<typename F>
class task_handle_task : public task {
    task_handle<F>& my_handle;
    task* execute() __TBB_override {
        my_handle();
        return NULL;
    }
public:
    task_handle_task( task_handle<F>& h ) : my_handle(h) { h.mark_scheduled(); }
};

class task_group_base : internal::no_copy {
    class ref_count_guard : internal::no_copy {
        task& my_task;
    public:
        ref_count_guard(task& t) : my_task(t) {
            my_task.increment_ref_count();
        }
        ~ref_count_guard() {
            my_task.decrement_ref_count();
        }
    };
protected:
    empty_task* my_root;
    task_group_context my_context;

    template<typename F>
    task_group_status internal_run_and_wait( F& f ) {
        __TBB_TRY {
            if ( !my_context.is_group_execution_cancelled() ) {
                // We need to increase the reference count of the root task to notify waiters that
                // this task group has some work in progress.
                ref_count_guard guard(*my_root);
                f();
            }
        } __TBB_CATCH( ... ) {
            my_context.register_pending_exception();
        }
        return wait();
    }

    template<typename Task, typename F>
    task* prepare_task( __TBB_FORWARDING_REF(F) f ) {
        return new( task::allocate_additional_child_of(*my_root) ) Task( internal::forward<F>(f) );
    }

public:
    task_group_base( uintptr_t traits = 0 )
        : my_context(task_group_context::bound, task_group_context::default_traits | traits)
    {
        my_root = new( task::allocate_root(my_context) ) empty_task;
        my_root->set_ref_count(1);
    }

    ~task_group_base() __TBB_NOEXCEPT(false) {
        if( my_root->ref_count() > 1 ) {
#if __TBB_CPP17_UNCAUGHT_EXCEPTIONS_PRESENT
            bool stack_unwinding_in_progress = std::uncaught_exceptions() > 0;
#else
            bool stack_unwinding_in_progress = std::uncaught_exception();
#endif
            // Always attempt to do proper cleanup to avoid inevitable memory corruption
            // in case of missing wait (for the sake of better testability & debuggability)
            if ( !is_canceling() )
                cancel();
            __TBB_TRY {
                my_root->wait_for_all();
            } __TBB_CATCH (...) {
                task::destroy(*my_root);
                __TBB_RETHROW();
            }
            task::destroy(*my_root);
            if ( !stack_unwinding_in_progress )
                internal::throw_exception( internal::eid_missing_wait );
        }
        else {
            task::destroy(*my_root);
        }
    }

    template<typename F>
    void run( task_handle<F>& h ) {
        task::spawn( *prepare_task< internal::task_handle_task<F> >(h) );
    }

    task_group_status wait() {
        __TBB_TRY {
            my_root->wait_for_all();
        } __TBB_CATCH( ... ) {
            my_context.reset();
            __TBB_RETHROW();
        }
        if ( my_context.is_group_execution_cancelled() ) {
            // TODO: the reset method is not thread-safe. Ensure the correct behavior.
            my_context.reset();
            return canceled;
        }
        return complete;
    }

    bool is_canceling() {
        return my_context.is_group_execution_cancelled();
    }

    void cancel() {
        my_context.cancel_group_execution();
    }
}; // class task_group_base

} // namespace internal

class task_group : public internal::task_group_base {
public:
    task_group () : task_group_base( task_group_context::concurrent_wait ) {}

#if __SUNPRO_CC
    template<typename F>
    void run( task_handle<F>& h ) {
        internal_run< internal::task_handle_task<F> >( h );
    }
#else
    using task_group_base::run;
#endif

#if __TBB_CPP11_RVALUE_REF_PRESENT
    template<typename F>
    void run( F&& f ) {
        task::spawn( *prepare_task< internal::function_task< typename internal::strip<F>::type > >(std::forward<F>(f)) );
    }
#else
    template<typename F>
    void run(const F& f) {
        task::spawn( *prepare_task< internal::function_task<F> >(f) );
    }
#endif

    template<typename F>
    task_group_status run_and_wait( const F& f ) {
        return internal_run_and_wait<const F>( f );
    }

    // TODO: add task_handle rvalues support
    template<typename F>
    task_group_status run_and_wait( task_handle<F>& h ) {
        h.mark_scheduled();
        return internal_run_and_wait< task_handle<F> >( h );
    }
}; // class task_group

class __TBB_DEPRECATED structured_task_group : public internal::task_group_base {
public:
    // TODO: add task_handle rvalues support
    template<typename F>
    task_group_status run_and_wait ( task_handle<F>& h ) {
        h.mark_scheduled();
        return internal_run_and_wait< task_handle<F> >( h );
    }

    task_group_status wait() {
        task_group_status res = task_group_base::wait();
        my_root->set_ref_count(1);
        return res;
    }
}; // class structured_task_group

#if TBB_PREVIEW_ISOLATED_TASK_GROUP && __TBB_TASK_ISOLATION
namespace internal {
    using interface7::internal::delegate_base;
    using interface7::internal::isolate_within_arena;

    class spawn_delegate : public delegate_base {
        task* task_to_spawn;
        void operator()() const __TBB_override {
            task::spawn(*task_to_spawn);
        }
    public:
        spawn_delegate(task* a_task) : task_to_spawn(a_task) {}
    };

    class wait_delegate : public delegate_base {
        void operator()() const __TBB_override {
            status = tg.wait();
        }
    protected:
        task_group& tg;
        task_group_status& status;
    public:
        wait_delegate(task_group& a_group, task_group_status& tgs)
            : tg(a_group), status(tgs) {}
    };

    template<typename F>
    class run_wait_delegate : public wait_delegate {
        F& func;
        void operator()() const __TBB_override {
            status = tg.run_and_wait( func );
        }
    public:
        run_wait_delegate(task_group& a_group, F& a_func, task_group_status& tgs)
            : wait_delegate(a_group, tgs), func(a_func) {}
    };
} // namespace internal

class isolated_task_group : public task_group {
    intptr_t this_isolation() {
        return reinterpret_cast<intptr_t>(this);
    }
public:
    isolated_task_group () : task_group() {}

#if __TBB_CPP11_RVALUE_REF_PRESENT
    template<typename F>
    void run( F&& f ) {
        internal::spawn_delegate sd(
            prepare_task< internal::function_task< typename internal::strip<F>::type > >(std::forward<F>(f))
        );
        internal::isolate_within_arena( sd, this_isolation() );
    }
#else
    template<typename F>
    void run(const F& f) {
        internal::spawn_delegate sd( prepare_task< internal::function_task<F> >(f) );
        internal::isolate_within_arena( sd, this_isolation() );
    }
#endif

    template<typename F>
    task_group_status run_and_wait( const F& f ) {
        task_group_status result = not_complete;
        internal::run_wait_delegate< const F > rwd( *this, f, result );
        internal::isolate_within_arena( rwd, this_isolation() );
        __TBB_ASSERT( result!=not_complete, "premature exit from wait?" );
        return result;
    }

    // TODO: add task_handle rvalues support
    template<typename F>
    void run( task_handle<F>& h ) {
        internal::spawn_delegate sd( prepare_task< internal::task_handle_task<F> >(h) );
        internal::isolate_within_arena( sd, this_isolation() );
    }

    template<typename F>
    task_group_status run_and_wait ( task_handle<F>& h ) {
        task_group_status result = not_complete;
        internal::run_wait_delegate< task_handle<F> > rwd( *this, h, result );
        internal::isolate_within_arena( rwd, this_isolation() );
        __TBB_ASSERT( result!=not_complete, "premature exit from wait?" );
        return result;
    }

    task_group_status wait() {
        task_group_status result = not_complete;
        internal::wait_delegate wd( *this, result );
        internal::isolate_within_arena( wd, this_isolation() );
        __TBB_ASSERT( result!=not_complete, "premature exit from wait?" );
        return result;
    }
}; // class isolated_task_group
#endif // TBB_PREVIEW_ISOLATED_TASK_GROUP && __TBB_TASK_ISOLATION

inline
bool is_current_task_group_canceling() {
    return task::self().is_cancelled();
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
template<class F>
task_handle< typename internal::strip<F>::type > make_task( F&& f ) {
    return task_handle< typename internal::strip<F>::type >( std::forward<F>(f) );
}
#else
template<class F>
task_handle<F> make_task( const F& f ) {
    return task_handle<F>( f );
}
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

} // namespace tbb

#endif /* __TBB_TASK_GROUP_CONTEXT */

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_task_group_H_include_area

#endif /* __TBB_task_group_H */
