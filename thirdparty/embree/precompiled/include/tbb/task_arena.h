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

#ifndef __TBB_task_arena_H
#define __TBB_task_arena_H

#define __TBB_task_arena_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "task.h"
#include "tbb_exception.h"
#include "internal/_template_helpers.h"
#if __TBB_NUMA_SUPPORT
#include "info.h"
#endif /*__TBB_NUMA_SUPPORT*/
#if TBB_USE_THREADING_TOOLS
#include "atomic.h" // for as_atomic
#endif
#include "aligned_space.h"

namespace tbb {

namespace this_task_arena {
    int max_concurrency();
} // namespace this_task_arena

//! @cond INTERNAL
namespace internal {
    //! Internal to library. Should not be used by clients.
    /** @ingroup task_scheduling */
    class arena;
    class task_scheduler_observer_v3;
} // namespace internal
//! @endcond

namespace interface7 {
class task_arena;

//! @cond INTERNAL
namespace internal {
using namespace tbb::internal; //e.g. function_task from task.h

class delegate_base : no_assign {
public:
    virtual void operator()() const = 0;
    virtual ~delegate_base() {}
};

// If decltype is available, the helper detects the return type of functor of specified type,
// otherwise it defines the void type.
template <typename F>
struct return_type_or_void {
#if __TBB_CPP11_DECLTYPE_PRESENT && !__TBB_CPP11_DECLTYPE_OF_FUNCTION_RETURN_TYPE_BROKEN
    typedef decltype(declval<F>()()) type;
#else
    typedef void type;
#endif
};

template<typename F, typename R>
class delegated_function : public delegate_base {
    F &my_func;
    tbb::aligned_space<R> my_return_storage;
    // The function should be called only once.
    void operator()() const __TBB_override {
        new (my_return_storage.begin()) R(my_func());
    }
public:
    delegated_function(F& f) : my_func(f) {}
    // The function can be called only after operator() and only once.
    R consume_result() const {
        return tbb::internal::move(*(my_return_storage.begin()));
    }
    ~delegated_function() {
        my_return_storage.begin()->~R();
    }
};

template<typename F>
class delegated_function<F,void> : public delegate_base {
    F &my_func;
    void operator()() const __TBB_override {
        my_func();
    }
public:
    delegated_function(F& f) : my_func(f) {}
    void consume_result() const {}

    friend class task_arena_base;
};

class task_arena_base {
#if __TBB_NUMA_SUPPORT
public:
    // TODO: consider version approach to resolve backward compatibility potential issues.
    struct constraints {
        constraints(numa_node_id id = automatic, int maximal_concurrency = automatic)
            : numa_id(id)
            , max_concurrency(maximal_concurrency)
        {}
        numa_node_id numa_id;
        int max_concurrency;
    };
#endif /*__TBB_NUMA_SUPPORT*/
protected:
    //! NULL if not currently initialized.
    internal::arena* my_arena;

#if __TBB_TASK_GROUP_CONTEXT
    //! default context of the arena
    task_group_context *my_context;
#endif

    //! Concurrency level for deferred initialization
    int my_max_concurrency;

    //! Reserved master slots
    unsigned my_master_slots;

    //! Special settings
    intptr_t my_version_and_traits;

    bool my_initialized;

#if __TBB_NUMA_SUPPORT
    //! The NUMA node index to which the arena will be attached
    numa_node_id my_numa_id;

    // Do not access my_numa_id without the following runtime check.
    // Despite my_numa_id is accesible, it does not exist in task_arena_base on user side
    // if TBB_PREVIEW_NUMA_SUPPORT macro is not defined by the user. To be sure that
    // my_numa_id exists in task_arena_base layout we check the traits.
    // TODO: Consider increasing interface version for task_arena_base instead of this runtime check.
    numa_node_id numa_id() {
        return (my_version_and_traits & numa_support_flag) == numa_support_flag ? my_numa_id : automatic;
    }
#endif

    enum {
        default_flags = 0
#if __TBB_TASK_GROUP_CONTEXT
        | (task_group_context::default_traits & task_group_context::exact_exception)  // 0 or 1 << 16
        , exact_exception_flag = task_group_context::exact_exception // used to specify flag for context directly
#endif
#if __TBB_NUMA_SUPPORT
        , numa_support_flag = 1
#endif
    };

    task_arena_base(int max_concurrency, unsigned reserved_for_masters)
        : my_arena(0)
#if __TBB_TASK_GROUP_CONTEXT
        , my_context(0)
#endif
        , my_max_concurrency(max_concurrency)
        , my_master_slots(reserved_for_masters)
#if __TBB_NUMA_SUPPORT
        , my_version_and_traits(default_flags | numa_support_flag)
#else
        , my_version_and_traits(default_flags)
#endif
        , my_initialized(false)
#if __TBB_NUMA_SUPPORT
        , my_numa_id(automatic)
#endif
        {}

#if __TBB_NUMA_SUPPORT
    task_arena_base(const constraints& constraints_, unsigned reserved_for_masters)
        : my_arena(0)
#if __TBB_TASK_GROUP_CONTEXT
        , my_context(0)
#endif
        , my_max_concurrency(constraints_.max_concurrency)
        , my_master_slots(reserved_for_masters)
        , my_version_and_traits(default_flags | numa_support_flag)
        , my_initialized(false)
        , my_numa_id(constraints_.numa_id )
        {}
#endif /*__TBB_NUMA_SUPPORT*/

    void __TBB_EXPORTED_METHOD internal_initialize();
    void __TBB_EXPORTED_METHOD internal_terminate();
    void __TBB_EXPORTED_METHOD internal_attach();
    void __TBB_EXPORTED_METHOD internal_enqueue( task&, intptr_t ) const;
    void __TBB_EXPORTED_METHOD internal_execute( delegate_base& ) const;
    void __TBB_EXPORTED_METHOD internal_wait() const;
    static int __TBB_EXPORTED_FUNC internal_current_slot();
    static int __TBB_EXPORTED_FUNC internal_max_concurrency( const task_arena * );
public:
    //! Typedef for number of threads that is automatic.
    static const int automatic = -1;
    static const int not_initialized = -2;

};

#if __TBB_TASK_ISOLATION
void __TBB_EXPORTED_FUNC isolate_within_arena( delegate_base& d, intptr_t isolation = 0 );

template<typename R, typename F>
R isolate_impl(F& f) {
    delegated_function<F, R> d(f);
    isolate_within_arena(d);
    return d.consume_result();
}
#endif /* __TBB_TASK_ISOLATION */
} // namespace internal
//! @endcond

/** 1-to-1 proxy representation class of scheduler's arena
 * Constructors set up settings only, real construction is deferred till the first method invocation
 * Destructor only removes one of the references to the inner arena representation.
 * Final destruction happens when all the references (and the work) are gone.
 */
class task_arena : public internal::task_arena_base {
    friend class tbb::internal::task_scheduler_observer_v3;
    friend void task::enqueue(task&, task_arena&
#if __TBB_TASK_PRIORITY
        , priority_t
#endif
    );
    friend int tbb::this_task_arena::max_concurrency();
    void mark_initialized() {
        __TBB_ASSERT( my_arena, "task_arena initialization is incomplete" );
#if __TBB_TASK_GROUP_CONTEXT
        __TBB_ASSERT( my_context, "task_arena initialization is incomplete" );
#endif
#if TBB_USE_THREADING_TOOLS
        // Actual synchronization happens in internal_initialize & internal_attach.
        // The race on setting my_initialized is benign, but should be hidden from Intel(R) Inspector
        internal::as_atomic(my_initialized).fetch_and_store<release>(true);
#else
        my_initialized = true;
#endif
    }

    template<typename F>
    void enqueue_impl( __TBB_FORWARDING_REF(F) f
#if __TBB_TASK_PRIORITY
        , priority_t p = priority_t(0)
#endif
    ) {
#if !__TBB_TASK_PRIORITY
        intptr_t p = 0;
#endif
        initialize();
#if __TBB_TASK_GROUP_CONTEXT
        internal_enqueue(*new(task::allocate_root(*my_context)) internal::function_task< typename internal::strip<F>::type >(internal::forward<F>(f)), p);
#else
        internal_enqueue(*new(task::allocate_root()) internal::function_task< typename internal::strip<F>::type >(internal::forward<F>(f)), p);
#endif /* __TBB_TASK_GROUP_CONTEXT */
    }

    template<typename R, typename F>
    R execute_impl(F& f) {
        initialize();
        internal::delegated_function<F, R> d(f);
        internal_execute(d);
        return d.consume_result();
    }

public:
    //! Creates task_arena with certain concurrency limits
    /** Sets up settings only, real construction is deferred till the first method invocation
     *  @arg max_concurrency specifies total number of slots in arena where threads work
     *  @arg reserved_for_masters specifies number of slots to be used by master threads only.
     *       Value of 1 is default and reflects behavior of implicit arenas.
     **/
    task_arena(int max_concurrency_ = automatic, unsigned reserved_for_masters = 1)
        : task_arena_base(max_concurrency_, reserved_for_masters)
    {}

#if __TBB_NUMA_SUPPORT
    //! Creates task arena pinned to certain NUMA node
    task_arena(const constraints& constraints_, unsigned reserved_for_masters = 1)
        : task_arena_base(constraints_, reserved_for_masters)
    {}

    //! Copies settings from another task_arena
    task_arena(const task_arena &s) // copy settings but not the reference or instance
        : task_arena_base(constraints(s.my_numa_id, s.my_max_concurrency), s.my_master_slots)
    {}
#else
    //! Copies settings from another task_arena
    task_arena(const task_arena &s) // copy settings but not the reference or instance
        : task_arena_base(s.my_max_concurrency, s.my_master_slots)
    {}
#endif /*__TBB_NUMA_SUPPORT*/

    //! Tag class used to indicate the "attaching" constructor
    struct attach {};

    //! Creates an instance of task_arena attached to the current arena of the thread
    explicit task_arena( attach )
        : task_arena_base(automatic, 1) // use default settings if attach fails
    {
        internal_attach();
        if( my_arena ) my_initialized = true;
    }

    //! Forces allocation of the resources for the task_arena as specified in constructor arguments
    inline void initialize() {
        if( !my_initialized ) {
            internal_initialize();
            mark_initialized();
        }
    }

    //! Overrides concurrency level and forces initialization of internal representation
    inline void initialize(int max_concurrency_, unsigned reserved_for_masters = 1) {
        // TODO: decide if this call must be thread-safe
        __TBB_ASSERT(!my_arena, "Impossible to modify settings of an already initialized task_arena");
        if( !my_initialized ) {
            my_max_concurrency = max_concurrency_;
            my_master_slots = reserved_for_masters;
            initialize();
        }
    }

#if __TBB_NUMA_SUPPORT
    inline void initialize(constraints constraints_, unsigned reserved_for_masters = 1) {
        // TODO: decide if this call must be thread-safe
        __TBB_ASSERT(!my_arena, "Impossible to modify settings of an already initialized task_arena");
        if( !my_initialized ) {
            my_numa_id = constraints_.numa_id;
            my_max_concurrency = constraints_.max_concurrency;
            my_master_slots = reserved_for_masters;
            initialize();
        }
    }
#endif /*__TBB_NUMA_SUPPORT*/

    //! Attaches this instance to the current arena of the thread
    inline void initialize(attach) {
        // TODO: decide if this call must be thread-safe
        __TBB_ASSERT(!my_arena, "Impossible to modify settings of an already initialized task_arena");
        if( !my_initialized ) {
            internal_attach();
            if ( !my_arena ) internal_initialize();
            mark_initialized();
        }
    }

    //! Removes the reference to the internal arena representation.
    //! Not thread safe wrt concurrent invocations of other methods.
    inline void terminate() {
        if( my_initialized ) {
            internal_terminate();
            my_initialized = false;
        }
    }

    //! Removes the reference to the internal arena representation, and destroys the external object.
    //! Not thread safe wrt concurrent invocations of other methods.
    ~task_arena() {
        terminate();
    }

    //! Returns true if the arena is active (initialized); false otherwise.
    //! The name was chosen to match a task_scheduler_init method with the same semantics.
    bool is_active() const { return my_initialized; }

    //! Enqueues a task into the arena to process a functor, and immediately returns.
    //! Does not require the calling thread to join the arena

#if __TBB_CPP11_RVALUE_REF_PRESENT
    template<typename F>
    void enqueue( F&& f ) {
        enqueue_impl(std::forward<F>(f));
    }
#else
    template<typename F>
    void enqueue( const F& f ) {
        enqueue_impl(f);
    }
#endif

#if __TBB_TASK_PRIORITY
    //! Enqueues a task with priority p into the arena to process a functor f, and immediately returns.
    //! Does not require the calling thread to join the arena
    template<typename F>
#if __TBB_CPP11_RVALUE_REF_PRESENT
    __TBB_DEPRECATED void enqueue( F&& f, priority_t p ) {
#if __TBB_PREVIEW_CRITICAL_TASKS
        __TBB_ASSERT(p == priority_low || p == priority_normal || p == priority_high
                     || p == internal::priority_critical, "Invalid priority level value");
#else
        __TBB_ASSERT(p == priority_low || p == priority_normal || p == priority_high, "Invalid priority level value");
#endif
        enqueue_impl(std::forward<F>(f), p);
    }
#else
    __TBB_DEPRECATED void enqueue( const F& f, priority_t p ) {
#if __TBB_PREVIEW_CRITICAL_TASKS
        __TBB_ASSERT(p == priority_low || p == priority_normal || p == priority_high
                     || p == internal::priority_critical, "Invalid priority level value");
#else
        __TBB_ASSERT(p == priority_low || p == priority_normal || p == priority_high, "Invalid priority level value");
#endif
        enqueue_impl(f,p);
    }
#endif
#endif// __TBB_TASK_PRIORITY

    //! Joins the arena and executes a mutable functor, then returns
    //! If not possible to join, wraps the functor into a task, enqueues it and waits for task completion
    //! Can decrement the arena demand for workers, causing a worker to leave and free a slot to the calling thread
    //! Since C++11, the method returns the value returned by functor (prior to C++11 it returns void).
    template<typename F>
    typename internal::return_type_or_void<F>::type execute(F& f) {
        return execute_impl<typename internal::return_type_or_void<F>::type>(f);
    }

    //! Joins the arena and executes a constant functor, then returns
    //! If not possible to join, wraps the functor into a task, enqueues it and waits for task completion
    //! Can decrement the arena demand for workers, causing a worker to leave and free a slot to the calling thread
    //! Since C++11, the method returns the value returned by functor (prior to C++11 it returns void).
    template<typename F>
    typename internal::return_type_or_void<F>::type execute(const F& f) {
        return execute_impl<typename internal::return_type_or_void<F>::type>(f);
    }

#if __TBB_EXTRA_DEBUG
    //! Wait for all work in the arena to be completed
    //! Even submitted by other application threads
    //! Joins arena if/when possible (in the same way as execute())
    void debug_wait_until_empty() {
        initialize();
        internal_wait();
    }
#endif //__TBB_EXTRA_DEBUG

    //! Returns the index, aka slot number, of the calling thread in its current arena
    //! This method is deprecated and replaced with this_task_arena::current_thread_index()
    inline static int current_thread_index() {
        return internal_current_slot();
    }

    //! Returns the maximal number of threads that can work inside the arena
    inline int max_concurrency() const {
        // Handle special cases inside the library
        return (my_max_concurrency>1) ? my_max_concurrency : internal_max_concurrency(this);
    }
};

namespace this_task_arena {
#if __TBB_TASK_ISOLATION
    //! Executes a mutable functor in isolation within the current task arena.
    //! Since C++11, the method returns the value returned by functor (prior to C++11 it returns void).
    template<typename F>
    typename internal::return_type_or_void<F>::type isolate(F& f) {
        return internal::isolate_impl<typename internal::return_type_or_void<F>::type>(f);
    }

    //! Executes a constant functor in isolation within the current task arena.
    //! Since C++11, the method returns the value returned by functor (prior to C++11 it returns void).
    template<typename F>
    typename internal::return_type_or_void<F>::type isolate(const F& f) {
        return internal::isolate_impl<typename internal::return_type_or_void<F>::type>(f);
    }
#endif /* __TBB_TASK_ISOLATION */
} // namespace this_task_arena
} // namespace interfaceX

using interface7::task_arena;

namespace this_task_arena {
    using namespace interface7::this_task_arena;

    //! Returns the index, aka slot number, of the calling thread in its current arena
    inline int current_thread_index() {
        int idx = tbb::task_arena::current_thread_index();
        return idx == -1 ? tbb::task_arena::not_initialized : idx;
    }

    //! Returns the maximal number of threads that can work inside the arena
    inline int max_concurrency() {
        return tbb::task_arena::internal_max_concurrency(NULL);
    }
} // namespace this_task_arena

//! Enqueue task in task_arena
#if __TBB_TASK_PRIORITY
void task::enqueue( task& t, task_arena& arena, priority_t p ) {
#else
void task::enqueue( task& t, task_arena& arena ) {
    intptr_t p = 0;
#endif
    arena.initialize();
    //! Note: the context of the task may differ from the context instantiated by task_arena
    arena.internal_enqueue(t, p);
}
} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_task_arena_H_include_area

#endif /* __TBB_task_arena_H */
