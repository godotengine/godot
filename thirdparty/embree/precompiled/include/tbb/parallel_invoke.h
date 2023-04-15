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

#ifndef __TBB_parallel_invoke_H
#define __TBB_parallel_invoke_H

#define __TBB_parallel_invoke_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "task.h"
#include "tbb_profiling.h"

#if __TBB_VARIADIC_PARALLEL_INVOKE
    #include <utility> // std::forward
#endif

namespace tbb {

#if !__TBB_TASK_GROUP_CONTEXT
    /** Dummy to avoid cluttering the bulk of the header with enormous amount of ifdefs. **/
    struct task_group_context {
        task_group_context(tbb::internal::string_index){}
    };
#endif /* __TBB_TASK_GROUP_CONTEXT */

//! @cond INTERNAL
namespace internal {
    // Simple task object, executing user method
    template<typename function>
    class function_invoker : public task{
    public:
        function_invoker(const function& _function) : my_function(_function) {}
    private:
        const function &my_function;
        task* execute() __TBB_override
        {
            my_function();
            return NULL;
        }
    };

    // The class spawns two or three child tasks
    template <size_t N, typename function1, typename function2, typename function3>
    class spawner : public task {
    private:
        const function1& my_func1;
        const function2& my_func2;
        const function3& my_func3;
        bool is_recycled;

        task* execute () __TBB_override {
            if(is_recycled){
                return NULL;
            }else{
                __TBB_ASSERT(N==2 || N==3, "Number of arguments passed to spawner is wrong");
                set_ref_count(N);
                recycle_as_safe_continuation();
                internal::function_invoker<function2>* invoker2 = new (allocate_child()) internal::function_invoker<function2>(my_func2);
                __TBB_ASSERT(invoker2, "Child task allocation failed");
                spawn(*invoker2);
                size_t n = N; // To prevent compiler warnings
                if (n>2) {
                    internal::function_invoker<function3>* invoker3 = new (allocate_child()) internal::function_invoker<function3>(my_func3);
                    __TBB_ASSERT(invoker3, "Child task allocation failed");
                    spawn(*invoker3);
                }
                my_func1();
                is_recycled = true;
                return NULL;
            }
        } // execute

    public:
        spawner(const function1& _func1, const function2& _func2, const function3& _func3) : my_func1(_func1), my_func2(_func2), my_func3(_func3), is_recycled(false) {}
    };

    // Creates and spawns child tasks
    class parallel_invoke_helper : public empty_task {
    public:
        // Dummy functor class
        class parallel_invoke_noop {
        public:
            void operator() () const {}
        };
        // Creates a helper object with user-defined number of children expected
        parallel_invoke_helper(int number_of_children)
        {
            set_ref_count(number_of_children + 1);
        }

#if __TBB_VARIADIC_PARALLEL_INVOKE
        void add_children() {}
        void add_children(tbb::task_group_context&) {}

        template <typename function>
        void add_children(function&& _func)
        {
            internal::function_invoker<function>* invoker = new (allocate_child()) internal::function_invoker<function>(std::forward<function>(_func));
            __TBB_ASSERT(invoker, "Child task allocation failed");
            spawn(*invoker);
        }

        template<typename function>
        void add_children(function&& _func, tbb::task_group_context&)
        {
            add_children(std::forward<function>(_func));
        }

        // Adds child(ren) task(s) and spawns them
        template <typename function1, typename function2, typename... function>
        void add_children(function1&& _func1, function2&& _func2, function&&... _func)
        {
            // The third argument is dummy, it is ignored actually.
            parallel_invoke_noop noop;
            typedef internal::spawner<2, function1, function2, parallel_invoke_noop> spawner_type;
            spawner_type & sub_root = *new(allocate_child()) spawner_type(std::forward<function1>(_func1), std::forward<function2>(_func2), noop);
            spawn(sub_root);
            add_children(std::forward<function>(_func)...);
        }
#else
        // Adds child task and spawns it
        template <typename function>
        void add_children (const function &_func)
        {
            internal::function_invoker<function>* invoker = new (allocate_child()) internal::function_invoker<function>(_func);
            __TBB_ASSERT(invoker, "Child task allocation failed");
            spawn(*invoker);
        }

        // Adds a task with multiple child tasks and spawns it
        // two arguments
        template <typename function1, typename function2>
        void add_children (const function1& _func1, const function2& _func2)
        {
            // The third argument is dummy, it is ignored actually.
            parallel_invoke_noop noop;
            internal::spawner<2, function1, function2, parallel_invoke_noop>& sub_root = *new(allocate_child())internal::spawner<2, function1, function2, parallel_invoke_noop>(_func1, _func2, noop);
            spawn(sub_root);
        }
        // three arguments
        template <typename function1, typename function2, typename function3>
        void add_children (const function1& _func1, const function2& _func2, const function3& _func3)
        {
            internal::spawner<3, function1, function2, function3>& sub_root = *new(allocate_child())internal::spawner<3, function1, function2, function3>(_func1, _func2, _func3);
            spawn(sub_root);
        }
#endif // __TBB_VARIADIC_PARALLEL_INVOKE

        // Waits for all child tasks
        template <typename F0>
        void run_and_finish(const F0& f0)
        {
            internal::function_invoker<F0>* invoker = new (allocate_child()) internal::function_invoker<F0>(f0);
            __TBB_ASSERT(invoker, "Child task allocation failed");
            spawn_and_wait_for_all(*invoker);
        }
    };
    // The class destroys root if exception occurred as well as in normal case
    class parallel_invoke_cleaner: internal::no_copy {
    public:
#if __TBB_TASK_GROUP_CONTEXT
        parallel_invoke_cleaner(int number_of_children, tbb::task_group_context& context)
            : root(*new(task::allocate_root(context)) internal::parallel_invoke_helper(number_of_children))
#else
        parallel_invoke_cleaner(int number_of_children, tbb::task_group_context&)
            : root(*new(task::allocate_root()) internal::parallel_invoke_helper(number_of_children))
#endif /* !__TBB_TASK_GROUP_CONTEXT */
        {}

        ~parallel_invoke_cleaner(){
            root.destroy(root);
        }
        internal::parallel_invoke_helper& root;
    };

#if __TBB_VARIADIC_PARALLEL_INVOKE
//  Determine whether the last parameter in a pack is task_group_context
    template<typename... T> struct impl_selector; // to workaround a GCC bug

    template<typename T1, typename... T> struct impl_selector<T1, T...> {
        typedef typename impl_selector<T...>::type type;
    };

    template<typename T> struct impl_selector<T> {
        typedef false_type type;
    };
    template<> struct impl_selector<task_group_context&> {
        typedef true_type  type;
    };

    // Select task_group_context parameter from the back of a pack
    inline task_group_context& get_context( task_group_context& tgc ) { return tgc; }

    template<typename T1, typename... T>
    task_group_context& get_context( T1&& /*ignored*/, T&&... t )
    { return get_context( std::forward<T>(t)... ); }

    // task_group_context is known to be at the back of the parameter pack
    template<typename F0, typename F1, typename... F>
    void parallel_invoke_impl(true_type, F0&& f0, F1&& f1, F&&... f) {
        __TBB_STATIC_ASSERT(sizeof...(F)>0, "Variadic parallel_invoke implementation broken?");
        // # of child tasks: f0, f1, and a task for each two elements of the pack except the last
        const size_t number_of_children = 2 + sizeof...(F)/2;
        parallel_invoke_cleaner cleaner(number_of_children, get_context(std::forward<F>(f)...));
        parallel_invoke_helper& root = cleaner.root;

        root.add_children(std::forward<F>(f)...);
        root.add_children(std::forward<F1>(f1));
        root.run_and_finish(std::forward<F0>(f0));
    }

    // task_group_context is not in the pack, needs to be added
    template<typename F0, typename F1, typename... F>
    void parallel_invoke_impl(false_type, F0&& f0, F1&& f1, F&&... f) {
        tbb::task_group_context context(PARALLEL_INVOKE);
        // Add context to the arguments, and redirect to the other overload
        parallel_invoke_impl(true_type(), std::forward<F0>(f0), std::forward<F1>(f1), std::forward<F>(f)..., context);
    }
#endif
} // namespace internal
//! @endcond

/** \name parallel_invoke
    **/
//@{
//! Executes a list of tasks in parallel and waits for all tasks to complete.
/** @ingroup algorithms */

#if __TBB_VARIADIC_PARALLEL_INVOKE

// parallel_invoke for two or more arguments via variadic templates
// presence of task_group_context is defined automatically
template<typename F0, typename F1, typename... F>
void parallel_invoke(F0&& f0, F1&& f1, F&&... f) {
    typedef typename internal::impl_selector<internal::false_type, F...>::type selector_type;
    internal::parallel_invoke_impl(selector_type(), std::forward<F0>(f0), std::forward<F1>(f1), std::forward<F>(f)...);
}

#else

// parallel_invoke with user-defined context
// two arguments
template<typename F0, typename F1 >
void parallel_invoke(const F0& f0, const F1& f1, tbb::task_group_context& context) {
    internal::parallel_invoke_cleaner cleaner(2, context);
    internal::parallel_invoke_helper& root = cleaner.root;

    root.add_children(f1);

    root.run_and_finish(f0);
}

// three arguments
template<typename F0, typename F1, typename F2 >
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, tbb::task_group_context& context) {
    internal::parallel_invoke_cleaner cleaner(3, context);
    internal::parallel_invoke_helper& root = cleaner.root;

    root.add_children(f2);
    root.add_children(f1);

    root.run_and_finish(f0);
}

// four arguments
template<typename F0, typename F1, typename F2, typename F3>
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3,
                     tbb::task_group_context& context)
{
    internal::parallel_invoke_cleaner cleaner(4, context);
    internal::parallel_invoke_helper& root = cleaner.root;

    root.add_children(f3);
    root.add_children(f2);
    root.add_children(f1);

    root.run_and_finish(f0);
}

// five arguments
template<typename F0, typename F1, typename F2, typename F3, typename F4 >
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3, const F4& f4,
                     tbb::task_group_context& context)
{
    internal::parallel_invoke_cleaner cleaner(3, context);
    internal::parallel_invoke_helper& root = cleaner.root;

    root.add_children(f4, f3);
    root.add_children(f2, f1);

    root.run_and_finish(f0);
}

// six arguments
template<typename F0, typename F1, typename F2, typename F3, typename F4, typename F5>
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3, const F4& f4, const F5& f5,
                     tbb::task_group_context& context)
{
    internal::parallel_invoke_cleaner cleaner(3, context);
    internal::parallel_invoke_helper& root = cleaner.root;

    root.add_children(f5, f4, f3);
    root.add_children(f2, f1);

    root.run_and_finish(f0);
}

// seven arguments
template<typename F0, typename F1, typename F2, typename F3, typename F4, typename F5, typename F6>
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3, const F4& f4,
                     const F5& f5, const F6& f6,
                     tbb::task_group_context& context)
{
    internal::parallel_invoke_cleaner cleaner(3, context);
    internal::parallel_invoke_helper& root = cleaner.root;

    root.add_children(f6, f5, f4);
    root.add_children(f3, f2, f1);

    root.run_and_finish(f0);
}

// eight arguments
template<typename F0, typename F1, typename F2, typename F3, typename F4,
         typename F5, typename F6, typename F7>
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3, const F4& f4,
                     const F5& f5, const F6& f6, const F7& f7,
                     tbb::task_group_context& context)
{
    internal::parallel_invoke_cleaner cleaner(4, context);
    internal::parallel_invoke_helper& root = cleaner.root;

    root.add_children(f7, f6, f5);
    root.add_children(f4, f3);
    root.add_children(f2, f1);

    root.run_and_finish(f0);
}

// nine arguments
template<typename F0, typename F1, typename F2, typename F3, typename F4,
         typename F5, typename F6, typename F7, typename F8>
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3, const F4& f4,
                     const F5& f5, const F6& f6, const F7& f7, const F8& f8,
                     tbb::task_group_context& context)
{
    internal::parallel_invoke_cleaner cleaner(4, context);
    internal::parallel_invoke_helper& root = cleaner.root;

    root.add_children(f8, f7, f6);
    root.add_children(f5, f4, f3);
    root.add_children(f2, f1);

    root.run_and_finish(f0);
}

// ten arguments
template<typename F0, typename F1, typename F2, typename F3, typename F4,
         typename F5, typename F6, typename F7, typename F8, typename F9>
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3, const F4& f4,
                     const F5& f5, const F6& f6, const F7& f7, const F8& f8, const F9& f9,
                     tbb::task_group_context& context)
{
    internal::parallel_invoke_cleaner cleaner(4, context);
    internal::parallel_invoke_helper& root = cleaner.root;

    root.add_children(f9, f8, f7);
    root.add_children(f6, f5, f4);
    root.add_children(f3, f2, f1);

    root.run_and_finish(f0);
}

// two arguments
template<typename F0, typename F1>
void parallel_invoke(const F0& f0, const F1& f1) {
    task_group_context context(internal::PARALLEL_INVOKE);
    parallel_invoke<F0, F1>(f0, f1, context);
}
// three arguments
template<typename F0, typename F1, typename F2>
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2) {
    task_group_context context(internal::PARALLEL_INVOKE);
    parallel_invoke<F0, F1, F2>(f0, f1, f2, context);
}
// four arguments
template<typename F0, typename F1, typename F2, typename F3 >
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3) {
    task_group_context context(internal::PARALLEL_INVOKE);
    parallel_invoke<F0, F1, F2, F3>(f0, f1, f2, f3, context);
}
// five arguments
template<typename F0, typename F1, typename F2, typename F3, typename F4>
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3, const F4& f4) {
    task_group_context context(internal::PARALLEL_INVOKE);
    parallel_invoke<F0, F1, F2, F3, F4>(f0, f1, f2, f3, f4, context);
}
// six arguments
template<typename F0, typename F1, typename F2, typename F3, typename F4, typename F5>
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3, const F4& f4, const F5& f5) {
    task_group_context context(internal::PARALLEL_INVOKE);
    parallel_invoke<F0, F1, F2, F3, F4, F5>(f0, f1, f2, f3, f4, f5, context);
}
// seven arguments
template<typename F0, typename F1, typename F2, typename F3, typename F4, typename F5, typename F6>
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3, const F4& f4,
                     const F5& f5, const F6& f6)
{
    task_group_context context(internal::PARALLEL_INVOKE);
    parallel_invoke<F0, F1, F2, F3, F4, F5, F6>(f0, f1, f2, f3, f4, f5, f6, context);
}
// eight arguments
template<typename F0, typename F1, typename F2, typename F3, typename F4,
         typename F5, typename F6, typename F7>
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3, const F4& f4,
                     const F5& f5, const F6& f6, const F7& f7)
{
    task_group_context context(internal::PARALLEL_INVOKE);
    parallel_invoke<F0, F1, F2, F3, F4, F5, F6, F7>(f0, f1, f2, f3, f4, f5, f6, f7, context);
}
// nine arguments
template<typename F0, typename F1, typename F2, typename F3, typename F4,
         typename F5, typename F6, typename F7, typename F8>
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3, const F4& f4,
                     const F5& f5, const F6& f6, const F7& f7, const F8& f8)
{
    task_group_context context(internal::PARALLEL_INVOKE);
    parallel_invoke<F0, F1, F2, F3, F4, F5, F6, F7, F8>(f0, f1, f2, f3, f4, f5, f6, f7, f8, context);
}
// ten arguments
template<typename F0, typename F1, typename F2, typename F3, typename F4,
         typename F5, typename F6, typename F7, typename F8, typename F9>
void parallel_invoke(const F0& f0, const F1& f1, const F2& f2, const F3& f3, const F4& f4,
                     const F5& f5, const F6& f6, const F7& f7, const F8& f8, const F9& f9)
{
    task_group_context context(internal::PARALLEL_INVOKE);
    parallel_invoke<F0, F1, F2, F3, F4, F5, F6, F7, F8, F9>(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, context);
}
#endif // __TBB_VARIADIC_PARALLEL_INVOKE
//@}

} // namespace

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_parallel_invoke_H_include_area

#endif /* __TBB_parallel_invoke_H */
