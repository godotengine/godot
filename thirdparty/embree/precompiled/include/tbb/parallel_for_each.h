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

#ifndef __TBB_parallel_for_each_H
#define __TBB_parallel_for_each_H

#include "parallel_do.h"
#include "parallel_for.h"

namespace tbb {

//! @cond INTERNAL
namespace internal {
    // The class calls user function in operator()
    template <typename Function, typename Iterator>
    class parallel_for_each_body_do : internal::no_assign {
        const Function &my_func;
    public:
        parallel_for_each_body_do(const Function &_func) : my_func(_func) {}

        void operator()(typename std::iterator_traits<Iterator>::reference value) const {
            my_func(value);
        }
    };

    // The class calls user function in operator()
    template <typename Function, typename Iterator>
    class parallel_for_each_body_for : internal::no_assign {
        const Function &my_func;
    public:
        parallel_for_each_body_for(const Function &_func) : my_func(_func) {}

        void operator()(tbb::blocked_range<Iterator> range) const {
#if __INTEL_COMPILER
#pragma ivdep
#endif
            for(Iterator it = range.begin(), end = range.end(); it != end; ++it) {
                my_func(*it);
            }
        }
    };

    template<typename Iterator, typename Function, typename Generic>
    struct parallel_for_each_impl {
#if __TBB_TASK_GROUP_CONTEXT
        static void doit(Iterator first, Iterator last, const Function& f, task_group_context &context) {
            internal::parallel_for_each_body_do<Function, Iterator> body(f);
            tbb::parallel_do(first, last, body, context);
        }
#endif
        static void doit(Iterator first, Iterator last, const Function& f) {
            internal::parallel_for_each_body_do<Function, Iterator> body(f);
            tbb::parallel_do(first, last, body);
        }
    };
    template<typename Iterator, typename Function>
    struct parallel_for_each_impl<Iterator, Function, std::random_access_iterator_tag> {
#if __TBB_TASK_GROUP_CONTEXT
        static void doit(Iterator first, Iterator last, const Function& f, task_group_context &context) {
            internal::parallel_for_each_body_for<Function, Iterator> body(f);
            tbb::parallel_for(tbb::blocked_range<Iterator>(first, last), body, context);
        }
#endif
        static void doit(Iterator first, Iterator last, const Function& f) {
            internal::parallel_for_each_body_for<Function, Iterator> body(f);
            tbb::parallel_for(tbb::blocked_range<Iterator>(first, last), body);
        }
    };
} // namespace internal
//! @endcond

/** \name parallel_for_each
    **/
//@{
//! Calls function f for all items from [first, last) interval using user-supplied context
/** @ingroup algorithms */
#if __TBB_TASK_GROUP_CONTEXT
template<typename Iterator, typename Function>
void parallel_for_each(Iterator first, Iterator last, const Function& f, task_group_context &context) {
    internal::parallel_for_each_impl<Iterator, Function, typename std::iterator_traits<Iterator>::iterator_category>::doit(first, last, f, context);
}

//! Calls function f for all items from rng using user-supplied context
/** @ingroup algorithms */
template<typename Range, typename Function>
void parallel_for_each(Range& rng, const Function& f, task_group_context& context) {
    parallel_for_each(tbb::internal::first(rng), tbb::internal::last(rng), f, context);
}

//! Calls function f for all items from const rng user-supplied context
/** @ingroup algorithms */
template<typename Range, typename Function>
void parallel_for_each(const Range& rng, const Function& f, task_group_context& context) {
    parallel_for_each(tbb::internal::first(rng), tbb::internal::last(rng), f, context);
}
#endif /* __TBB_TASK_GROUP_CONTEXT */

//! Uses default context
template<typename Iterator, typename Function>
void parallel_for_each(Iterator first, Iterator last, const Function& f) {
    internal::parallel_for_each_impl<Iterator, Function, typename std::iterator_traits<Iterator>::iterator_category>::doit(first, last, f);
}

//! Uses default context
template<typename Range, typename Function>
void parallel_for_each(Range& rng, const Function& f) {
    parallel_for_each(tbb::internal::first(rng), tbb::internal::last(rng), f);
}

//! Uses default context
template<typename Range, typename Function>
void parallel_for_each(const Range& rng, const Function& f) {
    parallel_for_each(tbb::internal::first(rng), tbb::internal::last(rng), f);
}

//@}

} // namespace

#endif /* __TBB_parallel_for_each_H */
