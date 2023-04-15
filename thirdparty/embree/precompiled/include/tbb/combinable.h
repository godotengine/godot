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

#ifndef __TBB_combinable_H
#define __TBB_combinable_H

#define __TBB_combinable_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "enumerable_thread_specific.h"
#include "cache_aligned_allocator.h"

namespace tbb {
/** \name combinable
    **/
//@{
//! Thread-local storage with optional reduction
/** @ingroup containers */
    template <typename T>
    class combinable {

    private:
        typedef typename tbb::cache_aligned_allocator<T> my_alloc;
        typedef typename tbb::enumerable_thread_specific<T, my_alloc, ets_no_key> my_ets_type;
        my_ets_type my_ets;

    public:

        combinable() { }

        template <typename finit>
        explicit combinable( finit _finit) : my_ets(_finit) { }

        //! destructor
        ~combinable() { }

        combinable( const combinable& other) : my_ets(other.my_ets) { }

#if __TBB_ETS_USE_CPP11
        combinable( combinable&& other) : my_ets( std::move(other.my_ets)) { }
#endif

        combinable & operator=( const combinable & other) {
            my_ets = other.my_ets;
            return *this;
        }

#if __TBB_ETS_USE_CPP11
        combinable & operator=( combinable && other) {
            my_ets=std::move(other.my_ets);
            return *this;
        }
#endif

        void clear() { my_ets.clear(); }

        T& local() { return my_ets.local(); }

        T& local(bool & exists) { return my_ets.local(exists); }

        // combine_func_t has signature T(T,T) or T(const T&, const T&)
        template <typename combine_func_t>
        T combine(combine_func_t f_combine) { return my_ets.combine(f_combine); }

        // combine_func_t has signature void(T) or void(const T&)
        template <typename combine_func_t>
        void combine_each(combine_func_t f_combine) { my_ets.combine_each(f_combine); }

    };
} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_combinable_H_include_area

#endif /* __TBB_combinable_H */
