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

#include "../../tbb/internal/_deprecated_header_message_guard.h"

#if !defined(__TBB_show_deprecation_message_parallel_for_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_parallel_for_H
#pragma message("TBB Warning: serial/tbb/parallel_for.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_SERIAL_parallel_for_H
#define __TBB_SERIAL_parallel_for_H

#include "tbb_annotate.h"

#ifndef __TBB_NORMAL_EXECUTION
#include "tbb/blocked_range.h"
#include "tbb/partitioner.h"
#endif

#if TBB_USE_EXCEPTIONS
#include <stdexcept>
#include <string> // required to construct std exception classes
#else
#include <cstdlib>
#include <iostream>
#endif

namespace tbb {
namespace serial {
namespace interface9 {

// parallel_for serial annotated implementation

template< typename Range, typename Body, typename Partitioner >
class start_for : tbb::internal::no_copy {
    Range my_range;
    const Body my_body;
    typename Partitioner::task_partition_type my_partition;
    void execute();

    //! Constructor for root task.
    start_for( const Range& range, const Body& body, Partitioner& partitioner ) :
        my_range( range ),
        my_body( body ),
        my_partition( partitioner )
    {
    }

    //! Splitting constructor used to generate children.
    /** this becomes left child.  Newly constructed object is right child. */
    start_for( start_for& parent_, typename Partitioner::split_type& split_obj ) :
        my_range( parent_.my_range, split_obj ),
        my_body( parent_.my_body ),
        my_partition( parent_.my_partition, split_obj )
    {
    }

public:
    static void run(  const Range& range, const Body& body, Partitioner& partitioner ) {
        if( !range.empty() ) {
            ANNOTATE_SITE_BEGIN( tbb_parallel_for );
            {
                start_for a( range, body, partitioner );
                a.execute();
            }
            ANNOTATE_SITE_END( tbb_parallel_for );
        }
    }
};

template< typename Range, typename Body, typename Partitioner >
void start_for< Range, Body, Partitioner >::execute() {
    if( !my_range.is_divisible() || !my_partition.is_divisible() ) {
        ANNOTATE_TASK_BEGIN( tbb_parallel_for_range );
        {
            my_body( my_range );
        }
        ANNOTATE_TASK_END( tbb_parallel_for_range );
    } else {
        typename Partitioner::split_type split_obj;
        start_for b( *this, split_obj );
        this->execute(); // Execute the left interval first to keep the serial order.
        b.execute();     // Execute the right interval then.
    }
}

//! Parallel iteration over range with default partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for( const Range& range, const Body& body ) {
    serial::interface9::start_for<Range,Body,const __TBB_DEFAULT_PARTITIONER>::run(range,body,__TBB_DEFAULT_PARTITIONER());
}

//! Parallel iteration over range with simple partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for( const Range& range, const Body& body, const simple_partitioner& partitioner ) {
    serial::interface9::start_for<Range,Body,const simple_partitioner>::run(range,body,partitioner);
}

//! Parallel iteration over range with auto_partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for( const Range& range, const Body& body, const auto_partitioner& partitioner ) {
    serial::interface9::start_for<Range,Body,const auto_partitioner>::run(range,body,partitioner);
}

//! Parallel iteration over range with static_partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for( const Range& range, const Body& body, const static_partitioner& partitioner ) {
    serial::interface9::start_for<Range,Body,const static_partitioner>::run(range,body,partitioner);
}

//! Parallel iteration over range with affinity_partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for( const Range& range, const Body& body, affinity_partitioner& partitioner ) {
    serial::interface9::start_for<Range,Body,affinity_partitioner>::run(range,body,partitioner);
}

//! Implementation of parallel iteration over stepped range of integers with explicit step and partitioner (ignored)
template <typename Index, typename Function, typename Partitioner>
void parallel_for_impl(Index first, Index last, Index step, const Function& f, Partitioner& ) {
    if (step <= 0 ) {
#if TBB_USE_EXCEPTIONS
        throw std::invalid_argument( "nonpositive_step" );
#else
        std::cerr << "nonpositive step in a call to parallel_for" << std::endl;
        std::abort();
#endif
    } else if (last > first) {
        // Above "else" avoids "potential divide by zero" warning on some platforms
        ANNOTATE_SITE_BEGIN( tbb_parallel_for );
        for( Index i = first; i < last; i = i + step ) {
            ANNOTATE_TASK_BEGIN( tbb_parallel_for_iteration );
            { f( i ); }
            ANNOTATE_TASK_END( tbb_parallel_for_iteration );
        }
        ANNOTATE_SITE_END( tbb_parallel_for );
    }
}

//! Parallel iteration over a range of integers with explicit step and default partitioner
template <typename Index, typename Function>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for(Index first, Index last, Index step, const Function& f) {
    parallel_for_impl<Index,Function,const auto_partitioner>(first, last, step, f, auto_partitioner());
}
//! Parallel iteration over a range of integers with explicit step and simple partitioner
template <typename Index, typename Function>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for(Index first, Index last, Index step, const Function& f, const simple_partitioner& p) {
    parallel_for_impl<Index,Function,const simple_partitioner>(first, last, step, f, p);
}
//! Parallel iteration over a range of integers with explicit step and auto partitioner
template <typename Index, typename Function>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for(Index first, Index last, Index step, const Function& f, const auto_partitioner& p) {
    parallel_for_impl<Index,Function,const auto_partitioner>(first, last, step, f, p);
}
//! Parallel iteration over a range of integers with explicit step and static partitioner
template <typename Index, typename Function>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for(Index first, Index last, Index step, const Function& f, const static_partitioner& p) {
    parallel_for_impl<Index,Function,const static_partitioner>(first, last, step, f, p);
}
//! Parallel iteration over a range of integers with explicit step and affinity partitioner
template <typename Index, typename Function>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for(Index first, Index last, Index step, const Function& f, affinity_partitioner& p) {
    parallel_for_impl(first, last, step, f, p);
}

//! Parallel iteration over a range of integers with default step and default partitioner
template <typename Index, typename Function>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for(Index first, Index last, const Function& f) {
    parallel_for_impl<Index,Function,const auto_partitioner>(first, last, static_cast<Index>(1), f, auto_partitioner());
}
//! Parallel iteration over a range of integers with default step and simple partitioner
template <typename Index, typename Function>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for(Index first, Index last, const Function& f, const simple_partitioner& p) {
    parallel_for_impl<Index,Function,const simple_partitioner>(first, last, static_cast<Index>(1), f, p);
}
//! Parallel iteration over a range of integers with default step and auto partitioner
template <typename Index, typename Function>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for(Index first, Index last, const Function& f, const auto_partitioner& p) {
    parallel_for_impl<Index,Function,const auto_partitioner>(first, last, static_cast<Index>(1), f, p);
}
//! Parallel iteration over a range of integers with default step and static partitioner
template <typename Index, typename Function>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for(Index first, Index last, const Function& f, const static_partitioner& p) {
    parallel_for_impl<Index,Function,const static_partitioner>(first, last, static_cast<Index>(1), f, p);
}
//! Parallel iteration over a range of integers with default step and affinity_partitioner
template <typename Index, typename Function>
__TBB_DEPRECATED_IN_VERBOSE_MODE void parallel_for(Index first, Index last, const Function& f, affinity_partitioner& p) {
    parallel_for_impl(first, last, static_cast<Index>(1), f, p);
}

} // namespace interfaceX

using interface9::parallel_for;

} // namespace serial

#ifndef __TBB_NORMAL_EXECUTION
using serial::interface9::parallel_for;
#endif

} // namespace tbb

#endif /* __TBB_SERIAL_parallel_for_H */
