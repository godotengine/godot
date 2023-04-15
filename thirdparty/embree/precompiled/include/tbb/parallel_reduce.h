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

#ifndef __TBB_parallel_reduce_H
#define __TBB_parallel_reduce_H

#define __TBB_parallel_reduce_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include <new>
#include "task.h"
#include "aligned_space.h"
#include "partitioner.h"
#include "tbb_profiling.h"

namespace tbb {

namespace interface9 {
//! @cond INTERNAL
namespace internal {

    using namespace tbb::internal;

    /** Values for reduction_context. */
    enum {
        root_task, left_child, right_child
    };

    /** Represented as a char, not enum, for compactness. */
    typedef char reduction_context;

    //! Task type used to combine the partial results of parallel_reduce.
    /** @ingroup algorithms */
    template<typename Body>
    class finish_reduce: public flag_task {
        //! Pointer to body, or NULL if the left child has not yet finished.
        bool has_right_zombie;
        const reduction_context my_context;
        Body* my_body;
        aligned_space<Body> zombie_space;
        finish_reduce( reduction_context context_ ) :
            has_right_zombie(false), // TODO: substitute by flag_task::child_stolen?
            my_context(context_),
            my_body(NULL)
        {
        }
        ~finish_reduce() {
            if( has_right_zombie )
                zombie_space.begin()->~Body();
        }
        task* execute() __TBB_override {
            if( has_right_zombie ) {
                // Right child was stolen.
                Body* s = zombie_space.begin();
                my_body->join( *s );
                // Body::join() won't be called if canceled. Defer destruction to destructor
            }
            if( my_context==left_child )
                itt_store_word_with_release( static_cast<finish_reduce*>(parent())->my_body, my_body );
            return NULL;
        }
        template<typename Range,typename Body_, typename Partitioner>
        friend class start_reduce;
    };

    //! allocate right task with new parent
    void allocate_sibling(task* start_reduce_task, task *tasks[], size_t start_bytes, size_t finish_bytes);

    //! Task type used to split the work of parallel_reduce.
    /** @ingroup algorithms */
    template<typename Range, typename Body, typename Partitioner>
    class start_reduce: public task {
        typedef finish_reduce<Body> finish_type;
        Body* my_body;
        Range my_range;
        typename Partitioner::task_partition_type my_partition;
        reduction_context my_context;
        task* execute() __TBB_override;
        //! Update affinity info, if any
        void note_affinity( affinity_id id ) __TBB_override {
            my_partition.note_affinity( id );
        }
        template<typename Body_>
        friend class finish_reduce;

public:
        //! Constructor used for root task
        start_reduce( const Range& range, Body* body, Partitioner& partitioner ) :
            my_body(body),
            my_range(range),
            my_partition(partitioner),
            my_context(root_task)
        {
        }
        //! Splitting constructor used to generate children.
        /** parent_ becomes left child.  Newly constructed object is right child. */
        start_reduce( start_reduce& parent_, typename Partitioner::split_type& split_obj ) :
            my_body(parent_.my_body),
            my_range(parent_.my_range, split_obj),
            my_partition(parent_.my_partition, split_obj),
            my_context(right_child)
        {
            my_partition.set_affinity(*this);
            parent_.my_context = left_child;
        }
        //! Construct right child from the given range as response to the demand.
        /** parent_ remains left child.  Newly constructed object is right child. */
        start_reduce( start_reduce& parent_, const Range& r, depth_t d ) :
            my_body(parent_.my_body),
            my_range(r),
            my_partition(parent_.my_partition, split()),
            my_context(right_child)
        {
            my_partition.set_affinity(*this);
            my_partition.align_depth( d ); // TODO: move into constructor of partitioner
            parent_.my_context = left_child;
        }
        static void run( const Range& range, Body& body, Partitioner& partitioner ) {
            if( !range.empty() ) {
#if !__TBB_TASK_GROUP_CONTEXT || TBB_JOIN_OUTER_TASK_GROUP
                task::spawn_root_and_wait( *new(task::allocate_root()) start_reduce(range,&body,partitioner) );
#else
                // Bound context prevents exceptions from body to affect nesting or sibling algorithms,
                // and allows users to handle exceptions safely by wrapping parallel_for in the try-block.
                task_group_context context(PARALLEL_REDUCE);
                task::spawn_root_and_wait( *new(task::allocate_root(context)) start_reduce(range,&body,partitioner) );
#endif /* __TBB_TASK_GROUP_CONTEXT && !TBB_JOIN_OUTER_TASK_GROUP */
            }
        }
#if __TBB_TASK_GROUP_CONTEXT
        static void run( const Range& range, Body& body, Partitioner& partitioner, task_group_context& context ) {
            if( !range.empty() )
                task::spawn_root_and_wait( *new(task::allocate_root(context)) start_reduce(range,&body,partitioner) );
        }
#endif /* __TBB_TASK_GROUP_CONTEXT */
        //! Run body for range
        void run_body( Range &r ) { (*my_body)( r ); }

        //! spawn right task, serves as callback for partitioner
        // TODO: remove code duplication from 'offer_work' methods
        void offer_work(typename Partitioner::split_type& split_obj) {
            task *tasks[2];
            allocate_sibling(static_cast<task*>(this), tasks, sizeof(start_reduce), sizeof(finish_type));
            new((void*)tasks[0]) finish_type(my_context);
            new((void*)tasks[1]) start_reduce(*this, split_obj);
            spawn(*tasks[1]);
        }
        //! spawn right task, serves as callback for partitioner
        void offer_work(const Range& r, depth_t d = 0) {
            task *tasks[2];
            allocate_sibling(static_cast<task*>(this), tasks, sizeof(start_reduce), sizeof(finish_type));
            new((void*)tasks[0]) finish_type(my_context);
            new((void*)tasks[1]) start_reduce(*this, r, d);
            spawn(*tasks[1]);
        }
    };

    //! allocate right task with new parent
    // TODO: 'inline' here is to avoid multiple definition error but for sake of code size this should not be inlined
    inline void allocate_sibling(task* start_reduce_task, task *tasks[], size_t start_bytes, size_t finish_bytes) {
        tasks[0] = &start_reduce_task->allocate_continuation().allocate(finish_bytes);
        start_reduce_task->set_parent(tasks[0]);
        tasks[0]->set_ref_count(2);
        tasks[1] = &tasks[0]->allocate_child().allocate(start_bytes);
    }

    template<typename Range, typename Body, typename Partitioner>
    task* start_reduce<Range,Body,Partitioner>::execute() {
        my_partition.check_being_stolen( *this );
        if( my_context==right_child ) {
            finish_type* parent_ptr = static_cast<finish_type*>(parent());
            if( !itt_load_word_with_acquire(parent_ptr->my_body) ) { // TODO: replace by is_stolen_task() or by parent_ptr->ref_count() == 2???
                my_body = new( parent_ptr->zombie_space.begin() ) Body(*my_body,split());
                parent_ptr->has_right_zombie = true;
            }
        } else __TBB_ASSERT(my_context==root_task,NULL);// because left leaf spawns right leafs without recycling
        my_partition.execute(*this, my_range);
        if( my_context==left_child ) {
            finish_type* parent_ptr = static_cast<finish_type*>(parent());
            __TBB_ASSERT(my_body!=parent_ptr->zombie_space.begin(),NULL);
            itt_store_word_with_release(parent_ptr->my_body, my_body );
        }
        return NULL;
    }

    //! Task type used to combine the partial results of parallel_deterministic_reduce.
    /** @ingroup algorithms */
    template<typename Body>
    class finish_deterministic_reduce: public task {
        Body &my_left_body;
        Body my_right_body;

        finish_deterministic_reduce( Body &body ) :
            my_left_body( body ),
            my_right_body( body, split() )
        {
        }
        task* execute() __TBB_override {
            my_left_body.join( my_right_body );
            return NULL;
        }
        template<typename Range,typename Body_, typename Partitioner>
        friend class start_deterministic_reduce;
    };

    //! Task type used to split the work of parallel_deterministic_reduce.
    /** @ingroup algorithms */
    template<typename Range, typename Body, typename Partitioner>
    class start_deterministic_reduce: public task {
        typedef finish_deterministic_reduce<Body> finish_type;
        Body &my_body;
        Range my_range;
        typename Partitioner::task_partition_type my_partition;
        task* execute() __TBB_override;

        //! Constructor used for root task
        start_deterministic_reduce( const Range& range, Body& body, Partitioner& partitioner ) :
            my_body( body ),
            my_range( range ),
            my_partition( partitioner )
        {
        }
        //! Splitting constructor used to generate children.
        /** parent_ becomes left child.  Newly constructed object is right child. */
        start_deterministic_reduce( start_deterministic_reduce& parent_, finish_type& c, typename Partitioner::split_type& split_obj ) :
            my_body( c.my_right_body ),
            my_range( parent_.my_range, split_obj ),
            my_partition( parent_.my_partition, split_obj )
        {
        }

public:
        static void run( const Range& range, Body& body, Partitioner& partitioner ) {
            if( !range.empty() ) {
#if !__TBB_TASK_GROUP_CONTEXT || TBB_JOIN_OUTER_TASK_GROUP
                task::spawn_root_and_wait( *new(task::allocate_root()) start_deterministic_reduce(range,&body,partitioner) );
#else
                // Bound context prevents exceptions from body to affect nesting or sibling algorithms,
                // and allows users to handle exceptions safely by wrapping parallel_for in the try-block.
                task_group_context context(PARALLEL_REDUCE);
                task::spawn_root_and_wait( *new(task::allocate_root(context)) start_deterministic_reduce(range,body,partitioner) );
#endif /* __TBB_TASK_GROUP_CONTEXT && !TBB_JOIN_OUTER_TASK_GROUP */
            }
        }
#if __TBB_TASK_GROUP_CONTEXT
        static void run( const Range& range, Body& body, Partitioner& partitioner, task_group_context& context ) {
            if( !range.empty() )
                task::spawn_root_and_wait( *new(task::allocate_root(context)) start_deterministic_reduce(range,body,partitioner) );
        }
#endif /* __TBB_TASK_GROUP_CONTEXT */

        void offer_work( typename Partitioner::split_type& split_obj) {
            task* tasks[2];
            allocate_sibling(static_cast<task*>(this), tasks, sizeof(start_deterministic_reduce), sizeof(finish_type));
            new((void*)tasks[0]) finish_type(my_body);
            new((void*)tasks[1]) start_deterministic_reduce(*this, *static_cast<finish_type*>(tasks[0]), split_obj);
            spawn(*tasks[1]);
        }

        void run_body( Range &r ) { my_body(r); }
    };

    template<typename Range, typename Body, typename Partitioner>
    task* start_deterministic_reduce<Range,Body, Partitioner>::execute() {
        my_partition.execute(*this, my_range);
        return NULL;
    }
} // namespace internal
//! @endcond
} //namespace interfaceX

//! @cond INTERNAL
namespace internal {
    using interface9::internal::start_reduce;
    using interface9::internal::start_deterministic_reduce;
    //! Auxiliary class for parallel_reduce; for internal use only.
    /** The adaptor class that implements \ref parallel_reduce_body_req "parallel_reduce Body"
        using given \ref parallel_reduce_lambda_req "anonymous function objects".
     **/
    /** @ingroup algorithms */
    template<typename Range, typename Value, typename RealBody, typename Reduction>
    class lambda_reduce_body {

//FIXME: decide if my_real_body, my_reduction, and identity_element should be copied or referenced
//       (might require some performance measurements)

        const Value&     identity_element;
        const RealBody&  my_real_body;
        const Reduction& my_reduction;
        Value            my_value;
        lambda_reduce_body& operator= ( const lambda_reduce_body& other );
    public:
        lambda_reduce_body( const Value& identity, const RealBody& body, const Reduction& reduction )
            : identity_element(identity)
            , my_real_body(body)
            , my_reduction(reduction)
            , my_value(identity)
        { }
        lambda_reduce_body( const lambda_reduce_body& other )
            : identity_element(other.identity_element)
            , my_real_body(other.my_real_body)
            , my_reduction(other.my_reduction)
            , my_value(other.my_value)
        { }
        lambda_reduce_body( lambda_reduce_body& other, tbb::split )
            : identity_element(other.identity_element)
            , my_real_body(other.my_real_body)
            , my_reduction(other.my_reduction)
            , my_value(other.identity_element)
        { }
        void operator()(Range& range) {
            my_value = my_real_body(range, const_cast<const Value&>(my_value));
        }
        void join( lambda_reduce_body& rhs ) {
            my_value = my_reduction(const_cast<const Value&>(my_value), const_cast<const Value&>(rhs.my_value));
        }
        Value result() const {
            return my_value;
        }
    };

} // namespace internal
//! @endcond

// Requirements on Range concept are documented in blocked_range.h

/** \page parallel_reduce_body_req Requirements on parallel_reduce body
    Class \c Body implementing the concept of parallel_reduce body must define:
    - \code Body::Body( Body&, split ); \endcode        Splitting constructor.
                                                        Must be able to run concurrently with operator() and method \c join
    - \code Body::~Body(); \endcode                     Destructor
    - \code void Body::operator()( Range& r ); \endcode Function call operator applying body to range \c r
                                                        and accumulating the result
    - \code void Body::join( Body& b ); \endcode        Join results.
                                                        The result in \c b should be merged into the result of \c this
**/

/** \page parallel_reduce_lambda_req Requirements on parallel_reduce anonymous function objects (lambda functions)
    TO BE DOCUMENTED
**/

/** \name parallel_reduce
    See also requirements on \ref range_req "Range" and \ref parallel_reduce_body_req "parallel_reduce Body". **/
//@{

//! Parallel iteration with reduction and default partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body ) {
    internal::start_reduce<Range,Body, const __TBB_DEFAULT_PARTITIONER>::run( range, body, __TBB_DEFAULT_PARTITIONER() );
}

//! Parallel iteration with reduction and simple_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const simple_partitioner& partitioner ) {
    internal::start_reduce<Range,Body,const simple_partitioner>::run( range, body, partitioner );
}

//! Parallel iteration with reduction and auto_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const auto_partitioner& partitioner ) {
    internal::start_reduce<Range,Body,const auto_partitioner>::run( range, body, partitioner );
}

//! Parallel iteration with reduction and static_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const static_partitioner& partitioner ) {
    internal::start_reduce<Range,Body,const static_partitioner>::run( range, body, partitioner );
}

//! Parallel iteration with reduction and affinity_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, affinity_partitioner& partitioner ) {
    internal::start_reduce<Range,Body,affinity_partitioner>::run( range, body, partitioner );
}

#if __TBB_TASK_GROUP_CONTEXT
//! Parallel iteration with reduction, default partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, task_group_context& context ) {
    internal::start_reduce<Range,Body,const __TBB_DEFAULT_PARTITIONER>::run( range, body, __TBB_DEFAULT_PARTITIONER(), context );
}

//! Parallel iteration with reduction, simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const simple_partitioner& partitioner, task_group_context& context ) {
    internal::start_reduce<Range,Body,const simple_partitioner>::run( range, body, partitioner, context );
}

//! Parallel iteration with reduction, auto_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const auto_partitioner& partitioner, task_group_context& context ) {
    internal::start_reduce<Range,Body,const auto_partitioner>::run( range, body, partitioner, context );
}

//! Parallel iteration with reduction, static_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const static_partitioner& partitioner, task_group_context& context ) {
    internal::start_reduce<Range,Body,const static_partitioner>::run( range, body, partitioner, context );
}

//! Parallel iteration with reduction, affinity_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, affinity_partitioner& partitioner, task_group_context& context ) {
    internal::start_reduce<Range,Body,affinity_partitioner>::run( range, body, partitioner, context );
}
#endif /* __TBB_TASK_GROUP_CONTEXT */

/** parallel_reduce overloads that work with anonymous function objects
    (see also \ref parallel_reduce_lambda_req "requirements on parallel_reduce anonymous function objects"). **/

//! Parallel iteration with reduction and default partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction ) {
    internal::lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    internal::start_reduce<Range,internal::lambda_reduce_body<Range,Value,RealBody,Reduction>,const __TBB_DEFAULT_PARTITIONER>
                          ::run(range, body, __TBB_DEFAULT_PARTITIONER() );
    return body.result();
}

//! Parallel iteration with reduction and simple_partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       const simple_partitioner& partitioner ) {
    internal::lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    internal::start_reduce<Range,internal::lambda_reduce_body<Range,Value,RealBody,Reduction>,const simple_partitioner>
                          ::run(range, body, partitioner );
    return body.result();
}

//! Parallel iteration with reduction and auto_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       const auto_partitioner& partitioner ) {
    internal::lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    internal::start_reduce<Range,internal::lambda_reduce_body<Range,Value,RealBody,Reduction>,const auto_partitioner>
                          ::run( range, body, partitioner );
    return body.result();
}

//! Parallel iteration with reduction and static_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       const static_partitioner& partitioner ) {
    internal::lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    internal::start_reduce<Range,internal::lambda_reduce_body<Range,Value,RealBody,Reduction>,const static_partitioner>
                                        ::run( range, body, partitioner );
    return body.result();
}

//! Parallel iteration with reduction and affinity_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       affinity_partitioner& partitioner ) {
    internal::lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    internal::start_reduce<Range,internal::lambda_reduce_body<Range,Value,RealBody,Reduction>,affinity_partitioner>
                                        ::run( range, body, partitioner );
    return body.result();
}

#if __TBB_TASK_GROUP_CONTEXT
//! Parallel iteration with reduction, default partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       task_group_context& context ) {
    internal::lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    internal::start_reduce<Range,internal::lambda_reduce_body<Range,Value,RealBody,Reduction>,const __TBB_DEFAULT_PARTITIONER>
                          ::run( range, body, __TBB_DEFAULT_PARTITIONER(), context );
    return body.result();
}

//! Parallel iteration with reduction, simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       const simple_partitioner& partitioner, task_group_context& context ) {
    internal::lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    internal::start_reduce<Range,internal::lambda_reduce_body<Range,Value,RealBody,Reduction>,const simple_partitioner>
                          ::run( range, body, partitioner, context );
    return body.result();
}

//! Parallel iteration with reduction, auto_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       const auto_partitioner& partitioner, task_group_context& context ) {
    internal::lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    internal::start_reduce<Range,internal::lambda_reduce_body<Range,Value,RealBody,Reduction>,const auto_partitioner>
                          ::run( range, body, partitioner, context );
    return body.result();
}

//! Parallel iteration with reduction, static_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       const static_partitioner& partitioner, task_group_context& context ) {
    internal::lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    internal::start_reduce<Range,internal::lambda_reduce_body<Range,Value,RealBody,Reduction>,const static_partitioner>
                                        ::run( range, body, partitioner, context );
    return body.result();
}

//! Parallel iteration with reduction, affinity_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
                       affinity_partitioner& partitioner, task_group_context& context ) {
    internal::lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    internal::start_reduce<Range,internal::lambda_reduce_body<Range,Value,RealBody,Reduction>,affinity_partitioner>
                                        ::run( range, body, partitioner, context );
    return body.result();
}
#endif /* __TBB_TASK_GROUP_CONTEXT */

//! Parallel iteration with deterministic reduction and default simple partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_deterministic_reduce( const Range& range, Body& body ) {
    internal::start_deterministic_reduce<Range, Body, const simple_partitioner>::run(range, body, simple_partitioner());
}

//! Parallel iteration with deterministic reduction and simple partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_deterministic_reduce( const Range& range, Body& body, const simple_partitioner& partitioner ) {
    internal::start_deterministic_reduce<Range, Body, const simple_partitioner>::run(range, body, partitioner);
}

//! Parallel iteration with deterministic reduction and static partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_deterministic_reduce( const Range& range, Body& body, const static_partitioner& partitioner ) {
    internal::start_deterministic_reduce<Range, Body, const static_partitioner>::run(range, body, partitioner);
}

#if __TBB_TASK_GROUP_CONTEXT
//! Parallel iteration with deterministic reduction, default simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_deterministic_reduce( const Range& range, Body& body, task_group_context& context ) {
    internal::start_deterministic_reduce<Range,Body, const simple_partitioner>::run( range, body, simple_partitioner(), context );
}

//! Parallel iteration with deterministic reduction, simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_deterministic_reduce( const Range& range, Body& body, const simple_partitioner& partitioner, task_group_context& context ) {
    internal::start_deterministic_reduce<Range, Body, const simple_partitioner>::run(range, body, partitioner, context);
}

//! Parallel iteration with deterministic reduction, static partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_deterministic_reduce( const Range& range, Body& body, const static_partitioner& partitioner, task_group_context& context ) {
    internal::start_deterministic_reduce<Range, Body, const static_partitioner>::run(range, body, partitioner, context);
}
#endif /* __TBB_TASK_GROUP_CONTEXT */

/** parallel_reduce overloads that work with anonymous function objects
    (see also \ref parallel_reduce_lambda_req "requirements on parallel_reduce anonymous function objects"). **/

//! Parallel iteration with deterministic reduction and default simple partitioner.
// TODO: consider making static_partitioner the default
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_deterministic_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction ) {
    return parallel_deterministic_reduce(range, identity, real_body, reduction, simple_partitioner());
}

//! Parallel iteration with deterministic reduction and simple partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_deterministic_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction, const simple_partitioner& partitioner ) {
    internal::lambda_reduce_body<Range,Value,RealBody,Reduction> body(identity, real_body, reduction);
    internal::start_deterministic_reduce<Range,internal::lambda_reduce_body<Range,Value,RealBody,Reduction>, const simple_partitioner>
                          ::run(range, body, partitioner);
    return body.result();
}

//! Parallel iteration with deterministic reduction and static partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_deterministic_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction, const static_partitioner& partitioner ) {
    internal::lambda_reduce_body<Range, Value, RealBody, Reduction> body(identity, real_body, reduction);
    internal::start_deterministic_reduce<Range, internal::lambda_reduce_body<Range, Value, RealBody, Reduction>, const static_partitioner>
        ::run(range, body, partitioner);
    return body.result();
}
#if __TBB_TASK_GROUP_CONTEXT
//! Parallel iteration with deterministic reduction, default simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_deterministic_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
    task_group_context& context ) {
    return parallel_deterministic_reduce(range, identity, real_body, reduction, simple_partitioner(), context);
}

//! Parallel iteration with deterministic reduction, simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_deterministic_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
    const simple_partitioner& partitioner, task_group_context& context ) {
    internal::lambda_reduce_body<Range, Value, RealBody, Reduction> body(identity, real_body, reduction);
    internal::start_deterministic_reduce<Range, internal::lambda_reduce_body<Range, Value, RealBody, Reduction>, const simple_partitioner>
        ::run(range, body, partitioner, context);
    return body.result();
}

//! Parallel iteration with deterministic reduction, static partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_deterministic_reduce( const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
    const static_partitioner& partitioner, task_group_context& context ) {
    internal::lambda_reduce_body<Range, Value, RealBody, Reduction> body(identity, real_body, reduction);
    internal::start_deterministic_reduce<Range, internal::lambda_reduce_body<Range, Value, RealBody, Reduction>, const static_partitioner>
        ::run(range, body, partitioner, context);
    return body.result();
}
#endif /* __TBB_TASK_GROUP_CONTEXT */
//@}

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_parallel_reduce_H_include_area

#endif /* __TBB_parallel_reduce_H */
