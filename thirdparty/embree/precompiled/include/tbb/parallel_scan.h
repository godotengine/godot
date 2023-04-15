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

#ifndef __TBB_parallel_scan_H
#define __TBB_parallel_scan_H

#define __TBB_parallel_scan_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "task.h"
#include "aligned_space.h"
#include <new>
#include "partitioner.h"

namespace tbb {

//! Used to indicate that the initial scan is being performed.
/** @ingroup algorithms */
struct pre_scan_tag {
    static bool is_final_scan() {return false;}
    operator bool() {return is_final_scan();}
};

//! Used to indicate that the final scan is being performed.
/** @ingroup algorithms */
struct final_scan_tag {
    static bool is_final_scan() {return true;}
    operator bool() {return is_final_scan();}
};

//! @cond INTERNAL
namespace internal {

    //! Performs final scan for a leaf
    /** @ingroup algorithms */
    template<typename Range, typename Body>
    class final_sum: public task {
    public:
        Body my_body;
    private:
        aligned_space<Range> my_range;
        //! Where to put result of last subrange, or NULL if not last subrange.
        Body* my_stuff_last;
    public:
        final_sum( Body& body_ ) :
            my_body(body_,split())
        {
            poison_pointer(my_stuff_last);
        }
        ~final_sum() {
            my_range.begin()->~Range();
        }
        void finish_construction( const Range& range_, Body* stuff_last_ ) {
            new( my_range.begin() ) Range(range_);
            my_stuff_last = stuff_last_;
        }
    private:
        task* execute() __TBB_override {
            my_body( *my_range.begin(), final_scan_tag() );
            if( my_stuff_last )
                my_stuff_last->assign(my_body);
            return NULL;
        }
    };

    //! Split work to be done in the scan.
    /** @ingroup algorithms */
    template<typename Range, typename Body>
    class sum_node: public task {
        typedef final_sum<Range,Body> final_sum_type;
    public:
        final_sum_type *my_incoming;
        final_sum_type *my_body;
        Body *my_stuff_last;
    private:
        final_sum_type *my_left_sum;
        sum_node *my_left;
        sum_node *my_right;
        bool my_left_is_final;
        Range my_range;
        sum_node( const Range range_, bool left_is_final_ ) :
            my_stuff_last(NULL),
            my_left_sum(NULL),
            my_left(NULL),
            my_right(NULL),
            my_left_is_final(left_is_final_),
            my_range(range_)
        {
            // Poison fields that will be set by second pass.
            poison_pointer(my_body);
            poison_pointer(my_incoming);
        }
        task* create_child( const Range& range_, final_sum_type& f, sum_node* n, final_sum_type* incoming_, Body* stuff_last_ ) {
            if( !n ) {
                f.recycle_as_child_of( *this );
                f.finish_construction( range_, stuff_last_ );
                return &f;
            } else {
                n->my_body = &f;
                n->my_incoming = incoming_;
                n->my_stuff_last = stuff_last_;
                return n;
            }
        }
        task* execute() __TBB_override {
            if( my_body ) {
                if( my_incoming )
                    my_left_sum->my_body.reverse_join( my_incoming->my_body );
                recycle_as_continuation();
                sum_node& c = *this;
                task* b = c.create_child(Range(my_range,split()),*my_left_sum,my_right,my_left_sum,my_stuff_last);
                task* a = my_left_is_final ? NULL : c.create_child(my_range,*my_body,my_left,my_incoming,NULL);
                set_ref_count( (a!=NULL)+(b!=NULL) );
                my_body = NULL;
                if( a ) spawn(*b);
                else a = b;
                return a;
            } else {
                return NULL;
            }
        }
        template<typename Range_,typename Body_,typename Partitioner_>
        friend class start_scan;

        template<typename Range_,typename Body_>
        friend class finish_scan;
    };

    //! Combine partial results
    /** @ingroup algorithms */
    template<typename Range, typename Body>
    class finish_scan: public task {
        typedef sum_node<Range,Body> sum_node_type;
        typedef final_sum<Range,Body> final_sum_type;
        final_sum_type** const my_sum;
        sum_node_type*& my_return_slot;
    public:
        final_sum_type* my_right_zombie;
        sum_node_type& my_result;

        task* execute() __TBB_override {
            __TBB_ASSERT( my_result.ref_count()==(my_result.my_left!=NULL)+(my_result.my_right!=NULL), NULL );
            if( my_result.my_left )
                my_result.my_left_is_final = false;
            if( my_right_zombie && my_sum )
                ((*my_sum)->my_body).reverse_join(my_result.my_left_sum->my_body);
            __TBB_ASSERT( !my_return_slot, NULL );
            if( my_right_zombie || my_result.my_right ) {
                my_return_slot = &my_result;
            } else {
                destroy( my_result );
            }
            if( my_right_zombie && !my_sum && !my_result.my_right ) {
                destroy(*my_right_zombie);
                my_right_zombie = NULL;
            }
            return NULL;
        }

        finish_scan( sum_node_type*& return_slot_, final_sum_type** sum_, sum_node_type& result_ ) :
            my_sum(sum_),
            my_return_slot(return_slot_),
            my_right_zombie(NULL),
            my_result(result_)
        {
            __TBB_ASSERT( !my_return_slot, NULL );
        }
    };

    //! Initial task to split the work
    /** @ingroup algorithms */
    template<typename Range, typename Body, typename Partitioner=simple_partitioner>
    class start_scan: public task {
        typedef sum_node<Range,Body> sum_node_type;
        typedef final_sum<Range,Body> final_sum_type;
        final_sum_type* my_body;
        /** Non-null if caller is requesting total. */
        final_sum_type** my_sum;
        sum_node_type** my_return_slot;
        /** Null if computing root. */
        sum_node_type* my_parent_sum;
        bool my_is_final;
        bool my_is_right_child;
        Range my_range;
        typename Partitioner::partition_type my_partition;
        task* execute() __TBB_override ;
    public:
        start_scan( sum_node_type*& return_slot_, start_scan& parent_, sum_node_type* parent_sum_ ) :
            my_body(parent_.my_body),
            my_sum(parent_.my_sum),
            my_return_slot(&return_slot_),
            my_parent_sum(parent_sum_),
            my_is_final(parent_.my_is_final),
            my_is_right_child(false),
            my_range(parent_.my_range,split()),
            my_partition(parent_.my_partition,split())
        {
            __TBB_ASSERT( !*my_return_slot, NULL );
        }

        start_scan( sum_node_type*& return_slot_, const Range& range_, final_sum_type& body_, const Partitioner& partitioner_) :
            my_body(&body_),
            my_sum(NULL),
            my_return_slot(&return_slot_),
            my_parent_sum(NULL),
            my_is_final(true),
            my_is_right_child(false),
            my_range(range_),
            my_partition(partitioner_)
        {
            __TBB_ASSERT( !*my_return_slot, NULL );
        }

        static void run( const Range& range_, Body& body_, const Partitioner& partitioner_ ) {
            if( !range_.empty() ) {
                typedef internal::start_scan<Range,Body,Partitioner> start_pass1_type;
                internal::sum_node<Range,Body>* root = NULL;
                final_sum_type* temp_body = new(task::allocate_root()) final_sum_type( body_ );
                start_pass1_type& pass1 = *new(task::allocate_root()) start_pass1_type(
                    /*my_return_slot=*/root,
                    range_,
                    *temp_body,
                    partitioner_ );
                temp_body->my_body.reverse_join(body_);
                task::spawn_root_and_wait( pass1 );
                if( root ) {
                    root->my_body = temp_body;
                    root->my_incoming = NULL;
                    root->my_stuff_last = &body_;
                    task::spawn_root_and_wait( *root );
                } else {
                    body_.assign(temp_body->my_body);
                    temp_body->finish_construction( range_, NULL );
                    temp_body->destroy(*temp_body);
                }
            }
        }
    };

    template<typename Range, typename Body, typename Partitioner>
    task* start_scan<Range,Body,Partitioner>::execute() {
        typedef internal::finish_scan<Range,Body> finish_pass1_type;
        finish_pass1_type* p = my_parent_sum ? static_cast<finish_pass1_type*>( parent() ) : NULL;
        // Inspecting p->result.left_sum would ordinarily be a race condition.
        // But we inspect it only if we are not a stolen task, in which case we
        // know that task assigning to p->result.left_sum has completed.
        bool treat_as_stolen = my_is_right_child && (is_stolen_task() || my_body!=p->my_result.my_left_sum);
        if( treat_as_stolen ) {
            // Invocation is for right child that has been really stolen or needs to be virtually stolen
            p->my_right_zombie = my_body = new( allocate_root() ) final_sum_type(my_body->my_body);
            my_is_final = false;
        }
        task* next_task = NULL;
        if( (my_is_right_child && !treat_as_stolen) || !my_range.is_divisible() || my_partition.should_execute_range(*this) ) {
            if( my_is_final )
                (my_body->my_body)( my_range, final_scan_tag() );
            else if( my_sum )
                (my_body->my_body)( my_range, pre_scan_tag() );
            if( my_sum )
                *my_sum = my_body;
            __TBB_ASSERT( !*my_return_slot, NULL );
        } else {
            sum_node_type* result;
            if( my_parent_sum )
                result = new(allocate_additional_child_of(*my_parent_sum)) sum_node_type(my_range,/*my_left_is_final=*/my_is_final);
            else
                result = new(task::allocate_root()) sum_node_type(my_range,/*my_left_is_final=*/my_is_final);
            finish_pass1_type& c = *new( allocate_continuation()) finish_pass1_type(*my_return_slot,my_sum,*result);
            // Split off right child
            start_scan& b = *new( c.allocate_child() ) start_scan( /*my_return_slot=*/result->my_right, *this, result );
            b.my_is_right_child = true;
            // Left child is recycling of *this.  Must recycle this before spawning b,
            // otherwise b might complete and decrement c.ref_count() to zero, which
            // would cause c.execute() to run prematurely.
            recycle_as_child_of(c);
            c.set_ref_count(2);
            c.spawn(b);
            my_sum = &result->my_left_sum;
            my_return_slot = &result->my_left;
            my_is_right_child = false;
            next_task = this;
            my_parent_sum = result;
            __TBB_ASSERT( !*my_return_slot, NULL );
        }
        return next_task;
    }

    template<typename Range, typename Value, typename Scan, typename ReverseJoin>
    class lambda_scan_body : no_assign {
        Value               my_sum;
        const Value&        identity_element;
        const Scan&         my_scan;
        const ReverseJoin&  my_reverse_join;
    public:
        lambda_scan_body( const Value& identity, const Scan& scan, const ReverseJoin& rev_join)
            : my_sum(identity)
            , identity_element(identity)
            , my_scan(scan)
            , my_reverse_join(rev_join) {}

        lambda_scan_body( lambda_scan_body& b, split )
            : my_sum(b.identity_element)
            , identity_element(b.identity_element)
            , my_scan(b.my_scan)
            , my_reverse_join(b.my_reverse_join) {}

        template<typename Tag>
        void operator()( const Range& r, Tag tag ) {
            my_sum = my_scan(r, my_sum, tag);
        }

        void reverse_join( lambda_scan_body& a ) {
            my_sum = my_reverse_join(a.my_sum, my_sum);
        }

        void assign( lambda_scan_body& b ) {
            my_sum = b.my_sum;
        }

        Value result() const {
            return my_sum;
        }
    };
} // namespace internal
//! @endcond

// Requirements on Range concept are documented in blocked_range.h

/** \page parallel_scan_body_req Requirements on parallel_scan body
    Class \c Body implementing the concept of parallel_scan body must define:
    - \code Body::Body( Body&, split ); \endcode    Splitting constructor.
                                                    Split \c b so that \c this and \c b can accumulate separately
    - \code Body::~Body(); \endcode                 Destructor
    - \code void Body::operator()( const Range& r, pre_scan_tag ); \endcode
                                                    Preprocess iterations for range \c r
    - \code void Body::operator()( const Range& r, final_scan_tag ); \endcode
                                                    Do final processing for iterations of range \c r
    - \code void Body::reverse_join( Body& a ); \endcode
                                                    Merge preprocessing state of \c a into \c this, where \c a was
                                                    created earlier from \c b by b's splitting constructor
**/

/** \name parallel_scan
    See also requirements on \ref range_req "Range" and \ref parallel_scan_body_req "parallel_scan Body". **/
//@{

//! Parallel prefix with default partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_scan( const Range& range, Body& body ) {
    internal::start_scan<Range,Body,__TBB_DEFAULT_PARTITIONER>::run(range,body,__TBB_DEFAULT_PARTITIONER());
}

//! Parallel prefix with simple_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_scan( const Range& range, Body& body, const simple_partitioner& partitioner ) {
    internal::start_scan<Range,Body,simple_partitioner>::run(range,body,partitioner);
}

//! Parallel prefix with auto_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_scan( const Range& range, Body& body, const auto_partitioner& partitioner ) {
    internal::start_scan<Range,Body,auto_partitioner>::run(range,body,partitioner);
}

//! Parallel prefix with default partitioner
/** @ingroup algorithms **/
template<typename Range, typename Value, typename Scan, typename ReverseJoin>
Value parallel_scan( const Range& range, const Value& identity, const Scan& scan, const ReverseJoin& reverse_join ) {
    internal::lambda_scan_body<Range, Value, Scan, ReverseJoin> body(identity, scan, reverse_join);
    tbb::parallel_scan(range,body,__TBB_DEFAULT_PARTITIONER());
    return body.result();
}

//! Parallel prefix with simple_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Value, typename Scan, typename ReverseJoin>
Value parallel_scan( const Range& range, const Value& identity, const Scan& scan, const ReverseJoin& reverse_join, const simple_partitioner& partitioner ) {
    internal::lambda_scan_body<Range, Value, Scan, ReverseJoin> body(identity, scan, reverse_join);
    tbb::parallel_scan(range,body,partitioner);
    return body.result();
}

//! Parallel prefix with auto_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Value, typename Scan, typename ReverseJoin>
Value parallel_scan( const Range& range, const Value& identity, const Scan& scan, const ReverseJoin& reverse_join, const auto_partitioner& partitioner ) {
    internal::lambda_scan_body<Range, Value, Scan, ReverseJoin> body(identity, scan, reverse_join);
    tbb::parallel_scan(range,body,partitioner);
    return body.result();
}

//@}

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_parallel_scan_H_include_area

#endif /* __TBB_parallel_scan_H */

