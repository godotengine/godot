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

#ifndef __TBB_parallel_do_H
#define __TBB_parallel_do_H

#define __TBB_parallel_do_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "internal/_range_iterator.h"
#include "internal/_template_helpers.h"
#include "task.h"
#include "aligned_space.h"
#include <iterator>

namespace tbb {
namespace interface9 {
//! @cond INTERNAL
namespace internal {
    template<typename Body, typename Item> class parallel_do_feeder_impl;
} // namespace internal
//! @endcond

//! Class the user supplied algorithm body uses to add new tasks
/** \param Item Work item type **/
    template<typename Item>
    class parallel_do_feeder: ::tbb::internal::no_copy
    {
        parallel_do_feeder() {}
        virtual ~parallel_do_feeder () {}
        virtual void internal_add_copy( const Item& item ) = 0;
#if __TBB_CPP11_RVALUE_REF_PRESENT
        virtual void internal_add_move( Item&& item ) = 0;
#endif
        template<typename Body_, typename Item_> friend class internal::parallel_do_feeder_impl;
    public:
        //! Add a work item to a running parallel_do.
        void add( const Item& item ) {internal_add_copy(item);}
#if __TBB_CPP11_RVALUE_REF_PRESENT
        void add( Item&& item ) {internal_add_move(std::move(item));}
#endif
    };

//! @cond INTERNAL
namespace internal {
    template<typename Body> class do_group_task;

    //! For internal use only.
    /** Selects one of the two possible forms of function call member operator.
        @ingroup algorithms **/
    template<class Body, typename Item>
    class parallel_do_operator_selector
    {
        typedef parallel_do_feeder<Item> Feeder;
        template<typename A1, typename A2, typename CvItem >
        static void internal_call( const Body& obj, __TBB_FORWARDING_REF(A1) arg1, A2&, void (Body::*)(CvItem) const ) {
            obj(tbb::internal::forward<A1>(arg1));
        }
        template<typename A1, typename A2, typename CvItem >
        static void internal_call( const Body& obj, __TBB_FORWARDING_REF(A1) arg1, A2& arg2, void (Body::*)(CvItem, parallel_do_feeder<Item>&) const ) {
            obj(tbb::internal::forward<A1>(arg1), arg2);
        }
        template<typename A1, typename A2, typename CvItem >
        static void internal_call( const Body& obj, __TBB_FORWARDING_REF(A1) arg1, A2&, void (Body::*)(CvItem&) const ) {
            obj(arg1);
        }
        template<typename A1, typename A2, typename CvItem >
        static void internal_call( const Body& obj, __TBB_FORWARDING_REF(A1) arg1, A2& arg2, void (Body::*)(CvItem&, parallel_do_feeder<Item>&) const ) {
            obj(arg1, arg2);
        }
    public:
        template<typename A1, typename A2>
        static void call( const Body& obj, __TBB_FORWARDING_REF(A1) arg1, A2& arg2 )
        {
            internal_call( obj, tbb::internal::forward<A1>(arg1), arg2, &Body::operator() );
        }
    };

    //! For internal use only.
    /** Executes one iteration of a do.
        @ingroup algorithms */
    template<typename Body, typename Item>
    class do_iteration_task: public task
    {
        typedef parallel_do_feeder_impl<Body, Item> feeder_type;

        Item my_value;
        feeder_type& my_feeder;

        do_iteration_task( const Item& value, feeder_type& feeder ) :
            my_value(value), my_feeder(feeder)
        {}

#if __TBB_CPP11_RVALUE_REF_PRESENT
        do_iteration_task( Item&& value, feeder_type& feeder ) :
            my_value(std::move(value)), my_feeder(feeder)
        {}
#endif

        task* execute() __TBB_override
        {
            parallel_do_operator_selector<Body, Item>::call(*my_feeder.my_body, tbb::internal::move(my_value), my_feeder);
            return NULL;
        }

        template<typename Body_, typename Item_> friend class parallel_do_feeder_impl;
    }; // class do_iteration_task

    template<typename Iterator, typename Body, typename Item>
    class do_iteration_task_iter: public task
    {
        typedef parallel_do_feeder_impl<Body, Item> feeder_type;

        Iterator my_iter;
        feeder_type& my_feeder;

        do_iteration_task_iter( const Iterator& iter, feeder_type& feeder ) :
            my_iter(iter), my_feeder(feeder)
        {}

        task* execute() __TBB_override
        {
            parallel_do_operator_selector<Body, Item>::call(*my_feeder.my_body, *my_iter, my_feeder);
            return NULL;
        }

        template<typename Iterator_, typename Body_, typename Item_> friend class do_group_task_forward;
        template<typename Body_, typename Item_> friend class do_group_task_input;
        template<typename Iterator_, typename Body_, typename Item_> friend class do_task_iter;
    }; // class do_iteration_task_iter

    //! For internal use only.
    /** Implements new task adding procedure.
        @ingroup algorithms **/
    template<class Body, typename Item>
    class parallel_do_feeder_impl : public parallel_do_feeder<Item>
    {
#if __TBB_CPP11_RVALUE_REF_PRESENT
        //Avoiding use of copy constructor in a virtual method if the type does not support it
        void internal_add_copy_impl(std::true_type, const Item& item) {
            typedef do_iteration_task<Body, Item> iteration_type;
            iteration_type& t = *new (task::allocate_additional_child_of(*my_barrier)) iteration_type(item, *this);
            task::spawn(t);
        }
        void internal_add_copy_impl(std::false_type, const Item&) {
            __TBB_ASSERT(false, "Overloading for r-value reference doesn't work or it's not movable and not copyable object");
        }
        void internal_add_copy( const Item& item ) __TBB_override
        {
#if __TBB_CPP11_IS_COPY_CONSTRUCTIBLE_PRESENT
            internal_add_copy_impl(typename std::is_copy_constructible<Item>::type(), item);
#else
            internal_add_copy_impl(std::true_type(), item);
#endif
        }
        void internal_add_move( Item&& item ) __TBB_override
        {
            typedef do_iteration_task<Body, Item> iteration_type;
            iteration_type& t = *new (task::allocate_additional_child_of(*my_barrier)) iteration_type(std::move(item), *this);
            task::spawn(t);
        }
#else /* ! __TBB_CPP11_RVALUE_REF_PRESENT */
        void internal_add_copy(const Item& item) __TBB_override {
            typedef do_iteration_task<Body, Item> iteration_type;
            iteration_type& t = *new (task::allocate_additional_child_of(*my_barrier)) iteration_type(item, *this);
            task::spawn(t);
        }
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */
    public:
        const Body* my_body;
        empty_task* my_barrier;

        parallel_do_feeder_impl()
        {
            my_barrier = new( task::allocate_root() ) empty_task();
            __TBB_ASSERT(my_barrier, "root task allocation failed");
        }

#if __TBB_TASK_GROUP_CONTEXT
        parallel_do_feeder_impl(tbb::task_group_context &context)
        {
            my_barrier = new( task::allocate_root(context) ) empty_task();
            __TBB_ASSERT(my_barrier, "root task allocation failed");
        }
#endif

        ~parallel_do_feeder_impl()
        {
            my_barrier->destroy(*my_barrier);
        }
    }; // class parallel_do_feeder_impl


    //! For internal use only
    /** Unpacks a block of iterations.
        @ingroup algorithms */

    template<typename Iterator, typename Body, typename Item>
    class do_group_task_forward: public task
    {
        static const size_t max_arg_size = 4;

        typedef parallel_do_feeder_impl<Body, Item> feeder_type;

        feeder_type& my_feeder;
        Iterator my_first;
        size_t my_size;

        do_group_task_forward( Iterator first, size_t size, feeder_type& feeder )
            : my_feeder(feeder), my_first(first), my_size(size)
        {}

        task* execute() __TBB_override
        {
            typedef do_iteration_task_iter<Iterator, Body, Item> iteration_type;
            __TBB_ASSERT( my_size>0, NULL );
            task_list list;
            task* t;
            size_t k=0;
            for(;;) {
                t = new( allocate_child() ) iteration_type( my_first, my_feeder );
                ++my_first;
                if( ++k==my_size ) break;
                list.push_back(*t);
            }
            set_ref_count(int(k+1));
            spawn(list);
            spawn_and_wait_for_all(*t);
            return NULL;
        }

        template<typename Iterator_, typename Body_, typename _Item> friend class do_task_iter;
    }; // class do_group_task_forward

    template<typename Body, typename Item>
    class do_group_task_input: public task
    {
        static const size_t max_arg_size = 4;

        typedef parallel_do_feeder_impl<Body, Item> feeder_type;

        feeder_type& my_feeder;
        size_t my_size;
        aligned_space<Item, max_arg_size> my_arg;

        do_group_task_input( feeder_type& feeder )
            : my_feeder(feeder), my_size(0)
        {}

        task* execute() __TBB_override
        {
#if __TBB_CPP11_RVALUE_REF_PRESENT
            typedef std::move_iterator<Item*> Item_iterator;
#else
            typedef Item* Item_iterator;
#endif
            typedef do_iteration_task_iter<Item_iterator, Body, Item> iteration_type;
            __TBB_ASSERT( my_size>0, NULL );
            task_list list;
            task* t;
            size_t k=0;
            for(;;) {
                t = new( allocate_child() ) iteration_type( Item_iterator(my_arg.begin() + k), my_feeder );
                if( ++k==my_size ) break;
                list.push_back(*t);
            }
            set_ref_count(int(k+1));
            spawn(list);
            spawn_and_wait_for_all(*t);
            return NULL;
        }

        ~do_group_task_input(){
            for( size_t k=0; k<my_size; ++k)
                (my_arg.begin() + k)->~Item();
        }

        template<typename Iterator_, typename Body_, typename Item_> friend class do_task_iter;
    }; // class do_group_task_input

    //! For internal use only.
    /** Gets block of iterations and packages them into a do_group_task.
        @ingroup algorithms */
    template<typename Iterator, typename Body, typename Item>
    class do_task_iter: public task
    {
        typedef parallel_do_feeder_impl<Body, Item> feeder_type;

    public:
        do_task_iter( Iterator first, Iterator last , feeder_type& feeder ) :
            my_first(first), my_last(last), my_feeder(feeder)
        {}

    private:
        Iterator my_first;
        Iterator my_last;
        feeder_type& my_feeder;

        /* Do not merge run(xxx) and run_xxx() methods. They are separated in order
            to make sure that compilers will eliminate unused argument of type xxx
            (that is will not put it on stack). The sole purpose of this argument
            is overload resolution.

            An alternative could be using template functions, but explicit specialization
            of member function templates is not supported for non specialized class
            templates. Besides template functions would always fall back to the least
            efficient variant (the one for input iterators) in case of iterators having
            custom tags derived from basic ones. */
        task* execute() __TBB_override
        {
            typedef typename std::iterator_traits<Iterator>::iterator_category iterator_tag;
            return run( (iterator_tag*)NULL );
        }

        /** This is the most restricted variant that operates on input iterators or
            iterators with unknown tags (tags not derived from the standard ones). **/
        inline task* run( void* ) { return run_for_input_iterator(); }

        task* run_for_input_iterator() {
            typedef do_group_task_input<Body, Item> block_type;

            block_type& t = *new( allocate_additional_child_of(*my_feeder.my_barrier) ) block_type(my_feeder);
            size_t k=0;
            while( !(my_first == my_last) ) {
                // Move semantics are automatically used when supported by the iterator
                new (t.my_arg.begin() + k) Item(*my_first);
                ++my_first;
                if( ++k==block_type::max_arg_size ) {
                    if ( !(my_first == my_last) )
                        recycle_to_reexecute();
                    break;
                }
            }
            if( k==0 ) {
                destroy(t);
                return NULL;
            } else {
                t.my_size = k;
                return &t;
            }
        }

        inline task* run( std::forward_iterator_tag* ) { return run_for_forward_iterator(); }

        task* run_for_forward_iterator() {
            typedef do_group_task_forward<Iterator, Body, Item> block_type;

            Iterator first = my_first;
            size_t k=0;
            while( !(my_first==my_last) ) {
                ++my_first;
                if( ++k==block_type::max_arg_size ) {
                    if ( !(my_first==my_last) )
                        recycle_to_reexecute();
                    break;
                }
            }
            return k==0 ? NULL : new( allocate_additional_child_of(*my_feeder.my_barrier) ) block_type(first, k, my_feeder);
        }

        inline task* run( std::random_access_iterator_tag* ) { return run_for_random_access_iterator(); }

        task* run_for_random_access_iterator() {
            typedef do_group_task_forward<Iterator, Body, Item> block_type;
            typedef do_iteration_task_iter<Iterator, Body, Item> iteration_type;

            size_t k = static_cast<size_t>(my_last-my_first);
            if( k > block_type::max_arg_size ) {
                Iterator middle = my_first + k/2;

                empty_task& c = *new( allocate_continuation() ) empty_task;
                do_task_iter& b = *new( c.allocate_child() ) do_task_iter(middle, my_last, my_feeder);
                recycle_as_child_of(c);

                my_last = middle;
                c.set_ref_count(2);
                c.spawn(b);
                return this;
            }else if( k != 0 ) {
                task_list list;
                task* t;
                size_t k1=0;
                for(;;) {
                    t = new( allocate_child() ) iteration_type(my_first, my_feeder);
                    ++my_first;
                    if( ++k1==k ) break;
                    list.push_back(*t);
                }
                set_ref_count(int(k+1));
                spawn(list);
                spawn_and_wait_for_all(*t);
            }
            return NULL;
        }
    }; // class do_task_iter

    //! For internal use only.
    /** Implements parallel iteration over a range.
        @ingroup algorithms */
    template<typename Iterator, typename Body, typename Item>
    void run_parallel_do( Iterator first, Iterator last, const Body& body
#if __TBB_TASK_GROUP_CONTEXT
        , task_group_context& context
#endif
        )
    {
        typedef do_task_iter<Iterator, Body, Item> root_iteration_task;
#if __TBB_TASK_GROUP_CONTEXT
        parallel_do_feeder_impl<Body, Item> feeder(context);
#else
        parallel_do_feeder_impl<Body, Item> feeder;
#endif
        feeder.my_body = &body;

        root_iteration_task &t = *new( feeder.my_barrier->allocate_child() ) root_iteration_task(first, last, feeder);

        feeder.my_barrier->set_ref_count(2);
        feeder.my_barrier->spawn_and_wait_for_all(t);
    }

    //! For internal use only.
    /** Detects types of Body's operator function arguments.
        @ingroup algorithms **/
    template<typename Iterator, typename Body, typename Item>
    void select_parallel_do( Iterator first, Iterator last, const Body& body, void (Body::*)(Item) const
#if __TBB_TASK_GROUP_CONTEXT
        , task_group_context& context
#endif
        )
    {
        run_parallel_do<Iterator, Body, typename ::tbb::internal::strip<Item>::type>( first, last, body
#if __TBB_TASK_GROUP_CONTEXT
            , context
#endif
            );
    }

    //! For internal use only.
    /** Detects types of Body's operator function arguments.
        @ingroup algorithms **/
    template<typename Iterator, typename Body, typename Item, typename _Item>
    void select_parallel_do( Iterator first, Iterator last, const Body& body, void (Body::*)(Item, parallel_do_feeder<_Item>&) const
#if __TBB_TASK_GROUP_CONTEXT
        , task_group_context& context
#endif
        )
    {
        run_parallel_do<Iterator, Body, typename ::tbb::internal::strip<Item>::type>( first, last, body
#if __TBB_TASK_GROUP_CONTEXT
            , context
#endif
            );
    }

} // namespace internal
} // namespace interface9
//! @endcond

/** \page parallel_do_body_req Requirements on parallel_do body
    Class \c Body implementing the concept of parallel_do body must define:
    - \code
        B::operator()(
                cv_item_type item,
                parallel_do_feeder<item_type>& feeder
        ) const

        OR

        B::operator()( cv_item_type& item ) const
      \endcode                                               Process item.
                                                             May be invoked concurrently  for the same \c this but different \c item.

    - \code item_type( const item_type& ) \endcode
                                                             Copy a work item.
    - \code ~item_type() \endcode                            Destroy a work item
**/

/** \name parallel_do
    See also requirements on \ref parallel_do_body_req "parallel_do Body". **/
//@{
//! Parallel iteration over a range, with optional addition of more work.
/** @ingroup algorithms */
template<typename Iterator, typename Body>
void parallel_do( Iterator first, Iterator last, const Body& body )
{
    if ( first == last )
        return;
#if __TBB_TASK_GROUP_CONTEXT
    task_group_context context(internal::PARALLEL_DO);
#endif
    interface9::internal::select_parallel_do( first, last, body, &Body::operator()
#if __TBB_TASK_GROUP_CONTEXT
        , context
#endif
        );
}

template<typename Range, typename Body>
void parallel_do(Range& rng, const Body& body) {
    parallel_do(tbb::internal::first(rng), tbb::internal::last(rng), body);
}

template<typename Range, typename Body>
void parallel_do(const Range& rng, const Body& body) {
    parallel_do(tbb::internal::first(rng), tbb::internal::last(rng), body);
}

#if __TBB_TASK_GROUP_CONTEXT
//! Parallel iteration over a range, with optional addition of more work and user-supplied context
/** @ingroup algorithms */
template<typename Iterator, typename Body>
void parallel_do( Iterator first, Iterator last, const Body& body, task_group_context& context  )
{
    if ( first == last )
        return;
    interface9::internal::select_parallel_do( first, last, body, &Body::operator(), context );
}

template<typename Range, typename Body>
void parallel_do(Range& rng, const Body& body, task_group_context& context) {
    parallel_do(tbb::internal::first(rng), tbb::internal::last(rng), body, context);
}

template<typename Range, typename Body>
void parallel_do(const Range& rng, const Body& body, task_group_context& context) {
    parallel_do(tbb::internal::first(rng), tbb::internal::last(rng), body, context);
}

#endif // __TBB_TASK_GROUP_CONTEXT

//@}

using interface9::parallel_do_feeder;

} // namespace

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_parallel_do_H_include_area

#endif /* __TBB_parallel_do_H */
