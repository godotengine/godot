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

#ifndef __TBB_concurrent_queue_H
#define __TBB_concurrent_queue_H

#define __TBB_concurrent_queue_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "internal/_concurrent_queue_impl.h"
#include "internal/_allocator_traits.h"

namespace tbb {

namespace strict_ppl {

//! A high-performance thread-safe non-blocking concurrent queue.
/** Multiple threads may each push and pop concurrently.
    Assignment construction is not allowed.
    @ingroup containers */
template<typename T, typename A = cache_aligned_allocator<T> >
class concurrent_queue: public internal::concurrent_queue_base_v3<T> {
    template<typename Container, typename Value> friend class internal::concurrent_queue_iterator;

    //! Allocator type
    typedef typename tbb::internal::allocator_rebind<A, char>::type page_allocator_type;
    page_allocator_type my_allocator;

    //! Allocates a block of size n (bytes)
    virtual void *allocate_block( size_t n ) __TBB_override {
        void *b = reinterpret_cast<void*>(my_allocator.allocate( n ));
        if( !b )
            internal::throw_exception(internal::eid_bad_alloc);
        return b;
    }

    //! Deallocates block created by allocate_block.
    virtual void deallocate_block( void *b, size_t n ) __TBB_override {
        my_allocator.deallocate( reinterpret_cast<char*>(b), n );
    }

    static void copy_construct_item(T* location, const void* src){
        new (location) T(*static_cast<const T*>(src));
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    static void move_construct_item(T* location, const void* src) {
        new (location) T( std::move(*static_cast<T*>(const_cast<void*>(src))) );
    }
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */
public:
    //! Element type in the queue.
    typedef T value_type;

    //! Reference type
    typedef T& reference;

    //! Const reference type
    typedef const T& const_reference;

    //! Integral type for representing size of the queue.
    typedef size_t size_type;

    //! Difference type for iterator
    typedef ptrdiff_t difference_type;

    //! Allocator type
    typedef A allocator_type;

    //! Construct empty queue
    explicit concurrent_queue(const allocator_type& a = allocator_type()) :
        my_allocator( a )
    {
    }

    //! [begin,end) constructor
    template<typename InputIterator>
    concurrent_queue( InputIterator begin, InputIterator end, const allocator_type& a = allocator_type()) :
        my_allocator( a )
    {
        for( ; begin != end; ++begin )
            this->push(*begin);
    }

    //! Copy constructor
    concurrent_queue( const concurrent_queue& src, const allocator_type& a = allocator_type()) :
        internal::concurrent_queue_base_v3<T>(), my_allocator( a )
    {
        this->assign( src, copy_construct_item );
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    //! Move constructors
    concurrent_queue( concurrent_queue&& src ) :
        internal::concurrent_queue_base_v3<T>(), my_allocator( std::move(src.my_allocator) )
    {
        this->internal_swap( src );
    }

    concurrent_queue( concurrent_queue&& src, const allocator_type& a ) :
        internal::concurrent_queue_base_v3<T>(), my_allocator( a )
    {
        // checking that memory allocated by one instance of allocator can be deallocated
        // with another
        if( my_allocator == src.my_allocator) {
            this->internal_swap( src );
        } else {
            // allocators are different => performing per-element move
            this->assign( src, move_construct_item );
            src.clear();
        }
    }
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

    //! Destroy queue
    ~concurrent_queue();

    //! Enqueue an item at tail of queue.
    void push( const T& source ) {
        this->internal_push( &source, copy_construct_item );
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    void push( T&& source ) {
        this->internal_push( &source, move_construct_item );
    }

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    template<typename... Arguments>
    void emplace( Arguments&&... args ) {
        push( T(std::forward<Arguments>( args )...) );
    }
#endif //__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

    //! Attempt to dequeue an item from head of queue.
    /** Does not wait for item to become available.
        Returns true if successful; false otherwise. */
    bool try_pop( T& result ) {
        return this->internal_try_pop( &result );
    }

    //! Return the number of items in the queue; thread unsafe
    size_type unsafe_size() const {return this->internal_size();}

    //! Equivalent to size()==0.
    bool empty() const {return this->internal_empty();}

    //! Clear the queue. not thread-safe.
    void clear() ;

    //! Return allocator object
    allocator_type get_allocator() const { return this->my_allocator; }

    typedef internal::concurrent_queue_iterator<concurrent_queue,T> iterator;
    typedef internal::concurrent_queue_iterator<concurrent_queue,const T> const_iterator;

    //------------------------------------------------------------------------
    // The iterators are intended only for debugging.  They are slow and not thread safe.
    //------------------------------------------------------------------------
    iterator unsafe_begin() {return iterator(*this);}
    iterator unsafe_end() {return iterator();}
    const_iterator unsafe_begin() const {return const_iterator(*this);}
    const_iterator unsafe_end() const {return const_iterator();}
} ;

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
// Deduction guide for the constructor from two iterators
template<typename InputIterator,
         typename T = typename std::iterator_traits<InputIterator>::value_type,
         typename A = cache_aligned_allocator<T>
> concurrent_queue(InputIterator, InputIterator, const A& = A())
-> concurrent_queue<T, A>;
#endif /* __TBB_CPP17_DEDUCTION_GUIDES_PRESENT */

template<typename T, class A>
concurrent_queue<T,A>::~concurrent_queue() {
    clear();
    this->internal_finish_clear();
}

template<typename T, class A>
void concurrent_queue<T,A>::clear() {
    T value;
    while( !empty() ) try_pop(value);
}

} // namespace strict_ppl

//! A high-performance thread-safe blocking concurrent bounded queue.
/** This is the pre-PPL TBB concurrent queue which supports boundedness and blocking semantics.
    Note that method names agree with the PPL-style concurrent queue.
    Multiple threads may each push and pop concurrently.
    Assignment construction is not allowed.
    @ingroup containers */
template<typename T, class A = cache_aligned_allocator<T> >
class concurrent_bounded_queue: public internal::concurrent_queue_base_v8 {
    template<typename Container, typename Value> friend class internal::concurrent_queue_iterator;
    typedef typename tbb::internal::allocator_rebind<A, char>::type page_allocator_type;

    //! Allocator type
    page_allocator_type my_allocator;

    typedef typename concurrent_queue_base_v3::padded_page<T> padded_page;
    typedef typename concurrent_queue_base_v3::copy_specifics copy_specifics;

    //! Class used to ensure exception-safety of method "pop"
    class destroyer: internal::no_copy {
        T& my_value;
    public:
        destroyer( T& value ) : my_value(value) {}
        ~destroyer() {my_value.~T();}
    };

    T& get_ref( page& p, size_t index ) {
        __TBB_ASSERT( index<items_per_page, NULL );
        return (&static_cast<padded_page*>(static_cast<void*>(&p))->last)[index];
    }

    virtual void copy_item( page& dst, size_t index, const void* src ) __TBB_override {
        new( &get_ref(dst,index) ) T(*static_cast<const T*>(src));
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    virtual void move_item( page& dst, size_t index, const void* src ) __TBB_override {
        new( &get_ref(dst,index) ) T( std::move(*static_cast<T*>(const_cast<void*>(src))) );
    }
#else
    virtual void move_item( page&, size_t, const void* ) __TBB_override {
        __TBB_ASSERT( false, "Unreachable code" );
    }
#endif

    virtual void copy_page_item( page& dst, size_t dindex, const page& src, size_t sindex ) __TBB_override {
        new( &get_ref(dst,dindex) ) T( get_ref( const_cast<page&>(src), sindex ) );
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    virtual void move_page_item( page& dst, size_t dindex, const page& src, size_t sindex ) __TBB_override {
        new( &get_ref(dst,dindex) ) T( std::move(get_ref( const_cast<page&>(src), sindex )) );
    }
#else
    virtual void move_page_item( page&, size_t, const page&, size_t ) __TBB_override {
        __TBB_ASSERT( false, "Unreachable code" );
    }
#endif

    virtual void assign_and_destroy_item( void* dst, page& src, size_t index ) __TBB_override {
        T& from = get_ref(src,index);
        destroyer d(from);
        *static_cast<T*>(dst) = tbb::internal::move( from );
    }

    virtual page *allocate_page() __TBB_override {
        size_t n = sizeof(padded_page) + (items_per_page-1)*sizeof(T);
        page *p = reinterpret_cast<page*>(my_allocator.allocate( n ));
        if( !p )
            internal::throw_exception(internal::eid_bad_alloc);
        return p;
    }

    virtual void deallocate_page( page *p ) __TBB_override {
        size_t n = sizeof(padded_page) + (items_per_page-1)*sizeof(T);
        my_allocator.deallocate( reinterpret_cast<char*>(p), n );
    }

public:
    //! Element type in the queue.
    typedef T value_type;

    //! Allocator type
    typedef A allocator_type;

    //! Reference type
    typedef T& reference;

    //! Const reference type
    typedef const T& const_reference;

    //! Integral type for representing size of the queue.
    /** Note that the size_type is a signed integral type.
        This is because the size can be negative if there are pending pops without corresponding pushes. */
    typedef std::ptrdiff_t size_type;

    //! Difference type for iterator
    typedef std::ptrdiff_t difference_type;

    //! Construct empty queue
    explicit concurrent_bounded_queue(const allocator_type& a = allocator_type()) :
        concurrent_queue_base_v8( sizeof(T) ), my_allocator( a )
    {
    }

    //! Copy constructor
    concurrent_bounded_queue( const concurrent_bounded_queue& src, const allocator_type& a = allocator_type())
        : concurrent_queue_base_v8( sizeof(T) ), my_allocator( a )
    {
        assign( src );
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    //! Move constructors
    concurrent_bounded_queue( concurrent_bounded_queue&& src )
        : concurrent_queue_base_v8( sizeof(T) ), my_allocator( std::move(src.my_allocator) )
    {
        internal_swap( src );
    }

    concurrent_bounded_queue( concurrent_bounded_queue&& src, const allocator_type& a )
        : concurrent_queue_base_v8( sizeof(T) ), my_allocator( a )
    {
        // checking that memory allocated by one instance of allocator can be deallocated
        // with another
        if( my_allocator == src.my_allocator) {
            this->internal_swap( src );
        } else {
            // allocators are different => performing per-element move
            this->move_content( src );
            src.clear();
        }
    }
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

    //! [begin,end) constructor
    template<typename InputIterator>
    concurrent_bounded_queue( InputIterator begin, InputIterator end,
                              const allocator_type& a = allocator_type())
        : concurrent_queue_base_v8( sizeof(T) ), my_allocator( a )
    {
        for( ; begin != end; ++begin )
            internal_push_if_not_full(&*begin);
    }

    //! Destroy queue
    ~concurrent_bounded_queue();

    //! Enqueue an item at tail of queue.
    void push( const T& source ) {
        internal_push( &source );
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    //! Move an item at tail of queue.
    void push( T&& source ) {
        internal_push_move( &source );
    }

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    template<typename... Arguments>
    void emplace( Arguments&&... args ) {
        push( T(std::forward<Arguments>( args )...) );
    }
#endif /* __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT */
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

    //! Dequeue item from head of queue.
    /** Block until an item becomes available, and then dequeue it. */
    void pop( T& destination ) {
        internal_pop( &destination );
    }

#if TBB_USE_EXCEPTIONS
    //! Abort all pending queue operations
    void abort() {
        internal_abort();
    }
#endif

    //! Enqueue an item at tail of queue if queue is not already full.
    /** Does not wait for queue to become not full.
        Returns true if item is pushed; false if queue was already full. */
    bool try_push( const T& source ) {
        return internal_push_if_not_full( &source );
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    //! Move an item at tail of queue if queue is not already full.
    /** Does not wait for queue to become not full.
        Returns true if item is pushed; false if queue was already full. */
    bool try_push( T&& source ) {
        return internal_push_move_if_not_full( &source );
    }
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    template<typename... Arguments>
    bool try_emplace( Arguments&&... args ) {
        return try_push( T(std::forward<Arguments>( args )...) );
    }
#endif /* __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT */
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

    //! Attempt to dequeue an item from head of queue.
    /** Does not wait for item to become available.
        Returns true if successful; false otherwise. */
    bool try_pop( T& destination ) {
        return internal_pop_if_present( &destination );
    }

    //! Return number of pushes minus number of pops.
    /** Note that the result can be negative if there are pops waiting for the
        corresponding pushes.  The result can also exceed capacity() if there
        are push operations in flight. */
    size_type size() const {return internal_size();}

    //! Equivalent to size()<=0.
    bool empty() const {return internal_empty();}

    //! Maximum number of allowed elements
    size_type capacity() const {
        return my_capacity;
    }

    //! Set the capacity
    /** Setting the capacity to 0 causes subsequent try_push operations to always fail,
        and subsequent push operations to block forever. */
    void set_capacity( size_type new_capacity ) {
        internal_set_capacity( new_capacity, sizeof(T) );
    }

    //! return allocator object
    allocator_type get_allocator() const { return this->my_allocator; }

    //! clear the queue. not thread-safe.
    void clear() ;

    typedef internal::concurrent_queue_iterator<concurrent_bounded_queue,T> iterator;
    typedef internal::concurrent_queue_iterator<concurrent_bounded_queue,const T> const_iterator;

    //------------------------------------------------------------------------
    // The iterators are intended only for debugging.  They are slow and not thread safe.
    //------------------------------------------------------------------------
    iterator unsafe_begin() {return iterator(*this);}
    iterator unsafe_end() {return iterator();}
    const_iterator unsafe_begin() const {return const_iterator(*this);}
    const_iterator unsafe_end() const {return const_iterator();}

};

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
// guide for concurrent_bounded_queue(InputIterator, InputIterator, ...)
template<typename InputIterator,
         typename T = typename std::iterator_traits<InputIterator>::value_type,
         typename A = cache_aligned_allocator<T>
> concurrent_bounded_queue(InputIterator, InputIterator, const A& = A())
-> concurrent_bounded_queue<T, A>;
#endif /* __TBB_CPP17_DEDUCTION_GUIDES_PRESENT */

template<typename T, class A>
concurrent_bounded_queue<T,A>::~concurrent_bounded_queue() {
    clear();
    internal_finish_clear();
}

template<typename T, class A>
void concurrent_bounded_queue<T,A>::clear() {
    T value;
    while( try_pop(value) ) /*noop*/;
}

using strict_ppl::concurrent_queue;

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_concurrent_queue_H_include_area

#endif /* __TBB_concurrent_queue_H */
