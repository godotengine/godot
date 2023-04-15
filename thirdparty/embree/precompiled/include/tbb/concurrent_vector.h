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

#ifndef __TBB_concurrent_vector_H
#define __TBB_concurrent_vector_H

#define __TBB_concurrent_vector_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "tbb_stddef.h"
#include "tbb_exception.h"
#include "atomic.h"
#include "cache_aligned_allocator.h"
#include "blocked_range.h"
#include "tbb_machine.h"
#include "tbb_profiling.h"
#include <new>
#include <cstring>   // for memset()
#include __TBB_STD_SWAP_HEADER
#include <algorithm>
#include <iterator>

#include "internal/_allocator_traits.h"

#if _MSC_VER==1500 && !__INTEL_COMPILER
    // VS2008/VC9 seems to have an issue; limits pull in math.h
    #pragma warning( push )
    #pragma warning( disable: 4985 )
#endif
#include <limits> /* std::numeric_limits */
#if _MSC_VER==1500 && !__INTEL_COMPILER
    #pragma warning( pop )
#endif

#if __TBB_INITIALIZER_LISTS_PRESENT
    #include <initializer_list>
#endif

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (push)
#if defined(_Wp64)
    #pragma warning (disable: 4267)
#endif
    #pragma warning (disable: 4127) //warning C4127: conditional expression is constant
#endif

namespace tbb {

template<typename T, class A = cache_aligned_allocator<T> >
class concurrent_vector;

//! @cond INTERNAL
namespace internal {

    template<typename Container, typename Value>
    class vector_iterator;

    //! Bad allocation marker
    static void *const vector_allocation_error_flag = reinterpret_cast<void*>(size_t(63));

    //! Exception helper function
    template<typename T>
    void handle_unconstructed_elements(T* array, size_t n_of_elements){
        std::memset( static_cast<void*>(array), 0, n_of_elements * sizeof( T ) );
    }

    //! Base class of concurrent vector implementation.
    /** @ingroup containers */
    class concurrent_vector_base_v3 {
    protected:

        // Basic types declarations
        typedef size_t segment_index_t;
        typedef size_t size_type;

        // Using enumerations due to Mac linking problems of static const variables
        enum {
            // Size constants
            default_initial_segments = 1, // 2 initial items
            //! Number of slots for segment pointers inside the class
            pointers_per_short_table = 3, // to fit into 8 words of entire structure
            pointers_per_long_table = sizeof(segment_index_t) * 8 // one segment per bit
        };

        struct segment_not_used {};
        struct segment_allocated {};
        struct segment_allocation_failed {};

        class segment_t;
        class segment_value_t {
            void* array;
        private:
            //TODO: More elegant way to grant access to selected functions _only_?
            friend class segment_t;
            explicit segment_value_t(void* an_array):array(an_array) {}
        public:
            friend bool operator==(segment_value_t const& lhs, segment_not_used ) { return lhs.array == 0;}
            friend bool operator==(segment_value_t const& lhs, segment_allocated) { return lhs.array > internal::vector_allocation_error_flag;}
            friend bool operator==(segment_value_t const& lhs, segment_allocation_failed) { return lhs.array == internal::vector_allocation_error_flag;}
            template<typename argument_type>
            friend bool operator!=(segment_value_t const& lhs, argument_type arg) { return ! (lhs == arg);}

            template<typename T>
            T* pointer() const {  return static_cast<T*>(const_cast<void*>(array)); }
        };

        friend void enforce_segment_allocated(segment_value_t const& s, internal::exception_id exception = eid_bad_last_alloc){
            if(s != segment_allocated()){
                internal::throw_exception(exception);
            }
        }

        // Segment pointer.
        class segment_t {
            atomic<void*> array;
        public:
            segment_t(){ store<relaxed>(segment_not_used());}
            //Copy ctor and assignment operator are defined to ease using of stl algorithms.
            //These algorithms usually not a synchronization point, so, semantic is
            //intentionally relaxed here.
            segment_t(segment_t const& rhs ){ array.store<relaxed>(rhs.array.load<relaxed>());}

            void swap(segment_t & rhs ){
                tbb::internal::swap<relaxed>(array, rhs.array);
            }

            segment_t& operator=(segment_t const& rhs ){
                array.store<relaxed>(rhs.array.load<relaxed>());
                return *this;
            }

            template<memory_semantics M>
            segment_value_t load() const { return segment_value_t(array.load<M>());}

            template<memory_semantics M>
            void store(segment_not_used) {
                array.store<M>(0);
            }

            template<memory_semantics M>
            void store(segment_allocation_failed) {
                __TBB_ASSERT(load<relaxed>() != segment_allocated(),"transition from \"allocated\" to \"allocation failed\" state looks non-logical");
                array.store<M>(internal::vector_allocation_error_flag);
            }

            template<memory_semantics M>
            void store(void* allocated_segment_pointer) __TBB_NOEXCEPT(true) {
                __TBB_ASSERT(segment_value_t(allocated_segment_pointer) == segment_allocated(),
                     "other overloads of store should be used for marking segment as not_used or allocation_failed" );
                array.store<M>(allocated_segment_pointer);
            }

#if TBB_USE_ASSERT
            ~segment_t() {
                __TBB_ASSERT(load<relaxed>() != segment_allocated(), "should have been freed by clear" );
            }
#endif /* TBB_USE_ASSERT */
        };
        friend void swap(segment_t & , segment_t & ) __TBB_NOEXCEPT(true);

        // Data fields

        //! allocator function pointer
        void* (*vector_allocator_ptr)(concurrent_vector_base_v3 &, size_t);

        //! count of segments in the first block
        atomic<size_type> my_first_block;

        //! Requested size of vector
        atomic<size_type> my_early_size;

        //! Pointer to the segments table
        atomic<segment_t*> my_segment;

        //! embedded storage of segment pointers
        segment_t my_storage[pointers_per_short_table];

        // Methods

        concurrent_vector_base_v3() {
            //Here the semantic is intentionally relaxed.
            //The reason this is next:
            //Object that is in middle of construction (i.e. its constructor is not yet finished)
            //cannot be used concurrently until the construction is finished.
            //Thus to flag other threads that construction is finished, some synchronization with
            //acquire-release semantic should be done by the (external) code that uses the vector.
            //So, no need to do the synchronization inside the vector.

            my_early_size.store<relaxed>(0);
            my_first_block.store<relaxed>(0); // here is not default_initial_segments
            my_segment.store<relaxed>(my_storage);
        }

        __TBB_EXPORTED_METHOD ~concurrent_vector_base_v3();

        //these helpers methods use the fact that segments are allocated so
        //that every segment size is a (increasing) power of 2.
        //with one exception 0 segment has size of 2 as well segment 1;
        //e.g. size of segment with index of 3 is 2^3=8;
        static segment_index_t segment_index_of( size_type index ) {
            return segment_index_t( __TBB_Log2( index|1 ) );
        }

        static segment_index_t segment_base( segment_index_t k ) {
            return (segment_index_t(1)<<k & ~segment_index_t(1));
        }

        static inline segment_index_t segment_base_index_of( segment_index_t &index ) {
            segment_index_t k = segment_index_of( index );
            index -= segment_base(k);
            return k;
        }

        static size_type segment_size( segment_index_t k ) {
            return segment_index_t(1)<<k; // fake value for k==0
        }


        static bool is_first_element_in_segment(size_type element_index){
            //check if element_index is a power of 2 that is at least 2.
            //The idea is to detect if the iterator crosses a segment boundary,
            //and 2 is the minimal index for which it's true
            __TBB_ASSERT(element_index, "there should be no need to call "
                                        "is_first_element_in_segment for 0th element" );
            return is_power_of_two_at_least( element_index, 2 );
        }

        //! An operation on an n-element array starting at begin.
        typedef void (__TBB_EXPORTED_FUNC *internal_array_op1)(void* begin, size_type n );

        //! An operation on n-element destination array and n-element source array.
        typedef void (__TBB_EXPORTED_FUNC *internal_array_op2)(void* dst, const void* src, size_type n );

        //! Internal structure for compact()
        struct internal_segments_table {
            segment_index_t first_block;
            segment_t table[pointers_per_long_table];
        };

        void __TBB_EXPORTED_METHOD internal_reserve( size_type n, size_type element_size, size_type max_size );
        size_type __TBB_EXPORTED_METHOD internal_capacity() const;
        void internal_grow( size_type start, size_type finish, size_type element_size, internal_array_op2 init, const void *src );
        size_type __TBB_EXPORTED_METHOD internal_grow_by( size_type delta, size_type element_size, internal_array_op2 init, const void *src );
        void* __TBB_EXPORTED_METHOD internal_push_back( size_type element_size, size_type& index );
        segment_index_t __TBB_EXPORTED_METHOD internal_clear( internal_array_op1 destroy );
        void* __TBB_EXPORTED_METHOD internal_compact( size_type element_size, void *table, internal_array_op1 destroy, internal_array_op2 copy );
        void __TBB_EXPORTED_METHOD internal_copy( const concurrent_vector_base_v3& src, size_type element_size, internal_array_op2 copy );
        void __TBB_EXPORTED_METHOD internal_assign( const concurrent_vector_base_v3& src, size_type element_size,
                              internal_array_op1 destroy, internal_array_op2 assign, internal_array_op2 copy );
        //! Obsolete
        void __TBB_EXPORTED_METHOD internal_throw_exception(size_type) const;
        void __TBB_EXPORTED_METHOD internal_swap(concurrent_vector_base_v3& v);

        void __TBB_EXPORTED_METHOD internal_resize( size_type n, size_type element_size, size_type max_size, const void *src,
                                                    internal_array_op1 destroy, internal_array_op2 init );
        size_type __TBB_EXPORTED_METHOD internal_grow_to_at_least_with_result( size_type new_size, size_type element_size, internal_array_op2 init, const void *src );

        //! Deprecated entry point for backwards compatibility to TBB 2.1.
        void __TBB_EXPORTED_METHOD internal_grow_to_at_least( size_type new_size, size_type element_size, internal_array_op2 init, const void *src );
private:
        //! Private functionality
        class helper;
        friend class helper;

        template<typename Container, typename Value>
        friend class vector_iterator;

    };

    inline void swap(concurrent_vector_base_v3::segment_t & lhs, concurrent_vector_base_v3::segment_t & rhs) __TBB_NOEXCEPT(true) {
        lhs.swap(rhs);
    }

    typedef concurrent_vector_base_v3 concurrent_vector_base;

    //! Meets requirements of a forward iterator for STL and a Value for a blocked_range.*/
    /** Value is either the T or const T type of the container.
        @ingroup containers */
    template<typename Container, typename Value>
    class vector_iterator
    {
        //! concurrent_vector over which we are iterating.
        Container* my_vector;

        //! Index into the vector
        size_t my_index;

        //! Caches my_vector-&gt;internal_subscript(my_index)
        /** NULL if cached value is not available */
        mutable Value* my_item;

        template<typename C, typename T>
        friend vector_iterator<C,T> operator+( ptrdiff_t offset, const vector_iterator<C,T>& v );

        template<typename C, typename T, typename U>
        friend bool operator==( const vector_iterator<C,T>& i, const vector_iterator<C,U>& j );

        template<typename C, typename T, typename U>
        friend bool operator<( const vector_iterator<C,T>& i, const vector_iterator<C,U>& j );

        template<typename C, typename T, typename U>
        friend ptrdiff_t operator-( const vector_iterator<C,T>& i, const vector_iterator<C,U>& j );

        template<typename C, typename U>
        friend class internal::vector_iterator;

#if !__TBB_TEMPLATE_FRIENDS_BROKEN
        template<typename T, class A>
        friend class tbb::concurrent_vector;
#else
public:
#endif

        vector_iterator( const Container& vector, size_t index, void *ptr = 0 ) :
            my_vector(const_cast<Container*>(&vector)),
            my_index(index),
            my_item(static_cast<Value*>(ptr))
        {}

    public:
        //! Default constructor
        vector_iterator() : my_vector(NULL), my_index(~size_t(0)), my_item(NULL) {}

        vector_iterator( const vector_iterator<Container,typename Container::value_type>& other ) :
            my_vector(other.my_vector),
            my_index(other.my_index),
            my_item(other.my_item)
        {}

        vector_iterator& operator=( const vector_iterator<Container,typename Container::value_type>& other )
        {
            my_vector=other.my_vector;
            my_index=other.my_index;
            my_item=other.my_item;
            return *this;
        }

        vector_iterator operator+( ptrdiff_t offset ) const {
            return vector_iterator( *my_vector, my_index+offset );
        }
        vector_iterator &operator+=( ptrdiff_t offset ) {
            my_index+=offset;
            my_item = NULL;
            return *this;
        }
        vector_iterator operator-( ptrdiff_t offset ) const {
            return vector_iterator( *my_vector, my_index-offset );
        }
        vector_iterator &operator-=( ptrdiff_t offset ) {
            my_index-=offset;
            my_item = NULL;
            return *this;
        }
        Value& operator*() const {
            Value* item = my_item;
            if( !item ) {
                item = my_item = &my_vector->internal_subscript(my_index);
            }
            __TBB_ASSERT( item==&my_vector->internal_subscript(my_index), "corrupt cache" );
            return *item;
        }
        Value& operator[]( ptrdiff_t k ) const {
            return my_vector->internal_subscript(my_index+k);
        }
        Value* operator->() const {return &operator*();}

        //! Pre increment
        vector_iterator& operator++() {
            size_t element_index = ++my_index;
            if( my_item ) {
                //TODO: consider using of knowledge about "first_block optimization" here as well?
                if( concurrent_vector_base::is_first_element_in_segment(element_index)) {
                    //if the iterator crosses a segment boundary, the pointer become invalid
                    //as possibly next segment is in another memory location
                    my_item= NULL;
                } else {
                    ++my_item;
                }
            }
            return *this;
        }

        //! Pre decrement
        vector_iterator& operator--() {
            __TBB_ASSERT( my_index>0, "operator--() applied to iterator already at beginning of concurrent_vector" );
            size_t element_index = my_index--;
            if( my_item ) {
                if(concurrent_vector_base::is_first_element_in_segment(element_index)) {
                    //if the iterator crosses a segment boundary, the pointer become invalid
                    //as possibly next segment is in another memory location
                    my_item= NULL;
                } else {
                    --my_item;
                }
            }
            return *this;
        }

        //! Post increment
        vector_iterator operator++(int) {
            vector_iterator result = *this;
            operator++();
            return result;
        }

        //! Post decrement
        vector_iterator operator--(int) {
            vector_iterator result = *this;
            operator--();
            return result;
        }

        // STL support

        typedef ptrdiff_t difference_type;
        typedef Value value_type;
        typedef Value* pointer;
        typedef Value& reference;
        typedef std::random_access_iterator_tag iterator_category;
    };

    template<typename Container, typename T>
    vector_iterator<Container,T> operator+( ptrdiff_t offset, const vector_iterator<Container,T>& v ) {
        return vector_iterator<Container,T>( *v.my_vector, v.my_index+offset );
    }

    template<typename Container, typename T, typename U>
    bool operator==( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return i.my_index==j.my_index && i.my_vector == j.my_vector;
    }

    template<typename Container, typename T, typename U>
    bool operator!=( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return !(i==j);
    }

    template<typename Container, typename T, typename U>
    bool operator<( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return i.my_index<j.my_index;
    }

    template<typename Container, typename T, typename U>
    bool operator>( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return j<i;
    }

    template<typename Container, typename T, typename U>
    bool operator>=( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return !(i<j);
    }

    template<typename Container, typename T, typename U>
    bool operator<=( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return !(j<i);
    }

    template<typename Container, typename T, typename U>
    ptrdiff_t operator-( const vector_iterator<Container,T>& i, const vector_iterator<Container,U>& j ) {
        return ptrdiff_t(i.my_index)-ptrdiff_t(j.my_index);
    }

    template<typename T, class A>
    class allocator_base {
    public:
        typedef typename tbb::internal::allocator_rebind<A, T>::type allocator_type;
        allocator_type my_allocator;
        allocator_base(const allocator_type &a = allocator_type() ) : my_allocator(a) {}
    };

} // namespace internal
//! @endcond

//! Concurrent vector container
/** concurrent_vector is a container having the following main properties:
    - It provides random indexed access to its elements. The index of the first element is 0.
    - It ensures safe concurrent growing its size (different threads can safely append new elements).
    - Adding new elements does not invalidate existing iterators and does not change indices of existing items.

@par Compatibility
    The class meets all Container Requirements and Reversible Container Requirements from
    C++ Standard (See ISO/IEC 14882:2003(E), clause 23.1). But it doesn't meet
    Sequence Requirements due to absence of insert() and erase() methods.

@par Exception Safety
    Methods working with memory allocation and/or new elements construction can throw an
    exception if allocator fails to allocate memory or element's default constructor throws one.
    Concurrent vector's element of type T must conform to the following requirements:
    - Throwing an exception is forbidden for destructor of T.
    - Default constructor of T must not throw an exception OR its non-virtual destructor must safely work when its object memory is zero-initialized.
    .
    Otherwise, the program's behavior is undefined.
@par
    If an exception happens inside growth or assignment operation, an instance of the vector becomes invalid unless it is stated otherwise in the method documentation.
    Invalid state means:
    - There are no guarantees that all items were initialized by a constructor. The rest of items is zero-filled, including item where exception happens.
    - An invalid vector instance cannot be repaired; it is unable to grow anymore.
    - Size and capacity reported by the vector are incorrect, and calculated as if the failed operation were successful.
    - Attempt to access not allocated elements using operator[] or iterators results in access violation or segmentation fault exception, and in case of using at() method a C++ exception is thrown.
    .
    If a concurrent grow operation successfully completes, all the elements it has added to the vector will remain valid and accessible even if one of subsequent grow operations fails.

@par Fragmentation
    Unlike an STL vector, a concurrent_vector does not move existing elements if it needs
    to allocate more memory. The container is divided into a series of contiguous arrays of
    elements. The first reservation, growth, or assignment operation determines the size of
    the first array. Using small number of elements as initial size incurs fragmentation that
    may increase element access time. Internal layout can be optimized by method compact() that
    merges several smaller arrays into one solid.

@par Changes since TBB 2.1
    - Fixed guarantees of concurrent_vector::size() and grow_to_at_least() methods to assure elements are allocated.
    - Methods end()/rbegin()/back() are partly thread-safe since they use size() to get the end of vector
    - Added resize() methods (not thread-safe)
    - Added cbegin/cend/crbegin/crend methods
    - Changed return type of methods grow* and push_back to iterator

@par Changes since TBB 2.0
    - Implemented exception-safety guarantees
    - Added template argument for allocator
    - Added allocator argument in constructors
    - Faster index calculation
    - First growth call specifies a number of segments to be merged in the first allocation.
    - Fixed memory blow up for swarm of vector's instances of small size
    - Added grow_by(size_type n, const_reference t) growth using copying constructor to init new items.
    - Added STL-like constructors.
    - Added operators ==, < and derivatives
    - Added at() method, approved for using after an exception was thrown inside the vector
    - Added get_allocator() method.
    - Added assign() methods
    - Added compact() method to defragment first segments
    - Added swap() method
    - range() defaults on grainsize = 1 supporting auto grainsize algorithms.

    @ingroup containers */
template<typename T, class A>
class concurrent_vector: protected internal::allocator_base<T, A>,
                         private internal::concurrent_vector_base {
private:
    template<typename I>
    class generic_range_type: public blocked_range<I> {
    public:
        typedef T value_type;
        typedef T& reference;
        typedef const T& const_reference;
        typedef I iterator;
        typedef ptrdiff_t difference_type;
        generic_range_type( I begin_, I end_, size_t grainsize_ = 1) : blocked_range<I>(begin_,end_,grainsize_) {}
        template<typename U>
        generic_range_type( const generic_range_type<U>& r) : blocked_range<I>(r.begin(),r.end(),r.grainsize()) {}
        generic_range_type( generic_range_type& r, split ) : blocked_range<I>(r,split()) {}
    };

    template<typename C, typename U>
    friend class internal::vector_iterator;

public:
    //------------------------------------------------------------------------
    // STL compatible types
    //------------------------------------------------------------------------
    typedef internal::concurrent_vector_base_v3::size_type size_type;
    typedef typename internal::allocator_base<T, A>::allocator_type allocator_type;

    typedef T value_type;
    typedef ptrdiff_t difference_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T *pointer;
    typedef const T *const_pointer;

    typedef internal::vector_iterator<concurrent_vector,T> iterator;
    typedef internal::vector_iterator<concurrent_vector,const T> const_iterator;

#if !defined(_MSC_VER) || _CPPLIB_VER>=300
    // Assume ISO standard definition of std::reverse_iterator
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
#else
    // Use non-standard std::reverse_iterator
    typedef std::reverse_iterator<iterator,T,T&,T*> reverse_iterator;
    typedef std::reverse_iterator<const_iterator,T,const T&,const T*> const_reverse_iterator;
#endif /* defined(_MSC_VER) && (_MSC_VER<1300) */

    //------------------------------------------------------------------------
    // Parallel algorithm support
    //------------------------------------------------------------------------
    typedef generic_range_type<iterator> range_type;
    typedef generic_range_type<const_iterator> const_range_type;

    //------------------------------------------------------------------------
    // STL compatible constructors & destructors
    //------------------------------------------------------------------------

    //! Construct empty vector.
    explicit concurrent_vector(const allocator_type &a = allocator_type())
        : internal::allocator_base<T, A>(a), internal::concurrent_vector_base()
    {
        vector_allocator_ptr = &internal_allocator;
    }

    //Constructors are not required to have synchronization
    //(for more details see comment in the concurrent_vector_base constructor).
#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Constructor from initializer_list
    concurrent_vector(std::initializer_list<T> init_list, const allocator_type &a = allocator_type())
        : internal::allocator_base<T, A>(a), internal::concurrent_vector_base()
    {
        vector_allocator_ptr = &internal_allocator;
        __TBB_TRY {
            internal_assign_iterators(init_list.begin(), init_list.end());
        } __TBB_CATCH(...) {
            segment_t *table = my_segment.load<relaxed>();;
            internal_free_segments( table, internal_clear(&destroy_array), my_first_block.load<relaxed>());
            __TBB_RETHROW();
        }

    }
#endif //# __TBB_INITIALIZER_LISTS_PRESENT

    //! Copying constructor
    concurrent_vector( const concurrent_vector& vector, const allocator_type& a = allocator_type() )
        : internal::allocator_base<T, A>(a), internal::concurrent_vector_base()
    {
        vector_allocator_ptr = &internal_allocator;
        __TBB_TRY {
            internal_copy(vector, sizeof(T), &copy_array);
        } __TBB_CATCH(...) {
            segment_t *table = my_segment.load<relaxed>();
            internal_free_segments( table, internal_clear(&destroy_array), my_first_block.load<relaxed>());
            __TBB_RETHROW();
        }
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    //! Move constructor
    //TODO add __TBB_NOEXCEPT(true) and static_assert(std::has_nothrow_move_constructor<A>::value)
    concurrent_vector( concurrent_vector&& source)
        : internal::allocator_base<T, A>(std::move(source)), internal::concurrent_vector_base()
    {
        vector_allocator_ptr = &internal_allocator;
        concurrent_vector_base_v3::internal_swap(source);
    }

    concurrent_vector( concurrent_vector&& source, const allocator_type& a)
        : internal::allocator_base<T, A>(a), internal::concurrent_vector_base()
    {
        vector_allocator_ptr = &internal_allocator;
        //C++ standard requires instances of an allocator being compared for equality,
        //which means that memory allocated by one instance is possible to deallocate with the other one.
        if (a == source.my_allocator) {
            concurrent_vector_base_v3::internal_swap(source);
        } else {
            __TBB_TRY {
                internal_copy(source, sizeof(T), &move_array);
            } __TBB_CATCH(...) {
                segment_t *table = my_segment.load<relaxed>();
                internal_free_segments( table, internal_clear(&destroy_array), my_first_block.load<relaxed>());
                __TBB_RETHROW();
            }
        }
    }

#endif

    //! Copying constructor for vector with different allocator type
    template<class M>
    __TBB_DEPRECATED concurrent_vector( const concurrent_vector<T, M>& vector, const allocator_type& a = allocator_type() )
        : internal::allocator_base<T, A>(a), internal::concurrent_vector_base()
    {
        vector_allocator_ptr = &internal_allocator;
        __TBB_TRY {
            internal_copy(vector.internal_vector_base(), sizeof(T), &copy_array);
        } __TBB_CATCH(...) {
            segment_t *table = my_segment.load<relaxed>();
            internal_free_segments( table, internal_clear(&destroy_array), my_first_block.load<relaxed>() );
            __TBB_RETHROW();
        }
    }

    //! Construction with initial size specified by argument n
    explicit concurrent_vector(size_type n)
    {
        vector_allocator_ptr = &internal_allocator;
        __TBB_TRY {
            internal_resize( n, sizeof(T), max_size(), NULL, &destroy_array, &initialize_array );
        } __TBB_CATCH(...) {
            segment_t *table = my_segment.load<relaxed>();
            internal_free_segments( table, internal_clear(&destroy_array), my_first_block.load<relaxed>() );
            __TBB_RETHROW();
        }
    }

    //! Construction with initial size specified by argument n, initialization by copying of t, and given allocator instance
    concurrent_vector(size_type n, const_reference t, const allocator_type& a = allocator_type())
        : internal::allocator_base<T, A>(a)
    {
        vector_allocator_ptr = &internal_allocator;
        __TBB_TRY {
            internal_resize( n, sizeof(T), max_size(), static_cast<const void*>(&t), &destroy_array, &initialize_array_by );
        } __TBB_CATCH(...) {
            segment_t *table = my_segment.load<relaxed>();
            internal_free_segments( table, internal_clear(&destroy_array), my_first_block.load<relaxed>() );
            __TBB_RETHROW();
        }
    }

    //! Construction with copying iteration range and given allocator instance
    template<class I>
    concurrent_vector(I first, I last, const allocator_type &a = allocator_type())
        : internal::allocator_base<T, A>(a)
    {
        vector_allocator_ptr = &internal_allocator;
        __TBB_TRY {
            internal_assign_range(first, last, static_cast<is_integer_tag<std::numeric_limits<I>::is_integer> *>(0) );
        } __TBB_CATCH(...) {
            segment_t *table = my_segment.load<relaxed>();
            internal_free_segments( table, internal_clear(&destroy_array), my_first_block.load<relaxed>() );
            __TBB_RETHROW();
        }
    }

    //! Assignment
    concurrent_vector& operator=( const concurrent_vector& vector ) {
        if( this != &vector )
            internal_assign(vector, sizeof(T), &destroy_array, &assign_array, &copy_array);
        return *this;
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    //TODO: add __TBB_NOEXCEPT()
    //! Move assignment
    concurrent_vector& operator=( concurrent_vector&& other ) {
        __TBB_ASSERT(this != &other, "Move assignment to itself is prohibited ");
        typedef typename tbb::internal::allocator_traits<A>::propagate_on_container_move_assignment pocma_t;
        if(pocma_t::value || this->my_allocator == other.my_allocator) {
            concurrent_vector trash (std::move(*this));
            internal_swap(other);
            tbb::internal::allocator_move_assignment(this->my_allocator, other.my_allocator, pocma_t());
        } else {
            internal_assign(other, sizeof(T), &destroy_array, &move_assign_array, &move_array);
        }
        return *this;
    }
#endif
    //TODO: add an template assignment operator? (i.e. with different element type)

    //! Assignment for vector with different allocator type
    template<class M>
    __TBB_DEPRECATED concurrent_vector& operator=( const concurrent_vector<T, M>& vector ) {
        if( static_cast<void*>( this ) != static_cast<const void*>( &vector ) )
            internal_assign(vector.internal_vector_base(),
                sizeof(T), &destroy_array, &assign_array, &copy_array);
        return *this;
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Assignment for initializer_list
    concurrent_vector& operator=( std::initializer_list<T> init_list ) {
        internal_clear(&destroy_array);
        internal_assign_iterators(init_list.begin(), init_list.end());
        return *this;
    }
#endif //#if __TBB_INITIALIZER_LISTS_PRESENT

    //------------------------------------------------------------------------
    // Concurrent operations
    //------------------------------------------------------------------------
    //! Grow by "delta" elements.
    /** Returns iterator pointing to the first new element. */
    iterator grow_by( size_type delta ) {
        return iterator(*this, delta ? internal_grow_by( delta, sizeof(T), &initialize_array, NULL ) : my_early_size.load());
    }

    //! Grow by "delta" elements using copying constructor.
    /** Returns iterator pointing to the first new element. */
    iterator grow_by( size_type delta, const_reference t ) {
        return iterator(*this, delta ? internal_grow_by( delta, sizeof(T), &initialize_array_by, static_cast<const void*>(&t) ) : my_early_size.load());
    }

    /** Returns iterator pointing to the first new element. */
    template<typename I>
    iterator grow_by( I first, I last ) {
        typename std::iterator_traits<I>::difference_type delta = std::distance(first, last);
        __TBB_ASSERT( delta >= 0, NULL);

        return iterator(*this, delta ? internal_grow_by(delta, sizeof(T), &copy_range<I>, static_cast<const void*>(&first)) : my_early_size.load());
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    /** Returns iterator pointing to the first new element. */
    iterator grow_by( std::initializer_list<T> init_list ) {
        return grow_by( init_list.begin(), init_list.end() );
    }
#endif //#if __TBB_INITIALIZER_LISTS_PRESENT

    //! Append minimal sequence of elements such that size()>=n.
    /** The new elements are default constructed.  Blocks until all elements in range [0..n) are allocated.
        May return while other elements are being constructed by other threads.
        Returns iterator that points to beginning of appended sequence.
        If no elements were appended, returns iterator pointing to nth element. */
    iterator grow_to_at_least( size_type n ) {
        size_type m=0;
        if( n ) {
            m = internal_grow_to_at_least_with_result( n, sizeof(T), &initialize_array, NULL );
            if( m>n ) m=n;
        }
        return iterator(*this, m);
    };

    /** Analogous to grow_to_at_least( size_type n ) with exception that the new
        elements are initialized by copying of t instead of default construction. */
    iterator grow_to_at_least( size_type n, const_reference t ) {
        size_type m=0;
        if( n ) {
            m = internal_grow_to_at_least_with_result( n, sizeof(T), &initialize_array_by, &t);
            if( m>n ) m=n;
        }
        return iterator(*this, m);
    };

    //! Push item
    /** Returns iterator pointing to the new element. */
    iterator push_back( const_reference item )
    {
        push_back_helper prolog(*this);
        new(prolog.internal_push_back_result()) T(item);
        return prolog.return_iterator_and_dismiss();
    }

#if    __TBB_CPP11_RVALUE_REF_PRESENT
    //! Push item, move-aware
    /** Returns iterator pointing to the new element. */
    iterator push_back(  T&& item )
    {
        push_back_helper prolog(*this);
        new(prolog.internal_push_back_result()) T(std::move(item));
        return prolog.return_iterator_and_dismiss();
    }
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    //! Push item, create item "in place" with provided arguments
    /** Returns iterator pointing to the new element. */
    template<typename... Args>
    iterator emplace_back(  Args&&... args )
    {
        push_back_helper prolog(*this);
        new(prolog.internal_push_back_result()) T(std::forward<Args>(args)...);
        return prolog.return_iterator_and_dismiss();
    }
#endif //__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif //__TBB_CPP11_RVALUE_REF_PRESENT
    //! Get reference to element at given index.
    /** This method is thread-safe for concurrent reads, and also while growing the vector,
        as long as the calling thread has checked that index < size(). */
    reference operator[]( size_type index ) {
        return internal_subscript(index);
    }

    //! Get const reference to element at given index.
    const_reference operator[]( size_type index ) const {
        return internal_subscript(index);
    }

    //! Get reference to element at given index. Throws exceptions on errors.
    reference at( size_type index ) {
        return internal_subscript_with_exceptions(index);
    }

    //! Get const reference to element at given index. Throws exceptions on errors.
    const_reference at( size_type index ) const {
        return internal_subscript_with_exceptions(index);
    }

    //! Get range for iterating with parallel algorithms
    range_type range( size_t grainsize = 1 ) {
        return range_type( begin(), end(), grainsize );
    }

    //! Get const range for iterating with parallel algorithms
    const_range_type range( size_t grainsize = 1 ) const {
        return const_range_type( begin(), end(), grainsize );
    }

    //------------------------------------------------------------------------
    // Capacity
    //------------------------------------------------------------------------
    //! Return size of vector. It may include elements under construction
    size_type size() const {
        size_type sz = my_early_size, cp = internal_capacity();
        return cp < sz ? cp : sz;
    }

    //! Return false if vector is not empty or has elements under construction at least.
    bool empty() const {return !my_early_size;}

    //! Maximum size to which array can grow without allocating more memory. Concurrent allocations are not included in the value.
    size_type capacity() const {return internal_capacity();}

    //! Allocate enough space to grow to size n without having to allocate more memory later.
    /** Like most of the methods provided for STL compatibility, this method is *not* thread safe.
        The capacity afterwards may be bigger than the requested reservation. */
    void reserve( size_type n ) {
        if( n )
            internal_reserve(n, sizeof(T), max_size());
    }

    //! Resize the vector. Not thread-safe.
    void resize( size_type n ) {
        internal_resize( n, sizeof(T), max_size(), NULL, &destroy_array, &initialize_array );
    }

    //! Resize the vector, copy t for new elements. Not thread-safe.
    void resize( size_type n, const_reference t ) {
        internal_resize( n, sizeof(T), max_size(), static_cast<const void*>(&t), &destroy_array, &initialize_array_by );
    }

    //! Optimize memory usage and fragmentation.
    void shrink_to_fit();

    //! Upper bound on argument to reserve.
    size_type max_size() const {return (~size_type(0))/sizeof(T);}

    //------------------------------------------------------------------------
    // STL support
    //------------------------------------------------------------------------

    //! start iterator
    iterator begin() {return iterator(*this,0);}
    //! end iterator
    iterator end() {return iterator(*this,size());}
    //! start const iterator
    const_iterator begin() const {return const_iterator(*this,0);}
    //! end const iterator
    const_iterator end() const {return const_iterator(*this,size());}
    //! start const iterator
    const_iterator cbegin() const {return const_iterator(*this,0);}
    //! end const iterator
    const_iterator cend() const {return const_iterator(*this,size());}
    //! reverse start iterator
    reverse_iterator rbegin() {return reverse_iterator(end());}
    //! reverse end iterator
    reverse_iterator rend() {return reverse_iterator(begin());}
    //! reverse start const iterator
    const_reverse_iterator rbegin() const {return const_reverse_iterator(end());}
    //! reverse end const iterator
    const_reverse_iterator rend() const {return const_reverse_iterator(begin());}
    //! reverse start const iterator
    const_reverse_iterator crbegin() const {return const_reverse_iterator(end());}
    //! reverse end const iterator
    const_reverse_iterator crend() const {return const_reverse_iterator(begin());}
    //! the first item
    reference front() {
        __TBB_ASSERT( size()>0, NULL);
        const segment_value_t& segment_value = my_segment[0].template load<relaxed>();
        return (segment_value.template pointer<T>())[0];
    }
    //! the first item const
    const_reference front() const {
        __TBB_ASSERT( size()>0, NULL);
        const segment_value_t& segment_value = my_segment[0].template load<relaxed>();
        return (segment_value.template pointer<const T>())[0];
    }
    //! the last item
    reference back() {
        __TBB_ASSERT( size()>0, NULL);
        return internal_subscript( size()-1 );
    }
    //! the last item const
    const_reference back() const {
        __TBB_ASSERT( size()>0, NULL);
        return internal_subscript( size()-1 );
    }
    //! return allocator object
    allocator_type get_allocator() const { return this->my_allocator; }

    //! assign n items by copying t item
    void assign(size_type n, const_reference t) {
        clear();
        internal_resize( n, sizeof(T), max_size(), static_cast<const void*>(&t), &destroy_array, &initialize_array_by );
    }

    //! assign range [first, last)
    template<class I>
    void assign(I first, I last) {
        clear(); internal_assign_range( first, last, static_cast<is_integer_tag<std::numeric_limits<I>::is_integer> *>(0) );
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! assigns an initializer list
    void assign(std::initializer_list<T> init_list) {
        clear(); internal_assign_iterators( init_list.begin(), init_list.end());
    }
#endif //# __TBB_INITIALIZER_LISTS_PRESENT

    //! swap two instances
    void swap(concurrent_vector &vector) {
        typedef typename tbb::internal::allocator_traits<A>::propagate_on_container_swap pocs_t;
        if( this != &vector && (this->my_allocator == vector.my_allocator || pocs_t::value) ) {
            concurrent_vector_base_v3::internal_swap(static_cast<concurrent_vector_base_v3&>(vector));
            tbb::internal::allocator_swap(this->my_allocator, vector.my_allocator, pocs_t());
        }
    }

    //! Clear container while keeping memory allocated.
    /** To free up the memory, use in conjunction with method compact(). Not thread safe **/
    void clear() {
        internal_clear(&destroy_array);
    }

    //! Clear and destroy vector.
    ~concurrent_vector() {
        segment_t *table = my_segment.load<relaxed>();
        internal_free_segments( table, internal_clear(&destroy_array), my_first_block.load<relaxed>() );
        // base class destructor call should be then
    }

    const internal::concurrent_vector_base_v3 &internal_vector_base() const { return *this; }
private:
    //! Allocate k items
    static void *internal_allocator(internal::concurrent_vector_base_v3 &vb, size_t k) {
        return static_cast<concurrent_vector<T, A>&>(vb).my_allocator.allocate(k);
    }
    //! Free k segments from table
    void internal_free_segments(segment_t table[], segment_index_t k, segment_index_t first_block);

    //! Get reference to element at given index.
    T& internal_subscript( size_type index ) const;

    //! Get reference to element at given index with errors checks
    T& internal_subscript_with_exceptions( size_type index ) const;

    //! assign n items by copying t
    void internal_assign_n(size_type n, const_pointer p) {
        internal_resize( n, sizeof(T), max_size(), static_cast<const void*>(p), &destroy_array, p? &initialize_array_by : &initialize_array );
    }

    //! True/false function override helper
    /* Functions declarations:
     *     void foo(is_integer_tag<true>*);
     *     void foo(is_integer_tag<false>*);
     * Usage example:
     *     foo(static_cast<is_integer_tag<std::numeric_limits<T>::is_integer>*>(0));
     */
    template<bool B> class is_integer_tag;

    //! assign integer items by copying when arguments are treated as iterators. See C++ Standard 2003 23.1.1p9
    template<class I>
    void internal_assign_range(I first, I last, is_integer_tag<true> *) {
        internal_assign_n(static_cast<size_type>(first), &static_cast<T&>(last));
    }
    //! inline proxy assign by iterators
    template<class I>
    void internal_assign_range(I first, I last, is_integer_tag<false> *) {
        internal_assign_iterators(first, last);
    }
    //! assign by iterators
    template<class I>
    void internal_assign_iterators(I first, I last);

    //these functions are marked __TBB_EXPORTED_FUNC as they are called from within the library

    //! Construct n instances of T, starting at "begin".
    static void __TBB_EXPORTED_FUNC initialize_array( void* begin, const void*, size_type n );

    //! Copy-construct n instances of T, starting at "begin".
    static void __TBB_EXPORTED_FUNC initialize_array_by( void* begin, const void* src, size_type n );

    //! Copy-construct n instances of T by copying single element pointed to by src, starting at "dst".
    static void __TBB_EXPORTED_FUNC copy_array( void* dst, const void* src, size_type n );

#if __TBB_MOVE_IF_NOEXCEPT_PRESENT
    //! Either opy or move-construct n instances of T, starting at "dst" by copying according element of src array.
    static void __TBB_EXPORTED_FUNC move_array_if_noexcept( void* dst, const void* src, size_type n );
#endif //__TBB_MOVE_IF_NO_EXCEPT_PRESENT

#if __TBB_CPP11_RVALUE_REF_PRESENT
    //! Move-construct n instances of T, starting at "dst" by copying according element of src array.
    static void __TBB_EXPORTED_FUNC move_array( void* dst, const void* src, size_type n );

    //! Move-assign (using operator=) n instances of T, starting at "dst" by assigning according element of src array.
    static void __TBB_EXPORTED_FUNC move_assign_array( void* dst, const void* src, size_type n );
#endif
    //! Copy-construct n instances of T, starting at "dst" by iterator range of [p_type_erased_iterator, p_type_erased_iterator+n).
    template<typename Iterator>
    static void __TBB_EXPORTED_FUNC copy_range( void* dst, const void* p_type_erased_iterator, size_type n );

    //! Assign (using operator=) n instances of T, starting at "dst" by assigning according element of src array.
    static void __TBB_EXPORTED_FUNC assign_array( void* dst, const void* src, size_type n );

    //! Destroy n instances of T, starting at "begin".
    static void __TBB_EXPORTED_FUNC destroy_array( void* begin, size_type n );

    //! Exception-aware helper class for filling a segment by exception-danger operators of user class
    class internal_loop_guide : internal::no_copy {
    public:
        const pointer array;
        const size_type n;
        size_type i;

        static const T* as_const_pointer(const void *ptr) { return static_cast<const T *>(ptr); }
        static T* as_pointer(const void *src) { return static_cast<T*>(const_cast<void *>(src)); }

        internal_loop_guide(size_type ntrials, void *ptr)
            : array(as_pointer(ptr)), n(ntrials), i(0) {}
        void init() {   for(; i < n; ++i) new( &array[i] ) T(); }
        void init(const void *src) { for(; i < n; ++i) new( &array[i] ) T(*as_const_pointer(src)); }
        void copy(const void *src) { for(; i < n; ++i) new( &array[i] ) T(as_const_pointer(src)[i]); }
        void assign(const void *src) { for(; i < n; ++i) array[i] = as_const_pointer(src)[i]; }
#if __TBB_CPP11_RVALUE_REF_PRESENT
        void move_assign(const void *src)       { for(; i < n; ++i) array[i]         =  std::move(as_pointer(src)[i]);   }
        void move_construct(const void *src)    { for(; i < n; ++i) new( &array[i] ) T( std::move(as_pointer(src)[i]) ); }
#endif
#if __TBB_MOVE_IF_NOEXCEPT_PRESENT
        void move_construct_if_noexcept(const void *src)    { for(; i < n; ++i) new( &array[i] ) T( std::move_if_noexcept(as_pointer(src)[i]) ); }
#endif //__TBB_MOVE_IF_NOEXCEPT_PRESENT

        //TODO: rename to construct_range
        template<class I> void iterate(I &src) { for(; i < n; ++i, ++src) new( &array[i] ) T( *src ); }
        ~internal_loop_guide() {
            if(i < n) {// if an exception was raised, fill the rest of items with zeros
                internal::handle_unconstructed_elements(array+i, n-i);
            }
        }
    };

    struct push_back_helper : internal::no_copy{
        struct element_construction_guard : internal::no_copy{
            pointer element;

            element_construction_guard(pointer an_element) : element (an_element){}
            void dismiss(){ element = NULL; }
            ~element_construction_guard(){
                if (element){
                    internal::handle_unconstructed_elements(element, 1);
                }
            }
        };

        concurrent_vector & v;
        size_type k;
        element_construction_guard g;

        push_back_helper(concurrent_vector & vector) :
            v(vector),
            g (static_cast<T*>(v.internal_push_back(sizeof(T),k)))
        {}

        pointer internal_push_back_result(){ return g.element;}
        iterator return_iterator_and_dismiss(){
            pointer ptr = g.element;
            g.dismiss();
            return iterator(v, k, ptr);
        }
    };
};

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
// Deduction guide for the constructor from two iterators
template<typename I,
         typename T = typename std::iterator_traits<I>::value_type,
         typename A = cache_aligned_allocator<T>
> concurrent_vector(I, I, const A& = A())
-> concurrent_vector<T, A>;

// Deduction guide for the constructor from a vector and allocator
template<typename T, typename A1, typename A2>
concurrent_vector(const concurrent_vector<T, A1> &, const A2 &)
-> concurrent_vector<T, A2>;

// Deduction guide for the constructor from an initializer_list
template<typename T, typename A = cache_aligned_allocator<T>
> concurrent_vector(std::initializer_list<T>, const A& = A())
-> concurrent_vector<T, A>;
#endif /* __TBB_CPP17_DEDUCTION_GUIDES_PRESENT */

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning (push)
#pragma warning (disable: 4701) // potentially uninitialized local variable "old"
#endif
template<typename T, class A>
void concurrent_vector<T, A>::shrink_to_fit() {
    internal_segments_table old;
    __TBB_TRY {
        internal_array_op2 copy_or_move_array =
#if __TBB_MOVE_IF_NOEXCEPT_PRESENT
                &move_array_if_noexcept
#else
                &copy_array
#endif
        ;
        if( internal_compact( sizeof(T), &old, &destroy_array, copy_or_move_array ) )
            internal_free_segments( old.table, pointers_per_long_table, old.first_block ); // free joined and unnecessary segments
    } __TBB_CATCH(...) {
        if( old.first_block ) // free segment allocated for compacting. Only for support of exceptions in ctor of user T[ype]
            internal_free_segments( old.table, 1, old.first_block );
        __TBB_RETHROW();
    }
}
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning (pop)
#endif // warning 4701 is back

template<typename T, class A>
void concurrent_vector<T, A>::internal_free_segments(segment_t table[], segment_index_t k, segment_index_t first_block) {
    // Free the arrays
    while( k > first_block ) {
        --k;
        segment_value_t segment_value = table[k].load<relaxed>();
        table[k].store<relaxed>(segment_not_used());
        if( segment_value == segment_allocated() ) // check for correct segment pointer
            this->my_allocator.deallocate( (segment_value.pointer<T>()), segment_size(k) );
    }
    segment_value_t segment_value = table[0].load<relaxed>();
    if( segment_value == segment_allocated() ) {
        __TBB_ASSERT( first_block > 0, NULL );
        while(k > 0) table[--k].store<relaxed>(segment_not_used());
        this->my_allocator.deallocate( (segment_value.pointer<T>()), segment_size(first_block) );
    }
}

template<typename T, class A>
T& concurrent_vector<T, A>::internal_subscript( size_type index ) const {
    //TODO: unify both versions of internal_subscript
    __TBB_ASSERT( index < my_early_size, "index out of bounds" );
    size_type j = index;
    segment_index_t k = segment_base_index_of( j );
    __TBB_ASSERT( my_segment.load<acquire>() != my_storage || k < pointers_per_short_table, "index is being allocated" );
    //no need in load with acquire (load<acquire>) since thread works in own space or gets
    //the information about added elements via some form of external synchronization
    //TODO: why not make a load of my_segment relaxed as well ?
    //TODO: add an assertion that my_segment[k] is properly aligned to please ITT
    segment_value_t segment_value =  my_segment[k].template load<relaxed>();
    __TBB_ASSERT( segment_value != segment_allocation_failed(), "the instance is broken by bad allocation. Use at() instead" );
    __TBB_ASSERT( segment_value != segment_not_used(), "index is being allocated" );
    return (( segment_value.pointer<T>()))[j];
}

template<typename T, class A>
T& concurrent_vector<T, A>::internal_subscript_with_exceptions( size_type index ) const {
    if( index >= my_early_size )
        internal::throw_exception(internal::eid_out_of_range); // throw std::out_of_range
    size_type j = index;
    segment_index_t k = segment_base_index_of( j );
    //TODO: refactor this condition into separate helper function, e.g. fits_into_small_table
    if( my_segment.load<acquire>() == my_storage && k >= pointers_per_short_table )
        internal::throw_exception(internal::eid_segment_range_error); // throw std::range_error
    // no need in load with acquire (load<acquire>) since thread works in own space or gets
    //the information about added elements via some form of external synchronization
    //TODO: why not make a load of my_segment relaxed as well ?
    //TODO: add an assertion that my_segment[k] is properly aligned to please ITT
    segment_value_t segment_value =  my_segment[k].template load<relaxed>();
    enforce_segment_allocated(segment_value, internal::eid_index_range_error);
    return (segment_value.pointer<T>())[j];
}

template<typename T, class A> template<class I>
void concurrent_vector<T, A>::internal_assign_iterators(I first, I last) {
    __TBB_ASSERT(my_early_size == 0, NULL);
    size_type n = std::distance(first, last);
    if( !n ) return;
    internal_reserve(n, sizeof(T), max_size());
    my_early_size = n;
    segment_index_t k = 0;
    //TODO: unify segment iteration code with concurrent_base_v3::helper
    size_type sz = segment_size( my_first_block );
    while( sz < n ) {
        internal_loop_guide loop(sz, my_segment[k].template load<relaxed>().template pointer<void>());
        loop.iterate(first);
        n -= sz;
        if( !k ) k = my_first_block;
        else { ++k; sz <<= 1; }
    }
    internal_loop_guide loop(n, my_segment[k].template load<relaxed>().template pointer<void>());
    loop.iterate(first);
}

template<typename T, class A>
void concurrent_vector<T, A>::initialize_array( void* begin, const void *, size_type n ) {
    internal_loop_guide loop(n, begin); loop.init();
}

template<typename T, class A>
void concurrent_vector<T, A>::initialize_array_by( void* begin, const void *src, size_type n ) {
    internal_loop_guide loop(n, begin); loop.init(src);
}

template<typename T, class A>
void concurrent_vector<T, A>::copy_array( void* dst, const void* src, size_type n ) {
    internal_loop_guide loop(n, dst); loop.copy(src);
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
template<typename T, class A>
void concurrent_vector<T, A>::move_array( void* dst, const void* src, size_type n ) {
    internal_loop_guide loop(n, dst); loop.move_construct(src);
}
template<typename T, class A>
void concurrent_vector<T, A>::move_assign_array( void* dst, const void* src, size_type n ) {
    internal_loop_guide loop(n, dst); loop.move_assign(src);
}
#endif

#if __TBB_MOVE_IF_NOEXCEPT_PRESENT
template<typename T, class A>
void concurrent_vector<T, A>::move_array_if_noexcept( void* dst, const void* src, size_type n ) {
    internal_loop_guide loop(n, dst); loop.move_construct_if_noexcept(src);
}
#endif //__TBB_MOVE_IF_NOEXCEPT_PRESENT

template<typename T, class A>
template<typename I>
void concurrent_vector<T, A>::copy_range( void* dst, const void* p_type_erased_iterator, size_type n ){
    internal_loop_guide loop(n, dst);
    loop.iterate( *(static_cast<I*>(const_cast<void*>(p_type_erased_iterator))) );
}

template<typename T, class A>
void concurrent_vector<T, A>::assign_array( void* dst, const void* src, size_type n ) {
    internal_loop_guide loop(n, dst); loop.assign(src);
}

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    // Workaround for overzealous compiler warning
    #pragma warning (push)
    #pragma warning (disable: 4189)
#endif
template<typename T, class A>
void concurrent_vector<T, A>::destroy_array( void* begin, size_type n ) {
    T* array = static_cast<T*>(begin);
    for( size_type j=n; j>0; --j )
        array[j-1].~T(); // destructors are supposed to not throw any exceptions
}
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #pragma warning (pop)
#endif // warning 4189 is back

// concurrent_vector's template functions
template<typename T, class A1, class A2>
inline bool operator==(const concurrent_vector<T, A1> &a, const concurrent_vector<T, A2> &b) {
    //TODO: call size() only once per vector (in operator==)
    // Simply:    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    if(a.size() != b.size()) return false;
    typename concurrent_vector<T, A1>::const_iterator i(a.begin());
    typename concurrent_vector<T, A2>::const_iterator j(b.begin());
    for(; i != a.end(); ++i, ++j)
        if( !(*i == *j) ) return false;
    return true;
}

template<typename T, class A1, class A2>
inline bool operator!=(const concurrent_vector<T, A1> &a, const concurrent_vector<T, A2> &b)
{    return !(a == b); }

template<typename T, class A1, class A2>
inline bool operator<(const concurrent_vector<T, A1> &a, const concurrent_vector<T, A2> &b)
{    return (std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end())); }

template<typename T, class A1, class A2>
inline bool operator>(const concurrent_vector<T, A1> &a, const concurrent_vector<T, A2> &b)
{    return b < a; }

template<typename T, class A1, class A2>
inline bool operator<=(const concurrent_vector<T, A1> &a, const concurrent_vector<T, A2> &b)
{    return !(b < a); }

template<typename T, class A1, class A2>
inline bool operator>=(const concurrent_vector<T, A1> &a, const concurrent_vector<T, A2> &b)
{    return !(a < b); }

template<typename T, class A>
inline void swap(concurrent_vector<T, A> &a, concurrent_vector<T, A> &b)
{    a.swap( b ); }

} // namespace tbb

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #pragma warning (pop)
#endif // warning 4267,4127 are back


#undef __TBB_concurrent_vector_H_include_area
#include "internal/_warning_suppress_disable_notice.h"

#endif /* __TBB_concurrent_vector_H */
