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

#ifndef __TBB_concurrent_hash_map_H
#define __TBB_concurrent_hash_map_H

#define __TBB_concurrent_hash_map_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "tbb_stddef.h"
#include <iterator>
#include <utility>      // Need std::pair
#include <cstring>      // Need std::memset
#include __TBB_STD_SWAP_HEADER

#include "tbb_allocator.h"
#include "spin_rw_mutex.h"
#include "atomic.h"
#include "tbb_exception.h"
#include "tbb_profiling.h"
#include "aligned_space.h"
#include "internal/_tbb_hash_compare_impl.h"
#include "internal/_template_helpers.h"
#include "internal/_allocator_traits.h"
#if __TBB_INITIALIZER_LISTS_PRESENT
#include <initializer_list>
#endif
#if TBB_USE_PERFORMANCE_WARNINGS || __TBB_STATISTICS
#include <typeinfo>
#endif
#if __TBB_STATISTICS
#include <stdio.h>
#endif
#if __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_TUPLE_PRESENT
// Definition of __TBB_CPP11_RVALUE_REF_PRESENT includes __TBB_CPP11_TUPLE_PRESENT
// for most of platforms, tuple present macro was added for logical correctness
#include <tuple>
#endif

namespace tbb {

namespace interface5 {

    template<typename Key, typename T, typename HashCompare = tbb_hash_compare<Key>, typename A = tbb_allocator<std::pair<const Key, T> > >
    class concurrent_hash_map;

    //! @cond INTERNAL
    namespace internal {
    using namespace tbb::internal;


    //! Type of a hash code.
    typedef size_t hashcode_t;
    //! Node base type
    struct hash_map_node_base : tbb::internal::no_copy {
        //! Mutex type
        typedef spin_rw_mutex mutex_t;
        //! Scoped lock type for mutex
        typedef mutex_t::scoped_lock scoped_t;
        //! Next node in chain
        hash_map_node_base *next;
        mutex_t mutex;
    };
    //! Incompleteness flag value
    static hash_map_node_base *const rehash_req = reinterpret_cast<hash_map_node_base*>(size_t(3));
    //! Rehashed empty bucket flag
    static hash_map_node_base *const empty_rehashed = reinterpret_cast<hash_map_node_base*>(size_t(0));
    //! base class of concurrent_hash_map
    class hash_map_base {
    public:
        //! Size type
        typedef size_t size_type;
        //! Type of a hash code.
        typedef size_t hashcode_t;
        //! Segment index type
        typedef size_t segment_index_t;
        //! Node base type
        typedef hash_map_node_base node_base;
        //! Bucket type
        struct bucket : tbb::internal::no_copy {
            //! Mutex type for buckets
            typedef spin_rw_mutex mutex_t;
            //! Scoped lock type for mutex
            typedef mutex_t::scoped_lock scoped_t;
            mutex_t mutex;
            node_base *node_list;
        };
        //! Count of segments in the first block
        static size_type const embedded_block = 1;
        //! Count of segments in the first block
        static size_type const embedded_buckets = 1<<embedded_block;
        //! Count of segments in the first block
        static size_type const first_block = 8; //including embedded_block. perfect with bucket size 16, so the allocations are power of 4096
        //! Size of a pointer / table size
        static size_type const pointers_per_table = sizeof(segment_index_t) * 8; // one segment per bit
        //! Segment pointer
        typedef bucket *segment_ptr_t;
        //! Segment pointers table type
        typedef segment_ptr_t segments_table_t[pointers_per_table];
        //! Hash mask = sum of allocated segment sizes - 1
        atomic<hashcode_t> my_mask;
        //! Segment pointers table. Also prevents false sharing between my_mask and my_size
        segments_table_t my_table;
        //! Size of container in stored items
        atomic<size_type> my_size; // It must be in separate cache line from my_mask due to performance effects
        //! Zero segment
        bucket my_embedded_segment[embedded_buckets];
#if __TBB_STATISTICS
        atomic<unsigned> my_info_resizes; // concurrent ones
        mutable atomic<unsigned> my_info_restarts; // race collisions
        atomic<unsigned> my_info_rehashes;  // invocations of rehash_bucket
#endif
        //! Constructor
        hash_map_base() {
            std::memset(my_table, 0, sizeof(my_table));
            my_mask = 0;
            my_size = 0;
            std::memset(my_embedded_segment, 0, sizeof(my_embedded_segment));
            for( size_type i = 0; i < embedded_block; i++ ) // fill the table
                my_table[i] = my_embedded_segment + segment_base(i);
            my_mask = embedded_buckets - 1;
            __TBB_ASSERT( embedded_block <= first_block, "The first block number must include embedded blocks");
#if __TBB_STATISTICS
            my_info_resizes = 0; // concurrent ones
            my_info_restarts = 0; // race collisions
            my_info_rehashes = 0;  // invocations of rehash_bucket
#endif
        }

        //! @return segment index of given index in the array
        static segment_index_t segment_index_of( size_type index ) {
            return segment_index_t( __TBB_Log2( index|1 ) );
        }

        //! @return the first array index of given segment
        static segment_index_t segment_base( segment_index_t k ) {
            return (segment_index_t(1)<<k & ~segment_index_t(1));
        }

        //! @return segment size except for @arg k == 0
        static size_type segment_size( segment_index_t k ) {
            return size_type(1)<<k; // fake value for k==0
        }

        //! @return true if @arg ptr is valid pointer
        static bool is_valid( void *ptr ) {
            return reinterpret_cast<uintptr_t>(ptr) > uintptr_t(63);
        }

        //! Initialize buckets
        static void init_buckets( segment_ptr_t ptr, size_type sz, bool is_initial ) {
            if( is_initial ) std::memset( static_cast<void*>(ptr), 0, sz*sizeof(bucket) );
            else for(size_type i = 0; i < sz; i++, ptr++) {
                *reinterpret_cast<intptr_t*>(&ptr->mutex) = 0;
                ptr->node_list = rehash_req;
            }
        }

        //! Add node @arg n to bucket @arg b
        static void add_to_bucket( bucket *b, node_base *n ) {
            __TBB_ASSERT(b->node_list != rehash_req, NULL);
            n->next = b->node_list;
            b->node_list = n; // its under lock and flag is set
        }

        //! Exception safety helper
        struct enable_segment_failsafe : tbb::internal::no_copy {
            segment_ptr_t *my_segment_ptr;
            enable_segment_failsafe(segments_table_t &table, segment_index_t k) : my_segment_ptr(&table[k]) {}
            ~enable_segment_failsafe() {
                if( my_segment_ptr ) *my_segment_ptr = 0; // indicate no allocation in progress
            }
        };

        //! Enable segment
        template<typename Allocator>
        void enable_segment( segment_index_t k, const Allocator& allocator, bool is_initial = false ) {
            typedef typename tbb::internal::allocator_rebind<Allocator, bucket>::type bucket_allocator_type;
            typedef tbb::internal::allocator_traits<bucket_allocator_type> bucket_allocator_traits;
            bucket_allocator_type bucket_allocator(allocator);
            __TBB_ASSERT( k, "Zero segment must be embedded" );
            enable_segment_failsafe watchdog( my_table, k );
            size_type sz;
            __TBB_ASSERT( !is_valid(my_table[k]), "Wrong concurrent assignment");
            if( k >= first_block ) {
                sz = segment_size( k );
                segment_ptr_t ptr = bucket_allocator_traits::allocate(bucket_allocator, sz);
                init_buckets( ptr, sz, is_initial );
                itt_hide_store_word( my_table[k], ptr );
                sz <<= 1;// double it to get entire capacity of the container
            } else { // the first block
                __TBB_ASSERT( k == embedded_block, "Wrong segment index" );
                sz = segment_size( first_block );
                segment_ptr_t ptr = bucket_allocator_traits::allocate(bucket_allocator, sz - embedded_buckets);
                init_buckets( ptr, sz - embedded_buckets, is_initial );
                ptr -= segment_base(embedded_block);
                for(segment_index_t i = embedded_block; i < first_block; i++) // calc the offsets
                    itt_hide_store_word( my_table[i], ptr + segment_base(i) );
            }
            itt_store_word_with_release( my_mask, sz-1 );
            watchdog.my_segment_ptr = 0;
        }

        template<typename Allocator>
        void delete_segment(segment_index_t s, const Allocator& allocator) {
            typedef typename tbb::internal::allocator_rebind<Allocator, bucket>::type bucket_allocator_type;
            typedef tbb::internal::allocator_traits<bucket_allocator_type> bucket_allocator_traits;
            bucket_allocator_type bucket_allocator(allocator);
            segment_ptr_t buckets_ptr = my_table[s];
            size_type sz = segment_size( s ? s : 1 );

            if( s >= first_block) // the first segment or the next
                bucket_allocator_traits::deallocate(bucket_allocator, buckets_ptr, sz);
            else if( s == embedded_block && embedded_block != first_block )
                bucket_allocator_traits::deallocate(bucket_allocator, buckets_ptr,
                                                    segment_size(first_block) - embedded_buckets);
            if( s >= embedded_block ) my_table[s] = 0;
        }

        //! Get bucket by (masked) hashcode
        bucket *get_bucket( hashcode_t h ) const throw() { // TODO: add throw() everywhere?
            segment_index_t s = segment_index_of( h );
            h -= segment_base(s);
            segment_ptr_t seg = my_table[s];
            __TBB_ASSERT( is_valid(seg), "hashcode must be cut by valid mask for allocated segments" );
            return &seg[h];
        }

        // internal serial rehashing helper
        void mark_rehashed_levels( hashcode_t h ) throw () {
            segment_index_t s = segment_index_of( h );
            while( segment_ptr_t seg = my_table[++s] )
                if( seg[h].node_list == rehash_req ) {
                    seg[h].node_list = empty_rehashed;
                    mark_rehashed_levels( h + ((hashcode_t)1<<s) ); // optimized segment_base(s)
                }
        }

        //! Check for mask race
        // Splitting into two functions should help inlining
        inline bool check_mask_race( const hashcode_t h, hashcode_t &m ) const {
            hashcode_t m_now, m_old = m;
            m_now = (hashcode_t) itt_load_word_with_acquire( my_mask );
            if( m_old != m_now )
                return check_rehashing_collision( h, m_old, m = m_now );
            return false;
        }

        //! Process mask race, check for rehashing collision
        bool check_rehashing_collision( const hashcode_t h, hashcode_t m_old, hashcode_t m ) const {
            __TBB_ASSERT(m_old != m, NULL); // TODO?: m arg could be optimized out by passing h = h&m
            if( (h & m_old) != (h & m) ) { // mask changed for this hashcode, rare event
                // condition above proves that 'h' has some other bits set beside 'm_old'
                // find next applicable mask after m_old    //TODO: look at bsl instruction
                for( ++m_old; !(h & m_old); m_old <<= 1 ) // at maximum few rounds depending on the first block size
                    ;
                m_old = (m_old<<1) - 1; // get full mask from a bit
                __TBB_ASSERT((m_old&(m_old+1))==0 && m_old <= m, NULL);
                // check whether it is rehashing/ed
                if( itt_load_word_with_acquire(get_bucket(h & m_old)->node_list) != rehash_req )
                {
#if __TBB_STATISTICS
                    my_info_restarts++; // race collisions
#endif
                    return true;
                }
            }
            return false;
        }

        //! Insert a node and check for load factor. @return segment index to enable.
        segment_index_t insert_new_node( bucket *b, node_base *n, hashcode_t mask ) {
            size_type sz = ++my_size; // prefix form is to enforce allocation after the first item inserted
            add_to_bucket( b, n );
            // check load factor
            if( sz >= mask ) { // TODO: add custom load_factor
                segment_index_t new_seg = __TBB_Log2( mask+1 ); //optimized segment_index_of
                __TBB_ASSERT( is_valid(my_table[new_seg-1]), "new allocations must not publish new mask until segment has allocated");
                static const segment_ptr_t is_allocating = (segment_ptr_t)2;
                if( !itt_hide_load_word(my_table[new_seg])
                  && as_atomic(my_table[new_seg]).compare_and_swap(is_allocating, NULL) == NULL )
                    return new_seg; // The value must be processed
            }
            return 0;
        }

        //! Prepare enough segments for number of buckets
        template<typename Allocator>
        void reserve(size_type buckets, const Allocator& allocator) {
            if( !buckets-- ) return;
            bool is_initial = !my_size;
            for( size_type m = my_mask; buckets > m; m = my_mask )
                enable_segment( segment_index_of( m+1 ), allocator, is_initial );
        }
        //! Swap hash_map_bases
        void internal_swap(hash_map_base &table) {
            using std::swap;
            swap(this->my_mask, table.my_mask);
            swap(this->my_size, table.my_size);
            for(size_type i = 0; i < embedded_buckets; i++)
                swap(this->my_embedded_segment[i].node_list, table.my_embedded_segment[i].node_list);
            for(size_type i = embedded_block; i < pointers_per_table; i++)
                swap(this->my_table[i], table.my_table[i]);
        }

#if __TBB_CPP11_RVALUE_REF_PRESENT
        void internal_move(hash_map_base&& other) {
            my_mask = other.my_mask;
            other.my_mask = embedded_buckets - 1;
            my_size = other.my_size;
            other.my_size = 0;

            for(size_type i = 0; i < embedded_buckets; ++i) {
                my_embedded_segment[i].node_list = other.my_embedded_segment[i].node_list;
                other.my_embedded_segment[i].node_list = NULL;
            }

            for(size_type i = embedded_block; i < pointers_per_table; ++i) {
                my_table[i] = other.my_table[i];
                other.my_table[i] = NULL;
            }
        }
#endif // __TBB_CPP11_RVALUE_REF_PRESENT
    };

    template<typename Iterator>
    class hash_map_range;

    //! Meets requirements of a forward iterator for STL */
    /** Value is either the T or const T type of the container.
        @ingroup containers */
    template<typename Container, typename Value>
    class hash_map_iterator
        : public std::iterator<std::forward_iterator_tag,Value>
    {
        typedef Container map_type;
        typedef typename Container::node node;
        typedef hash_map_base::node_base node_base;
        typedef hash_map_base::bucket bucket;

        template<typename C, typename T, typename U>
        friend bool operator==( const hash_map_iterator<C,T>& i, const hash_map_iterator<C,U>& j );

        template<typename C, typename T, typename U>
        friend bool operator!=( const hash_map_iterator<C,T>& i, const hash_map_iterator<C,U>& j );

        template<typename C, typename T, typename U>
        friend ptrdiff_t operator-( const hash_map_iterator<C,T>& i, const hash_map_iterator<C,U>& j );

        template<typename C, typename U>
        friend class hash_map_iterator;

        template<typename I>
        friend class hash_map_range;

        void advance_to_next_bucket() { // TODO?: refactor to iterator_base class
            size_t k = my_index+1;
            __TBB_ASSERT( my_bucket, "advancing an invalid iterator?");
            while( k <= my_map->my_mask ) {
                // Following test uses 2's-complement wizardry
                if( k&(k-2) ) // not the beginning of a segment
                    ++my_bucket;
                else my_bucket = my_map->get_bucket( k );
                my_node = static_cast<node*>( my_bucket->node_list );
                if( hash_map_base::is_valid(my_node) ) {
                    my_index = k; return;
                }
                ++k;
            }
            my_bucket = 0; my_node = 0; my_index = k; // the end
        }
#if !defined(_MSC_VER) || defined(__INTEL_COMPILER)
        template<typename Key, typename T, typename HashCompare, typename A>
        friend class interface5::concurrent_hash_map;
#else
    public: // workaround
#endif
        //! concurrent_hash_map over which we are iterating.
        const Container *my_map;

        //! Index in hash table for current item
        size_t my_index;

        //! Pointer to bucket
        const bucket *my_bucket;

        //! Pointer to node that has current item
        node *my_node;

        hash_map_iterator( const Container &map, size_t index, const bucket *b, node_base *n );

    public:
        //! Construct undefined iterator
        hash_map_iterator(): my_map(), my_index(), my_bucket(), my_node() {}
        hash_map_iterator( const hash_map_iterator<Container,typename Container::value_type> &other ) :
            my_map(other.my_map),
            my_index(other.my_index),
            my_bucket(other.my_bucket),
            my_node(other.my_node)
        {}

        hash_map_iterator& operator=( const hash_map_iterator<Container,typename Container::value_type> &other ) {
            my_map = other.my_map;
            my_index = other.my_index;
            my_bucket = other.my_bucket;
            my_node = other.my_node;
            return *this;
        }
        Value& operator*() const {
            __TBB_ASSERT( hash_map_base::is_valid(my_node), "iterator uninitialized or at end of container?" );
            return my_node->value();
        }
        Value* operator->() const {return &operator*();}
        hash_map_iterator& operator++();

        //! Post increment
        hash_map_iterator operator++(int) {
            hash_map_iterator old(*this);
            operator++();
            return old;
        }
    };

    template<typename Container, typename Value>
    hash_map_iterator<Container,Value>::hash_map_iterator( const Container &map, size_t index, const bucket *b, node_base *n ) :
        my_map(&map),
        my_index(index),
        my_bucket(b),
        my_node( static_cast<node*>(n) )
    {
        if( b && !hash_map_base::is_valid(n) )
            advance_to_next_bucket();
    }

    template<typename Container, typename Value>
    hash_map_iterator<Container,Value>& hash_map_iterator<Container,Value>::operator++() {
        my_node = static_cast<node*>( my_node->next );
        if( !my_node ) advance_to_next_bucket();
        return *this;
    }

    template<typename Container, typename T, typename U>
    bool operator==( const hash_map_iterator<Container,T>& i, const hash_map_iterator<Container,U>& j ) {
        return i.my_node == j.my_node && i.my_map == j.my_map;
    }

    template<typename Container, typename T, typename U>
    bool operator!=( const hash_map_iterator<Container,T>& i, const hash_map_iterator<Container,U>& j ) {
        return i.my_node != j.my_node || i.my_map != j.my_map;
    }

    //! Range class used with concurrent_hash_map
    /** @ingroup containers */
    template<typename Iterator>
    class hash_map_range {
        typedef typename Iterator::map_type map_type;
        Iterator my_begin;
        Iterator my_end;
        mutable Iterator my_midpoint;
        size_t my_grainsize;
        //! Set my_midpoint to point approximately half way between my_begin and my_end.
        void set_midpoint() const;
        template<typename U> friend class hash_map_range;
    public:
        //! Type for size of a range
        typedef std::size_t size_type;
        typedef typename Iterator::value_type value_type;
        typedef typename Iterator::reference reference;
        typedef typename Iterator::difference_type difference_type;
        typedef Iterator iterator;

        //! True if range is empty.
        bool empty() const {return my_begin==my_end;}

        //! True if range can be partitioned into two subranges.
        bool is_divisible() const {
            return my_midpoint!=my_end;
        }
        //! Split range.
        hash_map_range( hash_map_range& r, split ) :
            my_end(r.my_end),
            my_grainsize(r.my_grainsize)
        {
            r.my_end = my_begin = r.my_midpoint;
            __TBB_ASSERT( !empty(), "Splitting despite the range is not divisible" );
            __TBB_ASSERT( !r.empty(), "Splitting despite the range is not divisible" );
            set_midpoint();
            r.set_midpoint();
        }
        //! type conversion
        template<typename U>
        hash_map_range( hash_map_range<U>& r) :
            my_begin(r.my_begin),
            my_end(r.my_end),
            my_midpoint(r.my_midpoint),
            my_grainsize(r.my_grainsize)
        {}
        //! Init range with container and grainsize specified
        hash_map_range( const map_type &map, size_type grainsize_ = 1 ) :
            my_begin( Iterator( map, 0, map.my_embedded_segment, map.my_embedded_segment->node_list ) ),
            my_end( Iterator( map, map.my_mask + 1, 0, 0 ) ),
            my_grainsize( grainsize_ )
        {
            __TBB_ASSERT( grainsize_>0, "grainsize must be positive" );
            set_midpoint();
        }
        const Iterator& begin() const {return my_begin;}
        const Iterator& end() const {return my_end;}
        //! The grain size for this range.
        size_type grainsize() const {return my_grainsize;}
    };

    template<typename Iterator>
    void hash_map_range<Iterator>::set_midpoint() const {
        // Split by groups of nodes
        size_t m = my_end.my_index-my_begin.my_index;
        if( m > my_grainsize ) {
            m = my_begin.my_index + m/2u;
            hash_map_base::bucket *b = my_begin.my_map->get_bucket(m);
            my_midpoint = Iterator(*my_begin.my_map,m,b,b->node_list);
        } else {
            my_midpoint = my_end;
        }
        __TBB_ASSERT( my_begin.my_index <= my_midpoint.my_index,
            "my_begin is after my_midpoint" );
        __TBB_ASSERT( my_midpoint.my_index <= my_end.my_index,
            "my_midpoint is after my_end" );
        __TBB_ASSERT( my_begin != my_midpoint || my_begin == my_end,
            "[my_begin, my_midpoint) range should not be empty" );
    }

    } // internal
//! @endcond

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Suppress "conditional expression is constant" warning.
    #pragma warning( push )
    #pragma warning( disable: 4127 )
#endif

//! Unordered map from Key to T.
/** concurrent_hash_map is associative container with concurrent access.

@par Compatibility
    The class meets all Container Requirements from C++ Standard (See ISO/IEC 14882:2003(E), clause 23.1).

@par Exception Safety
    - Hash function is not permitted to throw an exception. User-defined types Key and T are forbidden from throwing an exception in destructors.
    - If exception happens during insert() operations, it has no effect (unless exception raised by HashCompare::hash() function during grow_segment).
    - If exception happens during operator=() operation, the container can have a part of source items, and methods size() and empty() can return wrong results.

@par Changes since TBB 2.1
    - Replaced internal algorithm and data structure. Patent is pending.
    - Added buckets number argument for constructor

@par Changes since TBB 2.0
    - Fixed exception-safety
    - Added template argument for allocator
    - Added allocator argument in constructors
    - Added constructor from a range of iterators
    - Added several new overloaded insert() methods
    - Added get_allocator()
    - Added swap()
    - Added count()
    - Added overloaded erase(accessor &) and erase(const_accessor&)
    - Added equal_range() [const]
    - Added [const_]pointer, [const_]reference, and allocator_type types
    - Added global functions: operator==(), operator!=(), and swap()

    @ingroup containers */
template<typename Key, typename T, typename HashCompare, typename Allocator>
class concurrent_hash_map : protected internal::hash_map_base {
    template<typename Container, typename Value>
    friend class internal::hash_map_iterator;

    template<typename I>
    friend class internal::hash_map_range;

public:
    typedef Key key_type;
    typedef T mapped_type;
    typedef std::pair<const Key,T> value_type;
    typedef hash_map_base::size_type size_type;
    typedef ptrdiff_t difference_type;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef value_type &reference;
    typedef const value_type &const_reference;
    typedef internal::hash_map_iterator<concurrent_hash_map,value_type> iterator;
    typedef internal::hash_map_iterator<concurrent_hash_map,const value_type> const_iterator;
    typedef internal::hash_map_range<iterator> range_type;
    typedef internal::hash_map_range<const_iterator> const_range_type;
    typedef Allocator allocator_type;

protected:
    friend class const_accessor;
    class node;
    typedef typename tbb::internal::allocator_rebind<Allocator, node>::type node_allocator_type;
    typedef tbb::internal::allocator_traits<node_allocator_type> node_allocator_traits;
    node_allocator_type my_allocator;
    HashCompare my_hash_compare;

    class node : public node_base {
        tbb::aligned_space<value_type> my_value;
    public:
        value_type* storage() { return my_value.begin(); }
        value_type& value() { return *storage(); }
    };

    void delete_node( node_base *n ) {
        node_allocator_traits::destroy(my_allocator, static_cast<node*>(n)->storage());
        node_allocator_traits::destroy(my_allocator, static_cast<node*>(n));
        node_allocator_traits::deallocate(my_allocator, static_cast<node*>(n), 1);
    }

    struct node_scoped_guard : tbb::internal::no_copy {
        node* my_node;
        node_allocator_type& my_alloc;

        node_scoped_guard(node* n, node_allocator_type& alloc) : my_node(n), my_alloc(alloc) {}
        ~node_scoped_guard() {
            if(my_node) {
                node_allocator_traits::destroy(my_alloc, my_node);
                node_allocator_traits::deallocate(my_alloc, my_node, 1);
            }
        }
        void dismiss() { my_node = NULL; }
    };

#if __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    template<typename... Args>
    static node* create_node(node_allocator_type& allocator, Args&&... args)
#else
    template<typename Arg1, typename Arg2>
    static node* create_node(node_allocator_type& allocator, __TBB_FORWARDING_REF(Arg1) arg1, __TBB_FORWARDING_REF(Arg2) arg2)
#endif
    {
        node* node_ptr = node_allocator_traits::allocate(allocator, 1);
        node_scoped_guard guard(node_ptr, allocator);
        node_allocator_traits::construct(allocator, node_ptr);
#if __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
        node_allocator_traits::construct(allocator, node_ptr->storage(), std::forward<Args>(args)...);
#else
        node_allocator_traits::construct(allocator, node_ptr->storage(), tbb::internal::forward<Arg1>(arg1), tbb::internal::forward<Arg2>(arg2));
#endif
        guard.dismiss();
        return node_ptr;
    }

    static node* allocate_node_copy_construct(node_allocator_type& allocator, const Key &key, const T * t){
        return create_node(allocator, key, *t);
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    static node* allocate_node_move_construct(node_allocator_type& allocator, const Key &key, const T * t){
        return create_node(allocator, key, std::move(*const_cast<T*>(t)));
    }
#endif

    static node* allocate_node_default_construct(node_allocator_type& allocator, const Key &key, const T * ){
#if __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_TUPLE_PRESENT
        // Emplace construct an empty T object inside the pair
        return create_node(allocator, std::piecewise_construct,
                           std::forward_as_tuple(key), std::forward_as_tuple());
#else
        // Use of a temporary object is impossible, because create_node takes a non-const reference.
        // copy-initialization is possible because T is already required to be CopyConstructible.
        T obj = T();
        return create_node(allocator, key, tbb::internal::move(obj));
#endif
    }

    static node* do_not_allocate_node(node_allocator_type& , const Key &, const T * ){
        __TBB_ASSERT(false,"this dummy function should not be called");
        return NULL;
    }

    node *search_bucket( const key_type &key, bucket *b ) const {
        node *n = static_cast<node*>( b->node_list );
        while( is_valid(n) && !my_hash_compare.equal(key, n->value().first) )
            n = static_cast<node*>( n->next );
        __TBB_ASSERT(n != internal::rehash_req, "Search can be executed only for rehashed bucket");
        return n;
    }

    //! bucket accessor is to find, rehash, acquire a lock, and access a bucket
    class bucket_accessor : public bucket::scoped_t {
        bucket *my_b;
    public:
        bucket_accessor( concurrent_hash_map *base, const hashcode_t h, bool writer = false ) { acquire( base, h, writer ); }
        //! find a bucket by masked hashcode, optionally rehash, and acquire the lock
        inline void acquire( concurrent_hash_map *base, const hashcode_t h, bool writer = false ) {
            my_b = base->get_bucket( h );
            // TODO: actually, notification is unnecessary here, just hiding double-check
            if( itt_load_word_with_acquire(my_b->node_list) == internal::rehash_req
                && try_acquire( my_b->mutex, /*write=*/true ) )
            {
                if( my_b->node_list == internal::rehash_req ) base->rehash_bucket( my_b, h ); //recursive rehashing
            }
            else bucket::scoped_t::acquire( my_b->mutex, writer );
            __TBB_ASSERT( my_b->node_list != internal::rehash_req, NULL);
        }
        //! check whether bucket is locked for write
        bool is_writer() { return bucket::scoped_t::is_writer; }
        //! get bucket pointer
        bucket *operator() () { return my_b; }
    };

    // TODO refactor to hash_base
    void rehash_bucket( bucket *b_new, const hashcode_t h ) {
        __TBB_ASSERT( *(intptr_t*)(&b_new->mutex), "b_new must be locked (for write)");
        __TBB_ASSERT( h > 1, "The lowermost buckets can't be rehashed" );
        __TBB_store_with_release(b_new->node_list, internal::empty_rehashed); // mark rehashed
        hashcode_t mask = ( 1u<<__TBB_Log2( h ) ) - 1; // get parent mask from the topmost bit
#if __TBB_STATISTICS
        my_info_rehashes++; // invocations of rehash_bucket
#endif

        bucket_accessor b_old( this, h & mask );

        mask = (mask<<1) | 1; // get full mask for new bucket
        __TBB_ASSERT( (mask&(mask+1))==0 && (h & mask) == h, NULL );
    restart:
        for( node_base **p = &b_old()->node_list, *n = __TBB_load_with_acquire(*p); is_valid(n); n = *p ) {
            hashcode_t c = my_hash_compare.hash( static_cast<node*>(n)->value().first );
#if TBB_USE_ASSERT
            hashcode_t bmask = h & (mask>>1);
            bmask = bmask==0? 1 : ( 1u<<(__TBB_Log2( bmask )+1 ) ) - 1; // minimal mask of parent bucket
            __TBB_ASSERT( (c & bmask) == (h & bmask), "hash() function changed for key in table" );
#endif
            if( (c & mask) == h ) {
                if( !b_old.is_writer() )
                    if( !b_old.upgrade_to_writer() ) {
                        goto restart; // node ptr can be invalid due to concurrent erase
                    }
                *p = n->next; // exclude from b_old
                add_to_bucket( b_new, n );
            } else p = &n->next; // iterate to next item
        }
    }

    struct call_clear_on_leave {
        concurrent_hash_map* my_ch_map;
        call_clear_on_leave( concurrent_hash_map* a_ch_map ) : my_ch_map(a_ch_map) {}
        void dismiss() {my_ch_map = 0;}
        ~call_clear_on_leave(){
            if (my_ch_map){
                my_ch_map->clear();
            }
        }
    };
public:

    class accessor;
    //! Combines data access, locking, and garbage collection.
    class const_accessor : private node::scoped_t /*which derived from no_copy*/ {
        friend class concurrent_hash_map<Key,T,HashCompare,Allocator>;
        friend class accessor;
    public:
        //! Type of value
        typedef const typename concurrent_hash_map::value_type value_type;

        //! True if result is empty.
        bool empty() const { return !my_node; }

        //! Set to null
        void release() {
            if( my_node ) {
                node::scoped_t::release();
                my_node = 0;
            }
        }

        //! Return reference to associated value in hash table.
        const_reference operator*() const {
            __TBB_ASSERT( my_node, "attempt to dereference empty accessor" );
            return my_node->value();
        }

        //! Return pointer to associated value in hash table.
        const_pointer operator->() const {
            return &operator*();
        }

        //! Create empty result
        const_accessor() : my_node(NULL) {}

        //! Destroy result after releasing the underlying reference.
        ~const_accessor() {
            my_node = NULL; // scoped lock's release() is called in its destructor
        }
    protected:
        bool is_writer() { return node::scoped_t::is_writer; }
        node *my_node;
        hashcode_t my_hash;
    };

    //! Allows write access to elements and combines data access, locking, and garbage collection.
    class accessor: public const_accessor {
    public:
        //! Type of value
        typedef typename concurrent_hash_map::value_type value_type;

        //! Return reference to associated value in hash table.
        reference operator*() const {
            __TBB_ASSERT( this->my_node, "attempt to dereference empty accessor" );
            return this->my_node->value();
        }

        //! Return pointer to associated value in hash table.
        pointer operator->() const {
            return &operator*();
        }
    };

    //! Construct empty table.
    explicit concurrent_hash_map( const allocator_type &a = allocator_type() )
        : internal::hash_map_base(), my_allocator(a)
    {}

    explicit concurrent_hash_map( const HashCompare& compare, const allocator_type& a = allocator_type() )
        : internal::hash_map_base(), my_allocator(a), my_hash_compare(compare)
    {}

    //! Construct empty table with n preallocated buckets. This number serves also as initial concurrency level.
    concurrent_hash_map( size_type n, const allocator_type &a = allocator_type() )
        : internal::hash_map_base(), my_allocator(a)
    {
        reserve( n, my_allocator );
    }

    concurrent_hash_map( size_type n, const HashCompare& compare, const allocator_type& a = allocator_type() )
        : internal::hash_map_base(), my_allocator(a), my_hash_compare(compare)
    {
        reserve( n, my_allocator );
    }

    //! Copy constructor
    concurrent_hash_map( const concurrent_hash_map &table )
        : internal::hash_map_base(),
          my_allocator(node_allocator_traits::select_on_container_copy_construction(table.get_allocator()))
    {
        call_clear_on_leave scope_guard(this);
        internal_copy(table);
        scope_guard.dismiss();
    }

    concurrent_hash_map( const concurrent_hash_map &table, const allocator_type &a)
        : internal::hash_map_base(), my_allocator(a)
    {
        call_clear_on_leave scope_guard(this);
        internal_copy(table);
        scope_guard.dismiss();
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    //! Move constructor
    concurrent_hash_map( concurrent_hash_map &&table )
        : internal::hash_map_base(), my_allocator(std::move(table.get_allocator()))
    {
        internal_move(std::move(table));
    }

    //! Move constructor
    concurrent_hash_map( concurrent_hash_map &&table, const allocator_type &a )
        : internal::hash_map_base(), my_allocator(a)
    {
        if (a == table.get_allocator()){
            internal_move(std::move(table));
        }else{
            call_clear_on_leave scope_guard(this);
            internal_copy(std::make_move_iterator(table.begin()), std::make_move_iterator(table.end()), table.size());
            scope_guard.dismiss();
        }
    }
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

    //! Construction with copying iteration range and given allocator instance
    template<typename I>
    concurrent_hash_map( I first, I last, const allocator_type &a = allocator_type() )
        : internal::hash_map_base(), my_allocator(a)
    {
        call_clear_on_leave scope_guard(this);
        internal_copy(first, last, std::distance(first, last));
        scope_guard.dismiss();
    }

    template<typename I>
    concurrent_hash_map( I first, I last, const HashCompare& compare, const allocator_type& a = allocator_type() )
        : internal::hash_map_base(), my_allocator(a), my_hash_compare(compare)
    {
        call_clear_on_leave scope_guard(this);
        internal_copy(first, last, std::distance(first, last));
        scope_guard.dismiss();
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Construct empty table with n preallocated buckets. This number serves also as initial concurrency level.
    concurrent_hash_map( std::initializer_list<value_type> il, const allocator_type &a = allocator_type() )
        : internal::hash_map_base(), my_allocator(a)
    {
        call_clear_on_leave scope_guard(this);
        internal_copy(il.begin(), il.end(), il.size());
        scope_guard.dismiss();
    }

    concurrent_hash_map( std::initializer_list<value_type> il, const HashCompare& compare, const allocator_type& a = allocator_type() )
        : internal::hash_map_base(), my_allocator(a), my_hash_compare(compare)
    {
        call_clear_on_leave scope_guard(this);
        internal_copy(il.begin(), il.end(), il.size());
        scope_guard.dismiss();
    }

#endif //__TBB_INITIALIZER_LISTS_PRESENT

    //! Assignment
    concurrent_hash_map& operator=( const concurrent_hash_map &table ) {
        if( this!=&table ) {
            typedef typename node_allocator_traits::propagate_on_container_copy_assignment pocca_type;
            clear();
            tbb::internal::allocator_copy_assignment(my_allocator, table.my_allocator, pocca_type());
            internal_copy(table);
        }
        return *this;
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    //! Move Assignment
    concurrent_hash_map& operator=( concurrent_hash_map &&table ) {
        if(this != &table) {
            typedef typename node_allocator_traits::propagate_on_container_move_assignment pocma_type;
            internal_move_assign(std::move(table), pocma_type());
        }
        return *this;
    }
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Assignment
    concurrent_hash_map& operator=( std::initializer_list<value_type> il ) {
        clear();
        internal_copy(il.begin(), il.end(), il.size());
        return *this;
    }
#endif //__TBB_INITIALIZER_LISTS_PRESENT


    //! Rehashes and optionally resizes the whole table.
    /** Useful to optimize performance before or after concurrent operations.
        Also enables using of find() and count() concurrent methods in serial context. */
    void rehash(size_type n = 0);

    //! Clear table
    void clear();

    //! Clear table and destroy it.
    ~concurrent_hash_map() { clear(); }

    //------------------------------------------------------------------------
    // Parallel algorithm support
    //------------------------------------------------------------------------
    range_type range( size_type grainsize=1 ) {
        return range_type( *this, grainsize );
    }
    const_range_type range( size_type grainsize=1 ) const {
        return const_range_type( *this, grainsize );
    }

    //------------------------------------------------------------------------
    // STL support - not thread-safe methods
    //------------------------------------------------------------------------
    iterator begin() { return iterator( *this, 0, my_embedded_segment, my_embedded_segment->node_list ); }
    iterator end() { return iterator( *this, 0, 0, 0 ); }
    const_iterator begin() const { return const_iterator( *this, 0, my_embedded_segment, my_embedded_segment->node_list ); }
    const_iterator end() const { return const_iterator( *this, 0, 0, 0 ); }
    std::pair<iterator, iterator> equal_range( const Key& key ) { return internal_equal_range( key, end() ); }
    std::pair<const_iterator, const_iterator> equal_range( const Key& key ) const { return internal_equal_range( key, end() ); }

    //! Number of items in table.
    size_type size() const { return my_size; }

    //! True if size()==0.
    bool empty() const { return my_size == 0; }

    //! Upper bound on size.
    size_type max_size() const {return (~size_type(0))/sizeof(node);}

    //! Returns the current number of buckets
    size_type bucket_count() const { return my_mask+1; }

    //! return allocator object
    allocator_type get_allocator() const { return this->my_allocator; }

    //! swap two instances. Iterators are invalidated
    void swap( concurrent_hash_map &table );

    //------------------------------------------------------------------------
    // concurrent map operations
    //------------------------------------------------------------------------

    //! Return count of items (0 or 1)
    size_type count( const Key &key ) const {
        return const_cast<concurrent_hash_map*>(this)->lookup(/*insert*/false, key, NULL, NULL, /*write=*/false, &do_not_allocate_node );
    }

    //! Find item and acquire a read lock on the item.
    /** Return true if item is found, false otherwise. */
    bool find( const_accessor &result, const Key &key ) const {
        result.release();
        return const_cast<concurrent_hash_map*>(this)->lookup(/*insert*/false, key, NULL, &result, /*write=*/false, &do_not_allocate_node );
    }

    //! Find item and acquire a write lock on the item.
    /** Return true if item is found, false otherwise. */
    bool find( accessor &result, const Key &key ) {
        result.release();
        return lookup(/*insert*/false, key, NULL, &result, /*write=*/true, &do_not_allocate_node );
    }

    //! Insert item (if not already present) and acquire a read lock on the item.
    /** Returns true if item is new. */
    bool insert( const_accessor &result, const Key &key ) {
        result.release();
        return lookup(/*insert*/true, key, NULL, &result, /*write=*/false, &allocate_node_default_construct );
    }

    //! Insert item (if not already present) and acquire a write lock on the item.
    /** Returns true if item is new. */
    bool insert( accessor &result, const Key &key ) {
        result.release();
        return lookup(/*insert*/true, key, NULL, &result, /*write=*/true, &allocate_node_default_construct );
    }

    //! Insert item by copying if there is no such key present already and acquire a read lock on the item.
    /** Returns true if item is new. */
    bool insert( const_accessor &result, const value_type &value ) {
        result.release();
        return lookup(/*insert*/true, value.first, &value.second, &result, /*write=*/false, &allocate_node_copy_construct );
    }

    //! Insert item by copying if there is no such key present already and acquire a write lock on the item.
    /** Returns true if item is new. */
    bool insert( accessor &result, const value_type &value ) {
        result.release();
        return lookup(/*insert*/true, value.first, &value.second, &result, /*write=*/true, &allocate_node_copy_construct );
    }

    //! Insert item by copying if there is no such key present already
    /** Returns true if item is inserted. */
    bool insert( const value_type &value ) {
        return lookup(/*insert*/true, value.first, &value.second, NULL, /*write=*/false, &allocate_node_copy_construct );
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    //! Insert item by copying if there is no such key present already and acquire a read lock on the item.
    /** Returns true if item is new. */
    bool insert( const_accessor &result, value_type && value ) {
        return generic_move_insert(result, std::move(value));
    }

    //! Insert item by copying if there is no such key present already and acquire a write lock on the item.
    /** Returns true if item is new. */
    bool insert( accessor &result, value_type && value ) {
        return generic_move_insert(result, std::move(value));
    }

    //! Insert item by copying if there is no such key present already
    /** Returns true if item is inserted. */
    bool insert( value_type && value ) {
        return generic_move_insert(accessor_not_used(), std::move(value));
    }

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    //! Insert item by copying if there is no such key present already and acquire a read lock on the item.
    /** Returns true if item is new. */
    template<typename... Args>
    bool emplace( const_accessor &result, Args&&... args ) {
        return generic_emplace(result, std::forward<Args>(args)...);
    }

    //! Insert item by copying if there is no such key present already and acquire a write lock on the item.
    /** Returns true if item is new. */
    template<typename... Args>
    bool emplace( accessor &result, Args&&... args ) {
        return generic_emplace(result, std::forward<Args>(args)...);
    }

    //! Insert item by copying if there is no such key present already
    /** Returns true if item is inserted. */
    template<typename... Args>
    bool emplace( Args&&... args ) {
        return generic_emplace(accessor_not_used(), std::forward<Args>(args)...);
    }
#endif //__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

    //! Insert range [first, last)
    template<typename I>
    void insert( I first, I last ) {
        for ( ; first != last; ++first )
            insert( *first );
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Insert initializer list
    void insert( std::initializer_list<value_type> il ) {
        insert( il.begin(), il.end() );
    }
#endif //__TBB_INITIALIZER_LISTS_PRESENT

    //! Erase item.
    /** Return true if item was erased by particularly this call. */
    bool erase( const Key& key );

    //! Erase item by const_accessor.
    /** Return true if item was erased by particularly this call. */
    bool erase( const_accessor& item_accessor ) {
        return exclude( item_accessor );
    }

    //! Erase item by accessor.
    /** Return true if item was erased by particularly this call. */
    bool erase( accessor& item_accessor ) {
        return exclude( item_accessor );
    }

protected:
    //! Insert or find item and optionally acquire a lock on the item.
    bool lookup(bool op_insert, const Key &key, const T *t, const_accessor *result, bool write,  node* (*allocate_node)(node_allocator_type& ,  const Key &, const T * ), node *tmp_n = 0  ) ;

    struct accessor_not_used { void release(){}};
    friend const_accessor* accessor_location( accessor_not_used const& ){ return NULL;}
    friend const_accessor* accessor_location( const_accessor & a )      { return &a;}

    friend bool is_write_access_needed( accessor const& )           { return true;}
    friend bool is_write_access_needed( const_accessor const& )     { return false;}
    friend bool is_write_access_needed( accessor_not_used const& )  { return false;}

#if __TBB_CPP11_RVALUE_REF_PRESENT
    template<typename Accessor>
    bool generic_move_insert( Accessor && result, value_type && value ) {
        result.release();
        return lookup(/*insert*/true, value.first, &value.second, accessor_location(result), is_write_access_needed(result), &allocate_node_move_construct );
    }

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    template<typename Accessor, typename... Args>
    bool generic_emplace( Accessor && result, Args &&... args ) {
        result.release();
        node * node_ptr = create_node(my_allocator, std::forward<Args>(args)...);
        return lookup(/*insert*/true, node_ptr->value().first, NULL, accessor_location(result), is_write_access_needed(result), &do_not_allocate_node, node_ptr );
    }
#endif //__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

    //! delete item by accessor
    bool exclude( const_accessor &item_accessor );

    //! Returns an iterator for an item defined by the key, or for the next item after it (if upper==true)
    template<typename I>
    std::pair<I, I> internal_equal_range( const Key& key, I end ) const;

    //! Copy "source" to *this, where *this must start out empty.
    void internal_copy( const concurrent_hash_map& source );

    template<typename I>
    void internal_copy( I first, I last, size_type reserve_size );

#if __TBB_CPP11_RVALUE_REF_PRESENT
    // A compile-time dispatch to allow move assignment of containers with non-movable value_type if POCMA is true_type
    void internal_move_assign(concurrent_hash_map&& other, tbb::internal::traits_true_type) {
        tbb::internal::allocator_move_assignment(my_allocator, other.my_allocator, tbb::internal::traits_true_type());
        internal_move(std::move(other));
    }

    void internal_move_assign(concurrent_hash_map&& other, tbb::internal::traits_false_type) {
        if (this->my_allocator == other.my_allocator) {
            internal_move(std::move(other));
        } else {
            //do per element move
            internal_copy(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()), other.size());
        }
    }
#endif

    //! Fast find when no concurrent erasure is used. For internal use inside TBB only!
    /** Return pointer to item with given key, or NULL if no such item exists.
        Must not be called concurrently with erasure operations. */
    const_pointer internal_fast_find( const Key& key ) const {
        hashcode_t h = my_hash_compare.hash( key );
        hashcode_t m = (hashcode_t) itt_load_word_with_acquire( my_mask );
        node *n;
    restart:
        __TBB_ASSERT((m&(m+1))==0, "data structure is invalid");
        bucket *b = get_bucket( h & m );
        // TODO: actually, notification is unnecessary here, just hiding double-check
        if( itt_load_word_with_acquire(b->node_list) == internal::rehash_req )
        {
            bucket::scoped_t lock;
            if( lock.try_acquire( b->mutex, /*write=*/true ) ) {
                if( b->node_list == internal::rehash_req)
                    const_cast<concurrent_hash_map*>(this)->rehash_bucket( b, h & m ); //recursive rehashing
            }
            else lock.acquire( b->mutex, /*write=*/false );
            __TBB_ASSERT(b->node_list!=internal::rehash_req,NULL);
        }
        n = search_bucket( key, b );
        if( n )
            return n->storage();
        else if( check_mask_race( h, m ) )
            goto restart;
        return 0;
    }
};

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
namespace internal {
using namespace tbb::internal;

template<template<typename...> typename Map, typename Key, typename T, typename... Args>
using hash_map_t = Map<
    Key, T,
    std::conditional_t< (sizeof...(Args)>0) && !is_allocator_v< pack_element_t<0, Args...> >,
                        pack_element_t<0, Args...>, tbb_hash_compare<Key> >,
    std::conditional_t< (sizeof...(Args)>0) && is_allocator_v< pack_element_t<sizeof...(Args)-1, Args...> >,
                         pack_element_t<sizeof...(Args)-1, Args...>, tbb_allocator<std::pair<const Key, T> > >
>;
}

// Deduction guide for the constructor from two iterators and hash_compare/ allocator
template<typename I, typename... Args>
concurrent_hash_map(I, I, Args...)
-> internal::hash_map_t<concurrent_hash_map, internal::iterator_key_t<I>,internal::iterator_mapped_t<I>, Args...>;

// Deduction guide for the constructor from an initializer_list and hash_compare/ allocator
// Deduction guide for an initializer_list, hash_compare and allocator is implicit
template<typename Key, typename T, typename CompareOrAllocator>
concurrent_hash_map(std::initializer_list<std::pair<const Key, T>>, CompareOrAllocator)
-> internal::hash_map_t<concurrent_hash_map, Key, T, CompareOrAllocator>;

#endif /* __TBB_CPP17_DEDUCTION_GUIDES_PRESENT */

template<typename Key, typename T, typename HashCompare, typename A>
bool concurrent_hash_map<Key,T,HashCompare,A>::lookup( bool op_insert, const Key &key, const T *t, const_accessor *result, bool write, node* (*allocate_node)(node_allocator_type& , const Key&, const T*), node *tmp_n ) {
    __TBB_ASSERT( !result || !result->my_node, NULL );
    bool return_value;
    hashcode_t const h = my_hash_compare.hash( key );
    hashcode_t m = (hashcode_t) itt_load_word_with_acquire( my_mask );
    segment_index_t grow_segment = 0;
    node *n;
    restart:
    {//lock scope
        __TBB_ASSERT((m&(m+1))==0, "data structure is invalid");
        return_value = false;
        // get bucket
        bucket_accessor b( this, h & m );

        // find a node
        n = search_bucket( key, b() );
        if( op_insert ) {
            // [opt] insert a key
            if( !n ) {
                if( !tmp_n ) {
                    tmp_n = allocate_node(my_allocator, key, t);
                }
                if( !b.is_writer() && !b.upgrade_to_writer() ) { // TODO: improved insertion
                    // Rerun search_list, in case another thread inserted the item during the upgrade.
                    n = search_bucket( key, b() );
                    if( is_valid(n) ) { // unfortunately, it did
                        b.downgrade_to_reader();
                        goto exists;
                    }
                }
                if( check_mask_race(h, m) )
                    goto restart; // b.release() is done in ~b().
                // insert and set flag to grow the container
                grow_segment = insert_new_node( b(), n = tmp_n, m );
                tmp_n = 0;
                return_value = true;
            }
        } else { // find or count
            if( !n ) {
                if( check_mask_race( h, m ) )
                    goto restart; // b.release() is done in ~b(). TODO: replace by continue
                return false;
            }
            return_value = true;
        }
    exists:
        if( !result ) goto check_growth;
        // TODO: the following seems as generic/regular operation
        // acquire the item
        if( !result->try_acquire( n->mutex, write ) ) {
            for( tbb::internal::atomic_backoff backoff(true);; ) {
                if( result->try_acquire( n->mutex, write ) ) break;
                if( !backoff.bounded_pause() ) {
                    // the wait takes really long, restart the operation
                    b.release();
                    __TBB_ASSERT( !op_insert || !return_value, "Can't acquire new item in locked bucket?" );
                    __TBB_Yield();
                    m = (hashcode_t) itt_load_word_with_acquire( my_mask );
                    goto restart;
                }
            }
        }
    }//lock scope
    result->my_node = n;
    result->my_hash = h;
check_growth:
    // [opt] grow the container
    if( grow_segment ) {
#if __TBB_STATISTICS
        my_info_resizes++; // concurrent ones
#endif
        enable_segment( grow_segment, my_allocator );
    }
    if( tmp_n ) // if op_insert only
        delete_node( tmp_n );
    return return_value;
}

template<typename Key, typename T, typename HashCompare, typename A>
template<typename I>
std::pair<I, I> concurrent_hash_map<Key,T,HashCompare,A>::internal_equal_range( const Key& key, I end_ ) const {
    hashcode_t h = my_hash_compare.hash( key );
    hashcode_t m = my_mask;
    __TBB_ASSERT((m&(m+1))==0, "data structure is invalid");
    h &= m;
    bucket *b = get_bucket( h );
    while( b->node_list == internal::rehash_req ) {
        m = ( 1u<<__TBB_Log2( h ) ) - 1; // get parent mask from the topmost bit
        b = get_bucket( h &= m );
    }
    node *n = search_bucket( key, b );
    if( !n )
        return std::make_pair(end_, end_);
    iterator lower(*this, h, b, n), upper(lower);
    return std::make_pair(lower, ++upper);
}

template<typename Key, typename T, typename HashCompare, typename A>
bool concurrent_hash_map<Key,T,HashCompare,A>::exclude( const_accessor &item_accessor ) {
    __TBB_ASSERT( item_accessor.my_node, NULL );
    node_base *const n = item_accessor.my_node;
    hashcode_t const h = item_accessor.my_hash;
    hashcode_t m = (hashcode_t) itt_load_word_with_acquire( my_mask );
    do {
        // get bucket
        bucket_accessor b( this, h & m, /*writer=*/true );
        node_base **p = &b()->node_list;
        while( *p && *p != n )
            p = &(*p)->next;
        if( !*p ) { // someone else was first
            if( check_mask_race( h, m ) )
                continue;
            item_accessor.release();
            return false;
        }
        __TBB_ASSERT( *p == n, NULL );
        *p = n->next; // remove from container
        my_size--;
        break;
    } while(true);
    if( !item_accessor.is_writer() ) // need to get exclusive lock
        item_accessor.upgrade_to_writer(); // return value means nothing here
    item_accessor.release();
    delete_node( n ); // Only one thread can delete it
    return true;
}

template<typename Key, typename T, typename HashCompare, typename A>
bool concurrent_hash_map<Key,T,HashCompare,A>::erase( const Key &key ) {
    node_base *n;
    hashcode_t const h = my_hash_compare.hash( key );
    hashcode_t m = (hashcode_t) itt_load_word_with_acquire( my_mask );
restart:
    {//lock scope
        // get bucket
        bucket_accessor b( this, h & m );
    search:
        node_base **p = &b()->node_list;
        n = *p;
        while( is_valid(n) && !my_hash_compare.equal(key, static_cast<node*>(n)->value().first ) ) {
            p = &n->next;
            n = *p;
        }
        if( !n ) { // not found, but mask could be changed
            if( check_mask_race( h, m ) )
                goto restart;
            return false;
        }
        else if( !b.is_writer() && !b.upgrade_to_writer() ) {
            if( check_mask_race( h, m ) ) // contended upgrade, check mask
                goto restart;
            goto search;
        }
        *p = n->next;
        my_size--;
    }
    {
        typename node::scoped_t item_locker( n->mutex, /*write=*/true );
    }
    // note: there should be no threads pretending to acquire this mutex again, do not try to upgrade const_accessor!
    delete_node( n ); // Only one thread can delete it due to write lock on the bucket
    return true;
}

template<typename Key, typename T, typename HashCompare, typename A>
void concurrent_hash_map<Key,T,HashCompare,A>::swap(concurrent_hash_map<Key,T,HashCompare,A> &table) {
    typedef typename node_allocator_traits::propagate_on_container_swap pocs_type;
    if (this != &table && (pocs_type::value || my_allocator == table.my_allocator)) {
        using std::swap;
        tbb::internal::allocator_swap(this->my_allocator, table.my_allocator, pocs_type());
        swap(this->my_hash_compare, table.my_hash_compare);
        internal_swap(table);
    }
}

template<typename Key, typename T, typename HashCompare, typename A>
void concurrent_hash_map<Key,T,HashCompare,A>::rehash(size_type sz) {
    reserve( sz, my_allocator ); // TODO: add reduction of number of buckets as well
    hashcode_t mask = my_mask;
    hashcode_t b = (mask+1)>>1; // size or first index of the last segment
    __TBB_ASSERT((b&(b-1))==0, NULL); // zero or power of 2
    bucket *bp = get_bucket( b ); // only the last segment should be scanned for rehashing
    for(; b <= mask; b++, bp++ ) {
        node_base *n = bp->node_list;
        __TBB_ASSERT( is_valid(n) || n == internal::empty_rehashed || n == internal::rehash_req, "Broken internal structure" );
        __TBB_ASSERT( *reinterpret_cast<intptr_t*>(&bp->mutex) == 0, "concurrent or unexpectedly terminated operation during rehash() execution" );
        if( n == internal::rehash_req ) { // rehash bucket, conditional because rehashing of a previous bucket may affect this one
            hashcode_t h = b; bucket *b_old = bp;
            do {
                __TBB_ASSERT( h > 1, "The lowermost buckets can't be rehashed" );
                hashcode_t m = ( 1u<<__TBB_Log2( h ) ) - 1; // get parent mask from the topmost bit
                b_old = get_bucket( h &= m );
            } while( b_old->node_list == internal::rehash_req );
            // now h - is index of the root rehashed bucket b_old
            mark_rehashed_levels( h ); // mark all non-rehashed children recursively across all segments
            for( node_base **p = &b_old->node_list, *q = *p; is_valid(q); q = *p ) {
                hashcode_t c = my_hash_compare.hash( static_cast<node*>(q)->value().first );
                if( (c & mask) != h ) { // should be rehashed
                    *p = q->next; // exclude from b_old
                    bucket *b_new = get_bucket( c & mask );
                    __TBB_ASSERT( b_new->node_list != internal::rehash_req, "hash() function changed for key in table or internal error" );
                    add_to_bucket( b_new, q );
                } else p = &q->next; // iterate to next item
            }
        }
    }
#if TBB_USE_PERFORMANCE_WARNINGS
    int current_size = int(my_size), buckets = int(mask)+1, empty_buckets = 0, overpopulated_buckets = 0; // usage statistics
    static bool reported = false;
#endif
#if TBB_USE_ASSERT || TBB_USE_PERFORMANCE_WARNINGS
    for( b = 0; b <= mask; b++ ) {// only last segment should be scanned for rehashing
        if( b & (b-2) ) ++bp; // not the beginning of a segment
        else bp = get_bucket( b );
        node_base *n = bp->node_list;
        __TBB_ASSERT( *reinterpret_cast<intptr_t*>(&bp->mutex) == 0, "concurrent or unexpectedly terminated operation during rehash() execution" );
        __TBB_ASSERT( is_valid(n) || n == internal::empty_rehashed, "Broken internal structure" );
#if TBB_USE_PERFORMANCE_WARNINGS
        if( n == internal::empty_rehashed ) empty_buckets++;
        else if( n->next ) overpopulated_buckets++;
#endif
#if TBB_USE_ASSERT
        for( ; is_valid(n); n = n->next ) {
            hashcode_t h = my_hash_compare.hash( static_cast<node*>(n)->value().first ) & mask;
            __TBB_ASSERT( h == b, "hash() function changed for key in table or internal error" );
        }
#endif
    }
#endif // TBB_USE_ASSERT || TBB_USE_PERFORMANCE_WARNINGS
#if TBB_USE_PERFORMANCE_WARNINGS
    if( buckets > current_size) empty_buckets -= buckets - current_size;
    else overpopulated_buckets -= current_size - buckets; // TODO: load_factor?
    if( !reported && buckets >= 512 && ( 2*empty_buckets > current_size || 2*overpopulated_buckets > current_size ) ) {
        tbb::internal::runtime_warning(
            "Performance is not optimal because the hash function produces bad randomness in lower bits in %s.\nSize: %d  Empties: %d  Overlaps: %d",
#if __TBB_USE_OPTIONAL_RTTI
            typeid(*this).name(),
#else
            "concurrent_hash_map",
#endif
            current_size, empty_buckets, overpopulated_buckets );
        reported = true;
    }
#endif
}

template<typename Key, typename T, typename HashCompare, typename A>
void concurrent_hash_map<Key,T,HashCompare,A>::clear() {
    hashcode_t m = my_mask;
    __TBB_ASSERT((m&(m+1))==0, "data structure is invalid");
#if TBB_USE_ASSERT || TBB_USE_PERFORMANCE_WARNINGS || __TBB_STATISTICS
#if TBB_USE_PERFORMANCE_WARNINGS || __TBB_STATISTICS
    int current_size = int(my_size), buckets = int(m)+1, empty_buckets = 0, overpopulated_buckets = 0; // usage statistics
    static bool reported = false;
#endif
    bucket *bp = 0;
    // check consistency
    for( segment_index_t b = 0; b <= m; b++ ) {
        if( b & (b-2) ) ++bp; // not the beginning of a segment
        else bp = get_bucket( b );
        node_base *n = bp->node_list;
        __TBB_ASSERT( is_valid(n) || n == internal::empty_rehashed || n == internal::rehash_req, "Broken internal structure" );
        __TBB_ASSERT( *reinterpret_cast<intptr_t*>(&bp->mutex) == 0, "concurrent or unexpectedly terminated operation during clear() execution" );
#if TBB_USE_PERFORMANCE_WARNINGS || __TBB_STATISTICS
        if( n == internal::empty_rehashed ) empty_buckets++;
        else if( n == internal::rehash_req ) buckets--;
        else if( n->next ) overpopulated_buckets++;
#endif
#if __TBB_EXTRA_DEBUG
        for(; is_valid(n); n = n->next ) {
            hashcode_t h = my_hash_compare.hash( static_cast<node*>(n)->value().first );
            h &= m;
            __TBB_ASSERT( h == b || get_bucket(h)->node_list == internal::rehash_req, "hash() function changed for key in table or internal error" );
        }
#endif
    }
#if TBB_USE_PERFORMANCE_WARNINGS || __TBB_STATISTICS
#if __TBB_STATISTICS
    printf( "items=%d buckets: capacity=%d rehashed=%d empty=%d overpopulated=%d"
        " concurrent: resizes=%u rehashes=%u restarts=%u\n",
        current_size, int(m+1), buckets, empty_buckets, overpopulated_buckets,
        unsigned(my_info_resizes), unsigned(my_info_rehashes), unsigned(my_info_restarts) );
    my_info_resizes = 0; // concurrent ones
    my_info_restarts = 0; // race collisions
    my_info_rehashes = 0;  // invocations of rehash_bucket
#endif
    if( buckets > current_size) empty_buckets -= buckets - current_size;
    else overpopulated_buckets -= current_size - buckets; // TODO: load_factor?
    if( !reported && buckets >= 512 && ( 2*empty_buckets > current_size || 2*overpopulated_buckets > current_size ) ) {
        tbb::internal::runtime_warning(
            "Performance is not optimal because the hash function produces bad randomness in lower bits in %s.\nSize: %d  Empties: %d  Overlaps: %d",
#if __TBB_USE_OPTIONAL_RTTI
            typeid(*this).name(),
#else
            "concurrent_hash_map",
#endif
            current_size, empty_buckets, overpopulated_buckets );
        reported = true;
    }
#endif
#endif // TBB_USE_ASSERT || TBB_USE_PERFORMANCE_WARNINGS || __TBB_STATISTICS
    my_size = 0;
    segment_index_t s = segment_index_of( m );
    __TBB_ASSERT( s+1 == pointers_per_table || !my_table[s+1], "wrong mask or concurrent grow" );
    do {
        __TBB_ASSERT( is_valid( my_table[s] ), "wrong mask or concurrent grow" );
        segment_ptr_t buckets_ptr = my_table[s];
        size_type sz = segment_size( s ? s : 1 );
        for( segment_index_t i = 0; i < sz; i++ )
            for( node_base *n = buckets_ptr[i].node_list; is_valid(n); n = buckets_ptr[i].node_list ) {
                buckets_ptr[i].node_list = n->next;
                delete_node( n );
            }
        delete_segment(s, my_allocator);
    } while(s-- > 0);
    my_mask = embedded_buckets - 1;
}

template<typename Key, typename T, typename HashCompare, typename A>
void concurrent_hash_map<Key,T,HashCompare,A>::internal_copy( const concurrent_hash_map& source ) {
    hashcode_t mask = source.my_mask;
    if( my_mask == mask ) { // optimized version
        reserve( source.my_size, my_allocator ); // TODO: load_factor?
        bucket *dst = 0, *src = 0;
        bool rehash_required = false;
        for( hashcode_t k = 0; k <= mask; k++ ) {
            if( k & (k-2) ) ++dst,src++; // not the beginning of a segment
            else { dst = get_bucket( k ); src = source.get_bucket( k ); }
            __TBB_ASSERT( dst->node_list != internal::rehash_req, "Invalid bucket in destination table");
            node *n = static_cast<node*>( src->node_list );
            if( n == internal::rehash_req ) { // source is not rehashed, items are in previous buckets
                rehash_required = true;
                dst->node_list = internal::rehash_req;
            } else for(; n; n = static_cast<node*>( n->next ) ) {
                node* node_ptr = create_node(my_allocator, n->value().first, n->value().second);
                add_to_bucket( dst, node_ptr);
                ++my_size; // TODO: replace by non-atomic op
            }
        }
        if( rehash_required ) rehash();
    } else internal_copy( source.begin(), source.end(), source.my_size );
}

template<typename Key, typename T, typename HashCompare, typename A>
template<typename I>
void concurrent_hash_map<Key,T,HashCompare,A>::internal_copy(I first, I last, size_type reserve_size) {
    reserve( reserve_size, my_allocator ); // TODO: load_factor?
    hashcode_t m = my_mask;
    for(; first != last; ++first) {
        hashcode_t h = my_hash_compare.hash( (*first).first );
        bucket *b = get_bucket( h & m );
        __TBB_ASSERT( b->node_list != internal::rehash_req, "Invalid bucket in destination table");
        node* node_ptr = create_node(my_allocator, (*first).first, (*first).second);
        add_to_bucket( b, node_ptr );
        ++my_size; // TODO: replace by non-atomic op
    }
}

} // namespace interface5

using interface5::concurrent_hash_map;


template<typename Key, typename T, typename HashCompare, typename A1, typename A2>
inline bool operator==(const concurrent_hash_map<Key, T, HashCompare, A1> &a, const concurrent_hash_map<Key, T, HashCompare, A2> &b) {
    if(a.size() != b.size()) return false;
    typename concurrent_hash_map<Key, T, HashCompare, A1>::const_iterator i(a.begin()), i_end(a.end());
    typename concurrent_hash_map<Key, T, HashCompare, A2>::const_iterator j, j_end(b.end());
    for(; i != i_end; ++i) {
        j = b.equal_range(i->first).first;
        if( j == j_end || !(i->second == j->second) ) return false;
    }
    return true;
}

template<typename Key, typename T, typename HashCompare, typename A1, typename A2>
inline bool operator!=(const concurrent_hash_map<Key, T, HashCompare, A1> &a, const concurrent_hash_map<Key, T, HashCompare, A2> &b)
{    return !(a == b); }

template<typename Key, typename T, typename HashCompare, typename A>
inline void swap(concurrent_hash_map<Key, T, HashCompare, A> &a, concurrent_hash_map<Key, T, HashCompare, A> &b)
{    a.swap( b ); }

#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning( pop )
#endif // warning 4127 is back

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_concurrent_hash_map_H_include_area

#endif /* __TBB_concurrent_hash_map_H */
