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

#ifndef __TBB_concurrent_priority_queue_H
#define __TBB_concurrent_priority_queue_H

#define __TBB_concurrent_priority_queue_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "atomic.h"
#include "cache_aligned_allocator.h"
#include "tbb_exception.h"
#include "tbb_stddef.h"
#include "tbb_profiling.h"
#include "internal/_aggregator_impl.h"
#include "internal/_template_helpers.h"
#include "internal/_allocator_traits.h"
#include <vector>
#include <iterator>
#include <functional>
#include __TBB_STD_SWAP_HEADER

#if __TBB_INITIALIZER_LISTS_PRESENT
    #include <initializer_list>
#endif

#if __TBB_CPP11_IS_COPY_CONSTRUCTIBLE_PRESENT
    #include <type_traits>
#endif

namespace tbb {
namespace interface5 {
namespace internal {
#if __TBB_CPP11_IS_COPY_CONSTRUCTIBLE_PRESENT
    template<typename T, bool C = std::is_copy_constructible<T>::value>
    struct use_element_copy_constructor {
        typedef tbb::internal::true_type type;
    };
    template<typename T>
    struct use_element_copy_constructor <T,false> {
        typedef tbb::internal::false_type type;
    };
#else
    template<typename>
    struct use_element_copy_constructor {
        typedef tbb::internal::true_type type;
    };
#endif
} // namespace internal

using namespace tbb::internal;

//! Concurrent priority queue
template <typename T, typename Compare=std::less<T>, typename A=cache_aligned_allocator<T> >
class concurrent_priority_queue {
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

    //! Constructs a new concurrent_priority_queue with default capacity
    explicit concurrent_priority_queue(const allocator_type& a = allocator_type()) : mark(0), my_size(0), compare(), data(a)
    {
        my_aggregator.initialize_handler(my_functor_t(this));
    }

    //! Constructs a new concurrent_priority_queue with default capacity
    explicit concurrent_priority_queue(const Compare& c, const allocator_type& a = allocator_type()) : mark(0), my_size(0), compare(c), data(a)
    {
        my_aggregator.initialize_handler(my_functor_t(this));
    }

    //! Constructs a new concurrent_priority_queue with init_sz capacity
    explicit concurrent_priority_queue(size_type init_capacity, const allocator_type& a = allocator_type()) :
        mark(0), my_size(0), compare(), data(a)
    {
        data.reserve(init_capacity);
        my_aggregator.initialize_handler(my_functor_t(this));
    }

    //! Constructs a new concurrent_priority_queue with init_sz capacity
    explicit concurrent_priority_queue(size_type init_capacity, const Compare& c, const allocator_type& a = allocator_type()) :
        mark(0), my_size(0), compare(c), data(a)
    {
        data.reserve(init_capacity);
        my_aggregator.initialize_handler(my_functor_t(this));
    }

    //! [begin,end) constructor
    template<typename InputIterator>
    concurrent_priority_queue(InputIterator begin, InputIterator end, const allocator_type& a = allocator_type()) :
        mark(0), compare(), data(begin, end, a)
    {
        my_aggregator.initialize_handler(my_functor_t(this));
        heapify();
        my_size = data.size();
    }

    //! [begin,end) constructor
    template<typename InputIterator>
    concurrent_priority_queue(InputIterator begin, InputIterator end, const Compare& c, const allocator_type& a = allocator_type()) :
        mark(0), compare(c), data(begin, end, a)
    {
        my_aggregator.initialize_handler(my_functor_t(this));
        heapify();
        my_size = data.size();
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Constructor from std::initializer_list
    concurrent_priority_queue(std::initializer_list<T> init_list, const allocator_type &a = allocator_type()) :
        mark(0), compare(), data(init_list.begin(), init_list.end(), a)
    {
        my_aggregator.initialize_handler(my_functor_t(this));
        heapify();
        my_size = data.size();
    }

    //! Constructor from std::initializer_list
    concurrent_priority_queue(std::initializer_list<T> init_list, const Compare& c, const allocator_type &a = allocator_type()) :
        mark(0), compare(c), data(init_list.begin(), init_list.end(), a)
    {
        my_aggregator.initialize_handler(my_functor_t(this));
        heapify();
        my_size = data.size();
    }
#endif //# __TBB_INITIALIZER_LISTS_PRESENT

    //! Copy constructor
    /** This operation is unsafe if there are pending concurrent operations on the src queue. */
    concurrent_priority_queue(const concurrent_priority_queue& src) : mark(src.mark),
        my_size(src.my_size), data(src.data.begin(), src.data.end(), src.data.get_allocator())
    {
        my_aggregator.initialize_handler(my_functor_t(this));
        heapify();
    }

    //! Copy constructor with specific allocator
    /** This operation is unsafe if there are pending concurrent operations on the src queue. */
    concurrent_priority_queue(const concurrent_priority_queue& src, const allocator_type& a) : mark(src.mark),
        my_size(src.my_size), data(src.data.begin(), src.data.end(), a)
    {
        my_aggregator.initialize_handler(my_functor_t(this));
        heapify();
    }

    //! Assignment operator
    /** This operation is unsafe if there are pending concurrent operations on the src queue. */
    concurrent_priority_queue& operator=(const concurrent_priority_queue& src) {
        if (this != &src) {
            vector_t(src.data.begin(), src.data.end(), src.data.get_allocator()).swap(data);
            mark = src.mark;
            my_size = src.my_size;
        }
        return *this;
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    //! Move constructor
    /** This operation is unsafe if there are pending concurrent operations on the src queue. */
    concurrent_priority_queue(concurrent_priority_queue&& src) : mark(src.mark),
        my_size(src.my_size), data(std::move(src.data))
    {
        my_aggregator.initialize_handler(my_functor_t(this));
    }

    //! Move constructor with specific allocator
    /** This operation is unsafe if there are pending concurrent operations on the src queue. */
    concurrent_priority_queue(concurrent_priority_queue&& src, const allocator_type& a) : mark(src.mark),
        my_size(src.my_size),
#if __TBB_ALLOCATOR_TRAITS_PRESENT
        data(std::move(src.data), a)
#else
    // Some early version of C++11 STL vector does not have a constructor of vector(vector&& , allocator).
    // It seems that the reason is absence of support of allocator_traits (stateful allocators).
        data(a)
#endif //__TBB_ALLOCATOR_TRAITS_PRESENT
    {
        my_aggregator.initialize_handler(my_functor_t(this));
#if !__TBB_ALLOCATOR_TRAITS_PRESENT
        if (a != src.data.get_allocator()){
            data.reserve(src.data.size());
            data.assign(std::make_move_iterator(src.data.begin()), std::make_move_iterator(src.data.end()));
        }else{
            data = std::move(src.data);
        }
#endif //!__TBB_ALLOCATOR_TRAITS_PRESENT
    }

    //! Move assignment operator
    /** This operation is unsafe if there are pending concurrent operations on the src queue. */
    concurrent_priority_queue& operator=( concurrent_priority_queue&& src) {
        if (this != &src) {
            mark = src.mark;
            my_size = src.my_size;
#if !__TBB_ALLOCATOR_TRAITS_PRESENT
            if (data.get_allocator() != src.data.get_allocator()){
                vector_t(std::make_move_iterator(src.data.begin()), std::make_move_iterator(src.data.end()), data.get_allocator()).swap(data);
            }else
#endif //!__TBB_ALLOCATOR_TRAITS_PRESENT
            {
                data = std::move(src.data);
            }
        }
        return *this;
    }
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

    //! Assign the queue from [begin,end) range, not thread-safe
    template<typename InputIterator>
    void assign(InputIterator begin, InputIterator end) {
        vector_t(begin, end, data.get_allocator()).swap(data);
        mark = 0;
        my_size = data.size();
        heapify();
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Assign the queue from std::initializer_list, not thread-safe
    void assign(std::initializer_list<T> il) { this->assign(il.begin(), il.end()); }

    //! Assign from std::initializer_list, not thread-safe
    concurrent_priority_queue& operator=(std::initializer_list<T> il) {
        this->assign(il.begin(), il.end());
        return *this;
    }
#endif //# __TBB_INITIALIZER_LISTS_PRESENT

    //! Returns true if empty, false otherwise
    /** Returned value may not reflect results of pending operations.
        This operation reads shared data and will trigger a race condition. */
    bool empty() const { return size()==0; }

    //! Returns the current number of elements contained in the queue
    /** Returned value may not reflect results of pending operations.
        This operation reads shared data and will trigger a race condition. */
    size_type size() const { return __TBB_load_with_acquire(my_size); }

    //! Pushes elem onto the queue, increasing capacity of queue if necessary
    /** This operation can be safely used concurrently with other push, try_pop or emplace operations. */
    void push(const_reference elem) {
#if __TBB_CPP11_IS_COPY_CONSTRUCTIBLE_PRESENT
        __TBB_STATIC_ASSERT( std::is_copy_constructible<value_type>::value, "The type is not copy constructible. Copying push operation is impossible." );
#endif
        cpq_operation op_data(elem, PUSH_OP);
        my_aggregator.execute(&op_data);
        if (op_data.status == FAILED) // exception thrown
            throw_exception(eid_bad_alloc);
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    //! Pushes elem onto the queue, increasing capacity of queue if necessary
    /** This operation can be safely used concurrently with other push, try_pop or emplace operations. */
    void push(value_type &&elem) {
        cpq_operation op_data(elem, PUSH_RVALUE_OP);
        my_aggregator.execute(&op_data);
        if (op_data.status == FAILED) // exception thrown
            throw_exception(eid_bad_alloc);
    }

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    //! Constructs a new element using args as the arguments for its construction and pushes it onto the queue */
    /** This operation can be safely used concurrently with other push, try_pop or emplace operations. */
    template<typename... Args>
    void emplace(Args&&... args) {
        push(value_type(std::forward<Args>(args)...));
    }
#endif /* __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT */
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

    //! Gets a reference to and removes highest priority element
    /** If a highest priority element was found, sets elem and returns true,
        otherwise returns false.
        This operation can be safely used concurrently with other push, try_pop or emplace operations. */
    bool try_pop(reference elem) {
        cpq_operation op_data(POP_OP);
        op_data.elem = &elem;
        my_aggregator.execute(&op_data);
        return op_data.status==SUCCEEDED;
    }

    //! Clear the queue; not thread-safe
    /** This operation is unsafe if there are pending concurrent operations on the queue.
        Resets size, effectively emptying queue; does not free space.
        May not clear elements added in pending operations. */
    void clear() {
        data.clear();
        mark = 0;
        my_size = 0;
    }

    //! Swap this queue with another; not thread-safe
    /** This operation is unsafe if there are pending concurrent operations on the queue. */
    void swap(concurrent_priority_queue& q) {
        using std::swap;
        data.swap(q.data);
        swap(mark, q.mark);
        swap(my_size, q.my_size);
    }

    //! Return allocator object
    allocator_type get_allocator() const { return data.get_allocator(); }

 private:
    enum operation_type {INVALID_OP, PUSH_OP, POP_OP, PUSH_RVALUE_OP};
    enum operation_status { WAIT=0, SUCCEEDED, FAILED };

    class cpq_operation : public aggregated_operation<cpq_operation> {
     public:
        operation_type type;
        union {
            value_type *elem;
            size_type sz;
        };
        cpq_operation(const_reference e, operation_type t) :
            type(t), elem(const_cast<value_type*>(&e)) {}
        cpq_operation(operation_type t) : type(t) {}
    };

    class my_functor_t {
        concurrent_priority_queue<T, Compare, A> *cpq;
     public:
        my_functor_t() {}
        my_functor_t(concurrent_priority_queue<T, Compare, A> *cpq_) : cpq(cpq_) {}
        void operator()(cpq_operation* op_list) {
            cpq->handle_operations(op_list);
        }
    };

    typedef tbb::internal::aggregator< my_functor_t, cpq_operation > aggregator_t;
    aggregator_t my_aggregator;
    //! Padding added to avoid false sharing
    char padding1[NFS_MaxLineSize - sizeof(aggregator_t)];
    //! The point at which unsorted elements begin
    size_type mark;
    __TBB_atomic size_type my_size;
    Compare compare;
    //! Padding added to avoid false sharing
    char padding2[NFS_MaxLineSize - (2*sizeof(size_type)) - sizeof(Compare)];
    //! Storage for the heap of elements in queue, plus unheapified elements
    /** data has the following structure:

         binary unheapified
          heap   elements
        ____|_______|____
        |       |       |
        v       v       v
        [_|...|_|_|...|_| |...| ]
         0       ^       ^       ^
                 |       |       |__capacity
                 |       |__my_size
                 |__mark

        Thus, data stores the binary heap starting at position 0 through
        mark-1 (it may be empty).  Then there are 0 or more elements
        that have not yet been inserted into the heap, in positions
        mark through my_size-1. */
    typedef std::vector<value_type, allocator_type> vector_t;
    vector_t data;

    void handle_operations(cpq_operation *op_list) {
        cpq_operation *tmp, *pop_list=NULL;

        __TBB_ASSERT(mark == data.size(), NULL);

        // First pass processes all constant (amortized; reallocation may happen) time pushes and pops.
        while (op_list) {
            // ITT note: &(op_list->status) tag is used to cover accesses to op_list
            // node. This thread is going to handle the operation, and so will acquire it
            // and perform the associated operation w/o triggering a race condition; the
            // thread that created the operation is waiting on the status field, so when
            // this thread is done with the operation, it will perform a
            // store_with_release to give control back to the waiting thread in
            // aggregator::insert_operation.
            call_itt_notify(acquired, &(op_list->status));
            __TBB_ASSERT(op_list->type != INVALID_OP, NULL);
            tmp = op_list;
            op_list = itt_hide_load_word(op_list->next);
            if (tmp->type == POP_OP) {
                if (mark < data.size() &&
                    compare(data[0], data[data.size()-1])) {
                    // there are newly pushed elems and the last one
                    // is higher than top
                    *(tmp->elem) = tbb::internal::move(data[data.size()-1]);
                    __TBB_store_with_release(my_size, my_size-1);
                    itt_store_word_with_release(tmp->status, uintptr_t(SUCCEEDED));
                    data.pop_back();
                    __TBB_ASSERT(mark<=data.size(), NULL);
                }
                else { // no convenient item to pop; postpone
                    itt_hide_store_word(tmp->next, pop_list);
                    pop_list = tmp;
                }
            } else { // PUSH_OP or PUSH_RVALUE_OP
                __TBB_ASSERT(tmp->type == PUSH_OP || tmp->type == PUSH_RVALUE_OP, "Unknown operation" );
                __TBB_TRY{
                    if (tmp->type == PUSH_OP) {
                        push_back_helper(*(tmp->elem), typename internal::use_element_copy_constructor<value_type>::type());
                    } else {
                        data.push_back(tbb::internal::move(*(tmp->elem)));
                    }
                    __TBB_store_with_release(my_size, my_size + 1);
                    itt_store_word_with_release(tmp->status, uintptr_t(SUCCEEDED));
                } __TBB_CATCH(...) {
                    itt_store_word_with_release(tmp->status, uintptr_t(FAILED));
                }
            }
        }

        // second pass processes pop operations
        while (pop_list) {
            tmp = pop_list;
            pop_list = itt_hide_load_word(pop_list->next);
            __TBB_ASSERT(tmp->type == POP_OP, NULL);
            if (data.empty()) {
                itt_store_word_with_release(tmp->status, uintptr_t(FAILED));
            }
            else {
                __TBB_ASSERT(mark<=data.size(), NULL);
                if (mark < data.size() &&
                    compare(data[0], data[data.size()-1])) {
                    // there are newly pushed elems and the last one is
                    // higher than top
                    *(tmp->elem) = tbb::internal::move(data[data.size()-1]);
                    __TBB_store_with_release(my_size, my_size-1);
                    itt_store_word_with_release(tmp->status, uintptr_t(SUCCEEDED));
                    data.pop_back();
                }
                else { // extract top and push last element down heap
                    *(tmp->elem) = tbb::internal::move(data[0]);
                    __TBB_store_with_release(my_size, my_size-1);
                    itt_store_word_with_release(tmp->status, uintptr_t(SUCCEEDED));
                    reheap();
                }
            }
        }

        // heapify any leftover pushed elements before doing the next
        // batch of operations
        if (mark<data.size()) heapify();
        __TBB_ASSERT(mark == data.size(), NULL);
    }

    //! Merge unsorted elements into heap
    void heapify() {
        if (!mark && data.size()>0) mark = 1;
        for (; mark<data.size(); ++mark) {
            // for each unheapified element under size
            size_type cur_pos = mark;
            value_type to_place = tbb::internal::move(data[mark]);
            do { // push to_place up the heap
                size_type parent = (cur_pos-1)>>1;
                if (!compare(data[parent], to_place)) break;
                data[cur_pos] = tbb::internal::move(data[parent]);
                cur_pos = parent;
            } while( cur_pos );
            data[cur_pos] = tbb::internal::move(to_place);
        }
    }

    //! Re-heapify after an extraction
    /** Re-heapify by pushing last element down the heap from the root. */
    void reheap() {
        size_type cur_pos=0, child=1;

        while (child < mark) {
            size_type target = child;
            if (child+1 < mark && compare(data[child], data[child+1]))
                ++target;
            // target now has the higher priority child
            if (compare(data[target], data[data.size()-1])) break;
            data[cur_pos] = tbb::internal::move(data[target]);
            cur_pos = target;
            child = (cur_pos<<1)+1;
        }
        if (cur_pos != data.size()-1)
            data[cur_pos] = tbb::internal::move(data[data.size()-1]);
        data.pop_back();
        if (mark > data.size()) mark = data.size();
    }

    void push_back_helper(const T& t, tbb::internal::true_type) {
        data.push_back(t);
    }

    void push_back_helper(const T&, tbb::internal::false_type) {
        __TBB_ASSERT( false, "The type is not copy constructible. Copying push operation is impossible." );
    }
};

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
namespace internal {

template<typename T, typename... Args>
using priority_queue_t = concurrent_priority_queue<
    T,
    std::conditional_t< (sizeof...(Args)>0) && !is_allocator_v< pack_element_t<0, Args...> >,
                        pack_element_t<0, Args...>, std::less<T> >,
    std::conditional_t< (sizeof...(Args)>0) && is_allocator_v< pack_element_t<sizeof...(Args)-1, Args...> >,
                         pack_element_t<sizeof...(Args)-1, Args...>, cache_aligned_allocator<T> >
>;
}

// Deduction guide for the constructor from two iterators
template<typename InputIterator,
         typename T = typename std::iterator_traits<InputIterator>::value_type,
         typename... Args
> concurrent_priority_queue(InputIterator, InputIterator, Args...)
-> internal::priority_queue_t<T, Args...>;

template<typename T, typename CompareOrAllocalor>
concurrent_priority_queue(std::initializer_list<T> init_list, CompareOrAllocalor)
-> internal::priority_queue_t<T, CompareOrAllocalor>;

#endif /* __TBB_CPP17_DEDUCTION_GUIDES_PRESENT */
} // namespace interface5

using interface5::concurrent_priority_queue;

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_concurrent_priority_queue_H_include_area

#endif /* __TBB_concurrent_priority_queue_H */
