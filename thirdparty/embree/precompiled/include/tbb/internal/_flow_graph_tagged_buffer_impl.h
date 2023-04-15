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

// a hash table buffer that can expand, and can support as many deletions as
// additions, list-based, with elements of list held in array (for destruction
// management), multiplicative hashing (like ets).  No synchronization built-in.
//

#ifndef __TBB__flow_graph_hash_buffer_impl_H
#define __TBB__flow_graph_hash_buffer_impl_H

#ifndef __TBB_flow_graph_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

// included in namespace tbb::flow::interfaceX::internal

// elements in the table are a simple list; we need pointer to next element to
// traverse the chain
template<typename ValueType>
struct buffer_element_type {
    // the second parameter below is void * because we can't forward-declare the type
    // itself, so we just reinterpret_cast below.
    typedef typename aligned_pair<ValueType, void *>::type type;
};

template
    <
     typename Key,         // type of key within ValueType
     typename ValueType,
     typename ValueToKey,  // abstract method that returns "const Key" or "const Key&" given ValueType
     typename HashCompare, // has hash and equal
     typename Allocator=tbb::cache_aligned_allocator< typename aligned_pair<ValueType, void *>::type >
    >
class hash_buffer : public HashCompare {
public:
    static const size_t INITIAL_SIZE = 8;  // initial size of the hash pointer table
    typedef ValueType value_type;
    typedef typename buffer_element_type< value_type >::type element_type;
    typedef value_type *pointer_type;
    typedef element_type *list_array_type;  // array we manage manually
    typedef list_array_type *pointer_array_type;
    typedef typename Allocator::template rebind<list_array_type>::other pointer_array_allocator_type;
    typedef typename Allocator::template rebind<element_type>::other elements_array_allocator;
    typedef typename tbb::internal::strip<Key>::type Knoref;

private:
    ValueToKey *my_key;
    size_t my_size;
    size_t nelements;
    pointer_array_type pointer_array;    // pointer_array[my_size]
    list_array_type elements_array;      // elements_array[my_size / 2]
    element_type* free_list;

    size_t mask() { return my_size - 1; }

    void set_up_free_list( element_type **p_free_list, list_array_type la, size_t sz) {
        for(size_t i=0; i < sz - 1; ++i ) {  // construct free list
            la[i].second = &(la[i+1]);
        }
        la[sz-1].second = NULL;
        *p_free_list = (element_type *)&(la[0]);
    }

    // cleanup for exceptions
    struct DoCleanup {
        pointer_array_type *my_pa;
        list_array_type *my_elements;
        size_t my_size;

        DoCleanup(pointer_array_type &pa, list_array_type &my_els, size_t sz) :
            my_pa(&pa), my_elements(&my_els), my_size(sz) {  }
        ~DoCleanup() {
            if(my_pa) {
                size_t dont_care = 0;
                internal_free_buffer(*my_pa, *my_elements, my_size, dont_care);
            }
        }
    };

    // exception-safety requires we do all the potentially-throwing operations first
    void grow_array() {
        size_t new_size = my_size*2;
        size_t new_nelements = nelements;  // internal_free_buffer zeroes this
        list_array_type new_elements_array = NULL;
        pointer_array_type new_pointer_array = NULL;
        list_array_type new_free_list = NULL;
        {
            DoCleanup my_cleanup(new_pointer_array, new_elements_array, new_size);
            new_elements_array = elements_array_allocator().allocate(my_size);
            new_pointer_array = pointer_array_allocator_type().allocate(new_size);
            for(size_t i=0; i < new_size; ++i) new_pointer_array[i] = NULL;
            set_up_free_list(&new_free_list, new_elements_array, my_size );

            for(size_t i=0; i < my_size; ++i) {
                for( element_type* op = pointer_array[i]; op; op = (element_type *)(op->second)) {
                    value_type *ov = reinterpret_cast<value_type *>(&(op->first));
                    // could have std::move semantics
                    internal_insert_with_key(new_pointer_array, new_size, new_free_list, *ov);
                }
            }
            my_cleanup.my_pa = NULL;
            my_cleanup.my_elements = NULL;
        }

        internal_free_buffer(pointer_array, elements_array, my_size, nelements);
        free_list = new_free_list;
        pointer_array = new_pointer_array;
        elements_array = new_elements_array;
        my_size = new_size;
        nelements = new_nelements;
    }

    // v should have perfect forwarding if std::move implemented.
    // we use this method to move elements in grow_array, so can't use class fields
    void internal_insert_with_key( element_type **p_pointer_array, size_t p_sz, list_array_type &p_free_list,
            const value_type &v) {
        size_t l_mask = p_sz-1;
        __TBB_ASSERT(my_key, "Error: value-to-key functor not provided");
        size_t h = this->hash((*my_key)(v)) & l_mask;
        __TBB_ASSERT(p_free_list, "Error: free list not set up.");
        element_type* my_elem = p_free_list; p_free_list = (element_type *)(p_free_list->second);
        (void) new(&(my_elem->first)) value_type(v);
        my_elem->second = p_pointer_array[h];
        p_pointer_array[h] = my_elem;
    }

    void internal_initialize_buffer() {
        pointer_array = pointer_array_allocator_type().allocate(my_size);
        for(size_t i = 0; i < my_size; ++i) pointer_array[i] = NULL;
        elements_array = elements_array_allocator().allocate(my_size / 2);
        set_up_free_list(&free_list, elements_array, my_size / 2);
    }

    // made static so an enclosed class can use to properly dispose of the internals
    static void internal_free_buffer( pointer_array_type &pa, list_array_type &el, size_t &sz, size_t &ne ) {
        if(pa) {
            for(size_t i = 0; i < sz; ++i ) {
                element_type *p_next;
                for( element_type *p = pa[i]; p; p = p_next) {
                    p_next = (element_type *)p->second;
                    internal::punned_cast<value_type *>(&(p->first))->~value_type();
                }
            }
            pointer_array_allocator_type().deallocate(pa, sz);
            pa = NULL;
        }
        // Separate test (if allocation of pa throws, el may be allocated.
        // but no elements will be constructed.)
        if(el) {
            elements_array_allocator().deallocate(el, sz / 2);
            el = NULL;
        }
        sz = INITIAL_SIZE;
        ne = 0;
    }

public:
    hash_buffer() : my_key(NULL), my_size(INITIAL_SIZE), nelements(0) {
        internal_initialize_buffer();
    }

    ~hash_buffer() {
        internal_free_buffer(pointer_array, elements_array, my_size, nelements);
        if(my_key) delete my_key;
    }

    void reset() {
        internal_free_buffer(pointer_array, elements_array, my_size, nelements);
        internal_initialize_buffer();
    }

    // Take ownership of func object allocated with new.
    // This method is only used internally, so can't be misused by user.
    void set_key_func(ValueToKey *vtk) { my_key = vtk; }
    // pointer is used to clone()
    ValueToKey* get_key_func() { return my_key; }

    bool insert_with_key(const value_type &v) {
        pointer_type p = NULL;
        __TBB_ASSERT(my_key, "Error: value-to-key functor not provided");
        if(find_ref_with_key((*my_key)(v), p)) {
            p->~value_type();
            (void) new(p) value_type(v);  // copy-construct into the space
            return false;
        }
        ++nelements;
        if(nelements*2 > my_size) grow_array();
        internal_insert_with_key(pointer_array, my_size, free_list, v);
        return true;
    }

    // returns true and sets v to array element if found, else returns false.
    bool find_ref_with_key(const Knoref& k, pointer_type &v) {
        size_t i = this->hash(k) & mask();
        for(element_type* p = pointer_array[i]; p; p = (element_type *)(p->second)) {
            pointer_type pv = reinterpret_cast<pointer_type>(&(p->first));
            __TBB_ASSERT(my_key, "Error: value-to-key functor not provided");
            if(this->equal((*my_key)(*pv), k)) {
                v = pv;
                return true;
            }
        }
        return false;
    }

    bool find_with_key( const Knoref& k, value_type &v) {
        value_type *p;
        if(find_ref_with_key(k, p)) {
            v = *p;
            return true;
        }
        else
            return false;
    }

    void delete_with_key(const Knoref& k) {
        size_t h = this->hash(k) & mask();
        element_type* prev = NULL;
        for(element_type* p = pointer_array[h]; p; prev = p, p = (element_type *)(p->second)) {
            value_type *vp = reinterpret_cast<value_type *>(&(p->first));
            __TBB_ASSERT(my_key, "Error: value-to-key functor not provided");
            if(this->equal((*my_key)(*vp), k)) {
                vp->~value_type();
                if(prev) prev->second = p->second;
                else pointer_array[h] = (element_type *)(p->second);
                p->second = free_list;
                free_list = p;
                --nelements;
                return;
            }
        }
        __TBB_ASSERT(false, "key not found for delete");
    }
};
#endif // __TBB__flow_graph_hash_buffer_impl_H
