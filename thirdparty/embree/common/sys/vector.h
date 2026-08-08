// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "alloc.h"
#include <algorithm>
#include <type_traits>

namespace embree
{
  class Device;
  
   template<typename T, typename allocator>
    class vector_t
    {
    public:
      typedef T value_type;
      typedef T* iterator;
      typedef const T* const_iterator;
    
      __forceinline vector_t () 
        : size_active(0), size_alloced(0), items(nullptr) {}
    
      __forceinline explicit vector_t (size_t sz) 
        : size_active(0), size_alloced(0), items(nullptr) { internal_resize_init(sz); }
    
      template<typename M>
      __forceinline explicit vector_t (M alloc, size_t sz) 
      : alloc(alloc), size_active(0), size_alloced(0), items(nullptr) { internal_resize_init(sz); }

      __forceinline vector_t (Device* alloc)
        : vector_t(alloc,0) {}

      __forceinline vector_t(void* data, size_t bytes)
        : size_active(0), size_alloced(bytes/sizeof(T)), items((T*)data) {}
    
      __forceinline ~vector_t() {
        clear();
      }
    
      __forceinline vector_t (const vector_t& other)
      {
        size_active = other.size_active;
        size_alloced = other.size_alloced;
        items = alloc.allocate(size_alloced);
        for (size_t i=0; i<size_active; i++) 
          ::new (&items[i]) value_type(other.items[i]);
      }
    
      __forceinline vector_t (vector_t&& other)
        : alloc(std::move(other.alloc))
      {
        size_active = other.size_active; other.size_active = 0;
        size_alloced = other.size_alloced; other.size_alloced = 0;
        items = other.items; other.items = nullptr;
      }

      __forceinline vector_t& operator=(const vector_t& other) 
      {
        resize(other.size_active);
        for (size_t i=0; i<size_active; i++)
          items[i] = value_type(other.items[i]);
        return *this;
      }

      __forceinline vector_t& operator=(vector_t&& other) 
      {
        clear();
        alloc = std::move(other.alloc);
        size_active = other.size_active; other.size_active = 0;
        size_alloced = other.size_alloced; other.size_alloced = 0;
        items = other.items; other.items = nullptr;
        return *this;
      }

      __forceinline allocator& getAlloc() {
	return alloc;
      }

      /********************** Iterators  ****************************/
    
      __forceinline       iterator begin()       { return items; };
      __forceinline const_iterator begin() const { return items; };

      __forceinline       iterator end  ()       { return items+size_active; };
      __forceinline const_iterator end  () const { return items+size_active; };


      /********************** Capacity ****************************/

      __forceinline bool   empty    () const { return size_active == 0; }
      __forceinline size_t size     () const { return size_active; }
      __forceinline size_t capacity () const { return size_alloced; }


      __forceinline void resize(size_t new_size) {
        internal_resize(new_size,internal_grow_size(new_size));
      }

      __forceinline void reserve(size_t new_alloced) 
      {
        /* do nothing if container already large enough */
        if (new_alloced <= size_alloced) 
          return;

        /* resize exact otherwise */
        internal_resize(size_active,new_alloced);
      }

      __forceinline void shrink_to_fit() {
        internal_resize(size_active,size_active);
      }

      /******************** Element access **************************/

      __forceinline       T& operator[](size_t i)       { assert(i < size_active); return items[i]; }
      __forceinline const T& operator[](size_t i) const { assert(i < size_active); return items[i]; }

      __forceinline       T& at(size_t i)       { assert(i < size_active); return items[i]; }
      __forceinline const T& at(size_t i) const { assert(i < size_active); return items[i]; }

      __forceinline T& front() const { assert(size_active > 0); return items[0]; };
      __forceinline T& back () const { assert(size_active > 0); return items[size_active-1]; };

      __forceinline       T* data()       { return items; };
      __forceinline const T* data() const { return items; };
      
      /* dangerous only use if you know what you're doing */
      __forceinline void setDataPtr(T* data) { items = data; }

      /******************** Modifiers **************************/

      __forceinline void push_back(const T& nt) 
      {
        const T v = nt; // need local copy as input reference could point to this vector
        internal_resize(size_active,internal_grow_size(size_active+1));
        ::new (&items[size_active++]) T(v);
      }

      __forceinline void pop_back() 
      {
        assert(!empty());
        size_active--;
        items[size_active].~T();
      }

      __forceinline void clear() 
      {
        /* destroy elements */
        for (size_t i=0; i<size_active; i++){
          items[i].~T();
        }
        
        /* free memory */
        alloc.deallocate(items,size_alloced); 
        items = nullptr;
        size_active = size_alloced = 0;
      }

    /******************** Comparisons **************************/
    
    friend bool operator== (const vector_t& a, const vector_t& b) 
    {
      if (a.size() != b.size()) return false;
      for (size_t i=0; i<a.size(); i++)
        if (a[i] != b[i])
          return false;
      return true;
    }

    friend bool operator!= (const vector_t& a, const vector_t& b) {
      return !(a==b);
    }

    private:

      __forceinline void internal_resize_init(size_t new_active)
      {
        assert(size_active == 0); 
        assert(size_alloced == 0);
        assert(items == nullptr);
        if (new_active == 0) return;
        items = alloc.allocate(new_active);
        for (size_t i=0; i<new_active; i++) ::new (&items[i]) T();
        size_active = new_active;
        size_alloced = new_active;
      }

      __forceinline void internal_resize(size_t new_active, size_t new_alloced)
      {
        assert(new_active <= new_alloced); 

        /* destroy elements */
        if (new_active < size_active) 
        {
          for (size_t i=new_active; i<size_active; i++){
            items[i].~T();
          }
          size_active = new_active;
        }

        /* only reallocate if necessary */
        if (new_alloced == size_alloced) {
          for (size_t i=size_active; i<new_active; i++) ::new (&items[i]) T;
          size_active = new_active;
          return;
        }

        /* reallocate and copy items */
        T* old_items = items;
        items = alloc.allocate(new_alloced);
        for (size_t i=0; i<size_active; i++) {
          ::new (&items[i]) T(std::move(old_items[i]));
          old_items[i].~T();
        }

        for (size_t i=size_active; i<new_active; i++) {
          ::new (&items[i]) T;
        }

        alloc.deallocate(old_items,size_alloced);
        size_active = new_active;
        size_alloced = new_alloced;
      }

      __forceinline size_t internal_grow_size(size_t new_alloced)
      {
        /* do nothing if container already large enough */
        if (new_alloced <= size_alloced) 
          return size_alloced;

        /* if current size is 0 allocate exact requested size */
        if (size_alloced == 0)
          return new_alloced;

        /* resize to next power of 2 otherwise */
        size_t new_size_alloced = size_alloced;
        while (new_size_alloced < new_alloced) {
          new_size_alloced = std::max(size_t(1),2*new_size_alloced);
        }
        return new_size_alloced;
      }

    private:
      allocator alloc;
      size_t size_active;    // number of valid items
      size_t size_alloced;   // number of items allocated
      T* items;              // data array
    };

  /*! vector class that performs standard allocations */
  template<typename T>
    using vector = vector_t<T,std::allocator<T>>;

  /*! vector class that performs aligned allocations */
  template<typename T>
    using avector = vector_t<T,aligned_allocator<T,std::alignment_of<T>::value> >;

  /*! vector class that performs OS allocations */
  template<typename T>
    using ovector = vector_t<T,os_allocator<T> >;

  /*! vector class with externally managed data buffer */
  template<typename T>
    using evector = vector_t<T,no_allocator<T>>;
}
