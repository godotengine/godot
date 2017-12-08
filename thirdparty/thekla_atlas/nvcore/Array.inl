// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_CORE_ARRAY_INL
#define NV_CORE_ARRAY_INL

#include "Array.h"

#include "Stream.h"
#include "Utils.h" // swap

#include <string.h>	// memmove
#include <new> // for placement new



namespace nv 
{
    template <typename T>
    NV_FORCEINLINE T & Array<T>::append()
    {
        uint old_size = m_size;
        uint new_size = m_size + 1;

        setArraySize(new_size);

        construct_range(m_buffer, new_size, old_size);

        return m_buffer[old_size]; // Return reference to last element.
    }

    // Push an element at the end of the vector.
    template <typename T>
    NV_FORCEINLINE void Array<T>::push_back( const T & val )
    {
#if 1
        nvDebugCheck(&val < m_buffer || &val >= m_buffer+m_size);

        uint old_size = m_size;
        uint new_size = m_size + 1;

        setArraySize(new_size);

        construct_range(m_buffer, new_size, old_size, val);
#else
        uint new_size = m_size + 1;

        if (new_size > m_capacity)
        {
            // @@ Is there any way to avoid this copy?
            // @@ Can we create a copy without side effects? Ie. without calls to constructor/destructor. Use alloca + memcpy?
            // @@ Assert instead of copy?
            const T copy(val);	// create a copy in case value is inside of this array.

            setArraySize(new_size);

            new (m_buffer+new_size-1) T(copy);
        }
        else
        {
            m_size = new_size;
            new(m_buffer+new_size-1) T(val);
        }
#endif // 0/1
    }
    template <typename T>
    NV_FORCEINLINE void Array<T>::pushBack( const T & val )
    {
        push_back(val);
    }
    template <typename T>
    NV_FORCEINLINE Array<T> & Array<T>::append( const T & val )
    {
        push_back(val);
        return *this;
    }

    // Qt like push operator.
    template <typename T>
    NV_FORCEINLINE Array<T> & Array<T>::operator<< ( T & t )
    {
        push_back(t);
        return *this;
    }

    // Pop the element at the end of the vector.
    template <typename T>
    NV_FORCEINLINE void Array<T>::pop_back()
    {
        nvDebugCheck( m_size > 0 );
        resize( m_size - 1 );
    }
    template <typename T>
    NV_FORCEINLINE void Array<T>::popBack(uint count)
    {
        nvDebugCheck(m_size >= count);
        resize(m_size - count);
    }

    template <typename T>
    NV_FORCEINLINE void Array<T>::popFront(uint count)
    {
        nvDebugCheck(m_size >= count);
        //resize(m_size - count);

        if (m_size == count) {
            clear();
        }
        else {
            destroy_range(m_buffer, 0, count);

            memmove(m_buffer, m_buffer + count, sizeof(T) * (m_size - count));

            m_size -= count;
        }

    }


    // Get back element.
    template <typename T>
    NV_FORCEINLINE const T & Array<T>::back() const
    {
        nvDebugCheck( m_size > 0 );
        return m_buffer[m_size-1];
    }

    // Get back element.
    template <typename T>
    NV_FORCEINLINE T & Array<T>::back()
    {
        nvDebugCheck( m_size > 0 );
        return m_buffer[m_size-1];
    }

    // Get front element.
    template <typename T>
    NV_FORCEINLINE const T & Array<T>::front() const
    {
        nvDebugCheck( m_size > 0 );
        return m_buffer[0];
    }

    // Get front element.
    template <typename T>
    NV_FORCEINLINE T & Array<T>::front()
    {
        nvDebugCheck( m_size > 0 );
        return m_buffer[0];
    }

    // Check if the given element is contained in the array.
    template <typename T>
    NV_FORCEINLINE bool Array<T>::contains(const T & e) const
    {
        return find(e, NULL);
    }

    // Return true if element found.
    template <typename T>
    NV_FORCEINLINE bool Array<T>::find(const T & element, uint * indexPtr) const
    {
        return find(element, 0, m_size, indexPtr);
    }

    // Return true if element found within the given range.
    template <typename T>
    NV_FORCEINLINE bool Array<T>::find(const T & element, uint begin, uint end, uint * indexPtr) const
    {
        return ::nv::find(element, m_buffer, begin, end, indexPtr);
    }


    // Remove the element at the given index. This is an expensive operation!
    template <typename T>
    void Array<T>::removeAt(uint index)
    {
        nvDebugCheck(index >= 0 && index < m_size);

        if (m_size == 1) {
            clear();
        }
        else {
            m_buffer[index].~T();

            memmove(m_buffer+index, m_buffer+index+1, sizeof(T) * (m_size - 1 - index));
            m_size--;
        }
    }

    // Remove the first instance of the given element.
    template <typename T>
    bool Array<T>::remove(const T & element)
    {
        uint index;
        if (find(element, &index)) {
            removeAt(index);
            return true;
        }
        return false;
    }

    // Insert the given element at the given index shifting all the elements up.
    template <typename T>
    void Array<T>::insertAt(uint index, const T & val/*=T()*/)
    {
        nvDebugCheck( index >= 0 && index <= m_size );

        setArraySize(m_size + 1);

        if (index < m_size - 1) {
            memmove(m_buffer+index+1, m_buffer+index, sizeof(T) * (m_size - 1 - index));
        }

        // Copy-construct into the newly opened slot.
        new(m_buffer+index) T(val);
    }

    // Append the given data to our vector.
    template <typename T>
    NV_FORCEINLINE void Array<T>::append(const Array<T> & other)
    {
        append(other.m_buffer, other.m_size);
    }

    // Append the given data to our vector.
    template <typename T>
    void Array<T>::append(const T other[], uint count)
    {
        if (count > 0) {
            const uint old_size = m_size;

            setArraySize(m_size + count);

            for (uint i = 0; i < count; i++ ) {
                new(m_buffer + old_size + i) T(other[i]);
            }
        }
    }


    // Remove the given element by replacing it with the last one.
    template <typename T> 
    void Array<T>::replaceWithLast(uint index)
    {
        nvDebugCheck( index < m_size );
        nv::swap(m_buffer[index], back());      // @@ Is this OK when index == size-1?
        (m_buffer+m_size-1)->~T();
        m_size--;
    }

    // Resize the vector preserving existing elements.
    template <typename T> 
    void Array<T>::resize(uint new_size)
    {
        uint old_size = m_size;

        // Destruct old elements (if we're shrinking).
        destroy_range(m_buffer, new_size, old_size);

        setArraySize(new_size);

        // Call default constructors
        construct_range(m_buffer, new_size, old_size);
    }


    // Resize the vector preserving existing elements and initializing the
    // new ones with the given value.
    template <typename T> 
    void Array<T>::resize(uint new_size, const T & elem)
    {
        nvDebugCheck(&elem < m_buffer || &elem > m_buffer+m_size);

        uint old_size = m_size;

        // Destruct old elements (if we're shrinking).
        destroy_range(m_buffer, new_size, old_size);

        setArraySize(new_size);

        // Call copy constructors
        construct_range(m_buffer, new_size, old_size, elem);
    }

    // Fill array with the given value.
    template <typename T>
    void Array<T>::fill(const T & elem)
    {
        fill(m_buffer, m_size, elem);
    }

    // Clear the buffer.
    template <typename T> 
    NV_FORCEINLINE void Array<T>::clear()
    {
        nvDebugCheck(isValidPtr(m_buffer));

        // Destruct old elements
        destroy_range(m_buffer, 0, m_size);

        m_size = 0;
    }

    // Shrink the allocated vector.
    template <typename T> 
    NV_FORCEINLINE void Array<T>::shrink()
    {
        if (m_size < m_capacity) {
            setArrayCapacity(m_size);
        }
    }

    // Preallocate space.
    template <typename T> 
    NV_FORCEINLINE void Array<T>::reserve(uint desired_size)
    {
        if (desired_size > m_capacity) {
            setArrayCapacity(desired_size);
        }
    }

    // Copy elements to this array. Resizes it if needed.
    template <typename T>
    NV_FORCEINLINE void Array<T>::copy(const T * data, uint count)
    {
#if 1   // More simple, but maybe not be as efficient?
        destroy_range(m_buffer, 0, m_size);

        setArraySize(count);

        construct_range(m_buffer, count, 0, data);
#else
        const uint old_size = m_size;

        destroy_range(m_buffer, count, old_size);

        setArraySize(count);

        copy_range(m_buffer, data, old_size);

        construct_range(m_buffer, count, old_size, data);
#endif
    }

    // Assignment operator.
    template <typename T>
    NV_FORCEINLINE Array<T> & Array<T>::operator=( const Array<T> & a )
    {
        copy(a.m_buffer, a.m_size);
        return *this;
    }

    // Release ownership of allocated memory and returns pointer to it.
    template <typename T>
    T * Array<T>::release() {
        T * tmp = m_buffer;
        m_buffer = NULL;
        m_capacity = 0;
        m_size = 0;
        return tmp;
    }



    // Change array size.
    template <typename T> 
    inline void Array<T>::setArraySize(uint new_size) {
        m_size = new_size;

        if (new_size > m_capacity) {
            uint new_buffer_size;
            if (m_capacity == 0) {
                // first allocation is exact
                new_buffer_size = new_size;
            }
            else {
                // following allocations grow array by 25%
                new_buffer_size = new_size + (new_size >> 2);
            }

            setArrayCapacity( new_buffer_size );
        }
    }

    // Change array capacity.
    template <typename T> 
    inline void Array<T>::setArrayCapacity(uint new_capacity) {
        nvDebugCheck(new_capacity >= m_size);

        if (new_capacity == 0) {
            // free the buffer.
            if (m_buffer != NULL) {
                free<T>(m_buffer);
                m_buffer = NULL;
            }
        }
        else {
            // realloc the buffer
            m_buffer = realloc<T>(m_buffer, new_capacity);
        }

        m_capacity = new_capacity;
    }

    // Array serialization.
    template <typename Typ> 
    inline Stream & operator<< ( Stream & s, Array<Typ> & p )
    {
        if (s.isLoading()) {
            uint size;
            s << size;
            p.resize( size );
        }
        else {
            s << p.m_size;
        }

        for (uint i = 0; i < p.m_size; i++) {
            s << p.m_buffer[i];
        }

        return s;
    }

    // Swap the members of the two given vectors.
    template <typename Typ>
    inline void swap(Array<Typ> & a, Array<Typ> & b)
    {
        nv::swap(a.m_buffer, b.m_buffer);
        nv::swap(a.m_capacity, b.m_capacity);
        nv::swap(a.m_size, b.m_size);
    }


} // nv namespace

// IC: These functions are for compatibility with the Foreach macro in The Witness.
template <typename T> inline int item_count(const nv::Array<T> & array) { return array.count(); }
template <typename T> inline const T & item_at(const nv::Array<T> & array, int i) { return array.at(i); }
template <typename T> inline T & item_at(nv::Array<T> & array, int i) { return array.at(i); }
template <typename T> inline int item_advance(const nv::Array<T> & array, int i) { return ++i; }
template <typename T> inline int item_remove(nv::Array<T> & array, int i) { array.replaceWithLast(i); return i - 1; }

template <typename T> inline int item_count(const nv::Array<T> * array) { return array->count(); }
template <typename T> inline const T & item_at(const nv::Array<T> * array, int i) { return array->at(i); }
template <typename T> inline T & item_at(nv::Array<T> * array, int i) { return array->at(i); }
template <typename T> inline int item_advance(const nv::Array<T> * array, int i) { return ++i; }
template <typename T> inline int item_remove(nv::Array<T> * array, int i) { array->replaceWithLast(i); return i - 1; }


#endif // NV_CORE_ARRAY_INL
