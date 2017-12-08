// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_CORE_ARRAY_H
#define NV_CORE_ARRAY_H

/*
This array class requires the elements to be relocable; it uses memmove and realloc. Ideally I should be 
using swap, but I honestly don't care. The only thing that you should be aware of is that internal pointers
are not supported.

Note also that push_back and resize does not support inserting arguments elements that are in the same 
container. This is forbidden to prevent an extra copy.
*/


#include "Memory.h"
#include "Debug.h"
#include "ForEach.h" // PseudoIndex


namespace nv 
{
    class Stream;

    /**
    * Replacement for std::vector that is easier to debug and provides
    * some nice foreach enumerators. 
    */
    template<typename T>
    class NVCORE_CLASS Array {
    public:
        typedef uint size_type;

        // Default constructor.
        NV_FORCEINLINE Array() : m_buffer(NULL), m_capacity(0), m_size(0) {}

        // Copy constructor.
        NV_FORCEINLINE Array(const Array & a) : m_buffer(NULL), m_capacity(0), m_size(0) {
            copy(a.m_buffer, a.m_size);
        }

        // Constructor that initializes the vector with the given elements.
        NV_FORCEINLINE Array(const T * ptr, uint num) : m_buffer(NULL), m_capacity(0), m_size(0) {
            copy(ptr, num);
        }

        // Allocate array.
        NV_FORCEINLINE explicit Array(uint capacity) : m_buffer(NULL), m_capacity(0), m_size(0) {
            setArrayCapacity(capacity);
        }

        // Destructor.
        NV_FORCEINLINE ~Array() {
            clear();
            free<T>(m_buffer);
        }


        /// Const element access.
        NV_FORCEINLINE const T & operator[]( uint index ) const
        {
            nvDebugCheck(index < m_size);
            return m_buffer[index];
        }
        NV_FORCEINLINE const T & at( uint index ) const
        {
            nvDebugCheck(index < m_size);
            return m_buffer[index];
        }

        /// Element access.
        NV_FORCEINLINE T & operator[] ( uint index )
        {
            nvDebugCheck(index < m_size);
            return m_buffer[index];
        }
        NV_FORCEINLINE T & at( uint index )
        {
            nvDebugCheck(index < m_size);
            return m_buffer[index];
        }

        /// Get vector size.
        NV_FORCEINLINE uint size() const { return m_size; }

        /// Get vector size.
        NV_FORCEINLINE uint count() const { return m_size; }

        /// Get vector capacity.
        NV_FORCEINLINE uint capacity() const { return m_capacity; }

        /// Get const vector pointer.
        NV_FORCEINLINE const T * buffer() const { return m_buffer; }

        /// Get vector pointer.
        NV_FORCEINLINE T * buffer() { return m_buffer; }

        /// Provide begin/end pointers for C++11 range-based for loops.
        NV_FORCEINLINE T * begin() { return m_buffer; }
        NV_FORCEINLINE T * end() { return m_buffer + m_size; }
        NV_FORCEINLINE const T * begin() const { return m_buffer; }
        NV_FORCEINLINE const T * end() const { return m_buffer + m_size; }

        /// Is vector empty.
        NV_FORCEINLINE bool isEmpty() const { return m_size == 0; }

        /// Is a null vector.
        NV_FORCEINLINE bool isNull() const { return m_buffer == NULL; }


        T & append();
        void push_back( const T & val );
        void pushBack( const T & val );
        Array<T> & append( const T & val );
        Array<T> & operator<< ( T & t );
        void pop_back();
        void popBack(uint count = 1);
        void popFront(uint count = 1);
        const T & back() const;
        T & back();
        const T & front() const;
        T & front();
        bool contains(const T & e) const;
        bool find(const T & element, uint * indexPtr) const;
        bool find(const T & element, uint begin, uint end, uint * indexPtr) const;
        void removeAt(uint index);
        bool remove(const T & element);
        void insertAt(uint index, const T & val = T());
        void append(const Array<T> & other);
        void append(const T other[], uint count);
        void replaceWithLast(uint index);
        void resize(uint new_size);
        void resize(uint new_size, const T & elem);
        void fill(const T & elem);
        void clear();
        void shrink();
        void reserve(uint desired_size);
        void copy(const T * data, uint count);
        Array<T> & operator=( const Array<T> & a );
        T * release();


        // Array enumerator.
        typedef uint PseudoIndex;

        NV_FORCEINLINE PseudoIndex start() const { return 0; }
        NV_FORCEINLINE bool isDone(const PseudoIndex & i) const { nvDebugCheck(i <= this->m_size); return i == this->m_size; }
        NV_FORCEINLINE void advance(PseudoIndex & i) const { nvDebugCheck(i <= this->m_size); i++; }

#if NV_NEED_PSEUDOINDEX_WRAPPER
        NV_FORCEINLINE T & operator[]( const PseudoIndexWrapper & i ) {
            return m_buffer[i(this)];
        }
        NV_FORCEINLINE const T & operator[]( const PseudoIndexWrapper & i ) const {
            return m_buffer[i(this)];
        }
#endif

        // Friends.
        template <typename Typ> 
        friend Stream & operator<< ( Stream & s, Array<Typ> & p );

        template <typename Typ>
        friend void swap(Array<Typ> & a, Array<Typ> & b);


    protected:

        void setArraySize(uint new_size);
        void setArrayCapacity(uint new_capacity);

        T * m_buffer;
        uint m_capacity;
        uint m_size;

    };


} // nv namespace

#endif // NV_CORE_ARRAY_H
