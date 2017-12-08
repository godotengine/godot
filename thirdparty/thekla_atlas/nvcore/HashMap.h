// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_CORE_HASHMAP_H
#define NV_CORE_HASHMAP_H

/*
HashMap based on Thatcher Ulrich <tu@tulrich.com> container, donated to the Public Domain.

I'd like to do something to reduce the amount of code generated with this template. The type of 
U is largely irrelevant to the generated code, except for calls to constructors and destructors,
but the combination of all T and U pairs, generate a large amounts of code.

HashMap is not used in NVTT, so it could be removed from the repository.
*/


#include "Memory.h"
#include "Debug.h"
#include "ForEach.h"
#include "Hash.h"

namespace nv 
{
    class Stream;

    /** Thatcher Ulrich's hash table.
    *
    * Hash table, linear probing, internal chaining.  One
    * interesting/nice thing about this implementation is that the table
    * itself is a flat chunk of memory containing no pointers, only
    * relative indices.  If the key and value types of the hash contain
    * no pointers, then the hash can be serialized using raw IO.  Could
    * come in handy.
    *
    * Never shrinks, unless you explicitly clear() it.  Expands on
    * demand, though.  For best results, if you know roughly how big your
    * table will be, default it to that size when you create it.
    */
    template<typename T, typename U, typename H = Hash<T>, typename E = Equal<T> >
    class NVCORE_CLASS HashMap
    {
        NV_FORBID_COPY(HashMap);
    public:

        /// Default ctor.
        HashMap() : entry_count(0), size_mask(-1), table(NULL) { }

        /// Ctor with size hint.
        explicit HashMap(int size_hint) : entry_count(0), size_mask(-1), table(NULL) { setCapacity(size_hint); }

        /// Dtor.
        ~HashMap() { clear(); }


        void set(const T& key, const U& value);
        void add(const T& key, const U& value);
        bool remove(const T& key);
        void clear();
        bool isEmpty() const;
        bool get(const T& key, U* value = NULL, T* other_key = NULL) const;
        bool contains(const T & key) const;
        int	size() const;
        int	count() const;
        int	capacity() const;
        void checkExpand();
        void resize(int n);

        void setCapacity(int new_size);

        // Behaves much like std::pair.
        struct Entry
        {
            int	next_in_chain;	// internal chaining for collisions
            uint hash_value;	// avoids recomputing.  Worthwhile?
            T key;
            U value;

            Entry() : next_in_chain(-2) {}
            Entry(const Entry& e) : next_in_chain(e.next_in_chain), hash_value(e.hash_value), key(e.key), value(e.value) {}
            Entry(const T& k, const U& v, int next, int hash) : next_in_chain(next), hash_value(hash), key(k), value(v) {}
            
            bool isEmpty() const { return next_in_chain == -2; }
            bool isEndOfChain() const { return next_in_chain == -1; }
            bool isTombstone() const { return hash_value == TOMBSTONE_HASH; }

            void clear() {
                key.~T();	// placement delete
                value.~U();	// placement delete
                next_in_chain = -2;
                hash_value = ~TOMBSTONE_HASH;
            }

            void makeTombstone() {
                key.~T();
                value.~U();
                hash_value = TOMBSTONE_HASH;
            }
        };


        // HashMap enumerator.
        typedef int PseudoIndex;
        PseudoIndex start() const { PseudoIndex i = 0; findNext(i); return i; }
        bool isDone(const PseudoIndex & i) const { nvDebugCheck(i <= size_mask+1); return i == size_mask+1; };
        void advance(PseudoIndex & i) const { nvDebugCheck(i <= size_mask+1); i++; findNext(i); }

#if NV_NEED_PSEUDOINDEX_WRAPPER
        Entry & operator[]( const PseudoIndexWrapper & i ) {
            Entry & e = entry(i(this));
            nvDebugCheck(e.isTombstone() == false);
            return e;
        }
        const Entry & operator[]( const PseudoIndexWrapper & i ) const {
            const Entry & e = entry(i(this));
            nvDebugCheck(e.isTombstone() == false);
            return e;
        }
#else
        Entry & operator[](const PseudoIndex & i) {
            Entry & e = entry(i);
            nvDebugCheck(e.isTombstone() == false);
            return e;
        }
        const Entry & operator[](const PseudoIndex & i) const {
            const Entry & e = entry(i);
            nvDebugCheck(e.isTombstone() == false);
            return e;
        }
#endif


        // By default we serialize the key-value pairs compactl	y.
        template<typename _T, typename _U, typename _H, typename _E>
        friend Stream & operator<< (Stream & s, HashMap<_T, _U, _H, _E> & map);

        // This requires more storage, but saves us from rehashing the elements.
        template<typename _T, typename _U, typename _H, typename _E>
        friend Stream & rawSerialize(Stream & s, HashMap<_T, _U, _H, _E> & map);

        /// Swap the members of this vector and the given vector.
        template<typename _T, typename _U, typename _H, typename _E>
        friend void swap(HashMap<_T, _U, _H, _E> & a, HashMap<_T, _U, _H, _E> & b);
	
    private:
        static const uint TOMBSTONE_HASH = (uint) -1;

        uint compute_hash(const T& key) const;

        // Find the index of the matching entry. If no match, then return -1.
        int	findIndex(const T& key) const;

        // Return the index of the newly cleared element.
        int removeTombstone(int index);

        // Helpers.
        Entry & entry(int index);
        const Entry & entry(int index) const;

        void setRawCapacity(int new_size);

        // Move the enumerator to the next valid element.
        void findNext(PseudoIndex & i) const;


        int	entry_count;
        int	size_mask;
        Entry * table;

    };

} // nv namespace

#endif // NV_CORE_HASHMAP_H
