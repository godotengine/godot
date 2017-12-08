// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_CORE_HASHMAP_INL
#define NV_CORE_HASHMAP_INL

#include "HashMap.h"

#include "Stream.h"
#include "Utils.h" // swap

#include <new> // for placement new


namespace nv 
{

    // Set a new or existing value under the key, to the value.
    template<typename T, typename U, typename H, typename E>
    void HashMap<T, U, H, E>::set(const T& key, const U& value)
    {
        int	index = findIndex(key);
        if (index >= 0)
        {
            entry(index).value = value;
            return;
        }

        // Entry under key doesn't exist.
        add(key, value);
    }


    // Add a new value to the hash table, under the specified key.
    template<typename T, typename U, typename H, typename E>
    void HashMap<T, U, H, E>::add(const T& key, const U& value)
    {
        nvCheck(findIndex(key) == -1);

        checkExpand();
        nvCheck(table != NULL);
        entry_count++;

        const uint hash_value = compute_hash(key);
        const int index = hash_value & size_mask;

        Entry * natural_entry = &(entry(index));

        if (natural_entry->isEmpty())
        {
            // Put the new entry in.
            new (natural_entry) Entry(key, value, -1, hash_value);
        } 
        else if (natural_entry->isTombstone()) {
            // Put the new entry in, without disturbing the rest of the chain.
            int next_in_chain = natural_entry->next_in_chain;
            new (natural_entry) Entry(key, value, next_in_chain, hash_value);
        }
        else
        {
            // Find a blank spot.
            int	blank_index = index;
            for (int search_count = 0; ; search_count++)
            {
                blank_index = (blank_index + 1) & size_mask;
                if (entry(blank_index).isEmpty()) break;	// found it
                if (entry(blank_index).isTombstone()) {
                    blank_index = removeTombstone(blank_index);
                    break;
                }
                nvCheck(search_count < this->size_mask);
            }
            Entry * blank_entry = &entry(blank_index);

            if (int(natural_entry->hash_value & size_mask) == index)
            {
                // Collision.  Link into this chain.

                // Move existing list head.
                new (blank_entry) Entry(*natural_entry);	// placement new, copy ctor

                // Put the new info in the natural entry.
                natural_entry->key = key;
                natural_entry->value = value;
                natural_entry->next_in_chain = blank_index;
                natural_entry->hash_value = hash_value;
            }
            else
            {
                // Existing entry does not naturally
                // belong in this slot.  Existing
                // entry must be moved.

                // Find natural location of collided element (i.e. root of chain)
                int	collided_index = natural_entry->hash_value & size_mask;
                for (int search_count = 0; ; search_count++)
                {
                    Entry * e = &entry(collided_index);
                    if (e->next_in_chain == index)
                    {
                        // Here's where we need to splice.
                        new (blank_entry) Entry(*natural_entry);
                        e->next_in_chain = blank_index;
                        break;
                    }
                    collided_index = e->next_in_chain;
                    nvCheck(collided_index >= 0 && collided_index <= size_mask);
                    nvCheck(search_count <= size_mask);
                }

                // Put the new data in the natural entry.
                natural_entry->key = key;
                natural_entry->value = value;
                natural_entry->hash_value = hash_value;
                natural_entry->next_in_chain = -1;
            }
        }
    }


    // Remove the first value under the specified key.
    template<typename T, typename U, typename H, typename E>
    bool HashMap<T, U, H, E>::remove(const T& key)
    {
        if (table == NULL)
        {
            return false;
        }

        int	index = findIndex(key);
        if (index < 0)
        {
            return false;
        }

        Entry * pos = &entry(index);

        int natural_index = (int) (pos->hash_value & size_mask);

        if (index != natural_index) {
            // We're not the head of our chain, so we can
            // be spliced out of it.

            // Iterate up the chain, and splice out when
            // we get to m_index.
            Entry* e = &entry(natural_index);
            while (e->next_in_chain != index) {
                nvDebugCheck(e->isEndOfChain() == false);
                e = &entry(e->next_in_chain);
            }

            if (e->isTombstone() && pos->isEndOfChain()) {
                // Tombstone has nothing else to point
                // to, so mark it empty.
                e->next_in_chain = -2;
            } else {
                e->next_in_chain = pos->next_in_chain;
            }

            pos->clear();
        }
        else if (pos->isEndOfChain() == false) {
            // We're the head of our chain, and there are
            // additional elements.
            //
            // We need to put a tombstone here.
            //
            // We can't clear the element, because the
            // rest of the elements in the chain must be
            // linked to this position.
            //
            // We can't move any of the succeeding
            // elements in the chain (i.e. to fill this
            // entry), because we don't want to invalidate
            // any other existing iterators.
            pos->makeTombstone();
        } else {
            // We're the head of the chain, but we're the
            // only member of the chain.
            pos->clear();
        }

        entry_count--;

        return true;
    }


    // Remove all entries from the hash table.
    template<typename T, typename U, typename H, typename E>
    void HashMap<T, U, H, E>::clear()
    {
        if (table != NULL)
        {
            // Delete the entries.
            for (int i = 0, n = size_mask; i <= n; i++)
            {
                Entry * e = &entry(i);
                if (e->isEmpty() == false && e->isTombstone() == false)
                {
                    e->clear();
                }
            }
            free(table);
            table = NULL;
            entry_count = 0;
            size_mask = -1;
        }
    }


    // Returns true if the hash is empty.
    template<typename T, typename U, typename H, typename E>
    bool HashMap<T, U, H, E>::isEmpty() const
    {
        return table == NULL || entry_count == 0;
    }


    // Retrieve the value under the given key.
    // - If there's no value under the key, then return false and leave *value alone.
    // - If there is a value, return true, and set *value to the entry's value.
    // - If value == NULL, return true or false according to the presence of the key, but don't touch *value.
    template<typename T, typename U, typename H, typename E>
    bool HashMap<T, U, H, E>::get(const T& key, U* value/*= NULL*/, T* other_key/*= NULL*/) const
    {
        int	index = findIndex(key);
        if (index >= 0)
        {
            if (value != NULL) {
                *value = entry(index).value;	// take care with side-effects!
            }
            if (other_key != NULL) {
                *other_key = entry(index).key;
            }
            return true;
        }
        return false;
    }

    // Determine if the given key is contained in the hash.
    template<typename T, typename U, typename H, typename E>
    bool HashMap<T, U, H, E>::contains(const T & key) const
    {
        return get(key);
    }

    // Number of entries in the hash.
    template<typename T, typename U, typename H, typename E>
    int	HashMap<T, U, H, E>::size() const
    {
        return entry_count;
    }

    // Number of entries in the hash.
    template<typename T, typename U, typename H, typename E>
    int	HashMap<T, U, H, E>::count() const
    {
        return size();
    }

    template<typename T, typename U, typename H, typename E>
    int	HashMap<T, U, H, E>::capacity() const
    {
        return size_mask+1;
    }


    // Resize the hash table to fit one more entry.  Often this doesn't involve any action.
    template<typename T, typename U, typename H, typename E>
    void HashMap<T, U, H, E>::checkExpand()
    {
        if (table == NULL) {
            // Initial creation of table.  Make a minimum-sized table.
            setRawCapacity(16);
        } 
        else if (entry_count * 3 > (size_mask + 1) * 2) {
            // Table is more than 2/3rds full.  Expand.
            setRawCapacity(entry_count * 2);
        }
    }


    // Hint the bucket count to >= n.
    template<typename T, typename U, typename H, typename E>
    void HashMap<T, U, H, E>::resize(int n)
    {
        // Not really sure what this means in relation to
        // STLport's hash_map... they say they "increase the
        // bucket count to at least n" -- but does that mean
        // their real capacity after resize(n) is more like
        // n*2 (since they do linked-list chaining within
        // buckets?).
        setCapacity(n);
    }


    // Size the hash so that it can comfortably contain the given number of elements.  If the hash already contains more
    // elements than new_size, then this may be a no-op.
    template<typename T, typename U, typename H, typename E>
    void HashMap<T, U, H, E>::setCapacity(int new_size)
    {
        int	new_raw_size = (new_size * 3) / 2;
        if (new_raw_size < size()) { return; }

        setRawCapacity(new_raw_size);
    }


    // By default we serialize the key-value pairs compactly.
    template<typename _T, typename _U, typename _H, typename _E>
    Stream & operator<< (Stream & s, HashMap<_T, _U, _H, _E> & map)
    {
        typedef typename HashMap<_T, _U, _H, _E>::Entry HashMapEntry;

        int entry_count = map.entry_count;
        s << entry_count;

        if (s.isLoading()) {
            map.clear();
            if(entry_count == 0) {
                return s;
            }
            map.entry_count = entry_count;
            map.size_mask = nextPowerOfTwo(U32(entry_count)) - 1;
            map.table = malloc<HashMapEntry>(map.size_mask + 1);

            for (int i = 0; i <= map.size_mask; i++) {
                map.table[i].next_in_chain = -2;	// mark empty
            }

            _T key;
            _U value;
            for (int i = 0; i < entry_count; i++) {
                s << key << value;
                map.add(key, value);
            }
        }
        else {
            int i = 0;
            map.findNext(i);
            while (i != map.size_mask+1) {
                HashMapEntry & e = map.entry(i);
                
                s << e.key << e.value;
                
                i++;
                map.findNext(i);
            }
            //for(HashMap<_T, _U, _H, _E>::PseudoIndex i((map).start()); !(map).isDone(i); (map).advance(i)) {
            //foreach(i, map) {
            //    s << map[i].key << map[i].value;
            //}
        }

        return s;
    }

    // This requires more storage, but saves us from rehashing the elements.
    template<typename _T, typename _U, typename _H, typename _E>
    Stream & rawSerialize(Stream & s, HashMap<_T, _U, _H, _E> & map)
    {
        typedef typename HashMap<_T, _U, _H, _E>::Entry HashMapEntry;

        if (s.isLoading()) {
            map.clear();
        }

        s << map.size_mask;

        if (map.size_mask != -1) {
            s << map.entry_count;

            if (s.isLoading()) {  
                map.table = new HashMapEntry[map.size_mask+1];
            }

            for (int i = 0; i <= map.size_mask; i++) {
                HashMapEntry & e = map.table[i];
                s << e.next_in_chain << e.hash_value;
                s << e.key;
                s << e.value;
            }
        }

        return s;
    }

    // Swap the members of this vector and the given vector.
    template<typename _T, typename _U, typename _H, typename _E>
    void swap(HashMap<_T, _U, _H, _E> & a, HashMap<_T, _U, _H, _E> & b)
    {
        swap(a.entry_count, b.entry_count);
        swap(a.size_mask, b.size_mask);
        swap(a.table, b.table);
    }


    template<typename T, typename U, typename H, typename E>
    uint HashMap<T, U, H, E>::compute_hash(const T& key) const
    {
        H hash;
        uint hash_value = hash(key);
        if (hash_value == TOMBSTONE_HASH) {
            hash_value ^= 0x8000;
        }
        return hash_value;
    }

    // Find the index of the matching entry. If no match, then return -1.
    template<typename T, typename U, typename H, typename E>
    int	HashMap<T, U, H, E>::findIndex(const T& key) const
    {
        if (table == NULL) return -1;

        E equal;

        uint hash_value = compute_hash(key);
        int	index = hash_value & size_mask;

        const Entry * e = &entry(index);
        if (e->isEmpty()) return -1;
        if (e->isTombstone() == false && int(e->hash_value & size_mask) != index) {
            // occupied by a collider
            return -1;
        }

        for (;;)
        {
            nvCheck(e->isTombstone() || (e->hash_value & size_mask) == (hash_value & size_mask));

            if (e->hash_value == hash_value && equal(e->key, key))
            {
                // Found it.
                return index;
            }
            nvDebugCheck(e->isTombstone() || !equal(e->key, key));   // keys are equal, but hash differs!

            // Keep looking through the chain.
            index = e->next_in_chain;
            if (index == -1) break;	// end of chain

            nvCheck(index >= 0 && index <= size_mask);
            e = &entry(index);

            nvCheck(e->isEmpty() == false || e->isTombstone());
        }
        return -1;
    }

    // Return the index of the newly cleared element.
    template<typename T, typename U, typename H, typename E>
    int HashMap<T, U, H, E>::removeTombstone(int index) {
        Entry* e = &entry(index);
        nvCheck(e->isTombstone());
        nvCheck(!e->isEndOfChain());

        // Move the next element of the chain into the
        // tombstone slot, and return the vacated element.
        int new_blank_index = e->next_in_chain;
        Entry* new_blank = &entry(new_blank_index);
        new (e) Entry(*new_blank);
        new_blank->clear();
        return new_blank_index;
    }

    // Helpers.
    template<typename T, typename U, typename H, typename E>
    typename HashMap<T, U, H, E>::Entry & HashMap<T, U, H, E>::entry(int index)
    {
        nvDebugCheck(table != NULL);
        nvDebugCheck(index >= 0 && index <= size_mask);
        return table[index];
    }
    template<typename T, typename U, typename H, typename E>
    const typename HashMap<T, U, H, E>::Entry & HashMap<T, U, H, E>::entry(int index) const
    {
        nvDebugCheck(table != NULL);
        nvDebugCheck(index >= 0 && index <= size_mask);
        return table[index];
    }


    // Resize the hash table to the given size (Rehash the contents of the current table).  The arg is the number of
    // hash table entries, not the number of elements we should actually contain (which will be less than this).
    template<typename T, typename U, typename H, typename E>
    void HashMap<T, U, H, E>::setRawCapacity(int new_size)
    {
        if (new_size <= 0) {
            // Special case.
            clear();
            return;
        }

        // Force new_size to be a power of two.
        new_size = nextPowerOfTwo(U32(new_size));

        HashMap<T, U, H, E> new_hash;
        new_hash.table = malloc<Entry>(new_size);
        nvDebugCheck(new_hash.table != NULL);

        new_hash.entry_count = 0;
        new_hash.size_mask = new_size - 1;
        for (int i = 0; i < new_size; i++)
        {
            new_hash.entry(i).next_in_chain = -2;	// mark empty
        }

        // Copy stuff to new_hash
        if (table != NULL)
        {
            for (int i = 0, n = size_mask; i <= n; i++)
            {
                Entry * e = &entry(i);
                if (e->isEmpty() == false && e->isTombstone() == false)
                {
                    // Insert old entry into new hash.
                    new_hash.add(e->key, e->value);
                    e->clear();	// placement delete of old element
                }
            }

            // Delete our old data buffer.
            free(table);
        }

        // Steal new_hash's data.
        entry_count = new_hash.entry_count;
        size_mask = new_hash.size_mask;
        table = new_hash.table;
        new_hash.entry_count = 0;
        new_hash.size_mask = -1;
        new_hash.table = NULL;
    }

    // Move the enumerator to the next valid element.
    template<typename T, typename U, typename H, typename E>
    void HashMap<T, U, H, E>::findNext(PseudoIndex & i) const {
        while (i <= size_mask) {
            const Entry & e = entry(i);
            if (e.isEmpty() == false && e.isTombstone() == false) {
                break;
            }
            i++;
        }
    }

} // nv namespace

#endif // NV_CORE_HASHMAP_INL
