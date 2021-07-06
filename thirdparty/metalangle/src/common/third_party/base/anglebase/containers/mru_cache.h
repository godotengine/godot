// Copyright (c) 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// This file contains a template for a Most Recently Used cache that allows
// constant-time access to items using a key, but easy identification of the
// least-recently-used items for removal.  Each key can only be associated with
// one payload item at a time.
//
// The key object will be stored twice, so it should support efficient copying.
//
// NOTE: While all operations are O(1), this code is written for
// legibility rather than optimality. If future profiling identifies this as
// a bottleneck, there is room for smaller values of 1 in the O(1). :]

#ifndef ANGLEBASE_CONTAINERS_MRU_CACHE_H_
#define ANGLEBASE_CONTAINERS_MRU_CACHE_H_

#include <stddef.h>

#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <unordered_map>
#include <utility>

#include "anglebase/logging.h"
#include "anglebase/macros.h"

namespace angle
{

namespace base
{

// MRUCacheBase ----------------------------------------------------------------

// This template is used to standardize map type containers that can be used
// by MRUCacheBase. This level of indirection is necessary because of the way
// that template template params and default template params interact.
template <class KeyType, class ValueType, class CompareType>
struct MRUCacheStandardMap
{
    typedef std::map<KeyType, ValueType, CompareType> Type;
};

// Base class for the MRU cache specializations defined below.
template <class KeyType,
          class PayloadType,
          class HashOrCompareType,
          template <typename, typename, typename> class MapType = MRUCacheStandardMap>
class MRUCacheBase
{
  public:
    // The payload of the list. This maintains a copy of the key so we can
    // efficiently delete things given an element of the list.
    typedef std::pair<KeyType, PayloadType> value_type;

  private:
    typedef std::list<value_type> PayloadList;
    typedef
        typename MapType<KeyType, typename PayloadList::iterator, HashOrCompareType>::Type KeyIndex;

  public:
    typedef typename PayloadList::size_type size_type;

    typedef typename PayloadList::iterator iterator;
    typedef typename PayloadList::const_iterator const_iterator;
    typedef typename PayloadList::reverse_iterator reverse_iterator;
    typedef typename PayloadList::const_reverse_iterator const_reverse_iterator;

    enum
    {
        NO_AUTO_EVICT = 0
    };

    // The max_size is the size at which the cache will prune its members to when
    // a new item is inserted. If the caller wants to manager this itself (for
    // example, maybe it has special work to do when something is evicted), it
    // can pass NO_AUTO_EVICT to not restrict the cache size.
    explicit MRUCacheBase(size_type max_size) : max_size_(max_size) {}

    virtual ~MRUCacheBase() {}

    size_type max_size() const { return max_size_; }

    // Inserts a payload item with the given key. If an existing item has
    // the same key, it is removed prior to insertion. An iterator indicating the
    // inserted item will be returned (this will always be the front of the list).
    //
    // The payload will be forwarded.
    template <typename Payload>
    iterator Put(const KeyType &key, Payload &&payload)
    {
        // Remove any existing payload with that key.
        typename KeyIndex::iterator index_iter = index_.find(key);
        if (index_iter != index_.end())
        {
            // Erase the reference to it. The index reference will be replaced in the
            // code below.
            Erase(index_iter->second);
        }
        else if (max_size_ != NO_AUTO_EVICT)
        {
            // New item is being inserted which might make it larger than the maximum
            // size: kick the oldest thing out if necessary.
            ShrinkToSize(max_size_ - 1);
        }

        ordering_.emplace_front(key, std::forward<Payload>(payload));
        index_.emplace(key, ordering_.begin());
        return ordering_.begin();
    }

    // Retrieves the contents of the given key, or end() if not found. This method
    // has the side effect of moving the requested item to the front of the
    // recency list.
    iterator Get(const KeyType &key)
    {
        typename KeyIndex::iterator index_iter = index_.find(key);
        if (index_iter == index_.end())
            return end();
        typename PayloadList::iterator iter = index_iter->second;

        // Move the touched item to the front of the recency ordering.
        ordering_.splice(ordering_.begin(), ordering_, iter);
        return ordering_.begin();
    }

    // Retrieves the payload associated with a given key and returns it via
    // result without affecting the ordering (unlike Get).
    iterator Peek(const KeyType &key)
    {
        typename KeyIndex::const_iterator index_iter = index_.find(key);
        if (index_iter == index_.end())
            return end();
        return index_iter->second;
    }

    const_iterator Peek(const KeyType &key) const
    {
        typename KeyIndex::const_iterator index_iter = index_.find(key);
        if (index_iter == index_.end())
            return end();
        return index_iter->second;
    }

    // Exchanges the contents of |this| by the contents of the |other|.
    void Swap(MRUCacheBase &other)
    {
        ordering_.swap(other.ordering_);
        index_.swap(other.index_);
        std::swap(max_size_, other.max_size_);
    }

    // Erases the item referenced by the given iterator. An iterator to the item
    // following it will be returned. The iterator must be valid.
    iterator Erase(iterator pos)
    {
        index_.erase(pos->first);
        return ordering_.erase(pos);
    }

    // MRUCache entries are often processed in reverse order, so we add this
    // convenience function (not typically defined by STL containers).
    reverse_iterator Erase(reverse_iterator pos)
    {
        // We have to actually give it the incremented iterator to delete, since
        // the forward iterator that base() returns is actually one past the item
        // being iterated over.
        return reverse_iterator(Erase((++pos).base()));
    }

    // Shrinks the cache so it only holds |new_size| items. If |new_size| is
    // bigger or equal to the current number of items, this will do nothing.
    void ShrinkToSize(size_type new_size)
    {
        for (size_type i = size(); i > new_size; i--)
            Erase(rbegin());
    }

    // Deletes everything from the cache.
    void Clear()
    {
        index_.clear();
        ordering_.clear();
    }

    // Returns the number of elements in the cache.
    size_type size() const
    {
        // We don't use ordering_.size() for the return value because
        // (as a linked list) it can be O(n).
        DCHECK(index_.size() == ordering_.size());
        return index_.size();
    }

    // Allows iteration over the list. Forward iteration starts with the most
    // recent item and works backwards.
    //
    // Note that since these iterators are actually iterators over a list, you
    // can keep them as you insert or delete things (as long as you don't delete
    // the one you are pointing to) and they will still be valid.
    iterator begin() { return ordering_.begin(); }
    const_iterator begin() const { return ordering_.begin(); }
    iterator end() { return ordering_.end(); }
    const_iterator end() const { return ordering_.end(); }

    reverse_iterator rbegin() { return ordering_.rbegin(); }
    const_reverse_iterator rbegin() const { return ordering_.rbegin(); }
    reverse_iterator rend() { return ordering_.rend(); }
    const_reverse_iterator rend() const { return ordering_.rend(); }

    bool empty() const { return ordering_.empty(); }

  private:
    PayloadList ordering_;
    KeyIndex index_;

    size_type max_size_;

    DISALLOW_COPY_AND_ASSIGN(MRUCacheBase);
};

// MRUCache --------------------------------------------------------------------

// A container that does not do anything to free its data. Use this when storing
// value types (as opposed to pointers) in the list.
template <class KeyType, class PayloadType, class CompareType = std::less<KeyType>>
class MRUCache : public MRUCacheBase<KeyType, PayloadType, CompareType>
{
  private:
    using ParentType = MRUCacheBase<KeyType, PayloadType, CompareType>;

  public:
    // See MRUCacheBase, noting the possibility of using NO_AUTO_EVICT.
    explicit MRUCache(typename ParentType::size_type max_size) : ParentType(max_size) {}
    virtual ~MRUCache() {}

  private:
    DISALLOW_COPY_AND_ASSIGN(MRUCache);
};

// HashingMRUCache ------------------------------------------------------------

template <class KeyType, class ValueType, class HashType>
struct MRUCacheHashMap
{
    typedef std::unordered_map<KeyType, ValueType, HashType> Type;
};

// This class is similar to MRUCache, except that it uses std::unordered_map as
// the map type instead of std::map. Note that your KeyType must be hashable to
// use this cache or you need to provide a hashing class.
template <class KeyType, class PayloadType, class HashType = std::hash<KeyType>>
class HashingMRUCache : public MRUCacheBase<KeyType, PayloadType, HashType, MRUCacheHashMap>
{
  private:
    using ParentType = MRUCacheBase<KeyType, PayloadType, HashType, MRUCacheHashMap>;

  public:
    // See MRUCacheBase, noting the possibility of using NO_AUTO_EVICT.
    explicit HashingMRUCache(typename ParentType::size_type max_size) : ParentType(max_size) {}
    virtual ~HashingMRUCache() {}

  private:
    DISALLOW_COPY_AND_ASSIGN(HashingMRUCache);
};

}  // namespace base

}  // namespace angle

#endif  // ANGLEBASE_CONTAINERS_MRU_CACHE_H_
