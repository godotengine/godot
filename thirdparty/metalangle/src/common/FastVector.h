//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FastVector.h:
//   A vector class with a initial fixed size and variable growth.
//   Based on FixedVector.
//

#ifndef COMMON_FASTVECTOR_H_
#define COMMON_FASTVECTOR_H_

#include "common/debug.h"

#include <algorithm>
#include <array>
#include <initializer_list>

namespace angle
{
template <class T, size_t N, class Storage = std::array<T, N>>
class FastVector final
{
  public:
    using value_type      = typename Storage::value_type;
    using size_type       = typename Storage::size_type;
    using reference       = typename Storage::reference;
    using const_reference = typename Storage::const_reference;
    using pointer         = typename Storage::pointer;
    using const_pointer   = typename Storage::const_pointer;
    using iterator        = T *;
    using const_iterator  = const T *;

    FastVector();
    FastVector(size_type count, const value_type &value);
    FastVector(size_type count);

    FastVector(const FastVector<T, N, Storage> &other);
    FastVector(FastVector<T, N, Storage> &&other);
    FastVector(std::initializer_list<value_type> init);
    FastVector(const T *src, size_type count);

    FastVector<T, N, Storage> &operator=(const FastVector<T, N, Storage> &other);
    FastVector<T, N, Storage> &operator=(FastVector<T, N, Storage> &&other);
    FastVector<T, N, Storage> &operator=(std::initializer_list<value_type> init);

    ~FastVector();

    reference at(size_type pos);
    const_reference at(size_type pos) const;

    reference operator[](size_type pos);
    const_reference operator[](size_type pos) const;

    pointer data();
    const_pointer data() const;

    iterator begin();
    const_iterator begin() const;

    iterator end();
    const_iterator end() const;

    bool empty() const;
    size_type size() const;

    void clear();

    void push_back(const value_type &value);
    void push_back(value_type &&value);

    void pop_back();

    reference front();
    const_reference front() const;

    reference back();
    const_reference back() const;

    void swap(FastVector<T, N, Storage> &other);

    void resize(size_type count);
    void resize(size_type count, const value_type &value);

    // Specialty function that removes a known element and might shuffle the list.
    void remove_and_permute(const value_type &element);

  private:
    void assign_from_initializer_list(std::initializer_list<value_type> init);
    void ensure_capacity(size_t capacity);
    bool uses_fixed_storage() const;

    Storage mFixedStorage;
    pointer mData           = mFixedStorage.data();
    size_type mSize         = 0;
    size_type mReservedSize = N;
};

template <class T, size_t N, class StorageN, size_t M, class StorageM>
bool operator==(const FastVector<T, N, StorageN> &a, const FastVector<T, M, StorageM> &b)
{
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

template <class T, size_t N, class StorageN, size_t M, class StorageM>
bool operator!=(const FastVector<T, N, StorageN> &a, const FastVector<T, M, StorageM> &b)
{
    return !(a == b);
}

template <class T, size_t N, class Storage>
ANGLE_INLINE bool FastVector<T, N, Storage>::uses_fixed_storage() const
{
    return mData == mFixedStorage.data();
}

template <class T, size_t N, class Storage>
FastVector<T, N, Storage>::FastVector()
{}

template <class T, size_t N, class Storage>
FastVector<T, N, Storage>::FastVector(size_type count, const value_type &value)
{
    ensure_capacity(count);
    mSize = count;
    std::fill(begin(), end(), value);
}

template <class T, size_t N, class Storage>
FastVector<T, N, Storage>::FastVector(size_type count)
{
    ensure_capacity(count);
    mSize = count;
}

template <class T, size_t N, class Storage>
FastVector<T, N, Storage>::FastVector(const FastVector<T, N, Storage> &other)
{
    ensure_capacity(other.mSize);
    mSize = other.mSize;
    std::copy(other.begin(), other.end(), begin());
}

template <class T, size_t N, class Storage>
FastVector<T, N, Storage>::FastVector(FastVector<T, N, Storage> &&other) : FastVector()
{
    swap(other);
}

template <class T, size_t N, class Storage>
FastVector<T, N, Storage>::FastVector(std::initializer_list<value_type> init)
{
    assign_from_initializer_list(init);
}

template <class T, size_t N, class Storage>
FastVector<T, N, Storage>::FastVector(const T *src, size_type count)
{
    ensure_capacity(count);
    mSize = count;
    std::copy(src, src + count, begin());
}

template <class T, size_t N, class Storage>
FastVector<T, N, Storage> &FastVector<T, N, Storage>::operator=(
    const FastVector<T, N, Storage> &other)
{
    ensure_capacity(other.mSize);
    mSize = other.mSize;
    std::copy(other.begin(), other.end(), begin());
    return *this;
}

template <class T, size_t N, class Storage>
FastVector<T, N, Storage> &FastVector<T, N, Storage>::operator=(FastVector<T, N, Storage> &&other)
{
    swap(other);
    return *this;
}

template <class T, size_t N, class Storage>
FastVector<T, N, Storage> &FastVector<T, N, Storage>::operator=(
    std::initializer_list<value_type> init)
{
    assign_from_initializer_list(init);
    return *this;
}

template <class T, size_t N, class Storage>
FastVector<T, N, Storage>::~FastVector()
{
    clear();
    if (!uses_fixed_storage())
    {
        delete[] mData;
    }
}

template <class T, size_t N, class Storage>
typename FastVector<T, N, Storage>::reference FastVector<T, N, Storage>::at(size_type pos)
{
    ASSERT(pos < mSize);
    return mData[pos];
}

template <class T, size_t N, class Storage>
typename FastVector<T, N, Storage>::const_reference FastVector<T, N, Storage>::at(
    size_type pos) const
{
    ASSERT(pos < mSize);
    return mData[pos];
}

template <class T, size_t N, class Storage>
ANGLE_INLINE typename FastVector<T, N, Storage>::reference FastVector<T, N, Storage>::operator[](
    size_type pos)
{
    ASSERT(pos < mSize);
    return mData[pos];
}

template <class T, size_t N, class Storage>
ANGLE_INLINE
    typename FastVector<T, N, Storage>::const_reference FastVector<T, N, Storage>::operator[](
        size_type pos) const
{
    ASSERT(pos < mSize);
    return mData[pos];
}

template <class T, size_t N, class Storage>
ANGLE_INLINE typename FastVector<T, N, Storage>::const_pointer
angle::FastVector<T, N, Storage>::data() const
{
    ASSERT(!empty());
    return mData;
}

template <class T, size_t N, class Storage>
ANGLE_INLINE typename FastVector<T, N, Storage>::pointer angle::FastVector<T, N, Storage>::data()
{
    ASSERT(!empty());
    return mData;
}

template <class T, size_t N, class Storage>
ANGLE_INLINE typename FastVector<T, N, Storage>::iterator FastVector<T, N, Storage>::begin()
{
    return mData;
}

template <class T, size_t N, class Storage>
ANGLE_INLINE typename FastVector<T, N, Storage>::const_iterator FastVector<T, N, Storage>::begin()
    const
{
    return mData;
}

template <class T, size_t N, class Storage>
ANGLE_INLINE typename FastVector<T, N, Storage>::iterator FastVector<T, N, Storage>::end()
{
    return mData + mSize;
}

template <class T, size_t N, class Storage>
ANGLE_INLINE typename FastVector<T, N, Storage>::const_iterator FastVector<T, N, Storage>::end()
    const
{
    return mData + mSize;
}

template <class T, size_t N, class Storage>
ANGLE_INLINE bool FastVector<T, N, Storage>::empty() const
{
    return mSize == 0;
}

template <class T, size_t N, class Storage>
ANGLE_INLINE typename FastVector<T, N, Storage>::size_type FastVector<T, N, Storage>::size() const
{
    return mSize;
}

template <class T, size_t N, class Storage>
void FastVector<T, N, Storage>::clear()
{
    resize(0);
}

template <class T, size_t N, class Storage>
ANGLE_INLINE void FastVector<T, N, Storage>::push_back(const value_type &value)
{
    if (mSize == mReservedSize)
        ensure_capacity(mSize + 1);
    mData[mSize++] = value;
}

template <class T, size_t N, class Storage>
ANGLE_INLINE void FastVector<T, N, Storage>::push_back(value_type &&value)
{
    if (mSize == mReservedSize)
        ensure_capacity(mSize + 1);
    mData[mSize++] = std::move(value);
}

template <class T, size_t N, class Storage>
ANGLE_INLINE void FastVector<T, N, Storage>::pop_back()
{
    ASSERT(mSize > 0);
    mSize--;
}

template <class T, size_t N, class Storage>
ANGLE_INLINE typename FastVector<T, N, Storage>::reference FastVector<T, N, Storage>::front()
{
    ASSERT(mSize > 0);
    return mData[0];
}

template <class T, size_t N, class Storage>
ANGLE_INLINE typename FastVector<T, N, Storage>::const_reference FastVector<T, N, Storage>::front()
    const
{
    ASSERT(mSize > 0);
    return mData[0];
}

template <class T, size_t N, class Storage>
ANGLE_INLINE typename FastVector<T, N, Storage>::reference FastVector<T, N, Storage>::back()
{
    ASSERT(mSize > 0);
    return mData[mSize - 1];
}

template <class T, size_t N, class Storage>
ANGLE_INLINE typename FastVector<T, N, Storage>::const_reference FastVector<T, N, Storage>::back()
    const
{
    ASSERT(mSize > 0);
    return mData[mSize - 1];
}

template <class T, size_t N, class Storage>
void FastVector<T, N, Storage>::swap(FastVector<T, N, Storage> &other)
{
    std::swap(mSize, other.mSize);

    pointer tempData = other.mData;
    if (uses_fixed_storage())
        other.mData = other.mFixedStorage.data();
    else
        other.mData = mData;
    if (tempData == other.mFixedStorage.data())
        mData = mFixedStorage.data();
    else
        mData = tempData;
    std::swap(mReservedSize, other.mReservedSize);

    if (uses_fixed_storage() || other.uses_fixed_storage())
        std::swap(mFixedStorage, other.mFixedStorage);
}

template <class T, size_t N, class Storage>
void FastVector<T, N, Storage>::resize(size_type count)
{
    if (count > mSize)
    {
        ensure_capacity(count);
    }
    mSize = count;
}

template <class T, size_t N, class Storage>
void FastVector<T, N, Storage>::resize(size_type count, const value_type &value)
{
    if (count > mSize)
    {
        ensure_capacity(count);
        std::fill(mData + mSize, mData + count, value);
    }
    mSize = count;
}

template <class T, size_t N, class Storage>
void FastVector<T, N, Storage>::assign_from_initializer_list(std::initializer_list<value_type> init)
{
    ensure_capacity(init.size());
    mSize        = init.size();
    size_t index = 0;
    for (auto &value : init)
    {
        mData[index++] = value;
    }
}

template <class T, size_t N, class Storage>
ANGLE_INLINE void FastVector<T, N, Storage>::remove_and_permute(const value_type &element)
{
    size_t len = mSize - 1;
    for (size_t index = 0; index < len; ++index)
    {
        if (mData[index] == element)
        {
            mData[index] = std::move(mData[len]);
            break;
        }
    }
    pop_back();
}

template <class T, size_t N, class Storage>
void FastVector<T, N, Storage>::ensure_capacity(size_t capacity)
{
    // We have a minimum capacity of N.
    if (mReservedSize < capacity)
    {
        ASSERT(capacity > N);
        size_type newSize = std::max(mReservedSize, N);
        while (newSize < capacity)
        {
            newSize *= 2;
        }

        pointer newData = new value_type[newSize];

        if (mSize > 0)
        {
            std::move(begin(), end(), newData);
        }

        if (!uses_fixed_storage())
        {
            delete[] mData;
        }

        mData         = newData;
        mReservedSize = newSize;
    }
}
}  // namespace angle

#endif  // COMMON_FASTVECTOR_H_
