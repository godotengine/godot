//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FixedVector.h:
//   A vector class with a maximum size and fixed storage.
//

#ifndef COMMON_FIXEDVECTOR_H_
#define COMMON_FIXEDVECTOR_H_

#include "common/debug.h"

#include <algorithm>
#include <array>
#include <initializer_list>

namespace angle
{
template <class T, size_t N, class Storage = std::array<T, N>>
class FixedVector final
{
  public:
    using value_type             = typename Storage::value_type;
    using size_type              = typename Storage::size_type;
    using reference              = typename Storage::reference;
    using const_reference        = typename Storage::const_reference;
    using pointer                = typename Storage::pointer;
    using const_pointer          = typename Storage::const_pointer;
    using iterator               = typename Storage::iterator;
    using const_iterator         = typename Storage::const_iterator;
    using reverse_iterator       = typename Storage::reverse_iterator;
    using const_reverse_iterator = typename Storage::const_reverse_iterator;

    FixedVector();
    FixedVector(size_type count, const value_type &value);
    FixedVector(size_type count);

    FixedVector(const FixedVector<T, N, Storage> &other);
    FixedVector(FixedVector<T, N, Storage> &&other);
    FixedVector(std::initializer_list<value_type> init);

    FixedVector<T, N, Storage> &operator=(const FixedVector<T, N, Storage> &other);
    FixedVector<T, N, Storage> &operator=(FixedVector<T, N, Storage> &&other);
    FixedVector<T, N, Storage> &operator=(std::initializer_list<value_type> init);

    ~FixedVector();

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
    static constexpr size_type max_size();

    void clear();

    void push_back(const value_type &value);
    void push_back(value_type &&value);

    template <class... Args>
    void emplace_back(Args &&... args);

    void pop_back();
    reference back();
    const_reference back() const;

    void swap(FixedVector<T, N, Storage> &other);

    void resize(size_type count);
    void resize(size_type count, const value_type &value);

    bool full() const;

  private:
    void assign_from_initializer_list(std::initializer_list<value_type> init);

    Storage mStorage;
    size_type mSize = 0;
};

template <class T, size_t N, class Storage>
bool operator==(const FixedVector<T, N, Storage> &a, const FixedVector<T, N, Storage> &b)
{
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

template <class T, size_t N, class Storage>
bool operator!=(const FixedVector<T, N, Storage> &a, const FixedVector<T, N, Storage> &b)
{
    return !(a == b);
}

template <class T, size_t N, class Storage>
FixedVector<T, N, Storage>::FixedVector() = default;

template <class T, size_t N, class Storage>
FixedVector<T, N, Storage>::FixedVector(size_type count, const value_type &value) : mSize(count)
{
    ASSERT(count <= N);
    std::fill(mStorage.begin(), mStorage.begin() + count, value);
}

template <class T, size_t N, class Storage>
FixedVector<T, N, Storage>::FixedVector(size_type count) : mSize(count)
{
    ASSERT(count <= N);
}

template <class T, size_t N, class Storage>
FixedVector<T, N, Storage>::FixedVector(const FixedVector<T, N, Storage> &other) = default;

template <class T, size_t N, class Storage>
FixedVector<T, N, Storage>::FixedVector(FixedVector<T, N, Storage> &&other) = default;

template <class T, size_t N, class Storage>
FixedVector<T, N, Storage>::FixedVector(std::initializer_list<value_type> init)
{
    ASSERT(init.size() <= N);
    assign_from_initializer_list(init);
}

template <class T, size_t N, class Storage>
FixedVector<T, N, Storage> &FixedVector<T, N, Storage>::operator=(
    const FixedVector<T, N, Storage> &other) = default;

template <class T, size_t N, class Storage>
FixedVector<T, N, Storage> &FixedVector<T, N, Storage>::operator=(
    FixedVector<T, N, Storage> &&other) = default;

template <class T, size_t N, class Storage>
FixedVector<T, N, Storage> &FixedVector<T, N, Storage>::operator=(
    std::initializer_list<value_type> init)
{
    clear();
    ASSERT(init.size() <= N);
    assign_from_initializer_list(init);
    return this;
}

template <class T, size_t N, class Storage>
FixedVector<T, N, Storage>::~FixedVector()
{
    clear();
}

template <class T, size_t N, class Storage>
typename FixedVector<T, N, Storage>::reference FixedVector<T, N, Storage>::at(size_type pos)
{
    ASSERT(pos < N);
    return mStorage.at(pos);
}

template <class T, size_t N, class Storage>
typename FixedVector<T, N, Storage>::const_reference FixedVector<T, N, Storage>::at(
    size_type pos) const
{
    ASSERT(pos < N);
    return mStorage.at(pos);
}

template <class T, size_t N, class Storage>
typename FixedVector<T, N, Storage>::reference FixedVector<T, N, Storage>::operator[](size_type pos)
{
    ASSERT(pos < N);
    return mStorage[pos];
}

template <class T, size_t N, class Storage>
typename FixedVector<T, N, Storage>::const_reference FixedVector<T, N, Storage>::operator[](
    size_type pos) const
{
    ASSERT(pos < N);
    return mStorage[pos];
}

template <class T, size_t N, class Storage>
typename FixedVector<T, N, Storage>::const_pointer angle::FixedVector<T, N, Storage>::data() const
{
    return mStorage.data();
}

template <class T, size_t N, class Storage>
typename FixedVector<T, N, Storage>::pointer angle::FixedVector<T, N, Storage>::data()
{
    return mStorage.data();
}

template <class T, size_t N, class Storage>
typename FixedVector<T, N, Storage>::iterator FixedVector<T, N, Storage>::begin()
{
    return mStorage.begin();
}

template <class T, size_t N, class Storage>
typename FixedVector<T, N, Storage>::const_iterator FixedVector<T, N, Storage>::begin() const
{
    return mStorage.begin();
}

template <class T, size_t N, class Storage>
typename FixedVector<T, N, Storage>::iterator FixedVector<T, N, Storage>::end()
{
    return mStorage.begin() + mSize;
}

template <class T, size_t N, class Storage>
typename FixedVector<T, N, Storage>::const_iterator FixedVector<T, N, Storage>::end() const
{
    return mStorage.begin() + mSize;
}

template <class T, size_t N, class Storage>
bool FixedVector<T, N, Storage>::empty() const
{
    return mSize == 0;
}

template <class T, size_t N, class Storage>
typename FixedVector<T, N, Storage>::size_type FixedVector<T, N, Storage>::size() const
{
    return mSize;
}

template <class T, size_t N, class Storage>
constexpr typename FixedVector<T, N, Storage>::size_type FixedVector<T, N, Storage>::max_size()
{
    return N;
}

template <class T, size_t N, class Storage>
void FixedVector<T, N, Storage>::clear()
{
    resize(0);
}

template <class T, size_t N, class Storage>
void FixedVector<T, N, Storage>::push_back(const value_type &value)
{
    ASSERT(mSize < N);
    mStorage[mSize] = value;
    mSize++;
}

template <class T, size_t N, class Storage>
void FixedVector<T, N, Storage>::push_back(value_type &&value)
{
    ASSERT(mSize < N);
    mStorage[mSize] = std::move(value);
    mSize++;
}

template <class T, size_t N, class Storage>
template <class... Args>
void FixedVector<T, N, Storage>::emplace_back(Args &&... args)
{
    ASSERT(mSize < N);
    new (&mStorage[mSize]) T{std::forward<Args>(args)...};
    mSize++;
}

template <class T, size_t N, class Storage>
void FixedVector<T, N, Storage>::pop_back()
{
    ASSERT(mSize > 0);
    mSize--;
}

template <class T, size_t N, class Storage>
typename FixedVector<T, N, Storage>::reference FixedVector<T, N, Storage>::back()
{
    ASSERT(mSize > 0);
    return mStorage[mSize - 1];
}

template <class T, size_t N, class Storage>
typename FixedVector<T, N, Storage>::const_reference FixedVector<T, N, Storage>::back() const
{
    ASSERT(mSize > 0);
    return mStorage[mSize - 1];
}

template <class T, size_t N, class Storage>
void FixedVector<T, N, Storage>::swap(FixedVector<T, N, Storage> &other)
{
    std::swap(mSize, other.mSize);
    std::swap(mStorage, other.mStorage);
}

template <class T, size_t N, class Storage>
void FixedVector<T, N, Storage>::resize(size_type count)
{
    ASSERT(count <= N);
    while (mSize > count)
    {
        mSize--;
        mStorage[mSize] = value_type();
    }
    while (mSize < count)
    {
        mStorage[mSize] = value_type();
        mSize++;
    }
}

template <class T, size_t N, class Storage>
void FixedVector<T, N, Storage>::resize(size_type count, const value_type &value)
{
    ASSERT(count <= N);
    while (mSize > count)
    {
        mSize--;
        mStorage[mSize] = value_type();
    }
    while (mSize < count)
    {
        mStorage[mSize] = value;
        mSize++;
    }
}

template <class T, size_t N, class Storage>
void FixedVector<T, N, Storage>::assign_from_initializer_list(
    std::initializer_list<value_type> init)
{
    for (auto element : init)
    {
        mStorage[mSize] = std::move(element);
        mSize++;
    }
}

template <class T, size_t N, class Storage>
bool FixedVector<T, N, Storage>::full() const
{
    return (mSize == N);
}
}  // namespace angle

#endif  // COMMON_FIXEDVECTOR_H_
