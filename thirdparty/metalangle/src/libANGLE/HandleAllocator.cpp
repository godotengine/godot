//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// HandleAllocator.cpp: Implements the gl::HandleAllocator class, which is used
// to allocate GL handles.

#include "libANGLE/HandleAllocator.h"

#include <algorithm>
#include <functional>

#include "common/debug.h"

namespace gl
{

struct HandleAllocator::HandleRangeComparator
{
    bool operator()(const HandleRange &range, GLuint handle) const { return (range.end < handle); }
};

HandleAllocator::HandleAllocator() : mBaseValue(1), mNextValue(1), mLoggingEnabled(false)
{
    mUnallocatedList.push_back(HandleRange(1, std::numeric_limits<GLuint>::max()));
}

HandleAllocator::HandleAllocator(GLuint maximumHandleValue)
    : mBaseValue(1), mNextValue(1), mLoggingEnabled(false)
{
    mUnallocatedList.push_back(HandleRange(1, maximumHandleValue));
}

HandleAllocator::~HandleAllocator() {}

void HandleAllocator::setBaseHandle(GLuint value)
{
    ASSERT(mBaseValue == mNextValue);
    mBaseValue = value;
    mNextValue = value;
}

GLuint HandleAllocator::allocate()
{
    ASSERT(!mUnallocatedList.empty() || !mReleasedList.empty());

    // Allocate from released list, logarithmic time for pop_heap.
    if (!mReleasedList.empty())
    {
        std::pop_heap(mReleasedList.begin(), mReleasedList.end(), std::greater<GLuint>());
        GLuint reusedHandle = mReleasedList.back();
        mReleasedList.pop_back();

        if (mLoggingEnabled)
        {
            WARN() << "HandleAllocator::allocate reusing " << reusedHandle << std::endl;
        }

        return reusedHandle;
    }

    // Allocate from unallocated list, constant time.
    auto listIt = mUnallocatedList.begin();

    GLuint freeListHandle = listIt->begin;
    ASSERT(freeListHandle > 0);

    if (listIt->begin == listIt->end)
    {
        mUnallocatedList.erase(listIt);
    }
    else
    {
        listIt->begin++;
    }

    if (mLoggingEnabled)
    {
        WARN() << "HandleAllocator::allocate allocating " << freeListHandle << std::endl;
    }

    return freeListHandle;
}

void HandleAllocator::release(GLuint handle)
{
    if (mLoggingEnabled)
    {
        WARN() << "HandleAllocator::release releasing " << handle << std::endl;
    }

    // Add to released list, logarithmic time for push_heap.
    mReleasedList.push_back(handle);
    std::push_heap(mReleasedList.begin(), mReleasedList.end(), std::greater<GLuint>());
}

void HandleAllocator::reserve(GLuint handle)
{
    if (mLoggingEnabled)
    {
        WARN() << "HandleAllocator::reserve reserving " << handle << std::endl;
    }

    // Clear from released list -- might be a slow operation.
    if (!mReleasedList.empty())
    {
        auto releasedIt = std::find(mReleasedList.begin(), mReleasedList.end(), handle);
        if (releasedIt != mReleasedList.end())
        {
            mReleasedList.erase(releasedIt);
            std::make_heap(mReleasedList.begin(), mReleasedList.end(), std::greater<GLuint>());
            return;
        }
    }

    // Not in released list, reserve in the unallocated list.
    auto boundIt = std::lower_bound(mUnallocatedList.begin(), mUnallocatedList.end(), handle,
                                    HandleRangeComparator());

    ASSERT(boundIt != mUnallocatedList.end());

    GLuint begin = boundIt->begin;
    GLuint end   = boundIt->end;

    if (handle == begin || handle == end)
    {
        if (begin == end)
        {
            mUnallocatedList.erase(boundIt);
        }
        else if (handle == begin)
        {
            boundIt->begin++;
        }
        else
        {
            ASSERT(handle == end);
            boundIt->end--;
        }
        return;
    }

    ASSERT(begin < handle && handle < end);

    // need to split the range
    auto placementIt = mUnallocatedList.erase(boundIt);
    placementIt      = mUnallocatedList.insert(placementIt, HandleRange(handle + 1, end));
    mUnallocatedList.insert(placementIt, HandleRange(begin, handle - 1));
}

void HandleAllocator::reset()
{
    mUnallocatedList.clear();
    mUnallocatedList.push_back(HandleRange(1, std::numeric_limits<GLuint>::max()));
    mReleasedList.clear();
    mBaseValue = 1;
    mNextValue = 1;
}

void HandleAllocator::enableLogging(bool enabled)
{
    mLoggingEnabled = enabled;
}

}  // namespace gl
