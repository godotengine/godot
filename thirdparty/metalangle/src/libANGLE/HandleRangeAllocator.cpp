//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// HandleRangeAllocator.cpp : Implementation for HandleRangeAllocator.h

#include "libANGLE/HandleRangeAllocator.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "common/angleutils.h"
#include "common/debug.h"

namespace gl
{

const GLuint HandleRangeAllocator::kInvalidHandle = 0;

HandleRangeAllocator::HandleRangeAllocator()
{
    // Simplify the code by making sure that lower_bound(id) never
    // returns the beginning of the map, if id is valid (eg != kInvalidHandle).
    mUsed.insert(std::make_pair(0u, 0u));
}

HandleRangeAllocator::~HandleRangeAllocator() {}

GLuint HandleRangeAllocator::allocate()
{
    return allocateRange(1u);
}

GLuint HandleRangeAllocator::allocateAtOrAbove(GLuint wanted)
{
    if (wanted == 0u || wanted == 1u)
        return allocateRange(1u);

    auto current = mUsed.lower_bound(wanted);
    auto next    = current;
    if (current == mUsed.end() || current->first > wanted)
    {
        current--;
    }
    else
    {
        next++;
    }

    GLuint firstId = current->first;
    GLuint lastId  = current->second;
    ASSERT(wanted >= firstId);

    if (wanted - 1u <= lastId)
    {
        // Append to current range.
        lastId++;
        if (lastId == 0)
        {
            // The increment overflowed.
            return allocateRange(1u);
        }

        current->second = lastId;

        if (next != mUsed.end() && next->first - 1u == lastId)
        {
            // Merge with next range.
            current->second = next->second;
            mUsed.erase(next);
        }
        return lastId;
    }
    else if (next != mUsed.end() && next->first - 1u == wanted)
    {
        // Prepend to next range.
        GLuint lastExisting = next->second;
        mUsed.erase(next);
        mUsed.insert(std::make_pair(wanted, lastExisting));
        return wanted;
    }
    mUsed.insert(std::make_pair(wanted, wanted));
    return wanted;
}

GLuint HandleRangeAllocator::allocateRange(GLuint range)
{
    ASSERT(range != 0);

    auto current = mUsed.begin();
    auto next    = current;

    while (++next != mUsed.end())
    {
        if (next->first - current->second > range)
            break;
        current = next;
    }
    const GLuint firstId = current->second + 1u;
    const GLuint lastId  = firstId + range - 1u;

    // deal with wraparound
    if (firstId == 0u || lastId < firstId)
        return kInvalidHandle;

    current->second = lastId;

    if (next != mUsed.end() && next->first - 1u == lastId)
    {
        // merge with next range
        current->second = next->second;
        mUsed.erase(next);
    }
    return firstId;
}

bool HandleRangeAllocator::markAsUsed(GLuint handle)
{
    ASSERT(handle);
    auto current = mUsed.lower_bound(handle);
    if (current != mUsed.end() && current->first == handle)
        return false;

    auto next = current;
    --current;

    if (current->second >= handle)
        return false;

    ASSERT(current->first < handle && current->second < handle);

    if (current->second + 1u == handle)
    {
        // Append to current range.
        current->second = handle;
        if (next != mUsed.end() && next->first - 1u == handle)
        {
            // Merge with next range.
            current->second = next->second;
            mUsed.erase(next);
        }
        return true;
    }
    else if (next != mUsed.end() && next->first - 1u == handle)
    {
        // Prepend to next range.
        GLuint lastExisting = next->second;
        mUsed.erase(next);
        mUsed.insert(std::make_pair(handle, lastExisting));
        return true;
    }

    mUsed.insert(std::make_pair(handle, handle));
    return true;
}

void HandleRangeAllocator::release(GLuint handle)
{
    releaseRange(handle, 1u);
}

void HandleRangeAllocator::releaseRange(GLuint first, GLuint range)
{
    if (range == 0u || (first == 0u && range == 1u))
        return;

    if (first == 0u)
    {
        first++;
        range--;
    }

    GLuint last = first + range - 1u;
    if (last < first)
        last = std::numeric_limits<GLuint>::max();

    while (true)
    {
        auto current = mUsed.lower_bound(last);
        if (current == mUsed.end() || current->first > last)
            --current;

        if (current->second < first)
            return;

        if (current->first >= first)
        {
            const GLuint lastExisting = current->second;
            mUsed.erase(current);
            if (last < lastExisting)
            {
                mUsed.insert(std::make_pair(last + 1u, lastExisting));
            }
        }
        else if (current->second <= last)
        {
            current->second = first - 1u;
        }
        else
        {
            ASSERT(current->first < first && current->second > last);
            const GLuint lastExisting = current->second;
            current->second           = first - 1u;
            mUsed.insert(std::make_pair(last + 1u, lastExisting));
        }
    }
}

bool HandleRangeAllocator::isUsed(GLuint handle) const
{
    if (handle == kInvalidHandle)
        return false;

    auto current = mUsed.lower_bound(handle);
    if (current != mUsed.end())
    {
        if (current->first == handle)
            return true;
    }
    --current;
    return current->second >= handle;
}

}  // namespace gl
