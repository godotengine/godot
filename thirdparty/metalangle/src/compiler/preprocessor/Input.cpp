//
// Copyright 2011 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/preprocessor/Input.h"

#include <algorithm>
#include <cstring>

#include "common/debug.h"

namespace angle
{

namespace pp
{

Input::Input() : mCount(0), mString(0) {}

Input::~Input() {}

Input::Input(size_t count, const char *const string[], const int length[])
    : mCount(count), mString(string)
{
    mLength.reserve(mCount);
    for (size_t i = 0; i < mCount; ++i)
    {
        int len = length ? length[i] : -1;
        mLength.push_back(len < 0 ? std::strlen(mString[i]) : len);
    }
}

const char *Input::skipChar()
{
    // This function should only be called when there is a character to skip.
    ASSERT(mReadLoc.cIndex < mLength[mReadLoc.sIndex]);
    ++mReadLoc.cIndex;
    if (mReadLoc.cIndex == mLength[mReadLoc.sIndex])
    {
        ++mReadLoc.sIndex;
        mReadLoc.cIndex = 0;
    }
    if (mReadLoc.sIndex >= mCount)
    {
        return nullptr;
    }
    return mString[mReadLoc.sIndex] + mReadLoc.cIndex;
}

size_t Input::read(char *buf, size_t maxSize, int *lineNo)
{
    size_t nRead = 0;
    // The previous call to read might have stopped copying the string when encountering a line
    // continuation. Check for this possibility first.
    if (mReadLoc.sIndex < mCount && maxSize > 0)
    {
        const char *c = mString[mReadLoc.sIndex] + mReadLoc.cIndex;
        if ((*c) == '\\')
        {
            c = skipChar();
            if (c != nullptr && (*c) == '\n')
            {
                // Line continuation of backslash + newline.
                skipChar();
                // Fake an EOF if the line number would overflow.
                if (*lineNo == INT_MAX)
                {
                    return 0;
                }
                ++(*lineNo);
            }
            else if (c != nullptr && (*c) == '\r')
            {
                // Line continuation. Could be backslash + '\r\n' or just backslash + '\r'.
                c = skipChar();
                if (c != nullptr && (*c) == '\n')
                {
                    skipChar();
                }
                // Fake an EOF if the line number would overflow.
                if (*lineNo == INT_MAX)
                {
                    return 0;
                }
                ++(*lineNo);
            }
            else
            {
                // Not line continuation, so write the skipped backslash to buf.
                *buf = '\\';
                ++nRead;
            }
        }
    }

    size_t maxRead = maxSize;
    while ((nRead < maxRead) && (mReadLoc.sIndex < mCount))
    {
        size_t size = mLength[mReadLoc.sIndex] - mReadLoc.cIndex;
        size        = std::min(size, maxSize);
        for (size_t i = 0; i < size; ++i)
        {
            // Stop if a possible line continuation is encountered.
            // It will be processed on the next call on input, which skips it
            // and increments line number if necessary.
            if (*(mString[mReadLoc.sIndex] + mReadLoc.cIndex + i) == '\\')
            {
                size    = i;
                maxRead = nRead + size;  // Stop reading right before the backslash.
            }
        }
        std::memcpy(buf + nRead, mString[mReadLoc.sIndex] + mReadLoc.cIndex, size);
        nRead += size;
        mReadLoc.cIndex += size;

        // Advance string if we reached the end of current string.
        if (mReadLoc.cIndex == mLength[mReadLoc.sIndex])
        {
            ++mReadLoc.sIndex;
            mReadLoc.cIndex = 0;
        }
    }
    return nRead;
}

}  // namespace pp

}  // namespace angle
