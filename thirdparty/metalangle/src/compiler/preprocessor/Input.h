//
// Copyright 2011 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_PREPROCESSOR_INPUT_H_
#define COMPILER_PREPROCESSOR_INPUT_H_

#include <cstddef>
#include <vector>

namespace angle
{

namespace pp
{

// Holds and reads input for Lexer.
class Input
{
  public:
    Input();
    ~Input();
    Input(size_t count, const char *const string[], const int length[]);

    size_t count() const { return mCount; }
    const char *string(size_t index) const { return mString[index]; }
    size_t length(size_t index) const { return mLength[index]; }

    size_t read(char *buf, size_t maxSize, int *lineNo);

    struct Location
    {
        size_t sIndex;  // String index;
        size_t cIndex;  // Char index.

        Location() : sIndex(0), cIndex(0) {}
    };
    const Location &readLoc() const { return mReadLoc; }

  private:
    // Skip a character and return the next character after the one that was skipped.
    // Return nullptr if data runs out.
    const char *skipChar();

    // Input.
    size_t mCount;
    const char *const *mString;
    std::vector<size_t> mLength;

    Location mReadLoc;
};

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_INPUT_H_
