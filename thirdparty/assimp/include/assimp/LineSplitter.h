/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/

/** @file  LineSplitter.h
 *  @brief LineSplitter, a helper class to iterate through all lines
 *    of a file easily. Works with StreamReader.
 */
#pragma once
#ifndef INCLUDED_LINE_SPLITTER_H
#define INCLUDED_LINE_SPLITTER_H

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <stdexcept>
#include <assimp/StreamReader.h>
#include <assimp/ParsingUtils.h>

namespace Assimp {

// ------------------------------------------------------------------------------------------------
/** Usage:
@code
for(LineSplitter splitter(stream);splitter;++splitter) {

    if (*splitter == "hi!") {
       ...
    }
    else if (splitter->substr(0,5) == "hello") {
       ...
       // access the third token in the line (tokens are space-separated)
       if (strtol(splitter[2]) > 5) { .. }
    }

    ASSIMP_LOG_DEBUG_F("Current line is: ", splitter.get_index());
}
@endcode
*/
// ------------------------------------------------------------------------------------------------
class LineSplitter {
public:
    typedef size_t line_idx;

    // -----------------------------------------
    /** construct from existing stream reader
    note: trim is *always* assumed true if skyp_empty_lines==true
    */
    LineSplitter(StreamReaderLE& stream, bool skip_empty_lines = true, bool trim = true);

    ~LineSplitter();

    // -----------------------------------------
    /** pseudo-iterator increment */
    LineSplitter& operator++();

    // -----------------------------------------
    LineSplitter& operator++(int);

    // -----------------------------------------
    /** get a pointer to the beginning of a particular token */
    const char* operator[] (size_t idx) const;

    // -----------------------------------------
    /** extract the start positions of N tokens from the current line*/
    template <size_t N>
    void get_tokens(const char* (&tokens)[N]) const;

    // -----------------------------------------
    /** member access */
    const std::string* operator -> () const;

    std::string operator* () const;

    // -----------------------------------------
    /** boolean context */
    operator bool() const;

    // -----------------------------------------
    /** line indices are zero-based, empty lines are included */
    operator line_idx() const;

    line_idx get_index() const;

    // -----------------------------------------
    /** access the underlying stream object */
    StreamReaderLE& get_stream();

    // -----------------------------------------
    /** !strcmp((*this)->substr(0,strlen(check)),check) */
    bool match_start(const char* check);

    // -----------------------------------------
    /** swallow the next call to ++, return the previous value. */
    void swallow_next_increment();

    LineSplitter( const LineSplitter & ) = delete;
    LineSplitter(LineSplitter &&) = delete;
    LineSplitter &operator = ( const LineSplitter & ) = delete;

private:
    line_idx mIdx;
    std::string mCur;
    StreamReaderLE& mStream;
    bool mSwallow, mSkip_empty_lines, mTrim;
};

AI_FORCE_INLINE
LineSplitter::LineSplitter(StreamReaderLE& stream, bool skip_empty_lines, bool trim )
: mIdx(0)
, mCur()
, mStream(stream)
, mSwallow()
, mSkip_empty_lines(skip_empty_lines)
, mTrim(trim) {
    mCur.reserve(1024);
    operator++();
    mIdx = 0;
}

AI_FORCE_INLINE
LineSplitter::~LineSplitter() {
    // empty
}

AI_FORCE_INLINE
LineSplitter& LineSplitter::operator++() {
    if (mSwallow) {
        mSwallow = false;
        return *this;
    }

    if (!*this) {
        throw std::logic_error("End of file, no more lines to be retrieved.");
    }

    char s;
    mCur.clear();
    while (mStream.GetRemainingSize() && (s = mStream.GetI1(), 1)) {
        if (s == '\n' || s == '\r') {
            if (mSkip_empty_lines) {
                while (mStream.GetRemainingSize() && ((s = mStream.GetI1()) == ' ' || s == '\r' || s == '\n'));
                if (mStream.GetRemainingSize()) {
                    mStream.IncPtr(-1);
                }
            } else {
                // skip both potential line terminators but don't read past this line.
                if (mStream.GetRemainingSize() && (s == '\r' && mStream.GetI1() != '\n')) {
                    mStream.IncPtr(-1);
                }
                if (mTrim) {
                    while (mStream.GetRemainingSize() && ((s = mStream.GetI1()) == ' ' || s == '\t'));
                    if (mStream.GetRemainingSize()) {
                        mStream.IncPtr(-1);
                    }
                }
            }
            break;
        }
        mCur += s;
    }
    ++mIdx;

    return *this;
}

AI_FORCE_INLINE
LineSplitter &LineSplitter::operator++(int) {
    return ++(*this);
}

AI_FORCE_INLINE
const char *LineSplitter::operator[] (size_t idx) const {
    const char* s = operator->()->c_str();

    SkipSpaces(&s);
    for (size_t i = 0; i < idx; ++i) {

        for (; !IsSpace(*s); ++s) {
            if (IsLineEnd(*s)) {
                throw std::range_error("Token index out of range, EOL reached");
            }
        }
        SkipSpaces(&s);
    }
    return s;
}

template <size_t N>
AI_FORCE_INLINE
void LineSplitter::get_tokens(const char* (&tokens)[N]) const {
    const char* s = operator->()->c_str();

    SkipSpaces(&s);
    for (size_t i = 0; i < N; ++i) {
        if (IsLineEnd(*s)) {
            throw std::range_error("Token count out of range, EOL reached");
        }
        tokens[i] = s;

        for (; *s && !IsSpace(*s); ++s);
        SkipSpaces(&s);
    }
}

AI_FORCE_INLINE
const std::string* LineSplitter::operator -> () const {
    return &mCur;
}

AI_FORCE_INLINE
std::string LineSplitter::operator* () const {
    return mCur;
}

AI_FORCE_INLINE
LineSplitter::operator bool() const {
    return mStream.GetRemainingSize() > 0;
}

AI_FORCE_INLINE
LineSplitter::operator line_idx() const {
    return mIdx;
}

AI_FORCE_INLINE
LineSplitter::line_idx LineSplitter::get_index() const {
    return mIdx;
}

AI_FORCE_INLINE
StreamReaderLE &LineSplitter::get_stream() {
    return mStream;
}

AI_FORCE_INLINE
bool LineSplitter::match_start(const char* check) {
    const size_t len = ::strlen(check);

    return len <= mCur.length() && std::equal(check, check + len, mCur.begin());
}

AI_FORCE_INLINE
void LineSplitter::swallow_next_increment() {
    mSwallow = true;
}

} // Namespace Assimp

#endif // INCLUDED_LINE_SPLITTER_H
