/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


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

/** @file  TinyFormatter.h
 *  @brief Utility to format log messages more easily. Introduced
 *    to get rid of the boost::format dependency. Much slinker,
 *    basically just extends stringstream.
 */
#ifndef INCLUDED_TINY_FORMATTER_H
#define INCLUDED_TINY_FORMATTER_H

#include <sstream>

namespace Assimp {
namespace Formatter {

// ------------------------------------------------------------------------------------------------
/** stringstream utility. Usage:
 *  @code
 *  void writelog(const std::string&s);
 *  void writelog(const std::wstring&s);
 *  ...
 *  writelog(format()<< "hi! this is a number: " << 4);
 *  writelog(wformat()<< L"hi! this is a number: " << 4);
 *
 *  @endcode */
template < typename T,
    typename CharTraits = std::char_traits<T>,
    typename Allocator  = std::allocator<T>
>
class basic_formatter
{

public:

    typedef class std::basic_string<
        T,CharTraits,Allocator
    > string;

    typedef class std::basic_ostringstream<
        T,CharTraits,Allocator
    > stringstream;

public:

    basic_formatter() {}

    /* Allow basic_formatter<T>'s to be used almost interchangeably
     * with std::(w)string or const (w)char* arguments because the
     * conversion c'tor is called implicitly. */
    template <typename TT>
    basic_formatter(const TT& sin)  {
        underlying << sin;
    }


    // The problem described here:
    // https://sourceforge.net/tracker/?func=detail&atid=1067632&aid=3358562&group_id=226462
    // can also cause trouble here. Apparently, older gcc versions sometimes copy temporaries
    // being bound to const ref& function parameters. Copying streams is not permitted, though.
    // This workaround avoids this by manually specifying a copy ctor.
#if !defined(__GNUC__) || !defined(__APPLE__) || __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
    explicit basic_formatter(const basic_formatter& other) {
        underlying << (string)other;
    }
#endif


public:

    operator string () const {
        return underlying.str();
    }


    /* note - this is declared const because binding temporaries does only
     * work for const references, so many function prototypes will
     * include const basic_formatter<T>& s but might still want to
     * modify the formatted string without the need for a full copy.*/
    template <typename TToken>
    const basic_formatter& operator << (const TToken& s) const {
        underlying << s;
        return *this;
    }

    template <typename TToken>
    basic_formatter& operator << (const TToken& s) {
        underlying << s;
        return *this;
    }


    // comma operator overloaded as well, choose your preferred way.
    template <typename TToken>
    const basic_formatter& operator, (const TToken& s) const {
        underlying << s;
        return *this;
    }

    template <typename TToken>
    basic_formatter& operator, (const TToken& s) {
        underlying << s;
        return *this;
    }

    // Fix for MSVC8
    // See https://sourceforge.net/projects/assimp/forums/forum/817654/topic/4372824
    template <typename TToken>
    basic_formatter& operator, (TToken& s) {
        underlying << s;
        return *this;
    }


private:
    mutable stringstream underlying;
};


typedef basic_formatter< char > format;
typedef basic_formatter< wchar_t > wformat;

} // ! namespace Formatter

} // ! namespace Assimp

#endif
