
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/internal/catch_stringref.hpp>

#include <algorithm>
#include <ostream>
#include <cstring>

namespace Catch {
    StringRef::StringRef( char const* rawChars ) noexcept
    : StringRef( rawChars, std::strlen(rawChars) )
    {}


    bool StringRef::operator<(StringRef rhs) const noexcept {
        if (m_size < rhs.m_size) {
            return strncmp(m_start, rhs.m_start, m_size) <= 0;
        }
        return strncmp(m_start, rhs.m_start, rhs.m_size) < 0;
    }

    int StringRef::compare( StringRef rhs ) const {
        auto cmpResult =
            strncmp( m_start, rhs.m_start, std::min( m_size, rhs.m_size ) );

        // This means that strncmp found a difference before the strings
        // ended, and we can return it directly
        if ( cmpResult != 0 ) {
            return cmpResult;
        }

        // If strings are equal up to length, then their comparison results on
        // their size
        if ( m_size < rhs.m_size ) {
            return -1;
        } else if ( m_size > rhs.m_size ) {
            return 1;
        } else {
            return 0;
        }
    }

    auto operator << ( std::ostream& os, StringRef str ) -> std::ostream& {
        return os.write(str.data(), static_cast<std::streamsize>(str.size()));
    }

    std::string operator+(StringRef lhs, StringRef rhs) {
        std::string ret;
        ret.reserve(lhs.size() + rhs.size());
        ret += lhs;
        ret += rhs;
        return ret;
    }

    auto operator+=( std::string& lhs, StringRef rhs ) -> std::string& {
        lhs.append(rhs.data(), rhs.size());
        return lhs;
    }

} // namespace Catch
