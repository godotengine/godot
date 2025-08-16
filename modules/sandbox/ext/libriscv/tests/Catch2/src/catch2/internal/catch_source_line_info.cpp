
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/internal/catch_source_line_info.hpp>

#include <cstring>
#include <ostream>

namespace Catch {

    bool SourceLineInfo::operator == ( SourceLineInfo const& other ) const noexcept {
        return line == other.line && (file == other.file || std::strcmp(file, other.file) == 0);
    }
    bool SourceLineInfo::operator < ( SourceLineInfo const& other ) const noexcept {
        // We can assume that the same file will usually have the same pointer.
        // Thus, if the pointers are the same, there is no point in calling the strcmp
        return line < other.line || ( line == other.line && file != other.file && (std::strcmp(file, other.file) < 0));
    }

    std::ostream& operator << ( std::ostream& os, SourceLineInfo const& info ) {
#ifndef __GNUG__
        os << info.file << '(' << info.line << ')';
#else
        os << info.file << ':' << info.line;
#endif
        return os;
    }

} // end namespace Catch
