
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TAG_ALIAS_HPP_INCLUDED
#define CATCH_TAG_ALIAS_HPP_INCLUDED

#include <catch2/internal/catch_source_line_info.hpp>

#include <string>

namespace Catch {

    struct TagAlias {
        TagAlias(std::string const& _tag, SourceLineInfo _lineInfo):
            tag(_tag),
            lineInfo(_lineInfo)
        {}

        std::string tag;
        SourceLineInfo lineInfo;
    };

} // end namespace Catch

#endif // CATCH_TAG_ALIAS_HPP_INCLUDED
