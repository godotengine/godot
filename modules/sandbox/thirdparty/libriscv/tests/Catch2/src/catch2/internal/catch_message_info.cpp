
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/internal/catch_message_info.hpp>

namespace Catch {

    MessageInfo::MessageInfo( StringRef _macroName,
                              SourceLineInfo const& _lineInfo,
                              ResultWas::OfType _type )
    :   macroName( _macroName ),
        lineInfo( _lineInfo ),
        type( _type ),
        sequence( ++globalCount )
    {}

    // Messages are owned by their individual threads, so the counter should be thread-local as well.
    // Alternative consideration: atomic, so threads don't share IDs and things are easier to debug.
    thread_local unsigned int MessageInfo::globalCount = 0;

} // end namespace Catch
