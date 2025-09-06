
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include "NullOStream.h"

void NullOStream::avoidOutOfLineVirtualCompilerWarning()
{
}

int NullStreambuf::overflow(int c){
    setp(dummyBuffer, dummyBuffer + sizeof(dummyBuffer));
    return (c == traits_type::eof()) ? '\0' : c;
}
