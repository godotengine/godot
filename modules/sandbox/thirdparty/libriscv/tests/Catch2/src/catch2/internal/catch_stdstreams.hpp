
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#ifndef CATCH_STDSTREAMS_HPP_INCLUDED
#define CATCH_STDSTREAMS_HPP_INCLUDED

#include <iosfwd>

namespace Catch {

    std::ostream& cout();
    std::ostream& cerr();
    std::ostream& clog();

} // namespace Catch

#endif
