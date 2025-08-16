
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/generators/catch_generator_exception.hpp>

namespace Catch {

    const char* GeneratorException::what() const noexcept {
        return m_msg;
    }

} // end namespace Catch
