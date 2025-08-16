
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/interfaces/catch_interfaces_reporter_factory.hpp>

namespace Catch {
    IReporterFactory::~IReporterFactory() = default;
    EventListenerFactory::~EventListenerFactory() = default;
}
