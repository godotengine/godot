
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_MOVE_AND_FORWARD_HPP_INCLUDED
#define CATCH_MOVE_AND_FORWARD_HPP_INCLUDED

#include <type_traits>

//! Replacement for std::move with better compile time performance
#define CATCH_MOVE(...) static_cast<std::remove_reference_t<decltype(__VA_ARGS__)>&&>(__VA_ARGS__)

//! Replacement for std::forward with better compile time performance
#define CATCH_FORWARD(...) static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)

#endif // CATCH_MOVE_AND_FORWARD_HPP_INCLUDED
