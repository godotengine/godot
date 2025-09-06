
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_CONTAINER_NONMEMBERS_HPP_INCLUDED
#define CATCH_CONTAINER_NONMEMBERS_HPP_INCLUDED

#include <catch2/internal/catch_compiler_capabilities.hpp>

#include <cstddef>
#include <initializer_list>

// We want a simple polyfill over `std::empty`, `std::size` and so on
// for C++14 or C++ libraries with incomplete support.
// We also have to handle that MSVC std lib will happily provide these
// under older standards.
#if defined(CATCH_CPP17_OR_GREATER) || defined(_MSC_VER)

// We are already using this header either way, so there shouldn't
// be much additional overhead in including it to get the feature
// test macros
#include <string>

#  if !defined(__cpp_lib_nonmember_container_access)
#      define CATCH_CONFIG_POLYFILL_NONMEMBER_CONTAINER_ACCESS
#  endif

#else
#define CATCH_CONFIG_POLYFILL_NONMEMBER_CONTAINER_ACCESS
#endif



namespace Catch {
namespace Detail {

#if defined(CATCH_CONFIG_POLYFILL_NONMEMBER_CONTAINER_ACCESS)
    template <typename Container>
    constexpr auto empty(Container const& cont) -> decltype(cont.empty()) {
        return cont.empty();
    }
    template <typename T, std::size_t N>
    constexpr bool empty(const T (&)[N]) noexcept {
        // GCC < 7 does not support the const T(&)[] parameter syntax
        // so we have to ignore the length explicitly
        (void)N;
        return false;
    }
    template <typename T>
    constexpr bool empty(std::initializer_list<T> list) noexcept {
        return list.size() > 0;
    }


    template <typename Container>
    constexpr auto size(Container const& cont) -> decltype(cont.size()) {
        return cont.size();
    }
    template <typename T, std::size_t N>
    constexpr std::size_t size(const T(&)[N]) noexcept {
        return N;
    }
#endif // CATCH_CONFIG_POLYFILL_NONMEMBER_CONTAINER_ACCESS

} // end namespace Detail
} // end namespace Catch



#endif // CATCH_CONTAINER_NONMEMBERS_HPP_INCLUDED
