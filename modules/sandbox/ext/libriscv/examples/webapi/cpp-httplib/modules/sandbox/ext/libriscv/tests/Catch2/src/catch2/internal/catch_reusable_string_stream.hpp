
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REUSABLE_STRING_STREAM_HPP_INCLUDED
#define CATCH_REUSABLE_STRING_STREAM_HPP_INCLUDED

#include <catch2/internal/catch_noncopyable.hpp>

#include <iosfwd>
#include <cstddef>
#include <ostream>
#include <string>

namespace Catch {

    class ReusableStringStream : Detail::NonCopyable {
        std::size_t m_index;
        std::ostream* m_oss;
    public:
        ReusableStringStream();
        ~ReusableStringStream();

        //! Returns the serialized state
        std::string str() const;
        //! Sets internal state to `str`
        void str(std::string const& str);

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
// Old versions of GCC do not understand -Wnonnull-compare
#pragma GCC diagnostic ignored "-Wpragmas"
// Streaming a function pointer triggers Waddress and Wnonnull-compare
// on GCC, because it implicitly converts it to bool and then decides
// that the check it uses (a? true : false) is tautological and cannot
// be null...
#pragma GCC diagnostic ignored "-Waddress"
#pragma GCC diagnostic ignored "-Wnonnull-compare"
#endif

        template<typename T>
        auto operator << ( T const& value ) -> ReusableStringStream& {
            *m_oss << value;
            return *this;
        }

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
        auto get() -> std::ostream& { return *m_oss; }
    };
}

#endif // CATCH_REUSABLE_STRING_STREAM_HPP_INCLUDED
