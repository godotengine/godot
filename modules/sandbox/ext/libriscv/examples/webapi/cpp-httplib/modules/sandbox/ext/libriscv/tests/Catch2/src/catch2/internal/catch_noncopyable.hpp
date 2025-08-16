
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_NONCOPYABLE_HPP_INCLUDED
#define CATCH_NONCOPYABLE_HPP_INCLUDED

namespace Catch {
    namespace Detail {

        //! Deriving classes become noncopyable and nonmovable
        class NonCopyable {
        public:
            NonCopyable( NonCopyable const& ) = delete;
            NonCopyable( NonCopyable&& ) = delete;
            NonCopyable& operator=( NonCopyable const& ) = delete;
            NonCopyable& operator=( NonCopyable&& ) = delete;

        protected:
            NonCopyable() noexcept = default;
        };

    } // namespace Detail
} // namespace Catch

#endif // CATCH_NONCOPYABLE_HPP_INCLUDED
