
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_ERRNO_GUARD_HPP_INCLUDED
#define CATCH_ERRNO_GUARD_HPP_INCLUDED

namespace Catch {

    //! Simple RAII class that stores the value of `errno`
    //! at construction and restores it at destruction.
    class ErrnoGuard {
    public:
        // Keep these outlined to avoid dragging in macros from <cerrno>

        ErrnoGuard();
        ~ErrnoGuard();
    private:
        int m_oldErrno;
    };

}

#endif // CATCH_ERRNO_GUARD_HPP_INCLUDED
