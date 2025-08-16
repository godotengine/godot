
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_OUTPUT_REDIRECT_HPP_INCLUDED
#define CATCH_OUTPUT_REDIRECT_HPP_INCLUDED

#include <catch2/internal/catch_unique_ptr.hpp>

#include <cassert>
#include <string>

namespace Catch {

    class OutputRedirect {
        bool m_redirectActive = false;
        virtual void activateImpl() = 0;
        virtual void deactivateImpl() = 0;
    public:
        enum Kind {
            //! No redirect (noop implementation)
            None,
            //! Redirect std::cout/std::cerr/std::clog streams internally
            Streams,
            //! Redirect the stdout/stderr file descriptors into files
            FileDescriptors,
        };

        virtual ~OutputRedirect(); // = default;

        // TODO: Do we want to check that redirect is not active before retrieving the output?
        virtual std::string getStdout() = 0;
        virtual std::string getStderr() = 0;
        virtual void clearBuffers() = 0;
        bool isActive() const { return m_redirectActive; }
        void activate() {
            assert( !m_redirectActive && "redirect is already active" );
            activateImpl();
            m_redirectActive = true;
        }
        void deactivate() {
            assert( m_redirectActive && "redirect is not active" );
            deactivateImpl();
            m_redirectActive = false;
        }
    };

    bool isRedirectAvailable( OutputRedirect::Kind kind);
    Detail::unique_ptr<OutputRedirect> makeOutputRedirect( bool actual );

    class RedirectGuard {
        OutputRedirect* m_redirect;
        bool m_activate;
        bool m_previouslyActive;
        bool m_moved = false;

    public:
        RedirectGuard( bool activate, OutputRedirect& redirectImpl );
        ~RedirectGuard() noexcept( false );

        RedirectGuard( RedirectGuard const& ) = delete;
        RedirectGuard& operator=( RedirectGuard const& ) = delete;

        // C++14 needs move-able guards to return them from functions
        RedirectGuard( RedirectGuard&& rhs ) noexcept;
        RedirectGuard& operator=( RedirectGuard&& rhs ) noexcept;
    };

    RedirectGuard scopedActivate( OutputRedirect& redirectImpl );
    RedirectGuard scopedDeactivate( OutputRedirect& redirectImpl );

} // end namespace Catch

#endif // CATCH_OUTPUT_REDIRECT_HPP_INCLUDED
