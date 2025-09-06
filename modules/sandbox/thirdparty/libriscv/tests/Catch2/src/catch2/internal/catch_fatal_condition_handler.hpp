
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_FATAL_CONDITION_HANDLER_HPP_INCLUDED
#define CATCH_FATAL_CONDITION_HANDLER_HPP_INCLUDED

#include <cassert>

namespace Catch {

    /**
     * Wrapper for platform-specific fatal error (signals/SEH) handlers
     *
     * Tries to be cooperative with other handlers, and not step over
     * other handlers. This means that unknown structured exceptions
     * are passed on, previous signal handlers are called, and so on.
     *
     * Can only be instantiated once, and assumes that once a signal
     * is caught, the binary will end up terminating. Thus, there
     */
    class FatalConditionHandler {
        bool m_started = false;

        // Install/disengage implementation for specific platform.
        // Should be if-defed to work on current platform, can assume
        // engage-disengage 1:1 pairing.
        void engage_platform();
        void disengage_platform() noexcept;
    public:
        // Should also have platform-specific implementations as needed
        FatalConditionHandler();
        ~FatalConditionHandler();

        void engage() {
            assert(!m_started && "Handler cannot be installed twice.");
            m_started = true;
            engage_platform();
        }

        void disengage() noexcept {
            assert(m_started && "Handler cannot be uninstalled without being installed first");
            m_started = false;
            disengage_platform();
        }
    };

    //! Simple RAII guard for (dis)engaging the FatalConditionHandler
    class FatalConditionHandlerGuard {
        FatalConditionHandler* m_handler;
    public:
        FatalConditionHandlerGuard(FatalConditionHandler* handler):
            m_handler(handler) {
            m_handler->engage();
        }
        ~FatalConditionHandlerGuard() {
            m_handler->disengage();
        }
    };

} // end namespace Catch

#endif // CATCH_FATAL_CONDITION_HANDLER_HPP_INCLUDED
