
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/internal/catch_leak_detector.hpp>
#include <catch2/interfaces/catch_interfaces_registry_hub.hpp>
#include <catch2/catch_user_config.hpp>

#ifdef CATCH_CONFIG_WINDOWS_CRTDBG
#include <crtdbg.h>

namespace Catch {

    LeakDetector::LeakDetector() {
        int flag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
        flag |= _CRTDBG_LEAK_CHECK_DF;
        flag |= _CRTDBG_ALLOC_MEM_DF;
        _CrtSetDbgFlag(flag);
        _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
        _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
        // Change this to leaking allocation's number to break there
        _CrtSetBreakAlloc(-1);
    }
}

#else // ^^ Windows crt debug heap enabled // Windows crt debug heap disabled vv

    Catch::LeakDetector::LeakDetector() = default;

#endif // CATCH_CONFIG_WINDOWS_CRTDBG

Catch::LeakDetector::~LeakDetector() {
    Catch::cleanUp();
}
