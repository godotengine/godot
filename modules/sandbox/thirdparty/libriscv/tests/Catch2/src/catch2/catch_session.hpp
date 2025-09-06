
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_SESSION_HPP_INCLUDED
#define CATCH_SESSION_HPP_INCLUDED

#include <catch2/internal/catch_commandline.hpp>
#include <catch2/internal/catch_noncopyable.hpp>
#include <catch2/catch_config.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>
#include <catch2/internal/catch_config_wchar.hpp>

namespace Catch {

    // TODO: Use C++17 `inline` variables
    constexpr int UnspecifiedErrorExitCode = 1;
    constexpr int NoTestsRunExitCode = 2;
    constexpr int UnmatchedTestSpecExitCode = 3;
    constexpr int AllTestsSkippedExitCode = 4;
    constexpr int InvalidTestSpecExitCode = 5;
    constexpr int TestFailureExitCode = 42;

    class Session : Detail::NonCopyable {
    public:

        Session();
        ~Session();

        void showHelp() const;
        void libIdentify();

        int applyCommandLine( int argc, char const * const * argv );
    #if defined(CATCH_CONFIG_WCHAR) && defined(_WIN32) && defined(UNICODE)
        int applyCommandLine( int argc, wchar_t const * const * argv );
    #endif

        void useConfigData( ConfigData const& configData );

        template<typename CharT>
        int run(int argc, CharT const * const argv[]) {
            if (m_startupExceptions)
                return 1;
            int returnCode = applyCommandLine(argc, argv);
            if (returnCode == 0)
                returnCode = run();
            return returnCode;
        }

        int run();

        Clara::Parser const& cli() const;
        void cli( Clara::Parser const& newParser );
        ConfigData& configData();
        Config& config();
    private:
        int runInternal();

        Clara::Parser m_cli;
        ConfigData m_configData;
        Detail::unique_ptr<Config> m_config;
        bool m_startupExceptions = false;
    };

} // end namespace Catch

#endif // CATCH_SESSION_HPP_INCLUDED
