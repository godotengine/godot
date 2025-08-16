
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REPORTER_HELPERS_HPP_INCLUDED
#define CATCH_REPORTER_HELPERS_HPP_INCLUDED

#include <iosfwd>
#include <string>
#include <vector>

#include <catch2/internal/catch_list.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/catch_totals.hpp>

namespace Catch {

    class IConfig;
    class TestCaseHandle;
    class ColourImpl;

    // Returns double formatted as %.3f (format expected on output)
    std::string getFormattedDuration( double duration );

    //! Should the reporter show duration of test given current configuration?
    bool shouldShowDuration( IConfig const& config, double duration );

    std::string serializeFilters( std::vector<std::string> const& filters );

    struct lineOfChars {
        char c;
        constexpr lineOfChars( char c_ ): c( c_ ) {}

        friend std::ostream& operator<<( std::ostream& out, lineOfChars value );
    };

    /**
     * Lists reporter descriptions to the provided stream in user-friendly
     * format
     *
     * Used as the default listing implementation by the first party reporter
     * bases. The output should be backwards compatible with the output of
     * Catch2 v2 binaries.
     */
    void
    defaultListReporters( std::ostream& out,
                          std::vector<ReporterDescription> const& descriptions,
                          Verbosity verbosity );

    /**
     * Lists listeners descriptions to the provided stream in user-friendly
     * format
     */
    void defaultListListeners( std::ostream& out,
                               std::vector<ListenerDescription> const& descriptions );

    /**
     * Lists tag information to the provided stream in user-friendly format
     *
     * Used as the default listing implementation by the first party reporter
     * bases. The output should be backwards compatible with the output of
     * Catch2 v2 binaries.
     */
    void defaultListTags( std::ostream& out, std::vector<TagInfo> const& tags, bool isFiltered );

    /**
     * Lists test case information to the provided stream in user-friendly
     * format
     *
     * Used as the default listing implementation by the first party reporter
     * bases. The output is backwards compatible with the output of Catch2
     * v2 binaries, and also supports the format specific to the old
     * `--list-test-names-only` option, for people who used it in integrations.
     */
    void defaultListTests( std::ostream& out,
                           ColourImpl* streamColour,
                           std::vector<TestCaseHandle> const& tests,
                           bool isFiltered,
                           Verbosity verbosity );

    /**
     * Prints test run totals to the provided stream in user-friendly format
     *
     * Used by the console and compact reporters.
     */
    void printTestRunTotals( std::ostream& stream,
                      ColourImpl& streamColour,
                      Totals const& totals );

} // end namespace Catch

#endif // CATCH_REPORTER_HELPERS_HPP_INCLUDED
