
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_REPORTER_SPEC_PARSER_HPP_INCLUDED
#define CATCH_REPORTER_SPEC_PARSER_HPP_INCLUDED

#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/internal/catch_optional.hpp>
#include <catch2/internal/catch_stringref.hpp>

#include <map>
#include <string>
#include <vector>

namespace Catch {

    enum class ColourMode : std::uint8_t;

    namespace Detail {
        //! Splits the reporter spec into reporter name and kv-pair options
        std::vector<std::string> splitReporterSpec( StringRef reporterSpec );

        Optional<ColourMode> stringToColourMode( StringRef colourMode );
    }

    /**
     * Structured reporter spec that a reporter can be created from
     *
     * Parsing has been validated, but semantics have not. This means e.g.
     * that the colour mode is known to Catch2, but it might not be
     * compiled into the binary, and the output filename might not be
     * openable.
     */
    class ReporterSpec {
        std::string m_name;
        Optional<std::string> m_outputFileName;
        Optional<ColourMode> m_colourMode;
        std::map<std::string, std::string> m_customOptions;

        friend bool operator==( ReporterSpec const& lhs,
                                ReporterSpec const& rhs );
        friend bool operator!=( ReporterSpec const& lhs,
                                ReporterSpec const& rhs ) {
            return !( lhs == rhs );
        }

    public:
        ReporterSpec(
            std::string name,
            Optional<std::string> outputFileName,
            Optional<ColourMode> colourMode,
            std::map<std::string, std::string> customOptions );

        std::string const& name() const { return m_name; }

        Optional<std::string> const& outputFile() const {
            return m_outputFileName;
        }

        Optional<ColourMode> const& colourMode() const { return m_colourMode; }

        std::map<std::string, std::string> const& customOptions() const {
            return m_customOptions;
        }
    };

    /**
     * Parses provided reporter spec string into
     *
     * Returns empty optional on errors, e.g.
     *  * field that is not first and not a key+value pair
     *  * duplicated keys in kv pair
     *  * unknown catch reporter option
     *  * empty key/value in an custom kv pair
     *  * ...
     */
    Optional<ReporterSpec> parseReporterSpec( StringRef reporterSpec );

}

#endif // CATCH_REPORTER_SPEC_PARSER_HPP_INCLUDED
