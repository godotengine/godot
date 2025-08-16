
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/internal/catch_reporter_spec_parser.hpp>

#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

#include <algorithm>

namespace Catch {

    namespace {
        struct kvPair {
            StringRef key, value;
        };

        kvPair splitKVPair(StringRef kvString) {
            auto splitPos = static_cast<size_t>(
                std::find( kvString.begin(), kvString.end(), '=' ) -
                kvString.begin() );

            return { kvString.substr( 0, splitPos ),
                     kvString.substr( splitPos + 1, kvString.size() ) };
        }
    }

    namespace Detail {
        std::vector<std::string> splitReporterSpec( StringRef reporterSpec ) {
            static constexpr auto separator = "::";
            static constexpr size_t separatorSize = 2;

            size_t separatorPos = 0;
            auto findNextSeparator = [&reporterSpec]( size_t startPos ) {
                static_assert(
                    separatorSize == 2,
                    "The code below currently assumes 2 char separator" );

                auto currentPos = startPos;
                do {
                    while ( currentPos < reporterSpec.size() &&
                            reporterSpec[currentPos] != separator[0] ) {
                        ++currentPos;
                    }
                    if ( currentPos + 1 < reporterSpec.size() &&
                         reporterSpec[currentPos + 1] == separator[1] ) {
                        return currentPos;
                    }
                    ++currentPos;
                } while ( currentPos < reporterSpec.size() );

                return static_cast<size_t>( -1 );
            };

            std::vector<std::string> parts;

            while ( separatorPos < reporterSpec.size() ) {
                const auto nextSeparator = findNextSeparator( separatorPos );
                parts.push_back( static_cast<std::string>( reporterSpec.substr(
                    separatorPos, nextSeparator - separatorPos ) ) );

                if ( nextSeparator == static_cast<size_t>( -1 ) ) {
                    break;
                }
                separatorPos = nextSeparator + separatorSize;
            }

            // Handle a separator at the end.
            // This is not a valid spec, but we want to do validation in a
            // centralized place
            if ( separatorPos == reporterSpec.size() ) {
                parts.emplace_back();
            }

            return parts;
        }

        Optional<ColourMode> stringToColourMode( StringRef colourMode ) {
            if ( colourMode == "default" ) {
                return ColourMode::PlatformDefault;
            } else if ( colourMode == "ansi" ) {
                return ColourMode::ANSI;
            } else if ( colourMode == "win32" ) {
                return ColourMode::Win32;
            } else if ( colourMode == "none" ) {
                return ColourMode::None;
            } else {
                return {};
            }
        }
    } // namespace Detail


    bool operator==( ReporterSpec const& lhs, ReporterSpec const& rhs ) {
        return lhs.m_name == rhs.m_name &&
               lhs.m_outputFileName == rhs.m_outputFileName &&
               lhs.m_colourMode == rhs.m_colourMode &&
               lhs.m_customOptions == rhs.m_customOptions;
    }

    Optional<ReporterSpec> parseReporterSpec( StringRef reporterSpec ) {
        auto parts = Detail::splitReporterSpec( reporterSpec );

        assert( parts.size() > 0 && "Split should never return empty vector" );

        std::map<std::string, std::string> kvPairs;
        Optional<std::string> outputFileName;
        Optional<ColourMode> colourMode;

        // First part is always reporter name, so we skip it
        for ( size_t i = 1; i < parts.size(); ++i ) {
            auto kv = splitKVPair( parts[i] );
            auto key = kv.key, value = kv.value;

            if ( key.empty() || value.empty() ) { // NOLINT(bugprone-branch-clone)
                return {};
            } else if ( key[0] == 'X' ) {
                // This is a reporter-specific option, we don't check these
                // apart from basic sanity checks
                if ( key.size() == 1 ) {
                    return {};
                }

                auto ret = kvPairs.emplace( std::string(kv.key), std::string(kv.value) );
                if ( !ret.second ) {
                    // Duplicated key. We might want to handle this differently,
                    // e.g. by overwriting the existing value?
                    return {};
                }
            } else if ( key == "out" ) {
                // Duplicated key
                if ( outputFileName ) {
                    return {};
                }
                outputFileName = static_cast<std::string>( value );
            } else if ( key == "colour-mode" ) {
                // Duplicated key
                if ( colourMode ) {
                    return {};
                }
                colourMode = Detail::stringToColourMode( value );
                // Parsing failed
                if ( !colourMode ) {
                    return {};
                }
            } else {
                // Unrecognized option
                return {};
            }
        }

        return ReporterSpec{ CATCH_MOVE( parts[0] ),
                             CATCH_MOVE( outputFileName ),
                             CATCH_MOVE( colourMode ),
                             CATCH_MOVE( kvPairs ) };
    }

ReporterSpec::ReporterSpec(
        std::string name,
        Optional<std::string> outputFileName,
        Optional<ColourMode> colourMode,
        std::map<std::string, std::string> customOptions ):
        m_name( CATCH_MOVE( name ) ),
        m_outputFileName( CATCH_MOVE( outputFileName ) ),
        m_colourMode( CATCH_MOVE( colourMode ) ),
        m_customOptions( CATCH_MOVE( customOptions ) ) {}

} // namespace Catch
