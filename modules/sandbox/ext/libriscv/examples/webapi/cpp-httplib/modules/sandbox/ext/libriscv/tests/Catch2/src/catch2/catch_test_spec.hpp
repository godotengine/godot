
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TEST_SPEC_HPP_INCLUDED
#define CATCH_TEST_SPEC_HPP_INCLUDED

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

#include <catch2/internal/catch_unique_ptr.hpp>
#include <catch2/internal/catch_wildcard_pattern.hpp>

#include <iosfwd>
#include <string>
#include <vector>

namespace Catch {

    class IConfig;
    struct TestCaseInfo;
    class TestCaseHandle;

    class TestSpec {

        class Pattern {
        public:
            explicit Pattern( std::string const& name );
            virtual ~Pattern();
            virtual bool matches( TestCaseInfo const& testCase ) const = 0;
            std::string const& name() const;
        private:
            virtual void serializeTo( std::ostream& out ) const = 0;
            // Writes string that would be reparsed into the pattern
            friend std::ostream& operator<<(std::ostream& out,
                                            Pattern const& pattern) {
                pattern.serializeTo( out );
                return out;
            }

            std::string const m_name;
        };

        class NamePattern : public Pattern {
        public:
            explicit NamePattern( std::string const& name, std::string const& filterString );
            bool matches( TestCaseInfo const& testCase ) const override;
        private:
            void serializeTo( std::ostream& out ) const override;

            WildcardPattern m_wildcardPattern;
        };

        class TagPattern : public Pattern {
        public:
            explicit TagPattern( std::string const& tag, std::string const& filterString );
            bool matches( TestCaseInfo const& testCase ) const override;
        private:
            void serializeTo( std::ostream& out ) const override;

            std::string m_tag;
        };

        struct Filter {
            std::vector<Detail::unique_ptr<Pattern>> m_required;
            std::vector<Detail::unique_ptr<Pattern>> m_forbidden;

            //! Serializes this filter into a string that would be parsed into
            //! an equivalent filter
            void serializeTo( std::ostream& out ) const;
            friend std::ostream& operator<<(std::ostream& out, Filter const& f) {
                f.serializeTo( out );
                return out;
            }

            bool matches( TestCaseInfo const& testCase ) const;
        };

        static std::string extractFilterName( Filter const& filter );

    public:
        struct FilterMatch {
            std::string name;
            std::vector<TestCaseHandle const*> tests;
        };
        using Matches = std::vector<FilterMatch>;
        using vectorStrings = std::vector<std::string>;

        bool hasFilters() const;
        bool matches( TestCaseInfo const& testCase ) const;
        Matches matchesByFilter( std::vector<TestCaseHandle> const& testCases, IConfig const& config ) const;
        const vectorStrings & getInvalidSpecs() const;

    private:
        std::vector<Filter> m_filters;
        std::vector<std::string> m_invalidSpecs;

        friend class TestSpecParser;
        //! Serializes this test spec into a string that would be parsed into
        //! equivalent test spec
        void serializeTo( std::ostream& out ) const;
        friend std::ostream& operator<<(std::ostream& out,
                                        TestSpec const& spec) {
            spec.serializeTo( out );
            return out;
        }
    };
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif // CATCH_TEST_SPEC_HPP_INCLUDED
