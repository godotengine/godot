
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/catch_test_case_info.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/internal/catch_string_manip.hpp>
#include <catch2/internal/catch_case_insensitive_comparisons.hpp>
#include <catch2/internal/catch_test_registry.hpp>

#include <cassert>
#include <cctype>
#include <algorithm>

namespace Catch {

    namespace {
        using TCP_underlying_type = uint8_t;
        static_assert(sizeof(TestCaseProperties) == sizeof(TCP_underlying_type),
                      "The size of the TestCaseProperties is different from the assumed size");

        constexpr TestCaseProperties operator|(TestCaseProperties lhs, TestCaseProperties rhs) {
            return static_cast<TestCaseProperties>(
                static_cast<TCP_underlying_type>(lhs) | static_cast<TCP_underlying_type>(rhs)
            );
        }

        constexpr TestCaseProperties& operator|=(TestCaseProperties& lhs, TestCaseProperties rhs) {
            lhs = static_cast<TestCaseProperties>(
                static_cast<TCP_underlying_type>(lhs) | static_cast<TCP_underlying_type>(rhs)
            );
            return lhs;
        }

        constexpr TestCaseProperties operator&(TestCaseProperties lhs, TestCaseProperties rhs) {
            return static_cast<TestCaseProperties>(
                static_cast<TCP_underlying_type>(lhs) & static_cast<TCP_underlying_type>(rhs)
            );
        }

        constexpr bool applies(TestCaseProperties tcp) {
            static_assert(static_cast<TCP_underlying_type>(TestCaseProperties::None) == 0,
                          "TestCaseProperties::None must be equal to 0");
            return tcp != TestCaseProperties::None;
        }

        TestCaseProperties parseSpecialTag( StringRef tag ) {
            if( !tag.empty() && tag[0] == '.' )
                return TestCaseProperties::IsHidden;
            else if( tag == "!throws"_sr )
                return TestCaseProperties::Throws;
            else if( tag == "!shouldfail"_sr )
                return TestCaseProperties::ShouldFail;
            else if( tag == "!mayfail"_sr )
                return TestCaseProperties::MayFail;
            else if( tag == "!nonportable"_sr )
                return TestCaseProperties::NonPortable;
            else if( tag == "!benchmark"_sr )
                return TestCaseProperties::Benchmark | TestCaseProperties::IsHidden;
            else
                return TestCaseProperties::None;
        }
        bool isReservedTag( StringRef tag ) {
            return parseSpecialTag( tag ) == TestCaseProperties::None
                && tag.size() > 0
                && !std::isalnum( static_cast<unsigned char>(tag[0]) );
        }
        void enforceNotReservedTag( StringRef tag, SourceLineInfo const& _lineInfo ) {
            CATCH_ENFORCE( !isReservedTag(tag),
                          "Tag name: [" << tag << "] is not allowed.\n"
                          << "Tag names starting with non alphanumeric characters are reserved\n"
                          << _lineInfo );
        }

        std::string makeDefaultName() {
            static size_t counter = 0;
            return "Anonymous test case " + std::to_string(++counter);
        }

        constexpr StringRef extractFilenamePart(StringRef filename) {
            size_t lastDot = filename.size();
            while (lastDot > 0 && filename[lastDot - 1] != '.') {
                --lastDot;
            }
            // In theory we could have filename without any extension in it
            if ( lastDot == 0 ) { return StringRef(); }

            --lastDot;
            size_t nameStart = lastDot;
            while (nameStart > 0 && filename[nameStart - 1] != '/' && filename[nameStart - 1] != '\\') {
                --nameStart;
            }

            return filename.substr(nameStart, lastDot - nameStart);
        }

        // Returns the upper bound on size of extra tags ([#file]+[.])
        constexpr size_t sizeOfExtraTags(StringRef filepath) {
            // [.] is 3, [#] is another 3
            const size_t extras = 3 + 3;
            return extractFilenamePart(filepath).size() + extras;
        }
    } // end unnamed namespace

    bool operator<(  Tag const& lhs, Tag const& rhs ) {
        Detail::CaseInsensitiveLess cmp;
        return cmp( lhs.original, rhs.original );
    }
    bool operator==( Tag const& lhs, Tag const& rhs ) {
        Detail::CaseInsensitiveEqualTo cmp;
        return cmp( lhs.original, rhs.original );
    }

    Detail::unique_ptr<TestCaseInfo>
        makeTestCaseInfo(StringRef _className,
                         NameAndTags const& nameAndTags,
                         SourceLineInfo const& _lineInfo ) {
        return Detail::make_unique<TestCaseInfo>(_className, nameAndTags, _lineInfo);
    }

    TestCaseInfo::TestCaseInfo(StringRef _className,
                               NameAndTags const& _nameAndTags,
                               SourceLineInfo const& _lineInfo):
        name( _nameAndTags.name.empty() ? makeDefaultName() : _nameAndTags.name ),
        className( _className ),
        lineInfo( _lineInfo )
    {
        StringRef originalTags = _nameAndTags.tags;
        // We need to reserve enough space to store all of the tags
        // (including optional hidden tag and filename tag)
        auto requiredSize = originalTags.size() + sizeOfExtraTags(_lineInfo.file);
        backingTags.reserve(requiredSize);

        // We cannot copy the tags directly, as we need to normalize
        // some tags, so that [.foo] is copied as [.][foo].
        size_t tagStart = 0;
        size_t tagEnd = 0;
        bool inTag = false;
        for (size_t idx = 0; idx < originalTags.size(); ++idx) {
            auto c = originalTags[idx];
            if (c == '[') {
                CATCH_ENFORCE(
                    !inTag,
                    "Found '[' inside a tag while registering test case '"
                        << _nameAndTags.name << "' at " << _lineInfo );

                inTag = true;
                tagStart = idx;
            }
            if (c == ']') {
                CATCH_ENFORCE(
                    inTag,
                    "Found unmatched ']' while registering test case '"
                        << _nameAndTags.name << "' at " << _lineInfo );

                inTag = false;
                tagEnd = idx;
                assert(tagStart < tagEnd);

                // We need to check the tag for special meanings, copy
                // it over to backing storage and actually reference the
                // backing storage in the saved tags
                StringRef tagStr = originalTags.substr(tagStart+1, tagEnd - tagStart - 1);
                CATCH_ENFORCE( !tagStr.empty(),
                               "Found an empty tag while registering test case '"
                                   << _nameAndTags.name << "' at "
                                   << _lineInfo );

                enforceNotReservedTag(tagStr, lineInfo);
                properties |= parseSpecialTag(tagStr);
                // When copying a tag to the backing storage, we need to
                // check if it is a merged hide tag, such as [.foo], and
                // if it is, we need to handle it as if it was [foo].
                if (tagStr.size() > 1 && tagStr[0] == '.') {
                    tagStr = tagStr.substr(1, tagStr.size() - 1);
                }
                // We skip over dealing with the [.] tag, as we will add
                // it later unconditionally and then sort and unique all
                // the tags.
                internalAppendTag(tagStr);
            }
        }
        CATCH_ENFORCE( !inTag,
                       "Found an unclosed tag while registering test case '"
                           << _nameAndTags.name << "' at " << _lineInfo );


        // Add [.] if relevant
        if (isHidden()) {
            internalAppendTag("."_sr);
        }

        // Sort and prepare tags
        std::sort(begin(tags), end(tags));
        tags.erase(std::unique(begin(tags), end(tags)),
                   end(tags));
    }

    bool TestCaseInfo::isHidden() const {
        return applies( properties & TestCaseProperties::IsHidden );
    }
    bool TestCaseInfo::throws() const {
        return applies( properties & TestCaseProperties::Throws );
    }
    bool TestCaseInfo::okToFail() const {
        return applies( properties & (TestCaseProperties::ShouldFail | TestCaseProperties::MayFail ) );
    }
    bool TestCaseInfo::expectedToFail() const {
        return applies( properties & (TestCaseProperties::ShouldFail) );
    }

    void TestCaseInfo::addFilenameTag() {
        std::string combined("#");
        combined += extractFilenamePart(lineInfo.file);
        internalAppendTag(combined);
    }

    std::string TestCaseInfo::tagsAsString() const {
        std::string ret;
        // '[' and ']' per tag
        std::size_t full_size = 2 * tags.size();
        for (const auto& tag : tags) {
            full_size += tag.original.size();
        }
        ret.reserve(full_size);
        for (const auto& tag : tags) {
            ret.push_back('[');
            ret += tag.original;
            ret.push_back(']');
        }

        return ret;
    }

    void TestCaseInfo::internalAppendTag(StringRef tagStr) {
        backingTags += '[';
        const auto backingStart = backingTags.size();
        backingTags += tagStr;
        const auto backingEnd = backingTags.size();
        backingTags += ']';
        tags.emplace_back(StringRef(backingTags.c_str() + backingStart, backingEnd - backingStart));
    }

    bool operator<( TestCaseInfo const& lhs, TestCaseInfo const& rhs ) {
        // We want to avoid redoing the string comparisons multiple times,
        // so we store the result of a three-way comparison before using
        // it in the actual comparison logic.
        const auto cmpName = lhs.name.compare( rhs.name );
        if ( cmpName != 0 ) {
            return cmpName < 0;
        }
        const auto cmpClassName = lhs.className.compare( rhs.className );
        if ( cmpClassName != 0 ) {
            return cmpClassName < 0;
        }
        return lhs.tags < rhs.tags;
    }

} // end namespace Catch
