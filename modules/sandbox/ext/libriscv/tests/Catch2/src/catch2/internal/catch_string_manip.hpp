
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_STRING_MANIP_HPP_INCLUDED
#define CATCH_STRING_MANIP_HPP_INCLUDED

#include <catch2/internal/catch_stringref.hpp>

#include <cstdint>
#include <string>
#include <iosfwd>
#include <vector>

namespace Catch {

    bool startsWith( std::string const& s, std::string const& prefix );
    bool startsWith( StringRef s, char prefix );
    bool endsWith( std::string const& s, std::string const& suffix );
    bool endsWith( std::string const& s, char suffix );
    bool contains( std::string const& s, std::string const& infix );
    void toLowerInPlace( std::string& s );
    std::string toLower( std::string const& s );
    char toLower( char c );
    //! Returns a new string without whitespace at the start/end
    std::string trim( std::string const& str );
    //! Returns a substring of the original ref without whitespace. Beware lifetimes!
    StringRef trim(StringRef ref);

    // !!! Be aware, returns refs into original string - make sure original string outlives them
    std::vector<StringRef> splitStringRef( StringRef str, char delimiter );
    bool replaceInPlace( std::string& str, std::string const& replaceThis, std::string const& withThis );

    /**
     * Helper for streaming a "count [maybe-plural-of-label]" human-friendly string
     *
     * Usage example:
     * ```cpp
     * std::cout << "Found " << pluralise(count, "error") << '\n';
     * ```
     *
     * **Important:** The provided string must outlive the instance
     */
    class pluralise {
        std::uint64_t m_count;
        StringRef m_label;

    public:
        constexpr pluralise(std::uint64_t count, StringRef label):
            m_count(count),
            m_label(label)
        {}

        friend std::ostream& operator << ( std::ostream& os, pluralise const& pluraliser );
    };
}

#endif // CATCH_STRING_MANIP_HPP_INCLUDED
