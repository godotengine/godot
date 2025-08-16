
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/catch_version.hpp>
#include <ostream>

namespace Catch {

    Version::Version
        (   unsigned int _majorVersion,
            unsigned int _minorVersion,
            unsigned int _patchNumber,
            char const * const _branchName,
            unsigned int _buildNumber )
    :   majorVersion( _majorVersion ),
        minorVersion( _minorVersion ),
        patchNumber( _patchNumber ),
        branchName( _branchName ),
        buildNumber( _buildNumber )
    {}

    std::ostream& operator << ( std::ostream& os, Version const& version ) {
        os  << version.majorVersion << '.'
            << version.minorVersion << '.'
            << version.patchNumber;
        // branchName is never null -> 0th char is \0 if it is empty
        if (version.branchName[0]) {
            os << '-' << version.branchName
               << '.' << version.buildNumber;
        }
        return os;
    }

    Version const& libraryVersion() {
        static Version version( 3, 9, 1, "", 0 );
        return version;
    }

}
