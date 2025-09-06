
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_INTERFACES_TAG_ALIAS_REGISTRY_HPP_INCLUDED
#define CATCH_INTERFACES_TAG_ALIAS_REGISTRY_HPP_INCLUDED

#include <string>

namespace Catch {

    struct TagAlias;

    class ITagAliasRegistry {
    public:
        virtual ~ITagAliasRegistry(); // = default
        // Nullptr if not present
        virtual TagAlias const* find( std::string const& alias ) const = 0;
        virtual std::string expandAliases( std::string const& unexpandedTestSpec ) const = 0;

        static ITagAliasRegistry const& get();
    };

} // end namespace Catch

#endif // CATCH_INTERFACES_TAG_ALIAS_REGISTRY_HPP_INCLUDED
