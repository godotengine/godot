
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TAG_ALIAS_REGISTRY_HPP_INCLUDED
#define CATCH_TAG_ALIAS_REGISTRY_HPP_INCLUDED

#include <catch2/interfaces/catch_interfaces_tag_alias_registry.hpp>
#include <catch2/catch_tag_alias.hpp>

#include <map>
#include <string>

namespace Catch {
    struct SourceLineInfo;

    class TagAliasRegistry : public ITagAliasRegistry {
    public:
        ~TagAliasRegistry() override;
        TagAlias const* find( std::string const& alias ) const override;
        std::string expandAliases( std::string const& unexpandedTestSpec ) const override;
        void add( std::string const& alias, std::string const& tag, SourceLineInfo const& lineInfo );

    private:
        std::map<std::string, TagAlias> m_registry;
    };

} // end namespace Catch

#endif // CATCH_TAG_ALIAS_REGISTRY_HPP_INCLUDED
