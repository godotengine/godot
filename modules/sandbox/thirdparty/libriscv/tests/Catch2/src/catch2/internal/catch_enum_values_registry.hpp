
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_ENUM_VALUES_REGISTRY_HPP_INCLUDED
#define CATCH_ENUM_VALUES_REGISTRY_HPP_INCLUDED

#include <catch2/interfaces/catch_interfaces_enum_values_registry.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>
#include <catch2/internal/catch_stringref.hpp>

#include <vector>

namespace Catch {

    namespace Detail {

        Catch::Detail::unique_ptr<EnumInfo> makeEnumInfo( StringRef enumName, StringRef allValueNames, std::vector<int> const& values );

        class EnumValuesRegistry : public IMutableEnumValuesRegistry {

            std::vector<Catch::Detail::unique_ptr<EnumInfo>> m_enumInfos;

            EnumInfo const& registerEnum( StringRef enumName, StringRef allValueNames, std::vector<int> const& values) override;
        };

        std::vector<StringRef> parseEnums( StringRef enums );

    } // Detail

} // Catch

#endif // CATCH_ENUM_VALUES_REGISTRY_HPP_INCLUDED
