
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/matchers/catch_matchers_templated.hpp>

namespace Catch {
namespace Matchers {
    MatcherGenericBase::~MatcherGenericBase() = default;

    namespace Detail {

        std::string describe_multi_matcher(StringRef combine, std::string const* descriptions_begin, std::string const* descriptions_end) {
            std::string description;
            std::size_t combined_size = 4;
            for ( auto desc = descriptions_begin; desc != descriptions_end; ++desc ) {
                combined_size += desc->size();
            }
            combined_size += static_cast<size_t>(descriptions_end - descriptions_begin - 1) * combine.size();

            description.reserve(combined_size);

            description += "( ";
            bool first = true;
            for( auto desc = descriptions_begin; desc != descriptions_end; ++desc ) {
                if( first )
                    first = false;
                else
                    description += combine;
                description += *desc;
            }
            description += " )";
            return description;
        }

    } // namespace Detail
} // namespace Matchers
} // namespace Catch
