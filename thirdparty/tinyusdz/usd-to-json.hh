// SPDX-License-Identifier: Apache 2.0
// Experimental USD to JSON converter

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include "nonstd/expected.hpp"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "tinyusdz.hh"

namespace tinyusdz {

///
/// Convert USD Stage to JSON
///
/// @returns JSON string or error message(std::string) when failed to convert.
///
nonstd::expected<std::string, std::string> ToJSON(const tinyusdz::Stage &stage);

} // namespace tinyusd
