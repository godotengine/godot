// SPDX-License-Identifier: Apache 2.0
// Experimental JSON to USD converter

#include <string>

#include "tinyusdz.hh"

namespace tinyusdz {

///
/// Convert JSON string to USD Stage
///
///
bool JSONToUSD(const std::string &json_string, tinyusdz::Stage *stage, std::string *warn, std::string *err);

} // namespace tinyusd
