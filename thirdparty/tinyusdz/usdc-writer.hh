// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
#pragma once

#include "tinyusdz.hh"

namespace tinyusdz {
namespace usdc {

///
/// Save scene as USDC(binary) to a file
///
/// @param[in] filename USDC filename
/// @param[in] stage Stage
/// @param[out] warn Warning message
/// @param[out] err Error message
///
/// @return true upon success.
///
bool SaveAsUSDCToFile(const std::string &filename, const Stage &stage,
                      std::string *warn, std::string *err);

///
/// Save scene as USDC(binary) to a memory
///
/// @param[in] stage Stage
/// @param[out] output Binary data
/// @param[out] warn Warning message
/// @param[out] err Error message
///
/// @return true upon success.
///
bool SaveAsUSDCToMemory(const Stage &stage, std::vector<uint8_t> *output,
                        std::string *warn, std::string *err);

}  // namespace usdc
}  // namespace tinyusdz
